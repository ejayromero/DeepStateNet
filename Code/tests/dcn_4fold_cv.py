'''
Script to train model on 50 subjects using 4-fold cross-validation,
training DeepConvNet on Microstates timeseries with subject-independent paradigm

'''

print('==================== Start of script dcn_indep_4fold.py! ===================')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

import mne
import random
import pickle

from braindecode.models import Deep4Net
from braindecode.classifier import EEGClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import KFold

# change directory go into Notebooks folder
if os.path.basename(os.getcwd()) != 'Notebooks':
    if os.path.basename(os.getcwd()) == 'lib':
        os.chdir(os.path.join(os.getcwd(), '..', 'Notebooks'))
    else:
        os.chdir(os.path.join(os.getcwd(), 'Notebooks'))
else:
    # if already in Notebooks folder, do nothing
    pass

from lib import my_functions as mf

# More explicit CUDA setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Specify GPU 0 explicitly
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Test if tensors are actually on GPU
test_tensor = torch.randn(10, 10).to(device)
print(f"Test tensor device: {test_tensor.device}")

# ---------------------------# Parameters ---------------------------
excluded_from_training = [2, 12, 14, 20, 22, 23, 30, 39, 46]
num_epochs = 100
type_of_subject = 'independent_clean_4fold'
n_folds = 4
# ---------------------------# Load files ---------------------------

data_path = '../Data/'
output_path = '../Output/ica_rest_all/'
do_all = False
n_subjects = 50
subject_list = list(range(n_subjects))
all_data, all_y = mf.load_all_data(subjects_list=None, do_all=do_all)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mf.set_seed(42)

# Remove excluded subjects from available subjects
available_subjects = [i for i in range(n_subjects) if i not in excluded_from_training]
print(f"Available subjects for cross-validation: {len(available_subjects)}")
print(f"Excluded subjects: {excluded_from_training}")

# ---------- 4-Fold Cross Validation Setup ----------
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_splits = list(kf.split(available_subjects))

output_file = os.path.join(output_path, f'{type_of_subject}_results_ica_rest_all.npy')

# Initialize results storage
if os.path.exists(output_file):
    results = np.load(output_file, allow_pickle=True).item()
    all_fold_results = results.get('fold_results', {})
else:
    all_fold_results = {}

# ---------- Loop Through 4 Folds ----------
for fold_idx, (train_val_indices, test_indices) in enumerate(fold_splits):
    fold_name = f"fold_{fold_idx}"
    
    if fold_name in all_fold_results:
        print(f"Skipping {fold_name} as it has already been processed.")
        continue
        
    print(f"\n\n==================== {fold_name.upper()} ====================")
    
    # Get actual subject IDs
    train_val_subjects = [available_subjects[i] for i in train_val_indices]
    test_subjects = [available_subjects[i] for i in test_indices]
    
    print(f"Train+Val subjects: {train_val_subjects} (n={len(train_val_subjects)})")
    print(f"Test subjects: {test_subjects} (n={len(test_subjects)})")
    
    # Further split train_val into train and validation (4 subjects for validation)
    np.random.seed(42 + fold_idx)  # Different seed for each fold
    val_subjects = np.random.choice(train_val_subjects, size=4, replace=False).tolist()
    train_subjects = [s for s in train_val_subjects if s not in val_subjects]
    
    print(f"Final train subjects: {train_subjects} (n={len(train_subjects)})")
    print(f"Final validation subjects: {val_subjects} (n={len(val_subjects)})")
    
    mf.set_seed(42)
    
    # Initialize fold-specific results
    fold_results = {
        'test_subjects': test_subjects,
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'subject_results': {}
    }
    
    # ---------- Loop Through Test Subjects in This Fold ----------
    for test_id in test_subjects:
        print(f"\n--- Processing Subject {test_id} in {fold_name} ---")
        
        # Concatenate training data
        x_train = torch.cat([torch.tensor(all_data[i], dtype=torch.float32).squeeze(1) for i in train_subjects], dim=0)
        y_train = torch.cat([torch.tensor(all_y[i], dtype=torch.long) for i in train_subjects], dim=0)

        # Concatenate validation data
        x_val = torch.cat([torch.tensor(all_data[i], dtype=torch.float32).squeeze(1) for i in val_subjects], dim=0)
        y_val = torch.cat([torch.tensor(all_y[i], dtype=torch.long) for i in val_subjects], dim=0)

        # Test subject
        x_test = torch.tensor(all_data[test_id], dtype=torch.float32).squeeze(1)
        y_test = torch.tensor(all_y[test_id], dtype=torch.long)

        print(f"Train data shape: {x_train.shape}, labels: {y_train.shape}")
        print(f"Val data shape: {x_val.shape}, labels: {y_val.shape}")
        print(f"Test data shape: {x_test.shape}, labels: {y_test.shape}")

        # ---------- DataLoaders ----------
        batch_size = 32
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

        # ---------- Model ----------
        base_model = Deep4Net(
            n_chans=x_train.shape[1],
            n_classes=len(torch.unique(y_train)),
            input_window_samples=x_train.shape[2],
            final_conv_length='auto'
        )

        model = EEGClassifier(
            base_model,
            criterion=nn.NLLLoss(),
            optimizer=torch.optim.Adam,
            optimizer__lr=1e-3,
            train_split=None,
            device=device
        )

        net = model.module.to(device)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        # ---------- Training ----------
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(num_epochs):
            net.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == batch_y).sum().item()
                train_total += batch_y.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total * 100
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # ---------- Validation ----------
            net.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = net(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total * 100
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1:02d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # ---------- Test ----------
        net.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = net(batch_x)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)

        test_acc = test_correct / test_total * 100
        print(f"✅ Subject {test_id} Test Accuracy: {test_acc:.2f}%")

        # Store results for this subject
        fold_results['subject_results'][test_id] = {
            'train_accuracies': train_accuracies,
            'train_losses': train_losses,
            'test_accuracy': test_acc,
            'val_accuracies': val_accuracies,
            'val_losses': val_losses,
            'model': model
        }

    # Store fold results
    all_fold_results[fold_name] = fold_results
    
    # Save intermediate results
    results = {'fold_results': all_fold_results}
    np.save(output_file, results)
    print(f"✅ {fold_name} completed and saved to {output_file}")

# ========================= MODEL TESTING =========================
print("\n==================== TESTING TRAINED MODELS ====================")

# Prepare model results in the expected format for the test function
# Create a list of models organized by subject_id (consistent with original script format)
all_models = []
all_train_accuracies = []
all_train_losses = []
all_test_accuracies = []
all_val_accuracies = []
all_val_losses = []

# Initialize lists for all subjects (including excluded ones as None)
for subject_id in range(n_subjects):
    all_models.append(None)
    all_train_accuracies.append(None)
    all_train_losses.append(None)
    all_test_accuracies.append(None)
    all_val_accuracies.append(None)
    all_val_losses.append(None)

# Fill in the results from the 4-fold CV
for fold_name, fold_data in all_fold_results.items():
    for subject_id, subject_results in fold_data['subject_results'].items():
        all_models[subject_id] = subject_results['model']
        all_train_accuracies[subject_id] = subject_results['train_accuracies']
        all_train_losses[subject_id] = subject_results['train_losses']
        all_test_accuracies[subject_id] = subject_results['test_accuracy']
        all_val_accuracies[subject_id] = subject_results['val_accuracies']
        all_val_losses[subject_id] = subject_results['val_losses']

# Create the results dictionary in the same format as original script
results = {
    'train_accuracies': all_train_accuracies,
    'train_losses': all_train_losses,
    'test_accuracies': all_test_accuracies,
    'val_accuracies': all_val_accuracies,
    'val_losses': all_val_losses,
    'models': all_models
}

# Call the unified testing function
test_results = mf.test_model_results(
    model_results=results,
    data=all_data,
    labels=all_y,
    type_of_subject='independent',
    output_path=output_path,
    model_type='single',
    device=device,
    batch_size=32
)

print(f"✅ Model testing completed. Results saved to test output files.")

# ------------------------- Analysis and Plotting -------------------------

print("\n==================== ANALYSIS ACROSS ALL FOLDS ====================")

# Collect training results across folds for plotting
all_test_accuracies_for_plotting = []
all_train_accuracies_per_epoch = []
all_val_accuracies_per_epoch = []
all_train_losses_per_epoch = []
all_val_losses_per_epoch = []

for fold_name, fold_data in all_fold_results.items():
    print(f"\n{fold_name.upper()} Results:")
    fold_test_accs = []
    fold_train_accs = []
    fold_val_accs = []
    fold_train_losses = []
    fold_val_losses = []
    
    for subject_id, subject_results in fold_data['subject_results'].items():
        test_acc = subject_results['test_accuracy']
        fold_test_accs.append(test_acc)
        fold_train_accs.append(subject_results['train_accuracies'])
        fold_val_accs.append(subject_results['val_accuracies'])
        fold_train_losses.append(subject_results['train_losses'])
        fold_val_losses.append(subject_results['val_losses'])
        print(f"  Subject {subject_id}: {test_acc:.2f}%")
    
    print(f"  {fold_name} mean test accuracy: {np.mean(fold_test_accs):.2f}% ± {np.std(fold_test_accs):.2f}%")
    
    all_test_accuracies_for_plotting.extend(fold_test_accs)
    all_train_accuracies_per_epoch.extend(fold_train_accs)
    all_val_accuracies_per_epoch.extend(fold_val_accs)
    all_train_losses_per_epoch.extend(fold_train_losses)
    all_val_losses_per_epoch.extend(fold_val_losses)

# Overall statistics
print(f"\n==================== OVERALL 4-FOLD CV RESULTS ====================")
print(f"Total subjects tested: {len(all_test_accuracies_for_plotting)}")
print(f"Mean test accuracy: {np.mean(all_test_accuracies_for_plotting):.2f}% ± {np.std(all_test_accuracies_for_plotting):.2f}%")
print(f"Min test accuracy: {np.min(all_test_accuracies_for_plotting):.2f}%")
print(f"Max test accuracy: {np.max(all_test_accuracies_for_plotting):.2f}%")

# Plot 1: Test accuracies for each subject across all folds
plt.figure(figsize=(15, 8))
colors = ['blue', 'red', 'green', 'orange']
x_positions = []
fold_labels = []

x_offset = 0
for fold_idx, (fold_name, fold_data) in enumerate(all_fold_results.items()):
    fold_test_accs = [fold_data['subject_results'][subj]['test_accuracy'] 
                     for subj in fold_data['subject_results'].keys()]
    test_subjects = list(fold_data['subject_results'].keys())
    
    x_pos = np.arange(len(fold_test_accs)) + x_offset
    plt.bar(x_pos, fold_test_accs, color=colors[fold_idx], alpha=0.7, 
            label=f'{fold_name.capitalize()}')
    
    # Add subject labels
    for i, (pos, subj_id) in enumerate(zip(x_pos, test_subjects)):
        plt.text(pos, fold_test_accs[i] + 1, f'S{subj_id}', 
                ha='center', va='bottom', fontsize=8, rotation=45)
    
    x_positions.extend(x_pos)
    fold_labels.extend([f'{fold_name}\nS{s}' for s in test_subjects])
    x_offset += len(fold_test_accs) + 1

plt.axhline(y=np.mean(all_test_accuracies_for_plotting), color='black', linestyle='--', 
           label=f'Overall Mean: {np.mean(all_test_accuracies_for_plotting):.2f}%')
plt.title(f'4-Fold Cross-Validation: DeepConvNet Test Accuracies by Subject')
plt.xlabel('Subjects by Fold')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.xticks(x_positions, [f'F{i//13}S{s}' for i, s in enumerate([int(label.split('S')[1]) for label in fold_labels if 'S' in label])], 
          rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_DCN_test_accuracies_by_fold.png'), dpi=300)
plt.show()

# Plot 2: Training curves across all subjects
df = pd.DataFrame({
    'Train Accuracy': all_train_accuracies_per_epoch,
    'Val Accuracy': all_val_accuracies_per_epoch,
    'Train Loss': all_train_losses_per_epoch,
    'Val Loss': all_val_losses_per_epoch
})

# Compute mean and std for each metric across epochs
metrics = ['Train Accuracy', 'Val Accuracy', 'Train Loss', 'Val Loss']
mean_std_df = pd.DataFrame({'Epoch': np.arange(1, num_epochs + 1)})

for metric in metrics:
    values = np.array(df[metric].tolist())  # shape: (n_subjects, n_epochs)
    mean_std_df[f"{metric} Mean"] = values.mean(axis=0)
    mean_std_df[f"{metric} Std"] = values.std(axis=0)

# --- Plotting Training Curves ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
fig.suptitle(f'4-Fold CV: DeepConvNet Training and Validation Metrics Over Epochs', fontsize=16)

plot_params = [
    ("Train Accuracy", axes[0, 0], "blue"),
    ("Val Accuracy", axes[0, 1], "green"),
    ("Train Loss", axes[1, 0], "red"),
    ("Val Loss", axes[1, 1], "orange"),
]

for metric, ax, color in plot_params:
    mean = mean_std_df[f"{metric} Mean"]
    std = mean_std_df[f"{metric} Std"]
    epoch = mean_std_df["Epoch"]

    ax.plot(epoch, mean, label=f'{metric} (Mean)', color=color, linewidth=2)
    ax.fill_between(epoch, mean - std, mean + std, color=color, alpha=0.3, 
                   label=f'{metric} (±1 std)')
    ax.set_title(f"{metric} over Epochs (n={len(all_test_accuracies)} subjects)")
    ax.set_xlabel("Epoch")
    ylabel = "Accuracy (%)" if "Accuracy" in metric else "Loss"
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_DCN_training_validation_metrics.png'), dpi=300)
plt.show()

# Plot 3: Box plot of test accuracies by fold
plt.figure(figsize=(10, 6))
fold_accuracies = []
fold_names = []

for fold_name, fold_data in all_fold_results.items():
    fold_test_accs = [fold_data['subject_results'][subj]['test_accuracy'] 
                     for subj in fold_data['subject_results'].keys()]
    fold_accuracies.append(fold_test_accs)
    fold_names.append(fold_name.replace('_', ' ').title())

plt.boxplot(fold_accuracies, labels=fold_names)
plt.axhline(y=np.mean(all_test_accuracies_for_plotting), color='red', linestyle='--', 
           label=f'Overall Mean: {np.mean(all_test_accuracies_for_plotting):.2f}%')
plt.title('Test Accuracy Distribution by Fold')
plt.xlabel('Fold')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_DCN_accuracy_distribution_by_fold.png'), dpi=300)
plt.show()

# Save final consolidated results
final_results = {
    'fold_results': all_fold_results,
    'summary': {
        'mean_test_accuracy': np.mean(all_test_accuracies_for_plotting),
        'std_test_accuracy': np.std(all_test_accuracies_for_plotting),
        'min_test_accuracy': np.min(all_test_accuracies_for_plotting),
        'max_test_accuracy': np.max(all_test_accuracies_for_plotting),
        'all_test_accuracies': all_test_accuracies_for_plotting,
        'n_subjects_tested': len(all_test_accuracies_for_plotting),
        'n_folds': n_folds
    }
}

final_output_file = os.path.join(output_path, f'{type_of_subject}_final_results.npy')
np.save(final_output_file, final_results)
print(f"Final results saved to {final_output_file}")

print('==================== End of script dcn_indep_4fold.py! ===================')