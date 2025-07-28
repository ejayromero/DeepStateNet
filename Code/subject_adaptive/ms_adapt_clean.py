'''
Script to train model on 50 subjects, training MicroSNet on Microstates timeseries
Modified to include fine-tuning with 90/10 split on test subject data
Clean version with embedded model support and new repository structure
'''

print('==================== Start of script ms_adapt_clean.py! ===================')

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import my_functions as mf
from lib import my_models as mm

# Explicit CUDA setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Specify GPU 0 explicitly
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# ---------------------------# Load files ---------------------------
data_path = 'Data/'
type_of_subject = 'adaptive_harmonize'  # 'independent' or 'adaptive'
model_name = 'embedded_microsnet'  # 'microsnet' or 'multiscale_microsnet' or 'embedded_microsnet'
output_path = f'Output/ica_rest_all/{type_of_subject}/'
input_path = 'Output/ica_rest_all/'

# Making sure all paths exist
if not os.path.exists(input_path):
    os.makedirs(input_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Parameters 
do_all = False
n_subjects = 50
num_epochs = 50
batch_size = 32  # or 256 if memory allows
excluded_from_training = [-1]  # No exclusions for adaptive clean
subject_list = list(range(n_subjects))
all_data, all_y = mf.load_all_data(subjects_list=None, do_all=do_all, data_path=data_path)

if 'embedded' in model_name:
    kmeans_path = os.path.join(input_path, 'modkmeans_results', 'modkmeans_sequence')
    ms_timeseries_path = os.path.join(kmeans_path, 'modkmeans_sequence_harmonize_indiv.pkl')
else:
    kmeans_path = os.path.join(input_path, 'modkmeans_results', 'ms_timeseries')
    ms_timeseries_path = os.path.join(kmeans_path, 'ms_timeseries_harmonize.pkl')
with open(ms_timeseries_path, 'rb') as f:
    finals_ls = pickle.load(f)

mf.set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Loop Through Subjects (Leave-One-Subject-Out with Fine-tuning) ----------
if n_subjects == 1:
    test_subjects = [0]
else:
    test_subjects = list(range(n_subjects))
    
output_file = os.path.join(output_path, f'{type_of_subject}_ms_{model_name}_results_ica_rest_all.npy')

for test_id in test_subjects:
    
    if os.path.exists(output_file):
        results = np.load(output_file, allow_pickle=True).item()
        all_train_accuracies = results['train_accuracies']
        all_train_losses = results['train_losses']
        all_test_accuracies = results['test_accuracies']
        all_val_accuracies = results['val_accuracies']
        all_val_losses = results['val_losses']
        all_models = results['models']
        # Load fine-tuning results if they exist
        all_finetune_accuracies = results.get('finetune_accuracies', [])
        all_finetune_losses = results.get('finetune_losses', [])
    else:
        all_train_accuracies = []
        all_train_losses = []
        all_test_accuracies = []
        all_val_accuracies = []
        all_val_losses = []
        all_models = []
        all_finetune_accuracies = []
        all_finetune_losses = []
    
    if len(all_train_accuracies) > test_id:
        print(f"Skipping Subject {test_id} as it has already been processed.")
        continue
    print(f"\n\n==================== Subject {test_id} ====================")

    # Choose validation subjects (4 random ones not equal to test_id)
    val_candidates, val_ids = mf.get_val_ids(42, test_id, excluded_from_training)

    # Remaining for training
    train_ids = [i for i in val_candidates if i not in val_ids]

    # Process and concatenate training data with embedded model support
    x_train_list = []
    y_train_list = []
    n_microstates = None  # Will be determined from data
    sequence_length = None
    global_min_val = None  # Track global min across all subjects
    global_max_val = None  # Track global max across all subjects
    
    # First pass: determine global min/max values across ALL subjects in the dataset
    # Since microstates are harmonized, we need to find the global range across all 50 subjects
    print("Determining global microstate range across all subjects...")
    all_possible_subjects = list(range(n_subjects))  # All 50 subjects
    
    for i in all_possible_subjects:
        x = torch.tensor(finals_ls[i], dtype=torch.float32)
        x_microstates = x[:, 0, :].long()
        
        current_min = torch.min(x_microstates).item()
        current_max = torch.max(x_microstates).item()
        
        if global_min_val is None or current_min < global_min_val:
            global_min_val = current_min
        if global_max_val is None or current_max > global_max_val:
            global_max_val = current_max
    
    print(f"Global microstate range across all {n_subjects} subjects: {global_min_val} to {global_max_val}")
    
    # Apply global shift if needed
    shift_amount = 0
    if global_min_val < 0:
        shift_amount = -global_min_val
        print(f"Applying global shift of {shift_amount} to handle negative indices")
    
    # Determine final parameters - must accommodate the full global range
    n_microstates = int(global_max_val + shift_amount) + 1
    print(f"Embedding layer size (n_microstates): {n_microstates}")
    print(f"This accounts for the full harmonized microstate space")
    
    for i in train_ids:
        x = torch.tensor(finals_ls[i], dtype=torch.float32)
        
        if 'embedded' in model_name:
            # For EmbeddedMicroSNet: use only the microstate sequences (first channel)
            x_microstates = x[:, 0, :].long()  # Shape: (batch_size, sequence_length)
            
            # Apply global shift
            if shift_amount > 0:
                x_microstates = x_microstates + shift_amount
            
            x_processed = x_microstates
            if sequence_length is None:
                sequence_length = x_microstates.shape[1]
            
            # Debug: check final range for this subject
            unique_microstates = torch.unique(x_microstates).tolist()
            print(f"Subject {i} uses microstates: {unique_microstates} (count: {len(unique_microstates)})")
            
        else:
            # For other models: convert microstate sequences to one-hot encoding
            x_microstates = x[:, 0, :].long()  # Extract microstate sequences
            
            # Apply global shift
            if shift_amount > 0:
                x_microstates = x_microstates + shift_amount
            
            if sequence_length is None:
                sequence_length = x_microstates.shape[1]
            
            # Convert to one-hot encoding
            x_onehot = torch.zeros(x_microstates.shape[0], n_microstates, sequence_length)
            x_onehot.scatter_(1, x_microstates.unsqueeze(1), 1)
            x_processed = x_onehot
        
        x_train_list.append(x_processed)
        y_train_list.append(torch.tensor(all_y[i], dtype=torch.long))
    
    x_train = torch.cat(x_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)

    # Process validation data
    x_val_list = []
    y_val_list = []
    for i in val_ids:
        x = torch.tensor(finals_ls[i], dtype=torch.float32)
        
        if 'embedded' in model_name:
            # For EmbeddedMicroSNet: use only the microstate sequences (first channel)
            x_microstates = x[:, 0, :].long()  # Shape: (batch_size, sequence_length)
            
            # Apply same global shift
            if shift_amount > 0:
                x_microstates = x_microstates + shift_amount
            
            x_processed = x_microstates
            
            # Debug: check unique microstates for this validation subject
            unique_microstates = torch.unique(x_microstates).tolist()
            print(f"Val Subject {i} uses microstates: {unique_microstates}")
            
        else:
            # For other models: convert microstate sequences to one-hot encoding
            x_microstates = x[:, 0, :].long()  # Extract microstate sequences
            
            # Apply same global shift
            if shift_amount > 0:
                x_microstates = x_microstates + shift_amount
            
            # Convert to one-hot encoding
            x_onehot = torch.zeros(x_microstates.shape[0], n_microstates, sequence_length)
            x_onehot.scatter_(1, x_microstates.unsqueeze(1), 1)
            x_processed = x_onehot
        
        x_val_list.append(x_processed)
        y_val_list.append(torch.tensor(all_y[i], dtype=torch.long))
    
    x_val = torch.cat(x_val_list, dim=0)
    y_val = torch.cat(y_val_list, dim=0)

    # Process test subject data - split into fine-tuning (90%) and final test (10%)
    x = torch.tensor(finals_ls[test_id], dtype=torch.float32)
    
    if 'embedded' in model_name:
        # For EmbeddedMicroSNet: use only the microstate sequences (first channel)
        x_microstates = x[:, 0, :].long()  # Shape: (batch_size, sequence_length)
        
        # Apply same global shift
        if shift_amount > 0:
            x_microstates = x_microstates + shift_amount
        
        x_test_full = x_microstates
        
        # Debug: check unique microstates for test subject
        unique_microstates = torch.unique(x_microstates).tolist()
        print(f"Test Subject {test_id} uses microstates: {unique_microstates}")
        
        # Final validation - check that all indices are within embedding bounds
        max_val = torch.max(x_test_full).item()
        min_val = torch.min(x_test_full).item()
        print(f"Test data microstate range: {min_val} to {max_val}")
        if max_val >= n_microstates:
            print(f"ERROR: Maximum microstate index ({max_val}) >= n_microstates ({n_microstates})")
            print(f"This will cause the embedding layer to fail!")
        else:
            print(f"✓ All microstate indices are within embedding bounds [0, {n_microstates-1}]")
        
    else:
        # For other models: convert microstate sequences to one-hot encoding
        x_microstates = x[:, 0, :].long()  # Extract microstate sequences
        
        # Apply same global shift
        if shift_amount > 0:
            x_microstates = x_microstates + shift_amount
        
        # Convert to one-hot encoding
        x_onehot = torch.zeros(x_microstates.shape[0], n_microstates, sequence_length)
        x_onehot.scatter_(1, x_microstates.unsqueeze(1), 1)
        x_test_full = x_onehot
    
    y_test_full = torch.tensor(all_y[test_id], dtype=torch.long)

    # Load pre-saved indices for ultimate consistency
    all_train_indices, all_test_indices = mf.load_split_indices(output_path, filename=f'{type_of_subject}_split_indices.pkl')
    if all_train_indices is not None:
        finetune_indices = all_train_indices[test_id]
        final_test_indices = all_test_indices[test_id]
        x_finetune = x_test_full[finetune_indices]
        y_finetune = y_test_full[finetune_indices]
        x_final_test = x_test_full[final_test_indices]
        y_final_test = y_test_full[final_test_indices]
    else:
        # If indices are not available, use utility function for consistent splitting
        print(f"Split indices not found for subject {test_id}. Using utility function to split data.")

        # Use utility function for consistent splitting
        x_finetune, y_finetune, x_final_test, y_final_test = mf.split_subject_data_consistently(
            x_test_full, y_test_full, test_id, train_ratio=0.9, base_seed=42
        )

    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Fine-tune data shape: {x_finetune.shape}")
    print(f"Final test data shape: {x_final_test.shape}")
    print(f"Embedding layer size (n_microstates): {n_microstates}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of classes: {len(torch.unique(y_train))}")
    
    if 'embedded' in model_name:
        print(f"Using microstate sequences for {model_name}")
        print(f"Each subject uses ~5 microstates from the global harmonized space")
        
        # Additional validation
        train_unique = len(torch.unique(x_train))
        val_unique = len(torch.unique(x_val)) 
        test_unique = len(torch.unique(x_test_full))
        print(f"Unique microstates in train/val/test: {train_unique}/{val_unique}/{test_unique}")
    else:
        print(f"Converted microstate sequences to one-hot for {model_name}")

    # ---------- DataLoaders ----------
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    finetune_loader = DataLoader(TensorDataset(x_finetune, y_finetune), batch_size=batch_size, shuffle=True)
    final_test_loader = DataLoader(TensorDataset(x_final_test, y_final_test), batch_size=batch_size, shuffle=False)

    # ---------- Model ----------
    n_classes = len(torch.unique(y_train))
    
    # Create model using factory function
    model = mm.get_model(
        model_name=model_name,
        n_microstates=n_microstates,
        n_classes=n_classes,
        sequence_length=sequence_length,
        dropout=0.25
    )
    
    net = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    print(f"Using model: {model_name}")
    print(f"Model description: {mm.MODEL_INFO[model_name]['description']}")

    # ---------- Initial Training ----------
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

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # ---------- Fine-tuning on Test Subject Data ----------
    print(f"Starting fine-tuning on {len(x_finetune)} samples from test subject {test_id}...")

    # Lower learning rate for fine-tuning
    finetune_optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    num_finetune_epochs = 20  # Fewer epochs for fine-tuning

    finetune_losses, finetune_accuracies = [], []

    for epoch in range(num_finetune_epochs):
        net.train()
        finetune_loss, finetune_correct, finetune_total = 0, 0, 0
        for batch_x, batch_y in finetune_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            finetune_optimizer.zero_grad()
            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            finetune_optimizer.step()
            finetune_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            finetune_correct += (preds == batch_y).sum().item()
            finetune_total += batch_y.size(0)

        finetune_loss /= finetune_total
        finetune_acc = finetune_correct / finetune_total * 100
        finetune_losses.append(finetune_loss)
        finetune_accuracies.append(finetune_acc)

        if (epoch + 1) % 5 == 0 or epoch == num_finetune_epochs - 1:
            print(f"Fine-tune Epoch {epoch+1:02d}/{num_finetune_epochs} | "
                f"Loss: {finetune_loss:.4f}, Acc: {finetune_acc:.2f}%")

    # ---------- Final Test on Remaining 10% ----------
    net.eval()
    final_test_correct, final_test_total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in final_test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = net(batch_x)
            preds = outputs.argmax(dim=1)
            final_test_correct += (preds == batch_y).sum().item()
            final_test_total += batch_y.size(0)

    final_test_acc = final_test_correct / final_test_total * 100
    print(f"✅ Subject {test_id} Final Test Accuracy (after fine-tuning): {final_test_acc:.2f}%")
    print(f"   Fine-tuning samples: {len(x_finetune)}, Final test samples: {len(x_final_test)}")

    # ---------- Save Results ----------
    all_train_accuracies.append(train_accuracies)
    all_train_losses.append(train_losses)
    all_test_accuracies.append(final_test_acc)  # This is now the final test accuracy after fine-tuning
    all_val_accuracies.append(val_accuracies)
    all_val_losses.append(val_losses)
    all_models.append(model)
    all_finetune_accuracies.append(finetune_accuracies)
    all_finetune_losses.append(finetune_losses)

    results = {
        'train_accuracies': all_train_accuracies,
        'train_losses': all_train_losses,
        'test_accuracies': all_test_accuracies,
        'val_accuracies': all_val_accuracies,
        'val_losses': all_val_losses,
        'finetune_accuracies': all_finetune_accuracies,
        'finetune_losses': all_finetune_losses,
        'models': all_models
    }
    np.save(output_file, results)
    print(f"Results saved to {output_file}")
    print(f"✅ Subject {test_id} processed successfully.\n\n")

# End of loop through subjects

# Generate and save split indices for all subjects for future consistency
print("Generating and saving split indices for all subjects...")
split_indices_file = os.path.join(output_path, f'{type_of_subject}_split_indices.pkl')
if not os.path.exists(split_indices_file):
    all_train_indices = []
    all_test_indices = []

    for subject_id in range(n_subjects):
        data_length = len(finals_ls[subject_id])
        train_indices, test_indices = mf.get_consistent_split_indices(
            data_length, subject_id, train_ratio=0.9, base_seed=42
        )
        all_train_indices.append(train_indices)
        all_test_indices.append(test_indices)

    # Save indices for reuse in other scripts
    mf.save_split_indices(output_path, all_train_indices, all_test_indices, f'{type_of_subject}_split_indices.pkl')
    print("Split indices saved for consistent reuse across scripts.")
else:
    print("Split indices already exist. Skipping generation.")

# ---------- Summary Results ----------
print(f"\n🎯 Overall Results:")
print(f"Mean Test Accuracy (after fine-tuning): {np.mean(all_test_accuracies):.2f}% ± {np.std(all_test_accuracies):.2f}%")
print(f"Best Subject Accuracy: {np.max(all_test_accuracies):.2f}%")
print(f"Worst Subject Accuracy: {np.min(all_test_accuracies):.2f}%")

# ------------------------- plotting results -------------------------
# plot all test accuracies
plt.figure(figsize=(10, 6))
plt.plot(all_test_accuracies, marker='o', linestyle='-')
plt.title(f'Subject {type_of_subject} {model_name.upper()} Test Accuracies for Each Subject (After Fine-tuning)')
plt.xlabel('Subject ID')
plt.ylabel('Test Accuracy (%)')
plt.xticks(range(n_subjects), [f'S{i}' for i in range(n_subjects)], rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_ms_{model_name}_test_accuracies_finetuned.png'))
plt.close()

# plot mean and std of train and val accuracies
df = pd.DataFrame({
    'Train Accuracy': all_train_accuracies,
    'Val Accuracy': all_val_accuracies,
    'Train loss': all_train_losses,
    'Val loss': all_val_losses
})

# Compute mean and std for each metric across epochs
metrics = ['Train Accuracy', 'Val Accuracy', 'Train loss', 'Val loss']
mean_std_df = pd.DataFrame({'Epoch': np.arange(1, num_epochs  + 1)})

for metric in metrics:
    values = np.array(df[metric].tolist())  # shape: (n_epochs, n_subjects)
    mean_std_df[f"{metric} Mean"] = values.mean(axis=0)
    mean_std_df[f"{metric} Std"] = values.std(axis=0)

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey='row')
fig.suptitle(f'Subject {type_of_subject} {model_name.upper()} Training and Validation Metrics Over Epochs', fontsize=16)

plot_params = [
    ("Train Accuracy", axes[0, 0], "blue"),
    ("Val Accuracy", axes[0, 1], "green"),
    ("Train loss", axes[1, 0], "red"),
    ("Val loss", axes[1, 1], "orange"),
]

for metric, ax, color in plot_params:
    mean = mean_std_df[f"{metric} Mean"]
    std = mean_std_df[f"{metric} Std"]
    epoch = mean_std_df["Epoch"]

    ax.plot(epoch, mean, label=metric, color=color)
    ax.fill_between(epoch, mean - std, mean + std, color=color, alpha=0.3)
    ax.set_title(f"{metric} over Epochs")
    ax.set_xlabel("Epoch")
    ylabel = "Accuracy (%)" if "Accuracy" in metric else "Loss"
    ax.set_ylabel(ylabel)

plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_ms_{model_name}_training_validation_metrics.png'))
plt.legend()
plt.subplots_adjust(top=0.9)  # Adjust top to make room for the suptitle
plt.close()

# Plot fine-tuning metrics
if all_finetune_accuracies:
    df_finetune = pd.DataFrame({
        'Finetune Accuracy': all_finetune_accuracies,
        'Finetune Loss': all_finetune_losses
    })
    
    num_finetune_epochs = 20
    metrics_finetune = ['Finetune Accuracy', 'Finetune Loss']
    mean_std_finetune_df = pd.DataFrame({'Epoch': np.arange(1, num_finetune_epochs + 1)})
    
    for metric in metrics_finetune:
        values = np.array(df_finetune[metric].tolist())
        mean_std_finetune_df[f"{metric} Mean"] = values.mean(axis=0)
        mean_std_finetune_df[f"{metric} Std"] = values.std(axis=0)
    
    # Plot fine-tuning metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Subject {type_of_subject} {model_name.upper()} Fine-tuning Metrics Over Epochs', fontsize=16)
    
    plot_params_finetune = [
        ("Finetune Accuracy", axes[0], "purple"),
        ("Finetune Loss", axes[1], "magenta"),
    ]
    
    for metric, ax, color in plot_params_finetune:
        mean = mean_std_finetune_df[f"{metric} Mean"]
        std = mean_std_finetune_df[f"{metric} Std"]
        epoch = mean_std_finetune_df["Epoch"]
        
        ax.plot(epoch, mean, label=metric, color=color)
        ax.fill_between(epoch, mean - std, mean + std, color=color, alpha=0.3)
        ax.set_title(f"{metric} over Epochs")
        ax.set_xlabel("Epoch")
        ylabel = "Accuracy (%)" if "Accuracy" in metric else "Loss"
        ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{type_of_subject}_ms_{model_name}_finetuning_metrics.png'))
    plt.close()

print('==================== End of script ms_adapt_clean.py! ===================')