'''
Script to train model on 50 subjects, training DeepConvNet on Microstates timeseries

'''
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


from braindecode.models import Deep4Net
from braindecode.classifier import EEGClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from lib import my_functions as mf

print(f'==================== Start of script {os.path.basename(__file__)}! ====================')

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
type_of_subject = 'independent'  # 'independent' or 'adaptive'
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
# excluded_from_training = [2, 12, 14, 20, 22, 23, 30, 39, 46]
excluded_from_training = [-1]
num_epochs = 100
subject_list = list(range(n_subjects))
all_data, all_y = mf.load_all_data(subjects_list=None, do_all=do_all, data_path=data_path)


kmeans_path = os.path.join(input_path, 'modkmeans_results', 'ms_timeseries')
ms_timeseries_path = os.path.join(kmeans_path, 'ms_timeseries_harmonize.pkl')
with open(ms_timeseries_path, 'rb') as f:
    finals_ls = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mf.set_seed(42)


# ---------- Loop Through Subjects (Leave-One-Subject-Out) ----------
# ---------- Loop Through First Subject Only ----------
if n_subjects == 1:
    test_subjects = [0]
else:
    test_subjects = list(range(n_subjects))
    
output_file = os.path.join(output_path, f'{type_of_subject}_results_ica_rest_all.npy')

for test_id in test_subjects:
    
    if os.path.exists(output_file):
        results = np.load(output_file, allow_pickle=True).item()
        all_train_accuracies = results['train_accuracies']
        all_train_losses = results['train_losses']
        all_test_accuracies = results['test_accuracies']
        all_val_accuracies = results['val_accuracies']
        all_val_losses = results['val_losses']
        all_models = results['models']
    else:
        all_train_accuracies = []
        all_train_losses = []
        all_test_accuracies = []
        all_val_accuracies = []
        all_val_losses = []
        all_models = []
    
    if len(all_train_accuracies) > test_id:
        print(f"Skipping Subject {test_id} as it has already been processed.")
        continue
    print(f"\n\n==================== Subject {test_id} ====================")

    mf.set_seed(42)
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for training.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU for training.")
        device = torch.device("cpu")


    # Choose validation subjects (4 random ones not equal to test_id)
    val_candidates, val_ids = mf.get_val_ids(42, test_id, excluded_from_training)

    # Remaining for training
    train_ids = [i for i in val_candidates if i not in val_ids]

    # Concatenate training data
    x_train = torch.cat([torch.tensor(all_data[i], dtype=torch.float32).squeeze(1) for i in train_ids], dim=0)
    y_train = torch.cat([torch.tensor(all_y[i], dtype=torch.long) for i in train_ids], dim=0)

    # Concatenate validation data
    x_val = torch.cat([torch.tensor(all_data[i], dtype=torch.float32).squeeze(1) for i in val_ids], dim=0)
    y_val = torch.cat([torch.tensor(all_y[i], dtype=torch.long) for i in val_ids], dim=0)

    # Test subject
    x_test = torch.tensor(all_data[test_id], dtype=torch.float32).squeeze(1)
    y_test = torch.tensor(all_y[test_id], dtype=torch.long)

    # ---------- DataLoaders ----------
    batch_size = 32
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # ---------- Model ----------
    base_model = Deep4Net(
        n_chans=x_train.shape[1],
        n_outputs=len(torch.unique(y_train)),
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

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
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

    # ---------- Save Results ----------
    all_train_accuracies.append(train_accuracies)
    all_train_losses.append(train_losses)
    all_test_accuracies.append(test_acc)
    all_val_accuracies.append(val_accuracies)
    all_val_losses.append(val_losses)
    all_models.append(model)

    results = {
        'train_accuracies': all_train_accuracies,
        'train_losses': all_train_losses,
        'test_accuracies': all_test_accuracies,
        'val_accuracies': all_val_accuracies,
        'val_losses': all_val_losses,
        'models': all_models
    }
    np.save(output_file, results)
    print(f"Results saved to {output_file}")
    print(f"✅ Subject {test_id} processed successfully.\n\n")

# End of loop through subjects

# ------------------------- plotting results -------------------------
# plot all test accuracies
plt.figure(figsize=(10, 6))
plt.plot(all_test_accuracies, marker='o', linestyle='-')
plt.title(f'Subject {type_of_subject} DeepConvNet Test Accuracies for Each Subject')
plt.xlabel('Subject ID')
plt.ylabel('Test Accuracy (%)')
plt.xticks(range(n_subjects), [f'S{i}' for i in range(n_subjects)], rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_DCN_test_accuracies.png'))

# plot mean and std of train and val accuracies
df = pd.DataFrame({
    'Train Accuracy': all_train_accuracies,
    'Val Accuracy': all_val_accuracies,
    'Train loss': all_train_losses,
    'Val loss': all_val_losses
})
num_epochs = 100
# Compute mean and std for each metric across epochs
metrics = ['Train Accuracy', 'Val Accuracy', 'Train loss', 'Val loss']
mean_std_df = pd.DataFrame({'Epoch': np.arange(1, num_epochs  + 1)})

for metric in metrics:
    values = np.array(df[metric].tolist())  # shape: (n_epochs, n_subjects)
    mean_std_df[f"{metric} Mean"] = values.mean(axis=0)
    mean_std_df[f"{metric} Std"] = values.std(axis=0)

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey='row')
fig.suptitle(f'Subject {type_of_subject} DeepConvNet Training and Validation Metrics Over Epochs', fontsize=16)

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
plt.savefig(os.path.join(output_path, f'{type_of_subject}_DCN_training_validation_metrics.png'))
plt.legend()
plt.subplots_adjust(top=0.9)  # Adjust top to make room for the suptitle
plt.close()

print('==================== End of script dcn_indep.py! ===================')