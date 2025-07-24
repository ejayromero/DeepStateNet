'''
Script to train MicroSNet on Microstates timeseries
Adapted from original ms_dcn.py with new model names but keeping original data handling
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

import mne
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

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
from lib import my_models as mm

# ---------------------------# Load files ---------------------------
data_path = '../Data/'

do_all = False
n_subjects = 50
num_epochs = 50
subject_list = list(range(n_subjects))
all_data, all_y = mf.load_all_data(subjects_list=None, do_all=do_all)
type_of_subject = 'dependent_harmonize'  # or 'dep' for dependent subjects
model_name = 'embedded_microsnet'  # 'microsnet' or 'multiscale_microsnet'
output_path = f'../Output/ica_rest_all/{type_of_subject}/'
if 'embedded' in model_name:
    input_path = '../Output/ica_rest_all/modkmeans_results/'
    ms_timeseries_path = os.path.join(input_path, 'modkmeans_sequence_harmonize_indiv.pkl')
else:
    input_path = '../Output/ica_rest_all/'
    ms_timeseries_path = os.path.join(input_path, 'ms_timeseries_harmonize.pkl')
with open(ms_timeseries_path, 'rb') as f:
    finals_ls = pickle.load(f)

# ---------- Seed & Device ----------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mf.set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Loop Through Subjects (Leave-One-Subject-Out) ----------
# ---------- Loop Through First Subject Only ----------
if n_subjects == 1:
    test_subjects = [0]
else:
    test_subjects = list(range(n_subjects))
    
output_file = os.path.join(output_path, f'{type_of_subject}_ms_{model_name}_results_ica_rest_all.npy')

# ---------- Loop Through Subjects ----------
for id in range(n_subjects):
    print(f"\n▶ Training Subject {id}")
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
    
    if len(all_train_accuracies) > id:
        print(f"Skipping Subject {id} as it has already been processed.")
        continue
    print(f"\n\n==================== Subject {id} ====================")

    # Keep original data handling - using finals_ls instead of all_data
    x = torch.tensor(finals_ls[id], dtype=torch.float32)

    if 'embedded' in model_name:
        # For EmbeddedMicroSNet: use only the microstate sequences (first channel)
        x_microstates = x[:, 0, :].long()  # Shape: (batch_size, sequence_length)
        
        # Handle negative indices (embedding layers require indices >= 0)
        min_val = torch.min(x_microstates).item()
        if min_val < 0:
            print(f"Found negative microstate indices ({min_val}), shifting to start from 0...")
            x_microstates = x_microstates - min_val  # Shift so minimum becomes 0
            print(f"New microstate range: {torch.min(x_microstates).item()} to {torch.max(x_microstates).item()}")
        
        x = x_microstates
        n_microstates = int(torch.max(x).item()) + 1
        sequence_length = x.shape[1]  # For categorical data: (batch_size, sequence_length)
        
        print(f"Using microstate sequences only for {model_name}")
        print(f"Microstate data shape: {x.shape}")
        print(f"Microstate range: {torch.min(x).item()} to {torch.max(x).item()}")
        
    else:
        # For other models: convert microstate sequences to one-hot encoding
        x_microstates = x[:, 0, :].long()  # Extract microstate sequences
        
        # Handle negative indices
        min_val = torch.min(x_microstates).item()
        if min_val < 0:
            print(f"Found negative microstate indices ({min_val}), shifting to start from 0...")
            x_microstates = x_microstates - min_val
        
        n_microstates = int(torch.max(x_microstates).item()) + 1
        sequence_length = x_microstates.shape[1]
        
        # Convert to one-hot encoding
        x_onehot = torch.zeros(x_microstates.shape[0], n_microstates, sequence_length)
        x_onehot.scatter_(1, x_microstates.unsqueeze(1), 1)
        x = x_onehot
        
        print(f"Converted microstate sequences to one-hot for {model_name}")
        print(f"One-hot data shape: {x.shape}")

    y = torch.tensor(all_y[id], dtype=torch.long)

    print(f"Final data shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of microstate categories: {n_microstates}")
    print(f"Number of classes: {len(torch.unique(y))}")

    # ---------- Data Splitting ----------
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)

    # ---------- DataLoaders ----------
    batch_size = 32
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # ---------- Model ----------
    # n_microstates = x.shape[1]  # Number of microstate categories
    n_classes = len(torch.unique(y))
    
    # Create model using factory function
    model = mm.get_model(
        model_name=model_name,
        n_microstates=n_microstates,
        n_classes=n_classes,
        sequence_length=sequence_length,
        dropout=0.25
    )
    
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Using model: {model_name}")
    print(f"Model description: {mm.MODEL_INFO[model_name]['description']}")

    # ---------- Training ----------
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
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
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total * 100
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:02d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # ---------- Test ----------
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)

    test_acc = test_correct / test_total * 100
    print(f"✅ Subject {id} Test Accuracy: {test_acc:.2f}%")

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
    print(f"✅ Subject {id} processed successfully.\n\n")

# ---------- Summary Results ----------
print(f"\n🎯 Overall Results:")
print(f"Mean Test Accuracy: {np.mean(all_test_accuracies):.2f}% ± {np.std(all_test_accuracies):.2f}%")
print(f"Best Subject Accuracy: {np.max(all_test_accuracies):.2f}%")
print(f"Worst Subject Accuracy: {np.min(all_test_accuracies):.2f}%")

# ------------------------- plotting results -------------------------
# plot all test accuracies
plt.figure(figsize=(10, 6))
plt.plot(all_test_accuracies, marker='o', linestyle='-')
plt.title(f'Subject {type_of_subject} {model_name.upper()} Test Accuracies for Each Subject')
plt.xlabel('Subject ID')
plt.ylabel('Test Accuracy (%)')
plt.xticks(range(n_subjects), [f'S{i}' for i in range(n_subjects)], rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_ms_{model_name}_test_accuracies.png'))

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

print('==================== End of script ms_dcn.py! ===================')