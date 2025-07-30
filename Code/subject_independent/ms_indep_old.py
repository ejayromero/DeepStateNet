'''
Script to train MicroSNet on 50 subjects using independent training (leave-one-subject-out)
Clean version using modular approach
'''


import os
import sys
import numpy as np
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import my_functions as mf
from lib import my_models as mm

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

print(f'==================== Start of script {os.path.basename(__file__)}! ===================')

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
type_of_subject = 'independent_harmonize'  # or 'dep' for dependent subjects
model_name = 'embedded_microsnet'  # 'microsnet' or 'multiscale_microsnet'
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
# excluded_from_training = [2, 12, 14, 20, 22, 23, 30, 39, 46]
excluded_from_training = [-1]  # No exclusions for independent clean
subject_list = list(range(n_subjects))
all_data, all_y = mf.load_all_data(subjects_list=None, do_all=do_all, data_path=data_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
if 'embedded' in model_name:
    kmeans_path = os.path.join(input_path, 'modkmeans_results', 'modkmeans_sequence')
    ms_timeseries_path = os.path.join(kmeans_path, 'modkmeans_sequence_harmonize_indiv.pkl')
else:
    kmeans_path = os.path.join(input_path, 'modkmeans_results', 'ms_timeseries')
    ms_timeseries_path = os.path.join(kmeans_path, 'ms_timeseries_harmonize.pkl')
with open(ms_timeseries_path, 'rb') as f:
    finals_ls = pickle.load(f)
mf.set_seed(42)

# ---------- Loop Through Subjects (Leave-One-Subject-Out) ----------
if n_subjects == 1:
    test_subjects = [0]
else:
    test_subjects = list(range(n_subjects))
    
output_file = os.path.join(output_path, f'{type_of_subject}_ms_results_ica_rest_all.npy')

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

    # Concatenate training data - Handle one-hot encoded microstate data
    x_train_list = []
    y_train_list = []
    for i in train_ids:
        x_data = torch.tensor(finals_ls[i], dtype=torch.float32)
        if x_data.ndim == 4:  # If shape is (n_trials, 1, n_microstates, sequence_length)
            x_data = x_data.squeeze(1)
        x_train_list.append(x_data)
        y_train_list.append(torch.tensor(all_y[i], dtype=torch.long))
    
    x_train = torch.cat(x_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)

    # Concatenate validation data
    x_val_list = []
    y_val_list = []
    for i in val_ids:
        x_data = torch.tensor(finals_ls[i], dtype=torch.float32)
        if x_data.ndim == 4:  # If shape is (n_trials, 1, n_microstates, sequence_length)
            x_data = x_data.squeeze(1)
        x_val_list.append(x_data)
        y_val_list.append(torch.tensor(all_y[i], dtype=torch.long))
    
    x_val = torch.cat(x_val_list, dim=0)
    y_val = torch.cat(y_val_list, dim=0)

    # Test subject
    x_test = torch.tensor(finals_ls[test_id], dtype=torch.float32)
    if x_test.ndim == 4:  # If shape is (n_trials, 1, n_microstates, sequence_length)
        x_test = x_test.squeeze(1)
    y_test = torch.tensor(all_y[test_id], dtype=torch.long)

    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {len(torch.unique(y_train))}")

    # ---------- DataLoaders ----------

    # After creating your datasets, move them to GPU
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # ---------- Model ----------
    n_microstates = x_train.shape[1]  # Number of microstate categories
    n_classes = len(torch.unique(y_train))
    sequence_length = x_train.shape[2]
    
    # Create model using factory function
    model = mm.get_model(
        model_name=model_name,
        n_microstates=n_microstates,
        n_classes=n_classes,
        sequence_length=sequence_length,
        dropout=0.25  # Optional parameter
    )
    
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Using model: {model_name}")
    print(f"Model description: {mm.MODEL_INFO[model_name]['description']}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # ---------- Training ----------
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
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
        val_loss, val_correct, val_total = 0, 0, 0
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

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # ---------- Test ----------
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
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
plt.title(f'Subject {type_of_subject} Test Accuracies for Each Subject')
plt.xlabel('Subject ID')
plt.ylabel('Test Accuracy (%)')
plt.xticks(range(n_subjects), [f'S{i}' for i in range(n_subjects)], rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_test_accuracies.png'))

# # plot mean and std of train and val accuracies
# df = pd.DataFrame({
#     'Train Accuracy': all_train_accuracies,
#     'Val Accuracy': all_val_accuracies,
#     'Train loss': all_train_losses,
#     'Val loss': all_val_losses
# })

# # Compute mean and std for each metric across epochs
# metrics = ['Train Accuracy', 'Val Accuracy', 'Train loss', 'Val loss']
# mean_std_df = pd.DataFrame({'Epoch': np.arange(1, num_epochs + 1)})

# for metric in metrics:
#     values = np.array(df[metric].tolist())  # shape: (n_epochs, n_subjects)
#     mean_std_df[f"{metric} Mean"] = values.mean(axis=0)
#     mean_std_df[f"{metric} Std"] = values.std(axis=0)

# # --- Plotting ---
# fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey='row')
# fig.suptitle(f'{model_name.upper()} Training and Validation Metrics Over Epochs', fontsize=16)

# plot_params = [
#     ("Train Accuracy", axes[0, 0], "blue"),
#     ("Val Accuracy", axes[0, 1], "green"),
#     ("Train loss", axes[1, 0], "red"),
#     ("Val loss", axes[1, 1], "orange"),
# ]

# for metric, ax, color in plot_params:
#     mean = mean_std_df[f"{metric} Mean"]
#     std = mean_std_df[f"{metric} Std"]
#     epoch = mean_std_df["Epoch"]

#     ax.plot(epoch, mean, label=metric, color=color)
#     ax.fill_between(epoch, mean - std, mean + std, color=color, alpha=0.3)
#     ax.set_title(f"{metric} over Epochs")
#     ax.set_xlabel("Epoch")
#     ylabel = "Accuracy (%)" if "Accuracy" in metric else "Loss"
#     ax.set_ylabel(ylabel)

# plt.tight_layout()
# plt.savefig(os.path.join(output_path, f'{type_of_subject}_training_validation_metrics.png'))
# plt.legend()
# plt.subplots_adjust(top=0.9)  # Adjust top to make room for the suptitle
# plt.close()

print('==================== End of script microsnet_indep.py! ===================')