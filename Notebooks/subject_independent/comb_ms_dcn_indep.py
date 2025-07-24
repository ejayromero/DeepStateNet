'''
Script to train a combined model of EEG and Microstates DeepConvNet
'''

print('==================== Start of script comb_ms_dcn_indep.py! ===================')

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



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
from lib.my_models import FeatureExtractor, MultiModalClassifier, MicroSNet

# ---------------------------# Parameters ---------------------------
# excluded_from_training = [2, 12, 14, 20, 22, 23, 30, 39, 46]
excluded_from_training = [-1]  # No exclusions for independent clean
num_epochs = 50
batch_size = 32
type_of_subject = 'independent_harmonize'  # 'independent' or 'adaptive'
# ---------------------------# Load files ---------------------------
data_path = '../Data/'
output_path = f'../Output/ica_rest_all/{type_of_subject}/'
input_path = '../Output/ica_rest_all/'
do_all = False
n_subjects = 50
subject_list = list(range(n_subjects))

if not os.path.exists(output_path):
    os.makedirs(output_path)
# class FeatureExtractor(nn.Module):
#     """Extracts features from pre-trained models by removing the final classification layer"""
    
#     def __init__(self, pretrained_model):
#         super().__init__()
#         self.pretrained_model = pretrained_model
        
#         # Get the feature extractor (everything except the final classifier)
#         if hasattr(pretrained_model, 'module'):
#             # If it's wrapped in EEGClassifier, get the underlying network
#             self.backbone = pretrained_model.module
#         else:
#             self.backbone = pretrained_model
            
#         # Remove the final classification layer
#         # For Deep4Net, the classifier is typically the last layer
#         if hasattr(self.backbone, 'final_layer'):
#             # If there's a specific final layer attribute
#             modules = list(self.backbone.children())[:-1]
#         elif hasattr(self.backbone, 'classifier'):
#             # If there's a classifier attribute
#             modules = [module for name, module in self.backbone.named_children() 
#                     if name != 'classifier']
#         else:
#             # Remove the last linear/conv layer
#             modules = list(self.backbone.children())[:-1]
            
#         self.feature_extractor = nn.Sequential(*modules)
        
#         # Freeze the feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
            
#     def forward(self, x):
#         with torch.no_grad():
#             features = self.feature_extractor(x)
#             if len(features.shape) > 2:
#                 # max pooling across time dimension
#                 # features = F.adaptive_max_pool1d(features.flatten(1, 2), 1).squeeze(-1)
#                 # globale average pooling
#                 features = F.adaptive_avg_pool1d(features.flatten(1, 2), 1).squeeze(-1)


#             return features

# class MultiModalClassifier(nn.Module):
#     """Classifier that takes features from multiple modalities"""
    
#     def __init__(self, raw_feature_dim, ms_feature_dim, n_classes, dropout=0.5):
#         super().__init__()
        
#         self.raw_feature_dim = raw_feature_dim
#         self.ms_feature_dim = ms_feature_dim
#         self.n_classes = n_classes
        
#         # Feature fusion layer
#         total_features = raw_feature_dim + ms_feature_dim
        
#         self.classifier = nn.Sequential(
#             nn.Linear(total_features, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, n_classes)
#         )
        
#     def forward(self, raw_features, ms_features):
#         # Concatenate features from both modalities
#         combined_features = torch.cat([raw_features, ms_features], dim=1)
#         return self.classifier(combined_features)

def extract_features_from_single_model(model, data, device):
    """Extract features from a single model for one subject's data"""
    
    # Convert to tensor if needed
    if not isinstance(data, torch.Tensor):
        x = torch.tensor(data, dtype=torch.float32)
    else:
        x = data.clone()
    
    print(f"  Original data shape: {x.shape}")
    
    # Handle different data formats based on the expected input
    if len(x.shape) == 4:
        # Could be (n_trials, 1, n_channels, timepoints) or (n_trials, n_channels, 1, timepoints)
        if x.shape[1] == 1:  # (n_trials, 1, n_channels, timepoints) - raw EEG format
            x = x.squeeze(1)  # Remove singleton dimension -> (n_trials, n_channels, timepoints)
            print(f"  Raw EEG format detected, shape after squeeze: {x.shape}")
        elif x.shape[2] == 1:  # (n_trials, n_channels, 1, timepoints)
            x = x.squeeze(2)  # Remove singleton dimension -> (n_trials, n_channels, timepoints)
            print(f"  Format with singleton at dim 2, shape after squeeze: {x.shape}")
        else:
            print(f"  4D format without singleton, keeping as is: {x.shape}")
    elif len(x.shape) == 3:
        # Could be (n_trials, n_channels, timepoints) - microstate format
        print(f"  3D format (likely microstate), shape: {x.shape}")
    else:
        print(f"  Unexpected shape: {x.shape}")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()
    
    # Extract features
    features_list = []

    
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            batch_features = feature_extractor(batch_x)
            features_list.append(batch_features.cpu())
    
    subject_features = torch.cat(features_list, dim=0)
    print(f"  Extracted features shape: {subject_features.shape}")
    
    return subject_features

def train_single_multimodal_classifier(raw_features, ms_features, labels, n_classes, 
                                    subject_id, test_size=0.2, val_size=0.25, 
                                    device='cuda', num_epochs=100, lr=0.001):
    """Train a classifier on combined features for a single subject"""
    
    print(f"\nTraining multimodal classifier for Subject {subject_id}")
    print(f"Raw features shape: {raw_features.shape}")
    print(f"MS features shape: {ms_features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {torch.unique(labels)}")
    
    # set_seed(42 + subject_id)  # Different seed for each subject
    mf.set_seed(42)  # Use a fixed seed for reproducibility
    
    # Check if we have enough samples for splitting
    if len(labels) < 10:
        print(f"Warning: Only {len(labels)} samples for subject {subject_id}. Using simple train/test split.")
        test_size = 0.3
        val_size = 0.0  # No validation set for small datasets
    
    # Split data
    indices = np.arange(len(labels))
    
    if val_size > 0:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, 
            stratify=labels.numpy() if len(np.unique(labels.numpy())) > 1 else None
        )
        
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size, random_state=42,
            stratify=labels[train_val_idx].numpy() if len(np.unique(labels[train_val_idx].numpy())) > 1 else None
        )
    else:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42,
            stratify=labels.numpy() if len(np.unique(labels.numpy())) > 1 else None
        )
        val_idx = []
    
    # Create datasets
    train_dataset = TensorDataset(
        raw_features[train_idx], ms_features[train_idx], labels[train_idx]
    )
    test_dataset = TensorDataset(
        raw_features[test_idx], ms_features[test_idx], labels[test_idx]
    )
    
    # Create data loaders
    batch_size = min(32, len(train_idx) // 2)  # Adjust batch size for small datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if val_idx.any():
        val_dataset = TensorDataset(
            raw_features[val_idx], ms_features[val_idx], labels[val_idx]
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    # Initialize model
    raw_feature_dim = raw_features.shape[1]
    ms_feature_dim = ms_features.shape[1]
    
    model = MultiModalClassifier(
        raw_feature_dim, ms_feature_dim, n_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if val_loader:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for raw_batch, ms_batch, label_batch in train_loader:
            raw_batch = raw_batch.to(device)
            ms_batch = ms_batch.to(device)
            label_batch = label_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(raw_batch, ms_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * raw_batch.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == label_batch).sum().item()
            train_total += label_batch.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total * 100
        
        # Validation (if available)
        val_acc = 0
        val_loss = 0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for raw_batch, ms_batch, label_batch in val_loader:
                    raw_batch = raw_batch.to(device)
                    ms_batch = ms_batch.to(device)
                    label_batch = label_batch.to(device)
                    
                    outputs = model(raw_batch, ms_batch)
                    loss = criterion(outputs, label_batch)
                    
                    val_loss += loss.item() * raw_batch.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == label_batch).sum().item()
                    val_total += label_batch.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total * 100
            scheduler.step(val_acc)
        else:
            val_acc = train_acc  # Use training accuracy as proxy
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Acc: {val_acc:.2f}%")
    
    # Load best model and test
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for raw_batch, ms_batch, label_batch in test_loader:
            raw_batch = raw_batch.to(device)
            ms_batch = ms_batch.to(device)
            label_batch = label_batch.to(device)
            
            outputs = model(raw_batch, ms_batch)
            preds = outputs.argmax(dim=1)
            
            test_correct += (preds == label_batch).sum().item()
            test_total += label_batch.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())
    
    test_acc = test_correct / test_total * 100
    
    print(f"  Subject {subject_id} Results:")
    print(f"    Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"    Test Accuracy: {test_acc:.2f}%")
    
    return {
        'model': model,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs,
        'predictions': all_preds,
        'true_labels': all_labels,
        'subject_id': subject_id
    }

def run_subject_specific_multimodal_pipeline(results, ms_results, all_data, finals_ls, all_y, 
                                            n_subjects=50, device='cuda'):
    """
    Leave-one-subject-out (LOSO) training:
    - 1 subject = test
    - 4 randomly sampled validation subjects (fixed seed) from non-excluded subjects
    - Remaining non-excluded subjects for training
    - Excludes specified subjects from training/validation
    Uses the same output format as the original function.
    """
    print(f"Starting subject-specific multimodal pipeline for {n_subjects} subjects...")

    # Define subjects to exclude from training
    # excluded_from_training = [2, 12, 14, 20, 22, 23, 30, 39, 46]
    print(f"Excluded subjects from training: {excluded_from_training}")
    
    output_file = os.path.join(output_path, f'{type_of_subject}_comb_results_ica_rest_all.npy')

    for subject_id in range(n_subjects):
        if os.path.exists(output_file):
            print(f"Loading existing results for Subject {subject_id} from {output_file}")
            all_subject_results = np.load(output_file, allow_pickle=True).tolist()
        else:
            all_subject_results = []
        if len(all_subject_results) > subject_id:
            print(f"Skipping Subject {subject_id}, results already exist.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing Subject {subject_id}")
        print(f"{'='*60}")

        mf.set_seed(42)  # Ensure reproducible splits

        # --- LOSO Subject Splits with Exclusions ---
        # Choose validation subjects (4 random ones not equal to test_id and not in excluded list)
        val_candidates, val_ids = mf.get_val_ids(42, subject_id, excluded_from_training)

        # Remaining for training (excluding test subject and excluded subjects)
        train_ids = [i for i in val_candidates if i not in val_ids]

        print(f"Train IDs: {train_ids} (total: {len(train_ids)})")
        print(f"Val IDs:   {val_ids}")
        print(f"Test ID:   {subject_id}")
        print(f"Excluded from training: {excluded_from_training}")

        # --- Extract Features ---
        def extract_group_features(ids, model_dict, data_source):
            features, labels = [], []
            for i in ids:
                feats = extract_features_from_single_model(model_dict['models'][i], data_source[i], device)
                features.append(feats)
                labels.append(torch.tensor(all_y[i], dtype=torch.long))
            return torch.cat(features), torch.cat(labels)

        raw_train_feats, y_train = extract_group_features(train_ids, results, all_data)
        ms_train_feats, _ = extract_group_features(train_ids, ms_results, finals_ls)

        raw_val_feats, y_val = extract_group_features(val_ids, results, all_data)
        ms_val_feats, _ = extract_group_features(val_ids, ms_results, finals_ls)

        raw_features = torch.cat([raw_train_feats, raw_val_feats], dim=0)
        ms_features = torch.cat([ms_train_feats, ms_val_feats], dim=0)
        labels = torch.cat([y_train, y_val], dim=0)

        val_size_ratio = len(y_val) / len(labels)

        # --- Train Classifier ---
        subject_results = train_single_multimodal_classifier(
            raw_features, ms_features, labels,
            n_classes=len(torch.unique(labels)),
            subject_id=subject_id,
            test_size=val_size_ratio,
            device=device
        )

        # --- Test on held-out subject ---
        raw_test_feats = extract_features_from_single_model(results['models'][subject_id], all_data[subject_id], device)
        ms_test_feats = extract_features_from_single_model(ms_results['models'][subject_id], finals_ls[subject_id], device)
        y_test = torch.tensor(all_y[subject_id], dtype=torch.long)

        model = subject_results['model'].to(device)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(y_test), 32):
                out = model(raw_test_feats[i:i+32].to(device), ms_test_feats[i:i+32].to(device))
                preds.extend(out.argmax(dim=1).cpu().numpy())

        test_acc = accuracy_score(y_test.numpy(), preds) * 100
        print(f"✅ Subject {subject_id} Test Accuracy: {test_acc:.2f}%")

        subject_results.update({
            'test_accuracy': test_acc,
            'true_labels': y_test.numpy(),
            'predictions': preds
        })

        all_subject_results.append(subject_results)
        print(f"Saving results for Subject {subject_id}...")
        np.save(output_file, all_subject_results, allow_pickle=True)


    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL SUBJECTS")
    print(f"{'='*60}")

    test_accuracies = [res['test_accuracy'] for res in all_subject_results]
    val_accuracies = [res['best_val_accuracy'] for res in all_subject_results]

    print(f"Test Accuracies:")
    for i, acc in enumerate(test_accuracies):
        print(f"  Subject {i}: {acc:.2f}%")

    print(f"\nAverage Test Accuracy: {np.mean(test_accuracies):.2f}% ± {np.std(test_accuracies):.2f}%")
    print(f"Average Validation Accuracy: {np.mean(val_accuracies):.2f}% ± {np.std(val_accuracies):.2f}%")

    return {
        'subject_results': all_subject_results,
        'summary': {
            'test_accuracies': test_accuracies,
            'val_accuracies': val_accuracies,
            'mean_test_acc': np.mean(test_accuracies),
            'std_test_acc': np.std(test_accuracies),
            'mean_val_acc': np.mean(val_accuracies),
            'std_val_acc': np.std(val_accuracies)
        }
    }

# ---------------------------# Load data ---------------------------
all_data, all_y = mf.load_all_data(subjects_list=None, do_all=do_all)

ms_timeseries_path = os.path.join(input_path, 'ms_timeseries_harmonize.pkl')
with open(ms_timeseries_path, 'rb') as f:
    finals_ls = pickle.load(f)


results = np.load(os.path.join(output_path, f'{type_of_subject}_results_ica_rest_all.npy'), allow_pickle=True).item()
ms_results = np.load(os.path.join(output_path, f'{type_of_subject}_ms_results_ica_rest_all.npy'), allow_pickle=True).item()
print(f'N_models in results: {len(results["models"])}')
print(f'N_models in ms_results: {len(ms_results["models"])}')

# ----------------------------------- Run training -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run the subject-specific multimodal pipeline for all subjects
multimodal_results = run_subject_specific_multimodal_pipeline(
    results, ms_results, all_data, finals_ls, all_y, 
    n_subjects=len(ms_results['models']), device=device
)
# Save the results
output_file = os.path.join(output_path, f'{type_of_subject}_multimodal_results_ica_rest_all.npy')
np.save(output_file, multimodal_results, allow_pickle=True)

print(f"\nMultimodal pipeline completed!")
print(f"Average performance across 50 subjects: {multimodal_results['summary']['mean_test_acc']:.2f}%")

# Access individual subject results
for i, subject_result in enumerate(multimodal_results['subject_results']):
    print(f"Subject {i}: {subject_result['test_accuracy']:.2f}% test accuracy")
    # Access the trained model: subject_result['model']

accuracy_list = [res['test_accuracy'] for res in multimodal_results['subject_results']]

# ------------------------------ Plotting the results ------------------------------
plt.figure(figsize=(15, 6))
plt.plot(range(len(accuracy_list)), accuracy_list, marker='o', linestyle='-')
plt.title(f'Subject {type_of_subject} Test Accuracies for Each Subject')
plt.xlabel('Subject ID')
plt.ylabel('Test Accuracy (%)')
plt.xticks(range(len(accuracy_list)), [f'S{i}' for i in range(len(accuracy_list))], rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_multimodal_test_accuracies.png'))
plt.close()


plt.figure(figsize=(15, 6))
plt.plot(results['test_accuracies'], marker='x', linestyle='--', label='DeepConvNet Raw EEG', color='orange')
plt.plot(ms_results['test_accuracies'], marker='s', linestyle=':', label='DeepConvNet Microstate', color='green')
plt.plot(accuracy_list, marker='o', linestyle='-', label='Combined', color='blue')
plt.xlabel('Subject ID')
plt.ylabel('Test Accuracy (%)')
plt.xticks(range(50), [f'S{i}' for i in range(50)], rotation=45)
plt.title(f'Subject {type_of_subject} Test Accuracies for Each Subject')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_multimodal_vs_individual_test_accuracies.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.violinplot(data=[results['test_accuracies'], ms_results['test_accuracies'], accuracy_list],
            palette=['orange', 'green', 'blue'], cut=0)
plt.xticks([0, 1, 2], ['DeepConvNet Raw EEG', 'DeepConvNet Microstate', 'Combined'])
plt.title(f'Subject {type_of_subject} Distribution of Test Accuracies')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Model Type')
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_test_accuracy_distribution.png'))
plt.close()
print('==================== End of script comb_ms_dcn_indep.py! ===================')