'''
Script to train a combined model of EEG DeepConvNet and MicroSNet
'''
import os
import gc
import sys
import numpy as np
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import my_functions as mf
from lib.my_models import MicroSNet, FeatureExtractor, MultiModalClassifier, EmbeddedMicroSNet


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
mf.print_memory_status("- INITIAL STARTUP")
# ---------------------------# Load files ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
project_root = os.path.dirname(os.path.dirname(script_dir))  # Up 2 levels: Code -> Master-Thesis
# Set absolute paths
data_path = os.path.join(project_root, 'Data') + os.sep
output_folder = os.path.join(project_root, 'Output') + os.sep

type_of_subject = 'dependent'  # 'independent' or 'adaptive'
model_name = 'attention_microsnet'  # 'microsnet' or 'multiscale_microsnet'
output_path = f'{output_folder}ica_rest_all/{type_of_subject}/'
input_path = f'{output_folder}ica_rest_all/'
# Making sure all paths exist
if not os.path.exists(input_path):
    os.makedirs(input_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Parameters
do_all = False
n_subjects = 50
batch_size = 32
subject_list = list(range(n_subjects))
all_data, all_y = mf.load_all_data(subjects_list=None, do_all=do_all, data_path=data_path)
mf.print_memory_status("- AFTER DATA LOADING") 
if 'embedded' in model_name or 'attention' in model_name:
    kmeans_path = os.path.join(input_path, 'modkmeans_results', 'modkmeans_sequence')
    ms_timeseries_path = os.path.join(kmeans_path, 'modkmeans_sequence_indiv.pkl')
else:
    kmeans_path = os.path.join(input_path, 'modkmeans_results', 'ms_timeseries')
    ms_timeseries_path = os.path.join(kmeans_path, 'ms_timeseries_harmonize.pkl')
with open(ms_timeseries_path, 'rb') as f:
    finals_ls = pickle.load(f)


results = np.load(os.path.join(output_path, f'{type_of_subject}_results_ica_rest_all.npy'), allow_pickle=True).item()
ms_results = np.load(os.path.join(output_path, f'{type_of_subject}_ms_{model_name}_results_ica_rest_all.npy'), allow_pickle=True).item()
print(f'N_models in results: {len(results["models"])}')
print(f'N_models in ms_results: {len(ms_results["models"])}')


def extract_features_from_single_model(model, data, device):
    """Extract features from a single model for one subject's data"""
    
    # Convert to tensor if needed
    if not isinstance(data, torch.Tensor):
        x = torch.tensor(data, dtype=torch.float32)
    else:
        x = data.clone()
    
    print(f"  Original data shape: {x.shape}")
    
    # Determine model type first to handle data conversion correctly
    model_backbone = model.module if hasattr(model, 'module') else model
    is_embedded_model = hasattr(model_backbone, 'microstate_embedding') or isinstance(model_backbone, EmbeddedMicroSNet)
    
    print(f"  Is embedded model: {is_embedded_model}")
    
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
        print(f"  3D format detected, shape: {x.shape}")
        
        # Check if this is microstate data with 2 channels (microstates + GFP)
        if x.shape[1] == 2:
            print("  Detected microstate + GFP format")
            
            if is_embedded_model:
                print("  Model is EmbeddedMicroSNet - extracting microstate sequences only")
                # Extract only microstate sequences (first channel) for embedded models
                x_microstates = x[:, 0, :]  # Shape: (batch_size, sequence_length)
                
                # Handle negative indices (embedding layers require indices >= 0)
                min_val = torch.min(x_microstates).item()
                if min_val < 0:
                    print(f"  Found negative microstate indices ({min_val}), shifting to start from 0...")
                    x_microstates = x_microstates - min_val
                    print(f"  New microstate range: {torch.min(x_microstates).item()} to {torch.max(x_microstates).item()}")
                
                # CRITICAL: Convert to integer type for embedding
                x = x_microstates.long()
                print(f"  Converted to integer type for embedding: {x.dtype}")
                print(f"  Preprocessed data shape for EmbeddedMicroSNet: {x.shape}")
            else:
                print("  Model expects one-hot encoded data - converting microstate sequences")
                # For other models, convert to one-hot encoding
                x_microstates = x[:, 0, :].long()
                
                # Handle negative indices
                min_val = torch.min(x_microstates).item()
                if min_val < 0:
                    x_microstates = x_microstates - min_val
                
                n_microstates = int(torch.max(x_microstates).item()) + 1
                sequence_length = x_microstates.shape[1]
                
                # Convert to one-hot encoding
                x_onehot = torch.zeros(x_microstates.shape[0], n_microstates, sequence_length)
                x_onehot.scatter_(1, x_microstates.unsqueeze(1), 1)
                x = x_onehot.float()
                print(f"  Converted to one-hot shape: {x.shape}")
        else:
            # Standard 3D format - ensure correct type
            if is_embedded_model and x.dtype == torch.float32:
                x = x.long()
                print(f"  Converted 3D data to integer type for embedding: {x.dtype}")
            elif not is_embedded_model and x.dtype != torch.float32:
                x = x.float()
                print(f"  Converted 3D data to float type: {x.dtype}")
                
    elif len(x.shape) == 2:
        # Could be (n_trials, sequence_length) - embedded microstate format
        print(f"  2D format (likely embedded microstate), shape: {x.shape}")
        
        if is_embedded_model:
            # Ensure integer type for embedding
            if x.dtype == torch.float32 or x.dtype == torch.float64:
                print(f"  Converting 2D float data to integer indices for embedded model")
                x = x.long()
            
            # Handle negative indices
            min_val = torch.min(x).item()
            if min_val < 0:
                print(f"  Found negative indices ({min_val}), shifting to start from 0...")
                x = x - min_val
                print(f"  New range: {torch.min(x).item()} to {torch.max(x).item()}")
                
        print(f"  Final data type: {x.dtype}")
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
            try:
                batch_features = feature_extractor(batch_x)
                features_list.append(batch_features.cpu())
            except Exception as e:
                print(f"  Error processing batch {i}: {e}")
                print(f"  Batch shape: {batch_x.shape}, dtype: {batch_x.dtype}")
                raise
    
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
                                           n_subjects=5, device='cuda'):
    """
    Main function to run subject-specific multimodal classifiers
    
    Args:
        results: Dictionary containing 'models' key with raw EEG models
        ms_results: Dictionary containing 'models' key with microstate models (first 5 subjects)
        all_data: List of arrays for raw EEG, shape (n_subjects, n_trials, 1, n_channels, timepoints)
        finals_ls: List of arrays for microstate timeseries, shape (n_subjects, n_trials, n_channels, timepoints)
        all_y: List of arrays, shape (n_subjects, n_trials)
        n_subjects: Number of subjects to process (default 5)
        device: Device to run computations on
    """
    
    print(f"Starting subject-specific multimodal pipeline for {n_subjects} subjects...")
    
    # Ensure we have enough models
    assert len(ms_results['models']) >= n_subjects, f"ms_results has only {len(ms_results['models'])} models, need {n_subjects}"
    assert len(results['models']) >= n_subjects, f"results has only {len(results['models'])} models, need {n_subjects}"
    
    all_subject_results = []
    
    for subject_id in range(n_subjects):
        mf.print_memory_status(f"- SUBJECT {id} START")  # Optional: at start of each subject
    
        print(f"\n{'='*60}")
        print(f"Processing Subject {subject_id}")
        print(f"{'='*60}")
        
        # Get data for this subject
        raw_data = all_data[subject_id]
        ms_data = finals_ls[subject_id]
        labels = torch.tensor(all_y[subject_id], dtype=torch.long)
        
        print(f"Raw data shape: {np.array(raw_data).shape}")
        print(f"MS data shape: {np.array(ms_data).shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Extract features from raw EEG model
        print(f"Extracting features from raw EEG model...")
        raw_features = extract_features_from_single_model(
            results['models'][subject_id], raw_data, device
        )
        
        # Extract features from microstate model
        print(f"Extracting features from microstate model...")
        ms_features = extract_features_from_single_model(
            ms_results['models'][subject_id], ms_data, device
        )
        
        # Get number of classes for this subject
        n_classes = len(torch.unique(labels))
        print(f"Number of classes: {n_classes}")
        
        # Print data distribution
        print(f"Total samples: {len(labels)}")
        for class_id in range(n_classes):
            count = (labels == class_id).sum().item()
            print(f"  Class {class_id}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        # Train subject-specific multimodal classifier
        subject_results = train_single_multimodal_classifier(
            raw_features, ms_features, labels, n_classes, subject_id, device=device
        )
        
        all_subject_results.append(subject_results)
        print(f"Subject {subject_id} results: {subject_results['test_accuracy']:.2f}% test accuracy")
        mf.print_memory_status(f"- SUBJECT {id} END")
    # Print summary
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
    summary_results = {
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
    return summary_results

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
mf.print_memory_status("- AFTER GARBAGE COLLECTION")

# Run the subject-specific multimodal pipeline for first 5 subjects
multimodal_results = run_subject_specific_multimodal_pipeline(
    results, ms_results, all_data, finals_ls, all_y, 
    n_subjects=len(ms_results['models']), device=device
)

print(f"\nMultimodal pipeline completed!")
print(f"Average performance across 50 subjects: {multimodal_results['summary']['mean_test_acc']:.2f}%")

# Access individual subject results
for i, subject_result in enumerate(multimodal_results['subject_results']):
    print(f"Subject {i}: {subject_result['test_accuracy']:.2f}% test accuracy")
    # Access the trained model: subject_result['model']

# Save the results
output_file = os.path.join(output_path, f'{type_of_subject}_multimodal_{model_name}_results_ica_rest_all.npy')
np.save(output_file, multimodal_results)

print(f'==================== End of script {os.path.basename(__file__)}! ===================')