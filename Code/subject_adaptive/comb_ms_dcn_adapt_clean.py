'''
Script to train a combined model of EEG and Microstates DeepConvNet with fine-tuning
Clean version with embedded model support and new repository structure
'''

print('==================== Start of script comb_ms_dcn_adapt_clean.py! ===================')

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
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import my_functions as mf
from lib.my_models import FeatureExtractor, MultiModalClassifier, get_model, EmbeddedMicroSNet

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

# ---------------------------# Parameters ---------------------------
excluded_from_training = [-1]  # No exclusions for adaptive clean
num_epochs = 100
batch_size = 32
type_of_subject = 'adaptive_harmonize'  # 'independent' or 'adaptive'
model_name = 'embedded_microsnet'  # 'microsnet' or 'multiscale_microsnet' or 'embedded_microsnet'
# Fine-tuning parameters
finetune_epochs = 50
finetune_lr = 0.0001  # Lower learning rate for fine-tuning

# ---------------------------# Load files ---------------------------
data_path = 'Data/'
output_path = f'Output/ica_rest_all/{type_of_subject}/'
input_path = 'Output/ica_rest_all/'
do_all = False
n_subjects = 50
subject_list = list(range(n_subjects))

# Making sure all paths exist
if not os.path.exists(input_path):
    os.makedirs(input_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

all_data, all_y = mf.load_all_data(subjects_list=None, do_all=do_all, data_path=data_path)

# Load microstate data based on model type
if 'embedded' in model_name:
    kmeans_path = os.path.join(input_path, 'modkmeans_results', 'modkmeans_sequence')
    ms_timeseries_path = os.path.join(kmeans_path, 'modkmeans_sequence_harmonize_indiv.pkl')
else:
    kmeans_path = os.path.join(input_path, 'modkmeans_results', 'ms_timeseries')
    ms_timeseries_path = os.path.join(kmeans_path, 'ms_timeseries_harmonize.pkl')

with open(ms_timeseries_path, 'rb') as f:
    finals_ls = pickle.load(f)

# Load results from individual models
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
        # Format: (n_trials, n_channels, timepoints)
        print(f"  3D format detected, shape: {x.shape}")
        
        # Check if this is microstate data with 2 channels (microstates + GFP)
        if x.shape[1] == 2:
            print("  Detected microstate + GFP format with 2 channels")
            
            if is_embedded_model:
                print("  Processing for EmbeddedMicroSNet - extracting microstate sequences only")
                # Extract only microstate sequences (first channel) for embedded models
                x_microstates = x[:, 0, :]  # Shape: (n_trials, sequence_length)
                print(f"  Extracted microstate sequences shape: {x_microstates.shape}")
                
                # Handle negative indices (embedding layers require indices >= 0)
                min_val = torch.min(x_microstates).item()
                max_val = torch.max(x_microstates).item()
                print(f"  Microstate value range: {min_val} to {max_val}")
                
                if min_val < 0:
                    print(f"  Shifting negative indices to start from 0...")
                    x_microstates = x_microstates - min_val
                    new_min = torch.min(x_microstates).item()
                    new_max = torch.max(x_microstates).item()
                    print(f"  New microstate range: {new_min} to {new_max}")
                
                # CRITICAL: Convert to integer type for embedding
                x = x_microstates.long()
                print(f"  Final data shape for EmbeddedMicroSNet: {x.shape}, dtype: {x.dtype}")
                
            else:
                print("  Processing for standard MicroSNet - converting to one-hot encoding")
                # For other models, convert to one-hot encoding
                x_microstates = x[:, 0, :].long()  # Extract microstate sequences
                
                # Handle negative indices
                min_val = torch.min(x_microstates).item()
                if min_val < 0:
                    print(f"  Shifting negative indices: {min_val}")
                    x_microstates = x_microstates - min_val
                
                n_microstates = int(torch.max(x_microstates).item()) + 1
                sequence_length = x_microstates.shape[1]
                print(f"  Creating one-hot encoding: {n_microstates} microstates, {sequence_length} timepoints")
                
                # Convert to one-hot encoding: (n_trials, n_microstates, sequence_length)
                x_onehot = torch.zeros(x_microstates.shape[0], n_microstates, sequence_length)
                x_onehot.scatter_(1, x_microstates.unsqueeze(1), 1)
                x = x_onehot.float()
                print(f"  One-hot encoded shape: {x.shape}")
        else:
            # Other 3D formats
            print(f"  Standard 3D format with {x.shape[1]} channels")
            if is_embedded_model:
                # For embedded models, assume this is already in the right format but wrong type
                if x.dtype in [torch.float32, torch.float64]:
                    x = x.long()
                    print(f"  Converted to integer type for embedding: {x.dtype}")
            else:
                # For non-embedded models, ensure float type
                if x.dtype != torch.float32:
                    x = x.float()
                    print(f"  Converted to float type: {x.dtype}")
                    
    elif len(x.shape) == 2:
        # Format: (n_trials, sequence_length) - embedded microstate format
        print(f"  2D format (embedded microstate sequences), shape: {x.shape}")
        
        if is_embedded_model:
            # Ensure integer type for embedding
            if x.dtype in [torch.float32, torch.float64]:
                print(f"  Converting 2D float data to integer indices for embedded model")
                x = x.long()
            
            # Handle negative indices
            min_val = torch.min(x).item()
            max_val = torch.max(x).item()
            print(f"  Value range: {min_val} to {max_val}")
            
            if min_val < 0:
                print(f"  Shifting negative indices to start from 0...")
                x = x - min_val
                new_min = torch.min(x).item()
                new_max = torch.max(x).item()
                print(f"  New range: {new_min} to {new_max}")
                
        print(f"  Final 2D data type: {x.dtype}, shape: {x.shape}")
    else:
        print(f"  Unexpected shape: {x.shape}")
        raise ValueError(f"Unexpected data shape: {x.shape}")
    
    # Validate final data format before feature extraction
    print(f"  Final preprocessed data - Shape: {x.shape}, Dtype: {x.dtype}")
    
    if is_embedded_model:
        if x.dtype not in [torch.int64, torch.int32, torch.long]:
            raise ValueError(f"EmbeddedMicroSNet requires integer data, got {x.dtype}")
        if len(x.shape) != 2:
            raise ValueError(f"EmbeddedMicroSNet requires 2D input (batch_size, sequence_length), got {x.shape}")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()
    
    # Extract features
    features_list = []
    
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            print(f"  Processing batch {i//batch_size + 1}: shape {batch_x.shape}, dtype {batch_x.dtype}")
            
            try:
                batch_features = feature_extractor(batch_x)
                features_list.append(batch_features.cpu())
            except Exception as e:
                print(f"  ERROR in batch {i//batch_size + 1}:")
                print(f"    Batch shape: {batch_x.shape}")
                print(f"    Batch dtype: {batch_x.dtype}")
                print(f"    Model type: {type(model_backbone)}")
                print(f"    Is embedded: {is_embedded_model}")
                print(f"    Error: {e}")
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
    
    if len(val_idx) > 0:
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

def finetune_multimodal_classifier(model, raw_features, ms_features, labels, 
                                 subject_id, device='cuda', num_epochs=50, lr=0.0001):
    """Fine-tune a pre-trained multimodal classifier on subject-specific data"""
    
    print(f"\nFine-tuning multimodal classifier for Subject {subject_id}")
    print(f"Raw features shape: {raw_features.shape}")
    print(f"MS features shape: {ms_features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Create dataset and loader
    finetune_dataset = TensorDataset(raw_features, ms_features, labels)
    finetune_batch_size = min(16, len(labels) // 2)  # Smaller batch size for fine-tuning
    finetune_loader = DataLoader(finetune_dataset, batch_size=finetune_batch_size, shuffle=True)
    
    # Set up optimizer with lower learning rate for fine-tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Fine-tuning loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for raw_batch, ms_batch, label_batch in finetune_loader:
            raw_batch = raw_batch.to(device)
            ms_batch = ms_batch.to(device)
            label_batch = label_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(raw_batch, ms_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * raw_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == label_batch).sum().item()
            total += label_batch.size(0)
        
        epoch_loss /= total
        epoch_acc = correct / total * 100
        
        if epoch % 10 == 0:
            print(f"  Fine-tune Epoch {epoch}/{num_epochs}: "
                f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    print(f"  Fine-tuning completed for Subject {subject_id}")
    return model

def run_subject_specific_multimodal_pipeline_with_finetuning(results, ms_results, all_data, finals_ls, all_y, 
                                                           n_subjects=50, device='cuda'):
    """
    Modified pipeline with fine-tuning:
    1. Train on other subjects (LOSO)
    2. Fine-tune on 90% of test subject's data
    3. Test on remaining 10% of test subject's data
    """
    print(f"Starting subject-specific multimodal pipeline with fine-tuning for {n_subjects} subjects...")

    print(f"Excluded subjects from training: {excluded_from_training}")
    
    output_file = os.path.join(output_path, f'{type_of_subject}_comb_{model_name}_results_ica_rest_all.npy')
    
    # Try to load existing split indices for consistency
    all_train_indices, all_test_indices = mf.load_split_indices(output_path, filename=f'{type_of_subject}_split_indices.pkl')
    
    # If no pre-saved indices, generate them
    if all_train_indices is None:
        all_train_indices = {}
        all_test_indices = {}
        for subject_id in range(n_subjects):
            data_length = len(all_y[subject_id])
            train_indices, test_indices = mf.get_consistent_split_indices(
                data_length, subject_id, train_ratio=0.9, base_seed=42
            )
            all_train_indices[subject_id] = train_indices
            all_test_indices[subject_id] = test_indices
        
        # Save the indices for future use
        mf.save_split_indices(output_path, all_train_indices, all_test_indices, filename=f'{type_of_subject}_split_indices.pkl')

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
        print(f"Processing Subject {subject_id} with Fine-tuning")
        print(f"{'='*60}")

        mf.set_seed(42)  # Ensure reproducible splits

        # --- LOSO Subject Splits with Exclusions ---
        val_candidates, val_ids = mf.get_val_ids(42, subject_id, excluded_from_training)
        train_ids = [i for i in val_candidates if i not in val_ids]

        print(f"Train IDs: {train_ids} (total: {len(train_ids)})")
        print(f"Val IDs:   {val_ids}")
        print(f"Test ID:   {subject_id}")

        # --- Extract Features for Training ---
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

        # --- Train Base Classifier ---
        print("Training base classifier...")
        subject_results = train_single_multimodal_classifier(
            raw_features, ms_features, labels,
            n_classes=len(torch.unique(labels)),
            subject_id=subject_id,
            test_size=val_size_ratio,
            device=device,
            num_epochs=num_epochs
        )

        # --- Prepare Test Subject Data for Fine-tuning ---
        # Extract features for the test subject
        raw_test_full_feats = extract_features_from_single_model(
            results['models'][subject_id], all_data[subject_id], device
        )
        ms_test_full_feats = extract_features_from_single_model(
            ms_results['models'][subject_id], finals_ls[subject_id], device
        )
        y_test_full = torch.tensor(all_y[subject_id], dtype=torch.long)

        # Split test subject data using consistent indices
        finetune_indices = all_train_indices[subject_id]
        final_test_indices = all_test_indices[subject_id]

        raw_finetune_feats = raw_test_full_feats[finetune_indices]
        ms_finetune_feats = ms_test_full_feats[finetune_indices]
        y_finetune = y_test_full[finetune_indices]

        raw_final_test_feats = raw_test_full_feats[final_test_indices]
        ms_final_test_feats = ms_test_full_feats[final_test_indices]
        y_final_test = y_test_full[final_test_indices]

        print(f"Fine-tuning data: {len(y_finetune)} samples")
        print(f"Final test data: {len(y_final_test)} samples")

        # --- Fine-tune the Model ---
        print("Fine-tuning the model...")
        finetuned_model = finetune_multimodal_classifier(
            subject_results['model'], raw_finetune_feats, ms_finetune_feats, y_finetune,
            subject_id=subject_id, device=device, num_epochs=finetune_epochs, lr=finetune_lr
        )

        # --- Test on Final Test Set ---
        print("Testing on final test set...")
        finetuned_model.eval()
        final_preds = []
        
        with torch.no_grad():
            for i in range(0, len(y_final_test), 32):
                raw_batch = raw_final_test_feats[i:i+32].to(device)
                ms_batch = ms_final_test_feats[i:i+32].to(device)
                outputs = finetuned_model(raw_batch, ms_batch)
                final_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        final_test_acc = accuracy_score(y_final_test.numpy(), final_preds) * 100
        print(f"✅ Subject {subject_id} Final Test Accuracy (after fine-tuning): {final_test_acc:.2f}%")

        # --- Test with Base Model (no fine-tuning) for comparison ---
        base_model = subject_results['model']
        base_model.eval()
        base_preds = []
        
        with torch.no_grad():
            for i in range(0, len(y_final_test), 32):
                raw_batch = raw_final_test_feats[i:i+32].to(device)
                ms_batch = ms_final_test_feats[i:i+32].to(device)
                outputs = base_model(raw_batch, ms_batch)
                base_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        base_test_acc = accuracy_score(y_final_test.numpy(), base_preds) * 100
        print(f"📊 Subject {subject_id} Base Test Accuracy (no fine-tuning): {base_test_acc:.2f}%")

        # Update results
        subject_results.update({
            'base_test_accuracy': base_test_acc,
            'finetuned_test_accuracy': final_test_acc,
            'base_predictions': base_preds,
            'finetuned_predictions': final_preds,
            'final_test_labels': y_final_test.numpy(),
            'finetune_improvement': final_test_acc - base_test_acc,
            'finetuned_model': finetuned_model
        })

        all_subject_results.append(subject_results)
        print(f"Saving results for Subject {subject_id}...")
        np.save(output_file, all_subject_results, allow_pickle=True)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL SUBJECTS (WITH FINE-TUNING)")
    print(f"{'='*60}")

    base_accuracies = [res['base_test_accuracy'] for res in all_subject_results]
    finetuned_accuracies = [res['finetuned_test_accuracy'] for res in all_subject_results]
    improvements = [res['finetune_improvement'] for res in all_subject_results]

    print(f"Base Test Accuracies (no fine-tuning):")
    for i, acc in enumerate(base_accuracies):
        print(f"  Subject {i}: {acc:.2f}%")

    print(f"\nFine-tuned Test Accuracies:")
    for i, acc in enumerate(finetuned_accuracies):
        print(f"  Subject {i}: {acc:.2f}%")

    print(f"\nImprovements from Fine-tuning:")
    for i, imp in enumerate(improvements):
        print(f"  Subject {i}: {imp:+.2f}%")

    print(f"\nAverage Base Test Accuracy: {np.mean(base_accuracies):.2f}% ± {np.std(base_accuracies):.2f}%")
    print(f"Average Fine-tuned Test Accuracy: {np.mean(finetuned_accuracies):.2f}% ± {np.std(finetuned_accuracies):.2f}%")
    print(f"Average Improvement: {np.mean(improvements):+.2f}% ± {np.std(improvements):.2f}%")

    return {
        'subject_results': all_subject_results,
        'summary': {
            'base_test_accuracies': base_accuracies,
            'finetuned_test_accuracies': finetuned_accuracies,
            'improvements': improvements,
            'mean_base_acc': np.mean(base_accuracies),
            'std_base_acc': np.std(base_accuracies),
            'mean_finetuned_acc': np.mean(finetuned_accuracies),
            'std_finetuned_acc': np.std(finetuned_accuracies),
            'mean_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements)
        }
    }

# ----------------------------------- Run training with fine-tuning -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run the subject-specific multimodal pipeline with fine-tuning
multimodal_results = run_subject_specific_multimodal_pipeline_with_finetuning(
    results, ms_results, all_data, finals_ls, all_y, 
    n_subjects=len(ms_results['models']), device=device
)

# Save the results
output_file = os.path.join(output_path, f'{type_of_subject}_multimodal_{model_name}_results_ica_rest_all.npy')
np.save(output_file, multimodal_results, allow_pickle=True)

print(f"\nMultimodal pipeline with fine-tuning completed!")
print(f"Average base performance: {multimodal_results['summary']['mean_base_acc']:.2f}%")
print(f"Average fine-tuned performance: {multimodal_results['summary']['mean_finetuned_acc']:.2f}%")
print(f"Average improvement: {multimodal_results['summary']['mean_improvement']:+.2f}%")

# Access individual subject results
for i, subject_result in enumerate(multimodal_results['subject_results']):
    print(f"Subject {i}: Base {subject_result['base_test_accuracy']:.2f}% -> "
          f"Fine-tuned {subject_result['finetuned_test_accuracy']:.2f}% "
          f"(improvement: {subject_result['finetune_improvement']:+.2f}%)")

base_accuracy_list = [res['base_test_accuracy'] for res in multimodal_results['subject_results']]
finetuned_accuracy_list = [res['finetuned_test_accuracy'] for res in multimodal_results['subject_results']]
improvement_list = [res['finetune_improvement'] for res in multimodal_results['subject_results']]

# ------------------------------ Plotting the results ------------------------------

# Plot 1: Base vs Fine-tuned accuracies
plt.figure(figsize=(15, 6))
x_positions = range(len(base_accuracy_list))
plt.plot(x_positions, base_accuracy_list, marker='s', linestyle='--', 
         label='Base Model (no fine-tuning)', color='red', alpha=0.7)
plt.plot(x_positions, finetuned_accuracy_list, marker='o', linestyle='-', 
         label='Fine-tuned Model', color='blue')
plt.title(f'Subject {type_of_subject} {model_name.upper()} Test Accuracies: Base vs Fine-tuned')
plt.xlabel('Subject ID')
plt.ylabel('Test Accuracy (%)')
plt.xticks(range(len(base_accuracy_list)), [f'S{i}' for i in range(len(base_accuracy_list))], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_base_vs_finetuned_accuracies.png'))
plt.close()

# Plot 2: Improvement from fine-tuning
plt.figure(figsize=(15, 6))
colors = ['green' if x > 0 else 'red' for x in improvement_list]
plt.bar(range(len(improvement_list)), improvement_list, color=colors, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.title(f'Subject {type_of_subject} {model_name.upper()} Improvement from Fine-tuning')
plt.xlabel('Subject ID')
plt.ylabel('Accuracy Improvement (%)')
plt.xticks(range(len(improvement_list)), [f'S{i}' for i in range(len(improvement_list))], rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_finetuning_improvements.png'))
plt.close()

# Plot 3: Comprehensive comparison with original models
plt.figure(figsize=(15, 6))
plt.plot(results['test_accuracies'], marker='x', linestyle='--', 
         label='DeepConvNet Raw EEG', color='orange')
plt.plot(ms_results['test_accuracies'], marker='s', linestyle=':', 
         label=f'DeepConvNet {model_name.replace("_", " ").title()}', color='green')
plt.plot(base_accuracy_list, marker='^', linestyle='-.', 
         label='Combined (Base)', color='purple')
plt.plot(finetuned_accuracy_list, marker='o', linestyle='-', 
         label='Combined (Fine-tuned)', color='blue')
plt.xlabel('Subject ID')
plt.ylabel('Test Accuracy (%)')
plt.xticks(range(50), [f'S{i}' for i in range(50)], rotation=45)
plt.title(f'Subject {type_of_subject} {model_name.upper()} Test Accuracies: All Models Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_all_models_comparison.png'))
plt.close()

# Plot 4: Distribution comparison
plt.figure(figsize=(12, 6))
data_to_plot = [
    results['test_accuracies'], 
    ms_results['test_accuracies'], 
    base_accuracy_list, 
    finetuned_accuracy_list
]
labels = ['Raw EEG', f'{model_name.replace("_", " ").title()}', 'Combined (Base)', 'Combined (Fine-tuned)']
colors = ['orange', 'green', 'purple', 'blue']

sns.violinplot(data=data_to_plot, palette=colors, cut=0)
plt.xticks(range(4), labels)
plt.title(f'Subject {type_of_subject} {model_name.upper()} Distribution of Test Accuracies')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Model Type')
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_accuracy_distributions_with_finetuning.png'))
plt.close()

# Plot 5: Statistical summary table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create summary statistics
summary_data = {
    'Model': ['Raw EEG', f'{model_name.replace("_", " ").title()}', 'Combined (Base)', 'Combined (Fine-tuned)'],
    'Mean (%)': [
        f"{np.mean(results['test_accuracies']):.2f}",
        f"{np.mean(ms_results['test_accuracies']):.2f}",
        f"{np.mean(base_accuracy_list):.2f}",
        f"{np.mean(finetuned_accuracy_list):.2f}"
    ],
    'Std (%)': [
        f"{np.std(results['test_accuracies']):.2f}",
        f"{np.std(ms_results['test_accuracies']):.2f}",
        f"{np.std(base_accuracy_list):.2f}",
        f"{np.std(finetuned_accuracy_list):.2f}"
    ],
    'Min (%)': [
        f"{np.min(results['test_accuracies']):.2f}",
        f"{np.min(ms_results['test_accuracies']):.2f}",
        f"{np.min(base_accuracy_list):.2f}",
        f"{np.min(finetuned_accuracy_list):.2f}"
    ],
    'Max (%)': [
        f"{np.max(results['test_accuracies']):.2f}",
        f"{np.max(ms_results['test_accuracies']):.2f}",
        f"{np.max(base_accuracy_list):.2f}",
        f"{np.max(finetuned_accuracy_list):.2f}"
    ]
}

# Add improvement statistics
summary_data['Improvement vs Base'] = [
    '-',
    '-',
    '0.00',
    f"{np.mean(improvement_list):+.2f} ± {np.std(improvement_list):.2f}"
]

table = ax.table(cellText=[[summary_data[col][i] for col in summary_data.keys()] 
                          for i in range(len(summary_data['Model']))],
                colLabels=list(summary_data.keys()),
                cellLoc='center',
                loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color code the rows
colors = ['lightcoral', 'lightgreen', 'plum', 'lightblue']
for i in range(len(summary_data['Model'])):
    for j in range(len(summary_data.keys())):
        table[(i+1, j)].set_facecolor(colors[i])

plt.title(f'Subject {type_of_subject} {model_name.upper()} Performance Summary Statistics', pad=20, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_performance_summary_table.png'), 
            bbox_inches='tight', dpi=300)
plt.close()

print(f"\nAll plots saved to {output_path}")
print("Files generated:")
print(f"- {type_of_subject}_{model_name}_base_vs_finetuned_accuracies.png")
print(f"- {type_of_subject}_{model_name}_finetuning_improvements.png") 
print(f"- {type_of_subject}_{model_name}_all_models_comparison.png")
print(f"- {type_of_subject}_{model_name}_accuracy_distributions_with_finetuning.png")
print(f"- {type_of_subject}_{model_name}_performance_summary_table.png")

print('==================== End of script comb_ms_dcn_adapt_clean.py! ===================')