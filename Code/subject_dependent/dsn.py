'''
DeepStateNet - Combined DCN and MicroStateNet Training Script
Combines DeepConvNet (raw EEG) and MicroStateNet (microstate) models using multimodal fusion
With K-fold cross-validation, balanced accuracy, F1 scores, and single-subject loading
'''
import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from braindecode.models import Deep4Net
from braindecode.classifier import EEGClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import my_functions as mf
from lib import my_models as mm
from lib import my_models_functions as mmf 


def load_subject_data(subject_id, args, data_path):
    """Load both raw EEG and microstate data for a single subject"""
    
    # Load raw EEG data and labels
    raw_data, y = mf.load_data(subject_id, data_path=data_path)
    
    # Load microstate timeseries
    input_path = os.path.abspath('Output/ica_rest_all')
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist.")
    
    kmeans_path = os.path.abspath(os.path.join(input_path, 'modkmeans_results'))

    # Determine ms_file_type based on model's input format
    model_info = mm.MODEL_INFO.get(args.model_name, {})
    input_format = model_info.get('input_format', 'one_hot')
    
    if args.use_embedding:
        ms_file_type = 'modk_sequence'
        input_format = 'categorical'
    else:  # input_format == 'one_hot'
        ms_file_type = 'ms_timeseries'
    
    print(f"Model {args.model_name} uses {input_format} format, loading from {ms_file_type}")
    
    finals_ls_folder = os.path.join(kmeans_path, ms_file_type)
    if not os.path.exists(finals_ls_folder):
        raise FileNotFoundError(f"{finals_ls_folder} does not exist.")
    
    ms_data = mf.load_ms(
        subject_id=subject_id,
        n_clusters=args.n_clusters, 
        seq_time_path=finals_ls_folder,
        seq_time_type=ms_file_type, 
        seq_time_specific=args.ms_file_specific
    )
    
    return raw_data, ms_data, y


def prepare_microstate_input(data, model_name, use_embedding=False):
    """Prepare input data based on model requirements and embedding flag"""
    x = torch.tensor(data, dtype=torch.float32)
    
    # Determine input format based on embedding flag or model default
    if use_embedding:
        input_format = 'categorical'
    else:
        model_info = mm.MODEL_INFO.get(model_name, {})
        input_format = model_info.get('input_format', 'one_hot')
    
    print(f"Input data shape: {x.shape}")
    print(f"Model {model_name} with use_embedding={use_embedding} expects {input_format} format")
    
    if input_format == 'categorical':
        # Data loaded is already categorical sequences
        # Handle potential negative indices
        min_val = torch.min(x).item()
        if min_val < 0:
            print(f"Found negative microstate indices ({min_val}), shifting to start from 0...")
            x = x - min_val
        
        n_microstates = int(torch.max(x).item()) + 1
        sequence_length = x.shape[1]
        
        print(f"Using categorical microstate sequences")
        print(f"Microstate data shape: {x.shape}")
        print(f"Microstate range: {torch.min(x).item()} to {torch.max(x).item()}")
        
    else:  # input_format == 'one_hot'
        # Data loaded is already one-hot encoded
        n_microstates = x.shape[1]
        sequence_length = x.shape[2]
        
        print(f"Using one-hot encoded microstate sequences")
        print(f"One-hot data shape: {x.shape}")
    
    return x, n_microstates, sequence_length


def extract_features_from_model(model, data, device, batch_size=32):
    """Extract features from a single model for one subject's data"""
    
    # Convert to tensor if needed
    if not isinstance(data, torch.Tensor):
        x = torch.tensor(data, dtype=torch.float32)
    else:
        x = data.clone()
    
    print(f"  Original data shape: {x.shape}")
    
    # Determine model type first to handle data conversion correctly
    model_backbone = model.module if hasattr(model, 'module') else model
    is_embedded_model = hasattr(model_backbone, 'microstate_embedding')
    
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
    feature_extractor = mm.FeatureExtractor(model).to(device)
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


def train_subject(subject_id, args, device, data_path, pretrained_models):
    """Train DeepStateNet for a single subject with K-fold CV on ALL data"""
    print(f"\n▶ Training Subject {subject_id}")
    
    # Load single subject data for memory efficiency
    raw_data, ms_data, y = load_subject_data(subject_id, args, data_path)
    
    # Prepare inputs for models
    raw_x = torch.tensor(raw_data, dtype=torch.float32).squeeze(1)  # (n_trials, n_channels, timepoints)
    ms_x, n_microstates, sequence_length = prepare_microstate_input(ms_data, args.model_name, args.use_embedding)
    y = torch.tensor(y, dtype=torch.long)
    
    print(f"Subject {subject_id} raw data shape: {raw_x.shape}")
    print(f"Subject {subject_id} microstate data shape: {ms_x.shape}")
    print(f"Subject {subject_id} labels shape: {y.shape}")
    print(f"Number of classes: {len(torch.unique(y))}")
    
    # Extract features from pretrained models for all data
    print(f"Extracting features from pretrained DCN model...")
    raw_features = extract_features_from_model(pretrained_models['dcn'][subject_id], raw_x, device)
    
    print(f"Extracting features from pretrained MicroStateNet model...")
    ms_features = extract_features_from_model(pretrained_models['ms'][subject_id], ms_x, device)
    
    # K-Fold Cross Validation on ALL data (no separate test set)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    cv_balanced_accs = []
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_features, y)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
        
        # Get fold data using extracted features ((K-1)/K train, 1/K val for each fold)
        raw_train_fold = raw_features[train_idx]
        ms_train_fold = ms_features[train_idx]
        y_train_fold = y[train_idx]
        raw_val_fold = raw_features[val_idx]
        ms_val_fold = ms_features[val_idx]
        y_val_fold = y[val_idx]
        
        train_pct = (args.n_folds - 1) / args.n_folds * 100
        val_pct = 1 / args.n_folds * 100
        print(f"Fold {fold + 1}: Train Raw {raw_train_fold.shape}, MS {ms_train_fold.shape} ({train_pct:.1f}%)")
        print(f"Fold {fold + 1}: Val Raw {raw_val_fold.shape}, MS {ms_val_fold.shape} ({val_pct:.1f}%)")
        
        # DataLoaders for this fold
        train_loader = DataLoader(TensorDataset(raw_train_fold, ms_train_fold, y_train_fold), 
                                 batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(raw_val_fold, ms_val_fold, y_val_fold), 
                               batch_size=args.batch_size, shuffle=False)
        
        # Model for this fold
        n_classes = len(torch.unique(y))
        raw_feature_dim = raw_train_fold.shape[1]
        ms_feature_dim = ms_train_fold.shape[1]
        
        model = mm.DeepStateNetClassifier(
            raw_feature_dim, ms_feature_dim, n_classes
        ).to(device)
        
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        print(f"Using DeepStateNet with DCN + {args.model_name} fusion")
        
        # Training for this fold
        fold_train_losses, fold_val_losses = [], []
        fold_train_balanced_accs, fold_val_balanced_accs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        
        for epoch in range(1, args.epochs + 1):
            # Train - USING SHARED FUNCTION STYLE
            model.train()
            train_loss = 0
            train_total = 0
            all_preds = []
            all_targets = []
            
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
                train_total += label_batch.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(label_batch.cpu().numpy())
            
            train_loss /= train_total
            train_balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
            train_f1 = f1_score(all_targets, all_preds, average='macro') * 100
            
            # Validate - USING SHARED FUNCTION STYLE
            model.eval()
            val_loss = 0
            val_total = 0
            all_val_preds = []
            all_val_targets = []
            
            with torch.no_grad():
                for raw_batch, ms_batch, label_batch in val_loader:
                    raw_batch = raw_batch.to(device)
                    ms_batch = ms_batch.to(device)
                    label_batch = label_batch.to(device)
                    
                    outputs = model(raw_batch, ms_batch)
                    loss = criterion(outputs, label_batch)
                    
                    val_loss += loss.item() * raw_batch.size(0)
                    preds = outputs.argmax(dim=1)
                    val_total += label_batch.size(0)
                    
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_targets.extend(label_batch.cpu().numpy())
            
            val_loss /= val_total
            val_balanced_acc = balanced_accuracy_score(all_val_targets, all_val_preds) * 100
            val_f1 = f1_score(all_val_targets, all_val_preds, average='macro') * 100
            
            fold_train_losses.append(train_loss)
            fold_train_balanced_accs.append(train_balanced_acc)
            fold_train_f1s.append(train_f1)
            fold_val_losses.append(val_loss)
            fold_val_balanced_accs.append(val_balanced_acc)
            fold_val_f1s.append(val_f1)
            
            if epoch % 20 == 0 or epoch == args.epochs:  # Print every 20 epochs
                print(f"Fold {fold + 1}, Epoch {epoch:02d}/{args.epochs} | "
                      f"Train Bal Acc: {train_balanced_acc:.2f}%, F1: {train_f1:.2f}% | "
                      f"Val Bal Acc: {val_balanced_acc:.2f}%, F1: {val_f1:.2f}%")
        
        # Store fold results
        fold_result = {
            'train_losses': fold_train_losses,
            'train_balanced_accuracies': fold_train_balanced_accs,
            'train_f1_macros': fold_train_f1s,
            'val_losses': fold_val_losses,
            'val_balanced_accuracies': fold_val_balanced_accs,
            'val_f1_macros': fold_val_f1s,
            'final_val_balanced_acc': val_balanced_acc,
            'final_val_f1': val_f1,
            'model': model
        }
        fold_results.append(fold_result)
        cv_balanced_accs.append(val_balanced_acc)
        cv_f1_scores.append(val_f1)
        
        print(f"✅ Fold {fold + 1} completed - Val Balanced Acc: {val_balanced_acc:.2f}%, F1: {val_f1:.2f}%")
    
    # Cross-validation summary - USING SHARED FUNCTION
    mean_cv_bal_acc, std_cv_bal_acc, mean_cv_f1, std_cv_f1 = mmf.print_cv_summary(
        cv_balanced_accs, cv_f1_scores, args.n_folds)
    
    # Select best fold model (highest validation balanced accuracy)
    best_fold_idx = np.argmax(cv_balanced_accs)
    best_model = fold_results[best_fold_idx]['model']
    print(f"Best fold: {best_fold_idx + 1} (Val Bal Acc: {cv_balanced_accs[best_fold_idx]:.2f}%)")
    
    # Aggregate training curves across folds - USING SHARED FUNCTION
    training_curves = mmf.aggregate_fold_training_curves(fold_results)
    
    # Clean up memory
    del raw_data, ms_data, raw_x, ms_x, y
    del raw_features, ms_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'n_folds': args.n_folds,
        'cv_balanced_accuracies': cv_balanced_accs,
        'cv_f1_scores': cv_f1_scores,
        'mean_cv_balanced_acc': mean_cv_bal_acc,
        'std_cv_balanced_acc': std_cv_bal_acc,
        'mean_cv_f1': mean_cv_f1,
        'std_cv_f1': std_cv_f1,
        'best_fold_idx': best_fold_idx,
        'fold_results': fold_results,
        'best_model': best_model,
        **training_curves  # ← UNPACKS ALL TRAINING CURVE DATA
    }


def load_pretrained_models(args, project_root):
    """Load pretrained DCN and MicroStateNet models using dynamic paths"""
    
    output_folder = os.path.join(project_root, 'Output') + os.sep
    
    # Build DCN path dynamically (updated path structure)
    dcn_path = f'{output_folder}ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_dcn_{args.n_folds}fold_results/'
    dcn_file = os.path.join(dcn_path, f'{args.type_of_subject}_dcn_{args.n_folds}fold_results.npy')
    
    # Build MicroStateNet path dynamically (like in msn.py)
    embedding_suffix = "_embedded" if args.use_embedding else ""
    ms_model_name_with_embedding = f"{args.model_name}{embedding_suffix}"
    ms_path = f'{output_folder}ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_{ms_model_name_with_embedding}_c{args.n_clusters}_{args.n_folds}fold_results/'
    ms_file = os.path.join(ms_path, f'{args.type_of_subject}_{ms_model_name_with_embedding}_c{args.n_clusters}_{args.n_folds}fold_results.npy')
    
    print(f"Loading pretrained DCN models from: {dcn_file}")
    if not os.path.exists(dcn_file):
        raise FileNotFoundError(f"DCN results file not found: {dcn_file}")
    dcn_results = np.load(dcn_file, allow_pickle=True).item()
    dcn_models = dcn_results['best_models']
    
    print(f"Loading pretrained MicroStateNet models from: {ms_file}")
    if not os.path.exists(ms_file):
        raise FileNotFoundError(f"MicroStateNet results file not found: {ms_file}")
    ms_results = np.load(ms_file, allow_pickle=True).item()
    ms_models = ms_results['best_models']
    
    print(f"Loaded {len(dcn_models)} DCN models and {len(ms_models)} MicroStateNet models")
    
    return {
        'dcn': dcn_models,
        'ms': ms_models
    }


def main():
    """Main training function"""
    # Argument parsing
    parser = argparse.ArgumentParser(description='PyTorch DeepStateNet (DCN + MicroStateNet) Classification with K-fold CV')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-subjects', type=int, default=50, metavar='N',
                        help='number of subjects to process (default: 50)')
    parser.add_argument('--type-of-subject', type=str, default='dependent',
                        choices=['independent', 'dependent', 'adaptive'],
                        help='type of subject analysis (default: dependent)')
    parser.add_argument('--n-folds', type=int, default=5, metavar='K',
                        help='number of CV folds (default: 5)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    # Microstate-specific arguments
    parser.add_argument('--model-name', type=str, default='msn',
                        choices=['msn', 'multiscale_msn', 'embedded_msn', 'attention_msn'],
                        help='microstate model architecture to use (default: msn)')
    parser.add_argument('--n-clusters', type=int, default=12, metavar='N',
                        help='number of microstate clusters (default: 12)')
    parser.add_argument('--ms-file-specific', type=str, default='indiv',
                        help='microstate file specific type (default: indiv)')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='D',
                        help='dropout rate (default: 0.25)')
    
    # Embedding support
    parser.add_argument('--use-embedding', action='store_true', default=False,
                        help='Use embedding-based input instead of one-hot encoding (default: False)')
    
    # Attention model specific arguments
    parser.add_argument('--embedding-dim', type=int, default=64, metavar='N',
                        help='embedding dimension for embedding models (default: 64)')
    parser.add_argument('--transformer-layers', type=int, default=4, metavar='N',
                        help='number of transformer layers for attention models (default: 4)')
    parser.add_argument('--transformer-heads', type=int, default=8, metavar='N',
                        help='number of transformer heads for attention models (default: 8)')
    
    args = parser.parse_args()
    
    print(f'==================== Start of script {os.path.basename(__file__)}! ====================')
    
    # Device setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    if use_cuda:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    mf.print_memory_status("- INITIAL STARTUP")
    
    # Set seed
    mf.set_seed(args.seed)
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, 'Data') + os.sep
    output_folder = os.path.join(project_root, 'Output') + os.sep
    
    # Create embedding suffix for output naming
    embedding_suffix = "_embedded" if args.use_embedding else ""
    model_name_with_embedding = f"dsn_{args.model_name}{embedding_suffix}"
    
    # Setup paths with embedding suffix
    output_path = f'{output_folder}ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_{model_name_with_embedding}_c{args.n_clusters}_{args.n_folds}fold_results/'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    print(f"Using DeepStateNet with DCN + {args.model_name} (embedding: {args.use_embedding})")
    if args.model_name in mm.MODEL_INFO:
        print(f"MicroStateNet model description: {mm.MODEL_INFO[args.model_name]['description']}")
    
    # Load pretrained models using dynamic paths
    print(f"\nLoading pretrained models...")
    pretrained_models = load_pretrained_models(args, project_root)
    
    # Verify we have enough models
    n_dcn_models = len(pretrained_models['dcn'])
    n_ms_models = len(pretrained_models['ms'])
    n_subjects_to_process = min(args.n_subjects, n_dcn_models, n_ms_models)
    
    if n_subjects_to_process < args.n_subjects:
        print(f"Warning: Requested {args.n_subjects} subjects but only have {n_dcn_models} DCN and {n_ms_models} MicroStateNet models.")
        print(f"Processing {n_subjects_to_process} subjects instead.")
    
    # Training loop
    all_results = []
    output_file = os.path.join(output_path, f'{args.type_of_subject}_{model_name_with_embedding}_c{args.n_clusters}_{args.n_folds}fold_results.npy')
    
    # Check for existing results and resume if needed
    if os.path.exists(output_file):
        print(f"Found existing results file: {output_file}")
        existing_results = np.load(output_file, allow_pickle=True).item()
        n_existing = len(existing_results.get('mean_cv_balanced_accs', []))
        print(f"Found {n_existing} existing subjects. Resuming from subject {n_existing}...")
        
        # Convert existing results to our format
        for i in range(n_existing):
            result = {
                'n_folds': existing_results.get('n_folds', [args.n_folds] * n_existing)[i],
                'cv_balanced_accuracies': existing_results['cv_balanced_accuracies'][i],
                'cv_f1_scores': existing_results['cv_f1_scores'][i],
                'mean_cv_balanced_acc': existing_results['mean_cv_balanced_accs'][i],
                'std_cv_balanced_acc': existing_results['std_cv_balanced_accs'][i],
                'mean_cv_f1': existing_results['mean_cv_f1s'][i],
                'std_cv_f1': existing_results['std_cv_f1s'][i],
                'best_fold_idx': existing_results['best_fold_indices'][i],
                'train_losses_mean': existing_results['train_curves_mean']['train_losses_mean'][i],
                'train_balanced_accuracies_mean': existing_results['train_curves_mean']['train_balanced_accuracies_mean'][i],
                'train_f1_macros_mean': existing_results['train_curves_mean']['train_f1_macros_mean'][i],
                'val_losses_mean': existing_results['train_curves_mean']['val_losses_mean'][i],
                'val_balanced_accuracies_mean': existing_results['train_curves_mean']['val_balanced_accuracies_mean'][i],
                'val_f1_macros_mean': existing_results['train_curves_mean']['val_f1_macros_mean'][i],
                'best_model': existing_results['best_models'][i]
            }
            # Add fold_results if available (for backward compatibility)
            if 'fold_results' in existing_results:
                result['fold_results'] = existing_results['fold_results'][i]
            
            all_results.append(result)
        start_subject = n_existing
    else:
        start_subject = 0
    
    for subject_id in range(start_subject, n_subjects_to_process):
        mf.print_memory_status(f"- SUBJECT {subject_id} START")
        
        result = train_subject(subject_id, args, device, data_path, pretrained_models)
        all_results.append(result)
        
        # Save intermediate results
        if args.save_model:
            results_dict = {
                'n_folds': [r['n_folds'] for r in all_results],
                'cv_balanced_accuracies': [r['cv_balanced_accuracies'] for r in all_results],
                'cv_f1_scores': [r['cv_f1_scores'] for r in all_results],
                'mean_cv_balanced_accs': [r['mean_cv_balanced_acc'] for r in all_results],
                'std_cv_balanced_accs': [r['std_cv_balanced_acc'] for r in all_results],
                'mean_cv_f1s': [r['mean_cv_f1'] for r in all_results],
                'std_cv_f1s': [r['std_cv_f1'] for r in all_results],
                'best_fold_indices': [r['best_fold_idx'] for r in all_results],
                'train_curves_mean': {
                    'train_losses_mean': [r['train_losses_mean'] for r in all_results],
                    'train_balanced_accuracies_mean': [r['train_balanced_accuracies_mean'] for r in all_results],
                    'train_f1_macros_mean': [r['train_f1_macros_mean'] for r in all_results],
                    'val_losses_mean': [r['val_losses_mean'] for r in all_results],
                    'val_balanced_accuracies_mean': [r['val_balanced_accuracies_mean'] for r in all_results],
                    'val_f1_macros_mean': [r['val_f1_macros_mean'] for r in all_results],
                },
                'best_models': [r['best_model'] for r in all_results],
                'fold_results': [r['fold_results'] for r in all_results]  # Store individual fold results
            }
            np.save(output_file, results_dict)
            print(f"Results saved to {output_file}")
        
        print(f"✅ Subject {subject_id} processed successfully.\n")
        mf.print_memory_status(f"- SUBJECT {subject_id} END")
    
    # Final summary - USING SHARED FUNCTION
    mmf.print_final_summary(all_results, model_name_with_embedding, args.n_folds)
    
    # Plot results - USING SHARED FUNCTION with embedding suffix
    mmf.plot_all_results(all_results, output_path, args.type_of_subject, model_name_with_embedding, len(all_results))
    
    print('==================== End of script! ====================')


if __name__ == '__main__':
    main()