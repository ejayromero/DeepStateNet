'''
Unified training, validation, testing, and plotting functions for neural network models
Clean, reusable abstractions that work with any model type and data format
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from braindecode.classifier import EEGClassifier
# Add this after your imports to suppress just this warning:
import warnings
warnings.filterwarnings("ignore", message="LogSoftmax final layer will be removed")

sys.path.append(os.path.abspath(__file__))
from lib import my_functions as mf
from lib import my_models as mm

# =============================================================================
# CORE TRAINING/VALIDATION/TESTING FUNCTIONS
# =============================================================================

def train_epoch(model, device, train_loader, optimizer, criterion, epoch, log_interval=10):
    """Train the model for one epoch - works with any model type"""
    model.train()
    train_loss = 0
    train_total = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Handle different data formats (single vs dual modal)
        if len(batch_data) == 2:  # Standard: (data, target)
            data, target = batch_data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
        elif len(batch_data) == 3:  # Dual modal: (data1, data2, target)
            data1, data2, target = batch_data
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data1, data2)
        else:
            raise ValueError(f"Unsupported batch format with {len(batch_data)} elements")
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * target.size(0)
        pred = output.argmax(dim=1)
        train_total += target.size(0)
        
        # Store predictions and targets for metric computation
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        if batch_idx % log_interval == 0 and epoch % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * target.size(0)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = train_loss / train_total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    
    return avg_loss, balanced_acc, f1_macro


def validate(model, device, val_loader, criterion):
    """Validate the model - works with any model type"""
    model.eval()
    val_loss = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            # Handle different data formats (single vs dual modal)
            if len(batch_data) == 2:  # Standard: (data, target)
                data, target = batch_data
                data, target = data.to(device), target.to(device)
                output = model(data)
            elif len(batch_data) == 3:  # Dual modal: (data1, data2, target)
                data1, data2, target = batch_data
                data1, data2, target = data1.to(device), data2.to(device), target.to(device)
                output = model(data1, data2)
            else:
                raise ValueError(f"Unsupported batch format with {len(batch_data)} elements")
            
            val_loss += criterion(output, target).item() * target.size(0)
            pred = output.argmax(dim=1)
            total += target.size(0)
            
            # Store predictions and targets for metric computation
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = val_loss / total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    
    return avg_loss, balanced_acc, f1_macro


def test(model, device, test_loader, criterion=None, verbose=True):
    """Test the model - works with any model type"""
    model.eval()
    test_loss = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # Use provided criterion or default to CrossEntropyLoss
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Handle different data formats (single vs dual modal)
            if len(batch_data) == 2:  # Standard: (data, target)
                data, target = batch_data
                data, target = data.to(device), target.to(device)
                output = model(data)
            elif len(batch_data) == 3:  # Dual modal: (data1, data2, target)
                data1, data2, target = batch_data
                data1, data2, target = data1.to(device), data2.to(device), target.to(device)
                output = model(data1, data2)
            else:
                raise ValueError(f"Unsupported batch format with {len(batch_data)} elements")
            
            if criterion.reduction == 'sum':
                test_loss += criterion(output, target).item()
            else:
                test_loss += criterion(output, target).item() * target.size(0)
            
            pred = output.argmax(dim=1)
            total += target.size(0)
            
            # Store predictions and targets for metric computation
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    if criterion.reduction == 'sum':
        test_loss /= total
    else:
        test_loss /= total
        
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    if verbose:
        print(f'Test set: Average loss: {test_loss:.4f}, '
              f'Balanced Accuracy: {balanced_acc:.2f}%, F1 Macro: {f1_macro:.2f}%')
    
    return balanced_acc, f1_macro, conf_matrix

# =============================================================================
# DATA LOADING STRATEGY PATTERN
# =============================================================================

class BaseDataLoader:
    """Abstract base for data loading strategies"""
    def load_subject_data(self, subject_id, args, data_path):
        raise NotImplementedError
    
    def prepare_input(self, data, args):
        raise NotImplementedError
    
    def get_data_info(self, data, args):
        """Return data-specific information needed for model creation"""
        raise NotImplementedError


class RawEEGDataLoader(BaseDataLoader):
    """Data loader for raw EEG data"""
    def load_subject_data(self, subject_id, args, data_path):
        data, y = mf.load_data(subject_id, data_path=data_path)
        return data, y
    
    def prepare_input(self, data, args):
        x = torch.tensor(data, dtype=torch.float32).squeeze(1)
        return x
    
    def get_data_info(self, data, args):
        x = self.prepare_input(data, args)
        return {
            'input_shape': x.shape[1:],  # (n_channels, timepoints)
            'n_channels': x.shape[1],
            'n_timepoints': x.shape[2]
        }


class MicrostateDataLoader(BaseDataLoader):
    """Data loader for microstate data"""
    def load_subject_data(self, subject_id, args, data_path):
        # Load labels
        _, y = mf.load_data(subject_id, data_path=data_path)
        
        # Load microstate timeseries
        input_path = os.path.abspath('Output/ica_rest_all')
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} does not exist.")
        
        kmeans_path = os.path.abspath(os.path.join(input_path, 'modkmeans_results'))

        # Determine file type based on model's input format
        model_info = mm.MODEL_INFO.get(args.model_name, {})
        input_format = model_info.get('input_format', 'one_hot')
        
        if args.use_embedding:
            file_type = 'modk_sequence'
        else:  # input_format == 'one_hot'
            file_type = 'ms_timeseries'
        
        print(f"Model {args.model_name} uses {input_format} format, loading from {file_type}")
        
        finals_ls_folder = os.path.join(kmeans_path, file_type)
        if not os.path.exists(finals_ls_folder):
            raise FileNotFoundError(f"{finals_ls_folder} does not exist.")
        
        data = mf.load_ms(
            subject_id=subject_id,
            n_clusters=args.n_clusters, 
            seq_time_path=finals_ls_folder,
            seq_time_type=file_type, 
            seq_time_specific=args.ms_file_specific
        )
        
        return data, y
    
    def prepare_input(self, data, args):
        x = torch.tensor(data, dtype=torch.float32)
        
        # Determine input format based on embedding flag or model default
        if args.use_embedding:
            input_format = 'categorical'
        else:
            model_info = mm.MODEL_INFO.get(args.model_name, {})
            input_format = model_info.get('input_format', 'one_hot')
        
        if input_format == 'categorical':
            # Handle potential negative indices
            min_val = torch.min(x).item()
            if min_val < 0:
                print(f"Found negative microstate indices ({min_val}), shifting to start from 0...")
                x = x - min_val
        
        return x
    
    def get_data_info(self, data, args):
        x = self.prepare_input(data, args)
        
        if args.use_embedding:
            n_microstates = int(torch.max(x).item()) + 1
            sequence_length = x.shape[1]
        else:
            n_microstates = x.shape[1]
            sequence_length = x.shape[2]
        
        return {
            'n_microstates': n_microstates,
            'sequence_length': sequence_length,
            'input_format': 'categorical' if args.use_embedding else 'one_hot'
        }


class DualModalDataLoader(BaseDataLoader):
    """Data loader for dual modal (raw EEG + microstate) data"""
    def __init__(self):
        self.raw_loader = RawEEGDataLoader()
        self.ms_loader = MicrostateDataLoader()
    
    def load_subject_data(self, subject_id, args, data_path):
        raw_data, y = self.raw_loader.load_subject_data(subject_id, args, data_path)
        ms_data, _ = self.ms_loader.load_subject_data(subject_id, args, data_path)
        return (raw_data, ms_data), y
    
    def prepare_input(self, data, args):
        raw_data, ms_data = data
        raw_x = self.raw_loader.prepare_input(raw_data, args)
        ms_x = self.ms_loader.prepare_input(ms_data, args)
        return raw_x, ms_x
    
    def get_data_info(self, data, args):
        raw_data, ms_data = data
        raw_info = self.raw_loader.get_data_info(raw_data, args)
        ms_info = self.ms_loader.get_data_info(ms_data, args)
        return {'raw': raw_info, 'microstate': ms_info}

# =============================================================================
# MODEL CREATION STRATEGY PATTERN
# =============================================================================

class ModelFactory:
    """Factory for creating different model types"""
    
    @staticmethod
    def create_model(model_type, data_info, args, device):
        """Create model based on type and data information"""
        if model_type == 'dcn':
            return ModelFactory._create_dcn_model(data_info, args, device)
        elif model_type == 'microstate':
            return ModelFactory._create_microstate_model(data_info, args, device)
        elif model_type == 'fusion':
            return ModelFactory._create_fusion_model(data_info, args, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _create_dcn_model(data_info, args, device):
        """Create DeepConvNet model"""
        from braindecode.models import Deep4Net
        
        base_model = Deep4Net(
            n_chans=data_info['n_channels'],
            n_outputs=3,  # rest, open, close
            n_times=data_info['n_timepoints'],
            final_conv_length='auto'
        )
        
        model = EEGClassifier(
            base_model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            optimizer__lr=args.lr,
            train_split=None,
            device=device
        )
        
        net = model.module.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        
        return net, criterion, optimizer
    
    @staticmethod
    def _create_microstate_model(data_info, args, device):
        """Create microstate model"""
        n_classes = 3  # rest, open, close
        
        # Create model using factory function
        if 'attention' in args.model_name:
            model = mm.get_model(
                model_name=args.model_name,
                n_microstates=data_info['n_microstates'],
                n_classes=n_classes,
                sequence_length=data_info['sequence_length'],
                dropout=args.dropout,
                embedding_dim=args.embedding_dim,
                transformer_layers=args.transformer_layers,
                transformer_heads=args.transformer_heads,
                use_embedding=args.use_embedding
            )
        else:
            model = mm.get_model(
                model_name=args.model_name,
                n_microstates=data_info['n_microstates'],
                n_classes=n_classes,
                sequence_length=data_info['sequence_length'],
                dropout=args.dropout,
                use_embedding=args.use_embedding
            )
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        return model, criterion, optimizer
    
    @staticmethod
    def _create_fusion_model(data_info, args, device):
        """Create fusion model for dual modal data"""
        raw_feature_dim = 256  # Typical DCN feature dimension
        ms_feature_dim = 128   # Typical microstate feature dimension
        n_classes = 3
        
        model = mm.DeepStateNetClassifier(
            raw_feature_dim, ms_feature_dim, n_classes
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        return model, criterion, optimizer

# =============================================================================
# UNIFIED TRAINING FUNCTIONS
# =============================================================================

def train_subject(subject_id, args, device, data_path, model_type='dcn'):
    """
    Unified training function for any model type with K-fold CV
    
    Args:
        subject_id: Subject ID to train
        args: Arguments namespace
        device: torch.device
        data_path: Path to data directory
        model_type: 'dcn', 'microstate', or 'fusion'
    """
    print(f"\n▶ Training Subject {subject_id}")
    
    # Get appropriate data loader
    if model_type == 'dcn':
        data_loader = RawEEGDataLoader()
    elif model_type == 'microstate':
        data_loader = MicrostateDataLoader()
    elif model_type == 'fusion':
        data_loader = DualModalDataLoader()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load and prepare data
    data, y = data_loader.load_subject_data(subject_id, args, data_path)
    
    if model_type == 'fusion':
        raw_x, ms_x = data_loader.prepare_input(data, args)
        x = (raw_x, ms_x)  # Keep as tuple for dual modal
    else:
        x = data_loader.prepare_input(data, args)
    
    y = torch.tensor(y, dtype=torch.long)
    
    mf.print_memory_status(f"After Loading Subject {subject_id}")

    print(f"Subject {subject_id} data prepared")
    print(f"Number of classes: {len(torch.unique(y))}")
    
    # Get data info for model creation
    data_info = data_loader.get_data_info(data, args)
    
    # Split: 90% for CV, 10% for final test
    if model_type == 'fusion':
        indices = np.arange(len(y))
        cv_indices, test_indices = train_test_split(
            indices, test_size=0.1, random_state=42, stratify=y.numpy())
        raw_cv, raw_test = raw_x[cv_indices], raw_x[test_indices]
        ms_cv, ms_test = ms_x[cv_indices], ms_x[test_indices]
        y_cv, y_test = y[cv_indices], y[test_indices]
        x_cv, x_test = (raw_cv, ms_cv), (raw_test, ms_test)
    else:
        x_cv, x_test, y_cv, y_test = train_test_split(
            x, y, test_size=0.1, random_state=42, stratify=y)
    
    print(f"CV/Test split completed")
    
    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    cv_balanced_accs = []
    cv_f1_scores = []
    
    # Determine stratification variable for dual modal
    stratify_var = y_cv if model_type != 'fusion' else y_cv
    split_var = x_cv if model_type != 'fusion' else cv_indices
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(split_var, stratify_var)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
        
        # Get fold data
        if model_type == 'fusion':
            raw_train_fold, ms_train_fold = raw_cv[train_idx], ms_cv[train_idx]
            raw_val_fold, ms_val_fold = raw_cv[val_idx], ms_cv[val_idx]
            y_train_fold, y_val_fold = y_cv[train_idx], y_cv[val_idx]
            train_dataset = TensorDataset(raw_train_fold, ms_train_fold, y_train_fold)
            val_dataset = TensorDataset(raw_val_fold, ms_val_fold, y_val_fold)
        else:
            x_train_fold = x_cv[train_idx]
            y_train_fold = y_cv[train_idx]
            x_val_fold = x_cv[val_idx]
            y_val_fold = y_cv[val_idx]
            train_dataset = TensorDataset(x_train_fold, y_train_fold)
            val_dataset = TensorDataset(x_val_fold, y_val_fold)
        
        # DataLoaders
        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model, criterion, optimizer = ModelFactory.create_model(model_type, data_info, args, device)
        
        # Training loop
        fold_train_losses, fold_val_losses = [], []
        fold_train_balanced_accs, fold_val_balanced_accs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss, train_balanced_acc, train_f1 = train_epoch(
                model, device, train_loader, optimizer, criterion, epoch, 
                args.log_interval if epoch % 10 == 1 else 999)
            
            # Validate
            val_loss, val_balanced_acc, val_f1 = validate(model, device, val_loader, criterion)
            
            fold_train_losses.append(train_loss)
            fold_train_balanced_accs.append(train_balanced_acc)
            fold_train_f1s.append(train_f1)
            fold_val_losses.append(val_loss)
            fold_val_balanced_accs.append(val_balanced_acc)
            fold_val_f1s.append(val_f1)
            
            if epoch % 20 == 0 or epoch == args.epochs:
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
    
    # Cross-validation summary
    mean_cv_bal_acc, std_cv_bal_acc, mean_cv_f1, std_cv_f1 = print_cv_summary(
        cv_balanced_accs, cv_f1_scores, args.n_folds)
    
    # Select best fold model
    best_fold_idx = np.argmax(cv_balanced_accs)
    best_model = fold_results[best_fold_idx]['model']
    print(f"Best fold: {best_fold_idx + 1} (Val Bal Acc: {cv_balanced_accs[best_fold_idx]:.2f}%)")
    
    # Final test
    if model_type == 'fusion':
        test_dataset = TensorDataset(raw_test, ms_test, y_test)
    else:
        test_dataset = TensorDataset(x_test, y_test)
    
    test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_balanced_acc, test_f1, conf_matrix = test(best_model, device, test_loader, criterion)
    print(f"🎯 Final Test Results - Balanced Acc: {test_balanced_acc:.2f}%, F1: {test_f1:.2f}%")
    
    # Aggregate training curves
    training_curves = aggregate_fold_training_curves(fold_results)
    
    # Clean up memory
    del data, x, y
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
        'test_balanced_accuracy': test_balanced_acc,
        'test_f1_macro': test_f1,
        'confusion_matrix': conf_matrix,
        'best_fold_idx': best_fold_idx,
        'fold_results': fold_results,
        'best_model': best_model,
        **training_curves
    }


def train_loso(test_subject_id, args, device, data_path, model_type='dcn'):
    """
    Unified LOSO training function for any model type
    
    Args:
        test_subject_id: Subject ID to use for testing
        args: Arguments namespace
        device: torch.device
        data_path: Path to data directory
        model_type: 'dcn', 'microstate', or 'fusion'
    """
    print(f"\n▶ {model_type.upper()} LOSO Training - Test Subject {test_subject_id}")
    
    # Get all remaining subjects (49 subjects)
    all_remaining_subjects = [i for i in range(args.n_subjects) if i != test_subject_id]
    
    # Get appropriate data loader
    if model_type == 'dcn':
        data_loader = RawEEGDataLoader()
    elif model_type == 'microstate':
        data_loader = MicrostateDataLoader()
    elif model_type == 'fusion':
        data_loader = DualModalDataLoader()
    
    # Load test data
    test_data, test_y = data_loader.load_subject_data(test_subject_id, args, data_path)
    
    if model_type == 'fusion':
        test_raw_x, test_ms_x = data_loader.prepare_input(test_data, args)
        test_x = (test_raw_x, test_ms_x)
    else:
        test_x = data_loader.prepare_input(test_data, args)
    
    test_y = torch.tensor(test_y, dtype=torch.long)
    data_info = data_loader.get_data_info(test_data, args)
    
    print(f"Test subject {test_subject_id} loaded")
    
    mf.print_memory_status(f"After Loading Test Subject {test_subject_id}")


    # 4-Fold Cross Validation on remaining subjects
    dummy_y = [0] * len(all_remaining_subjects)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    cv_balanced_accs = []
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_remaining_subjects, dummy_y)):
        print(f"\n--- {model_type.upper()} Fold {fold + 1}/{args.n_folds} ---")
        
        # Get subjects for this fold
        fold_train_subjects = [all_remaining_subjects[i] for i in train_idx]
        fold_val_subjects = [all_remaining_subjects[i] for i in val_idx]
        
        # Load batch data
        train_x, train_y = load_subjects_batch(fold_train_subjects, args, data_path, data_loader)
        val_x, val_y = load_subjects_batch(fold_val_subjects, args, data_path, data_loader)
        
        # Create DataLoaders
        if model_type == 'fusion':
            train_raw_x, train_ms_x = train_x
            val_raw_x, val_ms_x = val_x
            train_dataset = TensorDataset(train_raw_x, train_ms_x, train_y)
            val_dataset = TensorDataset(val_raw_x, val_ms_x, val_y)
        else:
            train_dataset = TensorDataset(train_x, train_y)
            val_dataset = TensorDataset(val_x, val_y)
        
        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model, criterion, optimizer = ModelFactory.create_model(model_type, data_info, args, device)
        
        # Training loop (same as train_subject)
        fold_train_losses, fold_val_losses = [], []
        fold_train_balanced_accs, fold_val_balanced_accs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_balanced_acc, train_f1 = train_epoch(
                model, device, train_loader, optimizer, criterion, epoch, 
                args.log_interval if epoch % 10 == 1 else 999)
            
            val_loss, val_balanced_acc, val_f1 = validate(model, device, val_loader, criterion)
            
            fold_train_losses.append(train_loss)
            fold_train_balanced_accs.append(train_balanced_acc)
            fold_train_f1s.append(train_f1)
            fold_val_losses.append(val_loss)
            fold_val_balanced_accs.append(val_balanced_acc)
            fold_val_f1s.append(val_f1)
            
            if epoch % 20 == 0 or epoch == args.epochs:
                print(f"{model_type.upper()} Fold {fold + 1}, Epoch {epoch:02d}/{args.epochs} | "
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
            'model': model,
            'fold_train_subjects': fold_train_subjects,
            'fold_val_subjects': fold_val_subjects
        }
        fold_results.append(fold_result)
        cv_balanced_accs.append(val_balanced_acc)
        cv_f1_scores.append(val_f1)
        
        print(f"✅ {model_type.upper()} Fold {fold + 1} completed - Val Bal Acc: {val_balanced_acc:.2f}%, F1: {val_f1:.2f}%")
        
        # Clean up fold memory
        del train_x, train_y, val_x, val_y, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Cross-validation summary
    mean_cv_bal_acc, std_cv_bal_acc, mean_cv_f1, std_cv_f1 = print_cv_summary(
        cv_balanced_accs, cv_f1_scores, args.n_folds)
    
    # Select best fold model
    best_fold_idx = np.argmax(cv_balanced_accs)
    best_model = fold_results[best_fold_idx]['model']
    print(f"Best {model_type.upper()} fold: {best_fold_idx + 1} (Val Bal Acc: {cv_balanced_accs[best_fold_idx]:.2f}%)")
    
    # Final test
    if model_type == 'fusion':
        test_dataset = TensorDataset(test_raw_x, test_ms_x, test_y)
    else:
        test_dataset = TensorDataset(test_x, test_y)
    
    test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_balanced_acc, test_f1, conf_matrix = test(best_model, device, test_loader, criterion)
    print(f"🎯 {model_type.upper()} Final Test Results - Balanced Acc: {test_balanced_acc:.2f}%, F1: {test_f1:.2f}%")
    
    # Aggregate training curves
    training_curves = aggregate_fold_training_curves(fold_results)
    
    # Clean up memory
    del test_x, test_y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'test_subject_id': test_subject_id,
        'remaining_subject_ids': all_remaining_subjects,
        'n_folds': args.n_folds,
        'cv_balanced_accuracies': cv_balanced_accs,
        'cv_f1_scores': cv_f1_scores,
        'mean_cv_balanced_acc': mean_cv_bal_acc,
        'std_cv_balanced_acc': std_cv_bal_acc,
        'mean_cv_f1': mean_cv_f1,
        'std_cv_f1': std_cv_f1,
        'test_balanced_accuracy': test_balanced_acc,
        'test_f1_macro': test_f1,
        'confusion_matrix': conf_matrix,
        'best_fold_idx': best_fold_idx,
        'fold_results': fold_results,
        'best_model': best_model,
        **training_curves
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_subjects_batch(subject_ids, args, data_path, data_loader):
    """Load multiple subjects and concatenate their data efficiently"""
    if not subject_ids:
        return None, None
    
    x_list, y_list = [], []
    
    for subject_id in subject_ids:
        data, y = data_loader.load_subject_data(subject_id, args, data_path)
        
        if isinstance(data, tuple):  # Dual modal data
            raw_data, ms_data = data
            raw_x = data_loader.raw_loader.prepare_input(raw_data, args)
            ms_x = data_loader.ms_loader.prepare_input(ms_data, args)
            x_list.append((raw_x, ms_x))
        else:  # Single modal data
            x = data_loader.prepare_input(data, args)
            x_list.append(x)
        
        y_list.append(torch.tensor(y, dtype=torch.long))
    
    # Concatenate all subjects
    if isinstance(x_list[0], tuple):  # Dual modal
        raw_combined = torch.cat([x[0] for x in x_list], dim=0)
        ms_combined = torch.cat([x[1] for x in x_list], dim=0)
        x_combined = (raw_combined, ms_combined)
    else:  # Single modal
        x_combined = torch.cat(x_list, dim=0)
    
    y_combined = torch.cat(y_list, dim=0)
    
    mf.print_memory_status(f"After Loading Batch of {len(subject_ids)} subjects")
    
    del x_list, y_list  # Free intermediate memory
    return x_combined, y_combined


def aggregate_fold_training_curves(fold_results):
    """Aggregate training curves across folds (calculate mean and std)"""
    if len(fold_results) == 0:
        return {}
    
    all_epochs = len(fold_results[0]['train_losses'])
    
    # Calculate mean and std across folds for each epoch
    train_losses_mean = np.mean([fold['train_losses'] for fold in fold_results], axis=0)
    train_losses_std = np.std([fold['train_losses'] for fold in fold_results], axis=0)
    train_bal_accs_mean = np.mean([fold['train_balanced_accuracies'] for fold in fold_results], axis=0)
    train_bal_accs_std = np.std([fold['train_balanced_accuracies'] for fold in fold_results], axis=0)
    train_f1s_mean = np.mean([fold['train_f1_macros'] for fold in fold_results], axis=0)
    train_f1s_std = np.std([fold['train_f1_macros'] for fold in fold_results], axis=0)
    
    val_losses_mean = np.mean([fold['val_losses'] for fold in fold_results], axis=0)
    val_losses_std = np.std([fold['val_losses'] for fold in fold_results], axis=0)
    val_bal_accs_mean = np.mean([fold['val_balanced_accuracies'] for fold in fold_results], axis=0)
    val_bal_accs_std = np.std([fold['val_balanced_accuracies'] for fold in fold_results], axis=0)
    val_f1s_mean = np.mean([fold['val_f1_macros'] for fold in fold_results], axis=0)
    val_f1s_std = np.std([fold['val_f1_macros'] for fold in fold_results], axis=0)
    
    return {
        'train_losses_mean': train_losses_mean.tolist(),
        'train_losses_std': train_losses_std.tolist(),
        'train_balanced_accuracies_mean': train_bal_accs_mean.tolist(),
        'train_balanced_accuracies_std': train_bal_accs_std.tolist(),
        'train_f1_macros_mean': train_f1s_mean.tolist(),
        'train_f1_macros_std': train_f1s_std.tolist(),
        'val_losses_mean': val_losses_mean.tolist(),
        'val_losses_std': val_losses_std.tolist(),
        'val_balanced_accuracies_mean': val_bal_accs_mean.tolist(),
        'val_balanced_accuracies_std': val_bal_accs_std.tolist(),
        'val_f1_macros_mean': val_f1s_mean.tolist(),
        'val_f1_macros_std': val_f1s_std.tolist(),
    }


def print_cv_summary(cv_balanced_accs, cv_f1_scores, n_folds):
    """Print cross-validation summary statistics"""
    mean_cv_bal_acc = np.mean(cv_balanced_accs)
    std_cv_bal_acc = np.std(cv_balanced_accs)
    mean_cv_f1 = np.mean(cv_f1_scores)
    std_cv_f1 = np.std(cv_f1_scores)
    
    print(f"\n📊 {n_folds}-Fold CV Results Summary:")
    print(f"CV Balanced Accuracy: {mean_cv_bal_acc:.2f}% ± {std_cv_bal_acc:.2f}%")
    print(f"CV F1 Macro: {mean_cv_f1:.2f}% ± {std_cv_f1:.2f}%")
    
    return mean_cv_bal_acc, std_cv_bal_acc, mean_cv_f1, std_cv_f1


def print_final_summary(all_results, model_name, n_folds):
    """Print final summary of all subjects' results"""
    test_bal_accs = [r['test_balanced_accuracy'] for r in all_results]
    test_f1s = [r['test_f1_macro'] for r in all_results]
    cv_bal_accs = [r['mean_cv_balanced_acc'] for r in all_results]
    cv_f1s = [r['mean_cv_f1'] for r in all_results]
    
    print(f"\n🎯 Overall Results Summary:")
    print(f"Model: {model_name}")
    print(f"Configuration: {n_folds}-fold CV with 10% test split")
    print(f"Test Results:")
    print(f"  Mean Test Balanced Accuracy: {np.mean(test_bal_accs):.2f}% ± {np.std(test_bal_accs):.2f}%")
    print(f"  Mean Test F1 Macro: {np.mean(test_f1s):.2f}% ± {np.std(test_f1s):.2f}%")
    print(f"  Best Subject Test Bal Acc: {np.max(test_bal_accs):.2f}%")
    print(f"  Worst Subject Test Bal Acc: {np.min(test_bal_accs):.2f}%")
    print(f"\n{n_folds}-Fold Cross-Validation Results:")
    print(f"  Mean CV Balanced Accuracy: {np.mean(cv_bal_accs):.2f}% ± {np.std(cv_bal_accs):.2f}%")
    print(f"  Mean CV F1 Macro: {np.mean(cv_f1s):.2f}% ± {np.std(cv_f1s):.2f}%")
    print(f"  Best Subject CV Bal Acc: {np.max(cv_bal_accs):.2f}%")
    print(f"  Worst Subject CV Bal Acc: {np.min(cv_bal_accs):.2f}%")
    
    # Correlation between CV and Test performance
    cv_test_corr = np.corrcoef(cv_bal_accs, test_bal_accs)[0, 1]
    print(f"\nCV-Test Correlation: {cv_test_corr:.3f}")


def save_results(all_results, output_file):
    """Save results in consistent format"""
    if 'test_subject_id' in all_results[0]:  # LOSO results
        results_dict = {
            'test_subject_ids': [r['test_subject_id'] for r in all_results],
            'remaining_subject_ids': [r['remaining_subject_ids'] for r in all_results],
            'n_folds': [r['n_folds'] for r in all_results],
            'cv_balanced_accuracies': [r['cv_balanced_accuracies'] for r in all_results],
            'cv_f1_scores': [r['cv_f1_scores'] for r in all_results],
            'mean_cv_balanced_accs': [r['mean_cv_balanced_acc'] for r in all_results],
            'std_cv_balanced_accs': [r['std_cv_balanced_acc'] for r in all_results],
            'mean_cv_f1s': [r['mean_cv_f1'] for r in all_results],
            'std_cv_f1s': [r['std_cv_f1'] for r in all_results],
            'test_balanced_accuracies': [r['test_balanced_accuracy'] for r in all_results],
            'test_f1_macros': [r['test_f1_macro'] for r in all_results],
            'confusion_matrices': [r['confusion_matrix'] for r in all_results],
            'best_fold_indices': [r['best_fold_idx'] for r in all_results],
            'train_curves_mean': {
                'train_losses_mean': [r['train_losses_mean'] for r in all_results],
                'train_balanced_accuracies_mean': [r['train_balanced_accuracies_mean'] for r in all_results],
                'train_f1_macros_mean': [r['train_f1_macros_mean'] for r in all_results],
                'val_losses_mean': [r['val_losses_mean'] for r in all_results],
                'val_balanced_accuracies_mean': [r['val_balanced_accuracies_mean'] for r in all_results],
                'val_f1_macros_mean': [r['val_f1_macros_mean'] for r in all_results],
            },
            'best_models': [r['best_model'] for r in all_results]
        }
    else:  # Subject-dependent results
        results_dict = {
            'n_folds': [r['n_folds'] for r in all_results],
            'cv_balanced_accuracies': [r['cv_balanced_accuracies'] for r in all_results],
            'cv_f1_scores': [r['cv_f1_scores'] for r in all_results],
            'mean_cv_balanced_accs': [r['mean_cv_balanced_acc'] for r in all_results],
            'std_cv_balanced_accs': [r['std_cv_balanced_acc'] for r in all_results],
            'mean_cv_f1s': [r['mean_cv_f1'] for r in all_results],
            'std_cv_f1s': [r['std_cv_f1'] for r in all_results],
            'test_balanced_accuracies': [r['test_balanced_accuracy'] for r in all_results],
            'test_f1_macros': [r['test_f1_macro'] for r in all_results],
            'confusion_matrices': [r['confusion_matrix'] for r in all_results],
            'best_fold_indices': [r['best_fold_idx'] for r in all_results],
            'train_curves_mean': {
                'train_losses_mean': [r['train_losses_mean'] for r in all_results],
                'train_balanced_accuracies_mean': [r['train_balanced_accuracies_mean'] for r in all_results],
                'train_f1_macros_mean': [r['train_f1_macros_mean'] for r in all_results],
                'val_losses_mean': [r['val_losses_mean'] for r in all_results],
                'val_balanced_accuracies_mean': [r['val_balanced_accuracies_mean'] for r in all_results],
                'val_f1_macros_mean': [r['val_f1_macros_mean'] for r in all_results],
            },
            'best_models': [r['best_model'] for r in all_results]
        }
    
    np.save(output_file, results_dict)
    return results_dict

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_cv_results(all_results, output_path, type_of_subject, model_name, n_subjects):
    """Plot training results with CV for any model type"""
    # Use colorblind-friendly colors
    colors = sns.color_palette("colorblind")
    
    all_test_balanced_accs = [result['test_balanced_accuracy'] for result in all_results]
    all_test_f1s = [result['test_f1_macro'] for result in all_results]
    all_cv_balanced_accs = [result['mean_cv_balanced_acc'] for result in all_results]
    all_cv_f1s = [result['mean_cv_f1'] for result in all_results]
    
    # Plot test metrics vs CV metrics
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{type_of_subject.title()} {model_name.upper()} - CV vs Test Performance Comparison', fontsize=16, y=0.98)
    
    # Test Balanced Accuracy
    axes[0, 0].plot(all_test_balanced_accs, marker='o', linestyle='-', color=colors[0], label='Test', linewidth=2)
    axes[0, 0].plot(all_cv_balanced_accs, marker='s', linestyle='--', color=colors[0], alpha=0.5, label='CV Mean', linewidth=2)
    axes[0, 0].set_title(f'Test vs CV Balanced Accuracy')
    axes[0, 0].set_xlabel('Subject ID')
    axes[0, 0].set_ylabel('Balanced Accuracy (%)')
    axes[0, 0].set_ylim(0, 102)
    axes[0, 0].legend()
    axes[0, 0].set_xticks(range(0, n_subjects, max(1, n_subjects//10)))
    axes[0, 0].set_xticklabels([f'S{i}' for i in range(0, n_subjects, max(1, n_subjects//10))], rotation=45)
    
    # Test F1 Macro
    axes[0, 1].plot(all_test_f1s, marker='o', linestyle='-', color=colors[1], label='Test', linewidth=2)
    axes[0, 1].plot(all_cv_f1s, marker='s', linestyle='--', color=colors[1], alpha=0.5, label='CV Mean', linewidth=2)
    axes[0, 1].set_title(f'Test vs CV F1 Macro')
    axes[0, 1].set_xlabel('Subject ID')
    axes[0, 1].set_ylabel('F1 Macro (%)')
    axes[0, 1].set_ylim(0, 102)
    axes[0, 1].legend()
    axes[0, 1].set_xticks(range(0, n_subjects, max(1, n_subjects//10)))
    axes[0, 1].set_xticklabels([f'S{i}' for i in range(0, n_subjects, max(1, n_subjects//10))], rotation=45)
    
    # CV Validation Scores Distribution
    all_cv_individual_scores = []
    for result in all_results:
        all_cv_individual_scores.extend(result['cv_balanced_accuracies'])
    
    axes[1, 0].hist(all_cv_individual_scores, bins=20, alpha=0.7, color=colors[2])
    axes[1, 0].set_title('Distribution of CV Fold Balanced Accuracies')
    axes[1, 0].set_xlabel('Balanced Accuracy (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xlim(0, 102)
    
    # Test vs CV correlation
    axes[1, 1].scatter(all_cv_balanced_accs, all_test_balanced_accs, alpha=0.6, color=colors[5], s=60)
    axes[1, 1].plot([0, 100], [0, 100], 'r--', alpha=0.5)
    axes[1, 1].set_title('CV vs Test Balanced Accuracy Correlation')
    axes[1, 1].set_xlabel('CV Mean Balanced Accuracy (%)')
    axes[1, 1].set_ylabel('Test Balanced Accuracy (%)')
    axes[1, 1].set_xlim(0, 102)
    axes[1, 1].set_ylim(0, 102)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_CV_test_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(all_results, output_path, type_of_subject, model_name):
    """Plot aggregated training curves (mean across subjects and folds)"""
    if len(all_results) == 0:
        return
    
    # Use colorblind-friendly colors
    colors = sns.color_palette("colorblind")
        
    num_epochs = len(all_results[0]['train_losses_mean'])
    
    # Calculate mean and std across subjects
    train_bal_accs_mean = np.mean([result['train_balanced_accuracies_mean'] for result in all_results], axis=0)
    train_bal_accs_std = np.std([result['train_balanced_accuracies_mean'] for result in all_results], axis=0)
    val_bal_accs_mean = np.mean([result['val_balanced_accuracies_mean'] for result in all_results], axis=0)
    val_bal_accs_std = np.std([result['val_balanced_accuracies_mean'] for result in all_results], axis=0)
    
    train_f1s_mean = np.mean([result['train_f1_macros_mean'] for result in all_results], axis=0)
    train_f1s_std = np.std([result['train_f1_macros_mean'] for result in all_results], axis=0)
    val_f1s_mean = np.mean([result['val_f1_macros_mean'] for result in all_results], axis=0)
    val_f1s_std = np.std([result['val_f1_macros_mean'] for result in all_results], axis=0)
    
    train_losses_mean = np.mean([result['train_losses_mean'] for result in all_results], axis=0)
    train_losses_std = np.std([result['train_losses_mean'] for result in all_results], axis=0)
    val_losses_mean = np.mean([result['val_losses_mean'] for result in all_results], axis=0)
    val_losses_std = np.std([result['val_losses_mean'] for result in all_results], axis=0)
    
    epochs = np.arange(1, num_epochs + 1)
    
    # Plot training curves
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'{type_of_subject.title()} {model_name.upper()} - Training Curves (Mean ± STD across subjects and CV folds)', fontsize=16, y=0.98)
    
    # Balanced Accuracy
    axes[0, 0].plot(epochs, train_bal_accs_mean, color=colors[0], linewidth=2, label='Train')
    axes[0, 0].fill_between(epochs, train_bal_accs_mean - train_bal_accs_std, 
                           train_bal_accs_mean + train_bal_accs_std, alpha=0.3, color=colors[0])
    axes[0, 0].set_title('Training Balanced Accuracy')
    axes[0, 0].set_ylabel('Balanced Accuracy (%)')
    axes[0, 0].set_ylim(0, 102)
    
    axes[0, 1].plot(epochs, val_bal_accs_mean, color=colors[1], linewidth=2, label='Validation')
    axes[0, 1].fill_between(epochs, val_bal_accs_mean - val_bal_accs_std, 
                           val_bal_accs_mean + val_bal_accs_std, alpha=0.3, color=colors[1])
    axes[0, 1].set_title('Validation Balanced Accuracy (CV)')
    axes[0, 1].set_ylabel('Balanced Accuracy (%)')
    axes[0, 1].set_ylim(0, 102)
    
    # F1 Macro
    axes[1, 0].plot(epochs, train_f1s_mean, color=colors[2], linewidth=2, label='Train F1')
    axes[1, 0].fill_between(epochs, train_f1s_mean - train_f1s_std, 
                           train_f1s_mean + train_f1s_std, alpha=0.3, color=colors[2])
    axes[1, 0].set_title('Training F1 Macro')
    axes[1, 0].set_ylabel('F1 Macro (%)')
    axes[1, 0].set_ylim(0, 102)
    
    axes[1, 1].plot(epochs, val_f1s_mean, color=colors[3], linewidth=2, label='Val F1')
    axes[1, 1].fill_between(epochs, val_f1s_mean - val_f1s_std, 
                           val_f1s_mean + val_f1s_std, alpha=0.3, color=colors[3])
    axes[1, 1].set_title('Validation F1 Macro (CV)')
    axes[1, 1].set_ylabel('F1 Macro (%)')
    axes[1, 1].set_ylim(0, 102)
    
    # Loss - with shared y-axis for better comparison
    loss_min = min(np.min(train_losses_mean), np.min(val_losses_mean))
    loss_max = max(np.max(train_losses_mean), np.max(val_losses_mean))
    
    axes[2, 0].plot(epochs, train_losses_mean, color=colors[4], linewidth=2, label='Train Loss')
    axes[2, 0].fill_between(epochs, train_losses_mean - train_losses_std, 
                           train_losses_mean + train_losses_std, alpha=0.3, color=colors[4])
    axes[2, 0].set_title('Training Loss')
    axes[2, 0].set_ylabel('Loss')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylim(loss_min * 0.9, loss_max * 1.5)
    
    axes[2, 1].plot(epochs, val_losses_mean, color=colors[5], linewidth=2, label='Val Loss')
    axes[2, 1].fill_between(epochs, val_losses_mean - val_losses_std, 
                           val_losses_mean + val_losses_std, alpha=0.3, color=colors[5])
    axes[2, 1].set_title('Validation Loss (CV)')
    axes[2, 1].set_ylabel('Loss')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylim(loss_min * 0.9, loss_max * 1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_CV_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(all_results, output_path, type_of_subject, model_name):
    """Plot average confusion matrix across all subjects"""
    if len(all_results) == 0:
        return
        
    all_conf_matrices = [result['confusion_matrix'] for result in all_results]
    avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
    # To normalize by row (show percentage of true class predictions)
    conf_matrix_pct = avg_conf_matrix.astype('float') / avg_conf_matrix.sum(axis=1)[:, np.newaxis]
    name_classes = {0: 'rest', 1: 'open', 2: 'close'}
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_pct, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=[name_classes[i] for i in range(conf_matrix_pct.shape[1])],
                yticklabels=[name_classes[i] for i in range(conf_matrix_pct.shape[0])])
    plt.title(f'Average Confusion Matrix - {type_of_subject} {model_name.upper()} (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_avg_confusion_matrix.png'))
    plt.close()


def plot_all_results(all_results, output_path, type_of_subject, model_name, n_subjects=None):
    """Plot all result visualizations (convenience function)"""
    if n_subjects is None:
        n_subjects = len(all_results)
        
    plot_cv_results(all_results, output_path, type_of_subject, model_name, n_subjects)
    plot_training_curves(all_results, output_path, type_of_subject, model_name)
    plot_confusion_matrix(all_results, output_path, type_of_subject, model_name)

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# These maintain the original function names for existing scripts
# =============================================================================

# Keep original function names as aliases for backward compatibility
def train_microstate_subject(subject_id, args, device, data_path):
    """Backward compatibility wrapper for microstate training"""
    return train_subject(subject_id, args, device, data_path, model_type='microstate')

def train_microstate_loso_subject(test_subject_id, args, device, data_path):
    """Backward compatibility wrapper for microstate LOSO training"""
    return train_loso(test_subject_id, args, device, data_path, model_type='microstate')

def train_loso_subject(test_subject_id, args, device, data_path, model_class, model_name="Model"):
    """Backward compatibility wrapper for DCN LOSO training"""
    return train_loso(test_subject_id, args, device, data_path, model_type='dcn')

def train_fusion_subject(subject_id, args, device, data_path, pretrained_models):
    """Backward compatibility wrapper for fusion training"""
    # Note: pretrained_models parameter is ignored in new unified approach
    return train_subject(subject_id, args, device, data_path, model_type='fusion')

def train_fusion_loso_subject(test_subject_id, args, device, data_path, pretrained_models):
    """Backward compatibility wrapper for fusion LOSO training"""
    # Note: pretrained_models parameter is ignored in new unified approach
    return train_loso(test_subject_id, args, device, data_path, model_type='fusion')

def save_loso_results(all_results, output_file):
    """Backward compatibility wrapper for saving LOSO results"""
    return save_results(all_results, output_file)

# =============================================================================
# FEATURE EXTRACTION UTILITIES (for fusion models that need pretrained features)
# =============================================================================

def extract_features_from_model(model, data, device, batch_size=32):
    """Extract features from a pretrained model"""
    
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
                print("  Model is EmbeddedMicroStateNet - extracting microstate sequences only")
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
                print(f"  Preprocessed data shape for EmbeddedMicroStateNet: {x.shape}")
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


def load_pretrained_models(args, project_root):
    """Load pretrained DCN and MicroStateNet models using dynamic paths"""
    
    output_folder = os.path.join(project_root, 'Output') + os.sep
    
    # Build DCN path dynamically
    dcn_path = f'{output_folder}ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_dcn_{args.n_folds}fold_results/'
    dcn_file = os.path.join(dcn_path, f'{args.type_of_subject}_dcn_{args.n_folds}fold_results.npy')
    
    # Build MicroStateNet path dynamically
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