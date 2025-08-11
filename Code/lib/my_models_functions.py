'''
Reusable training, validation, testing, and plotting functions for neural network models
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from braindecode.classifier import EEGClassifier
# Add this after your imports to suppress just this warning:
import warnings
warnings.filterwarnings("ignore", message="LogSoftmax final layer will be removed")

sys.path.append(os.path.abspath(__file__))
from lib import my_functions as mf

def train_epoch(model, device, train_loader, optimizer, criterion, epoch, log_interval=10):
    """Train the model for one epoch"""
    model.train()
    train_loss = 0
    train_total = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        train_total += target.size(0)
        
        # Store predictions and targets for metric computation
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    avg_loss = train_loss / train_total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    
    return avg_loss, balanced_acc, f1_macro


def validate(model, device, val_loader, criterion):
    """Validate the model"""
    model.eval()
    val_loss = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            total += target.size(0)
            
            # Store predictions and targets for metric computation
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = val_loss / total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    
    return avg_loss, balanced_acc, f1_macro


def test(model, device, test_loader, verbose=True):
    """Test the model"""
    model.eval()
    test_loss = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            total += target.size(0)
            
            # Store predictions and targets for metric computation
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    if verbose:
        print(f'Test set: Average loss: {test_loss:.4f}, '
              f'Balanced Accuracy: {balanced_acc:.2f}%, F1 Macro: {f1_macro:.2f}%')
    
    return balanced_acc, f1_macro, conf_matrix


def plot_cv_results(all_results, output_path, type_of_subject, model_name, n_subjects):
    """
    Plot training results with CV for any model type
    
    Parameters:
    -----------
    all_results : list
        List of result dictionaries from each subject
    output_path : str
        Path to save plots
    type_of_subject : str
        Subject type (dependent, independent, adaptive)
    model_name : str
        Name of the model (DCN, microsnet, etc.)
    n_subjects : int
        Number of subjects processed
    """
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
    axes[0, 0].set_xticks(range(0, n_subjects, max(1, n_subjects//10)))  # Show every 5th or 10th subject
    axes[0, 0].set_xticklabels([f'S{i}' for i in range(0, n_subjects, max(1, n_subjects//10))], rotation=45)
    # axes[0, 0].grid(True, alpha=0.3)
    
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
    # axes[0, 1].grid(True, alpha=0.3)
    
    # CV Validation Scores Distribution
    all_cv_individual_scores = []
    for result in all_results:
        all_cv_individual_scores.extend(result['cv_balanced_accuracies'])
    
    axes[1, 0].hist(all_cv_individual_scores, bins=20, alpha=0.7, color=colors[2])
    axes[1, 0].set_title('Distribution of CV Fold Balanced Accuracies')
    axes[1, 0].set_xlabel('Balanced Accuracy (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xlim(0, 102)
    # axes[1, 0].grid(True, alpha=0.3)
    
    # Test vs CV correlation
    axes[1, 1].scatter(all_cv_balanced_accs, all_test_balanced_accs, alpha=0.6, color=colors[5], s=60)
    axes[1, 1].plot([0, 100], [0, 100], 'r--', alpha=0.5)
    axes[1, 1].set_title('CV vs Test Balanced Accuracy Correlation')
    axes[1, 1].set_xlabel('CV Mean Balanced Accuracy (%)')
    axes[1, 1].set_ylabel('Test Balanced Accuracy (%)')
    axes[1, 1].set_xlim(0, 102)
    axes[1, 1].set_ylim(0, 102)
    # axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_CV_test_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(all_results, output_path, type_of_subject, model_name):
    """
    Plot aggregated training curves (mean across subjects and folds)
    
    Parameters:
    -----------
    all_results : list
        List of result dictionaries from each subject
    output_path : str
        Path to save plots
    type_of_subject : str
        Subject type (dependent, independent, adaptive)
    model_name : str
        Name of the model (DCN, microsnet, etc.)
    """
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
    # axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, val_bal_accs_mean, color=colors[1], linewidth=2, label='Validation')
    axes[0, 1].fill_between(epochs, val_bal_accs_mean - val_bal_accs_std, 
                           val_bal_accs_mean + val_bal_accs_std, alpha=0.3, color=colors[1])
    axes[0, 1].set_title('Validation Balanced Accuracy (CV)')
    axes[0, 1].set_ylabel('Balanced Accuracy (%)')
    axes[0, 1].set_ylim(0, 102)
    # axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Macro
    axes[1, 0].plot(epochs, train_f1s_mean, color=colors[2], linewidth=2, label='Train F1')
    axes[1, 0].fill_between(epochs, train_f1s_mean - train_f1s_std, 
                           train_f1s_mean + train_f1s_std, alpha=0.3, color=colors[2])
    axes[1, 0].set_title('Training F1 Macro')
    axes[1, 0].set_ylabel('F1 Macro (%)')
    axes[1, 0].set_ylim(0, 102)
    # axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, val_f1s_mean, color=colors[3], linewidth=2, label='Val F1')
    axes[1, 1].fill_between(epochs, val_f1s_mean - val_f1s_std, 
                           val_f1s_mean + val_f1s_std, alpha=0.3, color=colors[3])
    axes[1, 1].set_title('Validation F1 Macro (CV)')
    axes[1, 1].set_ylabel('F1 Macro (%)')
    axes[1, 1].set_ylim(0, 102)
    # axes[1, 1].grid(True, alpha=0.3)
    
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
    # axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(epochs, val_losses_mean, color=colors[5], linewidth=2, label='Val Loss')
    axes[2, 1].fill_between(epochs, val_losses_mean - val_losses_std, 
                           val_losses_mean + val_losses_std, alpha=0.3, color=colors[5])
    axes[2, 1].set_title('Validation Loss (CV)')
    axes[2, 1].set_ylabel('Loss')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylim(loss_min * 0.9, loss_max * 1.5)
    # axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{type_of_subject}_{model_name}_CV_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(all_results, output_path, type_of_subject, model_name):
    """
    Plot average confusion matrix across all subjects
    
    Parameters:
    -----------
    all_results : list
        List of result dictionaries from each subject
    output_path : str
        Path to save plots
    type_of_subject : str
        Subject type (dependent, independent, adaptive)
    model_name : str
        Name of the model (DCN, microsnet, etc.)
    """
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
    """
    Plot all result visualizations (convenience function)
    
    Parameters:
    -----------
    all_results : list
        List of result dictionaries from each subject
    output_path : str
        Path to save plots
    type_of_subject : str
        Subject type (dependent, independent, adaptive)
    model_name : str
        Name of the model (DCN, microsnet, etc.)
    n_subjects : int, optional
        Number of subjects (defaults to len(all_results))
    """
    if n_subjects is None:
        n_subjects = len(all_results)
        
    plot_cv_results(all_results, output_path, type_of_subject, model_name, n_subjects)
    plot_training_curves(all_results, output_path, type_of_subject, model_name)
    plot_confusion_matrix(all_results, output_path, type_of_subject, model_name)


def aggregate_fold_training_curves(fold_results):
    """
    Aggregate training curves across folds (calculate mean and std)
    
    Parameters:
    -----------
    fold_results : list
        List of fold result dictionaries
        
    Returns:
    --------
    dict : Dictionary with aggregated training curves
    """
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
    """
    Print cross-validation summary statistics
    
    Parameters:
    -----------
    cv_balanced_accs : list
        List of balanced accuracies from each CV fold
    cv_f1_scores : list
        List of F1 scores from each CV fold
    n_folds : int
        Number of CV folds
    """
    mean_cv_bal_acc = np.mean(cv_balanced_accs)
    std_cv_bal_acc = np.std(cv_balanced_accs)
    mean_cv_f1 = np.mean(cv_f1_scores)
    std_cv_f1 = np.std(cv_f1_scores)
    
    print(f"\n📊 {n_folds}-Fold CV Results Summary:")
    print(f"CV Balanced Accuracy: {mean_cv_bal_acc:.2f}% ± {std_cv_bal_acc:.2f}%")
    print(f"CV F1 Macro: {mean_cv_f1:.2f}% ± {std_cv_f1:.2f}%")
    
    return mean_cv_bal_acc, std_cv_bal_acc, mean_cv_f1, std_cv_f1


def print_final_summary(all_results, model_name, n_folds):
    """
    Print final summary of all subjects' results
    
    Parameters:
    -----------
    all_results : list
        List of result dictionaries from each subject
    model_name : str
        Name of the model
    n_folds : int
        Number of CV folds
    """
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

# =============================== Independent functions ==============================


def load_subjects_batch(subject_ids, data_path):
    """
    Load multiple subjects and concatenate their data efficiently
    
    Args:
        subject_ids: List of subject IDs to load
        data_path: Path to data directory
    
    Returns:
        x_combined: Concatenated tensor of all subjects' data
        y_combined: Concatenated tensor of all subjects' labels
    """
    if not subject_ids:
        return None, None
    
    x_list, y_list = [], []
    
    for subject_id in subject_ids:
        data, y = mf.load_data(subject_id, data_path=data_path)
        x_list.append(torch.tensor(data, dtype=torch.float32).squeeze(1))
        y_list.append(torch.tensor(y, dtype=torch.long))
    
    # Concatenate all subjects
    x_combined = torch.cat(x_list, dim=0)
    y_combined = torch.cat(y_list, dim=0)
    
    del x_list, y_list  # Free intermediate memory
    return x_combined, y_combined


def create_braindecode_model(model_class, x_shape, n_classes, device, lr=1e-3):
    """
    Create a Braindecode model with EEGClassifier wrapper
    
    Args:
        model_class: Braindecode model class (Deep4Net, EEGNet, etc.)
        x_shape: Input data shape (batch, channels, timepoints)
        n_classes: Number of output classes
        device: torch.device
        lr: Learning rate
    
    Returns:
        model: EEGClassifier wrapped model
        net: Underlying PyTorch model
        criterion: Loss function
        optimizer: Optimizer
    """
    # Determine model parameters based on class name
    model_name = model_class.__name__
    
    if model_name == "Deep4Net":
        base_model = model_class(
            n_chans=x_shape[1],
            n_outputs=n_classes,
            n_times=x_shape[2],
            final_conv_length='auto'
        )
    elif model_name == "EEGNet":
        base_model = model_class(
            n_chans=x_shape[1],
            n_outputs=n_classes,
            n_times=x_shape[2]
        )
    else:
        # Generic fallback
        base_model = model_class(
            n_chans=x_shape[1],
            n_outputs=n_classes,
            n_times=x_shape[2]
        )
    
    model = EEGClassifier(
        base_model,
        criterion=nn.CrossEntropyLoss(),  # Changed from NLLLoss
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
        train_split=None,
        device=device
    )
    
    net = model.module.to(device)
    criterion = nn.CrossEntropyLoss()  # Changed from NLLLoss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    return model, net, criterion, optimizer


def train_loso_subject(test_subject_id, args, device, data_path, model_class, model_name="Model"):
    """
    Standard LOSO training with 4-fold CV on all 49 remaining subjects
    
    Args:
        test_subject_id: Subject ID to use for testing (0-49)
        args: Argument namespace with training parameters
        device: torch.device
        data_path: Path to data directory
        model_class: Braindecode model class (Deep4Net, EEGNet, etc.)
        model_name: String name for logging purposes
        
    Returns:
        dict: Complete results dictionary with all metrics and models
    """
    print(f"\n▶ {model_name} LOSO Training - Test Subject {test_subject_id}")
    
    # Get all remaining subjects (49 subjects)
    all_remaining_subjects = [i for i in range(args.n_subjects) if i != test_subject_id]
    
    print(f"Test subject: {test_subject_id}")
    print(f"Remaining subjects for CV: {len(all_remaining_subjects)} subjects")
    
    # Load test data ONCE (1 subject - keep in memory) 
    print(f"Loading test subject {test_subject_id}...")
    test_data, test_y = mf.load_data(test_subject_id, data_path=data_path)
    x_test = torch.tensor(test_data, dtype=torch.float32).squeeze(1)
    y_test = torch.tensor(test_y, dtype=torch.long)
    print(f"Test data shape: {x_test.shape}, labels shape: {y_test.shape}")
    
    # 4-Fold Cross Validation on all 49 remaining subjects
    dummy_y = [0] * len(all_remaining_subjects)  # Dummy for StratifiedKFold
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Store results for each fold
    fold_results = []
    cv_balanced_accs = []
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_remaining_subjects, dummy_y)):
        print(f"\n--- {model_name} Fold {fold + 1}/{args.n_folds} ---")
        
        # Get subjects for this fold (~37 train, ~12 val)
        fold_train_subjects = [all_remaining_subjects[i] for i in train_idx]
        fold_val_subjects = [all_remaining_subjects[i] for i in val_idx]
        
        print(f"Loading {len(fold_train_subjects)} training subjects and {len(fold_val_subjects)} validation subjects for fold {fold + 1}")
        
        # Load training and validation data for this fold
        x_train_fold, y_train_fold = load_subjects_batch(fold_train_subjects, data_path)
        x_val_fold, y_val_fold = load_subjects_batch(fold_val_subjects, data_path)
        
        print(f"Fold {fold + 1} - Train: {x_train_fold.shape}, Val: {x_val_fold.shape}")
        print(f"Memory usage: {len(fold_train_subjects)} train + {len(fold_val_subjects)} val + 1 test = {len(fold_train_subjects) + len(fold_val_subjects) + 1} subjects")
        
        # DataLoaders for this fold
        train_loader = DataLoader(TensorDataset(x_train_fold, y_train_fold), 
                                 batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val_fold, y_val_fold), 
                               batch_size=args.batch_size, shuffle=False)
        
        # Create model for this fold
        model, net, criterion, optimizer = create_braindecode_model(
            model_class, x_train_fold.shape, len(torch.unique(y_train_fold)), device, args.lr
        )
        
        # Training for this fold
        fold_train_losses, fold_val_losses = [], []
        fold_train_balanced_accs, fold_val_balanced_accs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        
        for epoch in range(1, args.epochs + 1):
            # Train - using existing mmf function
            train_loss, train_balanced_acc, train_f1 = train_epoch(
                net, device, train_loader, optimizer, criterion, epoch, 
                args.log_interval if epoch % 10 == 1 else 999)
            
            # Validate - using existing mmf function
            val_loss, val_balanced_acc, val_f1 = validate(net, device, val_loader, criterion)
            
            fold_train_losses.append(train_loss)
            fold_train_balanced_accs.append(train_balanced_acc)
            fold_train_f1s.append(train_f1)
            fold_val_losses.append(val_loss)
            fold_val_balanced_accs.append(val_balanced_acc)
            fold_val_f1s.append(val_f1)
            
            if epoch % 20 == 0 or epoch == args.epochs:  # Print every 20 epochs
                print(f"{model_name} Fold {fold + 1}, Epoch {epoch:02d}/{args.epochs} | "
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
        
        print(f"✅ {model_name} Fold {fold + 1} completed - Val Bal Acc: {val_balanced_acc:.2f}%, F1: {val_f1:.2f}%")
        
        # FREE this fold's data immediately
        del x_train_fold, y_train_fold, x_val_fold, y_val_fold, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"🧹 Fold {fold + 1} memory cleaned")
    
    # Cross-validation summary - using existing mmf function
    mean_cv_bal_acc, std_cv_bal_acc, mean_cv_f1, std_cv_f1 = print_cv_summary(
        cv_balanced_accs, cv_f1_scores, args.n_folds)
    
    # Select best fold model
    best_fold_idx = np.argmax(cv_balanced_accs)
    best_model = fold_results[best_fold_idx]['model']
    print(f"Best {model_name} fold: {best_fold_idx + 1} (Val Bal Acc: {cv_balanced_accs[best_fold_idx]:.2f}%)")
    
    # Final test on held-out test subject - using existing mmf function
    test_loader = DataLoader(TensorDataset(x_test, y_test), 
                            batch_size=args.batch_size, shuffle=False)
    
    test_balanced_acc, test_f1, conf_matrix = test(best_model.module, device, test_loader)
    print(f"🎯 {model_name} Final Test Results - Balanced Acc: {test_balanced_acc:.2f}%, F1: {test_f1:.2f}%")
    
    # Aggregate training curves - using existing mmf function
    training_curves = aggregate_fold_training_curves(fold_results)
    
    # Clean up remaining memory
    del x_test, y_test
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


def save_loso_results(all_results, output_file):
    """
    Save LOSO results in consistent format for standard 4-fold CV
    
    Args:
        all_results: List of result dictionaries from train_loso_subject
        output_file: Path to save .npy file
    """
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
    np.save(output_file, results_dict)
    return results_dict