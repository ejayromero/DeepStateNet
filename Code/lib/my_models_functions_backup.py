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
from sklearn.model_selection import train_test_split, StratifiedKFold
from braindecode.classifier import EEGClassifier
# Add this after your imports to suppress just this warning:
import warnings
warnings.filterwarnings("ignore", message="LogSoftmax final layer will be removed")

sys.path.append(os.path.abspath(__file__))
from lib import my_functions as mf
from lib import my_models as mm

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
        Name of the model (DCN, MicroStateNet, etc.)
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
        Name of the model (DCN, MicroStateNet, etc.)
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
        Name of the model (DCN, MicroStateNet, etc.)
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
        Name of the model (DCN, MicroStateNet, etc.)
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

# MSN
def load_subject_microstate_data(subject_id, args, data_path):
    """Load microstate data for a single subject"""
    # Load labels
    _, y = mf.load_data(subject_id, data_path=data_path)
    
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
    finals_ls = mf.load_ms(
        subject_id=subject_id,
        n_clusters=args.n_clusters, 
        seq_time_path=finals_ls_folder,
        seq_time_type=ms_file_type, 
        seq_time_specific=args.ms_file_specific
    )
    
    return finals_ls, y


def prepare_model_input(data, model_name, use_embedding=False):
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


def create_microstate_model(model_name, n_microstates, n_classes, sequence_length, args, device):
    """Create microstate model using the model factory"""
    # Create model using factory function
    if 'attention' in model_name:
        model = mm.get_model(
            model_name=model_name,
            n_microstates=n_microstates,
            n_classes=n_classes,
            sequence_length=sequence_length,
            dropout=args.dropout,
            embedding_dim=args.embedding_dim,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            use_embedding=args.use_embedding
        )
    else:
        # For other models, use the original parameters
        model = mm.get_model(
            model_name=model_name,
            n_microstates=n_microstates,
            n_classes=n_classes,
            sequence_length=sequence_length,
            dropout=args.dropout,
            use_embedding=args.use_embedding
        )
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for consistency
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Using model: {model_name} with embedding: {args.use_embedding}")
    print(f"Model description: {mm.MODEL_INFO[model_name]['description']}")
    
    return model, criterion, optimizer


def load_microstate_subjects_batch(subject_ids, args, data_path):
    """
    Load multiple subjects' microstate data and concatenate efficiently
    
    Args:
        subject_ids: List of subject IDs to load
        args: Arguments containing model configuration
        data_path: Path to data directory
    
    Returns:
        x_combined: Concatenated tensor of all subjects' microstate data
        y_combined: Concatenated tensor of all subjects' labels
        n_microstates: Number of microstate categories
        sequence_length: Length of microstate sequences
    """
    if not subject_ids:
        return None, None, None, None
    
    x_list, y_list = [], []
    n_microstates = None
    sequence_length = None
    
    for subject_id in subject_ids:
        # Load microstate data for this subject
        data, y = load_subject_microstate_data(subject_id, args, data_path)
        x, n_ms, seq_len = prepare_model_input(data, args.model_name, args.use_embedding)
        
        # Store dimensions from first subject
        if n_microstates is None:
            n_microstates = n_ms
            sequence_length = seq_len
        
        x_list.append(x)
        y_list.append(torch.tensor(y, dtype=torch.long))
    
    # Concatenate all subjects
    x_combined = torch.cat(x_list, dim=0)
    y_combined = torch.cat(y_list, dim=0)
    
    del x_list, y_list  # Free intermediate memory
    return x_combined, y_combined, n_microstates, sequence_length


def train_microstate_subject(subject_id, args, device, data_path):
    """Train microstate model for a single subject with K-fold CV and 10% test split (subject-dependent)"""
    print(f"\n▶ Training Subject {subject_id}")
    
    # Load single subject data for memory efficiency
    data, y = load_subject_microstate_data(subject_id, args, data_path)
    x, n_microstates, sequence_length = prepare_model_input(data, args.model_name, args.use_embedding)
    y = torch.tensor(y, dtype=torch.long)
    
    print(f"Subject {subject_id} data shape: {x.shape}, labels shape: {y.shape}")
    print(f"Number of microstate categories: {n_microstates}")
    print(f"Number of classes: {len(torch.unique(y))}")
    
    # Split: 90% for CV, 10% for final test
    x_cv, x_test, y_cv, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42, stratify=y)
    
    print(f"CV data: {x_cv.shape}, Test data: {x_test.shape}")
    
    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    cv_balanced_accs = []
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_cv, y_cv)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
        
        # Get fold data
        x_train_fold = x_cv[train_idx]
        y_train_fold = y_cv[train_idx]
        x_val_fold = x_cv[val_idx]
        y_val_fold = y_cv[val_idx]
        
        print(f"Fold {fold + 1}: Train {x_train_fold.shape}, Val {x_val_fold.shape}")
        
        # DataLoaders for this fold
        train_loader = DataLoader(TensorDataset(x_train_fold, y_train_fold), 
                                 batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val_fold, y_val_fold), 
                               batch_size=args.batch_size, shuffle=False)
        
        # Model for this fold
        n_classes = len(torch.unique(y))
        model, criterion, optimizer = create_microstate_model(
            args.model_name, n_microstates, n_classes, sequence_length, args, device
        )
        
        # Training for this fold
        fold_train_losses, fold_val_losses = [], []
        fold_train_balanced_accs, fold_val_balanced_accs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        
        for epoch in range(1, args.epochs + 1):
            # Train - USING SHARED FUNCTION
            train_loss, train_balanced_acc, train_f1 = train_epoch(
                model, device, train_loader, optimizer, criterion, epoch, 
                args.log_interval if epoch % 10 == 1 else 999)  # Reduce logging frequency
            
            # Validate - USING SHARED FUNCTION
            val_loss, val_balanced_acc, val_f1 = validate(model, device, val_loader, criterion)
            
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
    mean_cv_bal_acc, std_cv_bal_acc, mean_cv_f1, std_cv_f1 = print_cv_summary(
        cv_balanced_accs, cv_f1_scores, args.n_folds)
    
    # Select best fold model (highest validation balanced accuracy)
    best_fold_idx = np.argmax(cv_balanced_accs)
    best_model = fold_results[best_fold_idx]['model']
    print(f"Best fold: {best_fold_idx + 1} (Val Bal Acc: {cv_balanced_accs[best_fold_idx]:.2f}%)")
    
    # Final test on held-out test set using best model - USING SHARED FUNCTION
    test_loader = DataLoader(TensorDataset(x_test, y_test), 
                            batch_size=args.batch_size, shuffle=False)
    
    test_balanced_acc, test_f1, conf_matrix = test(best_model, device, test_loader)
    print(f"🎯 Final Test Results - Balanced Acc: {test_balanced_acc:.2f}%, F1: {test_f1:.2f}%")
    
    # Aggregate training curves across folds - USING SHARED FUNCTION
    training_curves = aggregate_fold_training_curves(fold_results)
    
    # Clean up memory
    del data, x, y, x_cv, x_test, y_cv, y_test
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
        **training_curves  # ← UNPACKS ALL TRAINING CURVE DATA
    }


def train_microstate_loso_subject(test_subject_id, args, device, data_path):
    """
    LOSO training for microstate models with 4-fold CV on all 49 remaining subjects
    
    Args:
        test_subject_id: Subject ID to use for testing (0-49)
        args: Argument namespace with training parameters
        device: torch.device
        data_path: Path to data directory
        
    Returns:
        dict: Complete results dictionary with all metrics and models
    """
    embedding_suffix = "_embedded" if args.use_embedding else ""
    model_name_with_embedding = f"{args.model_name}{embedding_suffix}"
    
    print(f"\n▶ {model_name_with_embedding} LOSO Training - Test Subject {test_subject_id}")
    
    # Get all remaining subjects (49 subjects)
    all_remaining_subjects = [i for i in range(args.n_subjects) if i != test_subject_id]
    
    print(f"Test subject: {test_subject_id}")
    print(f"Remaining subjects for CV: {len(all_remaining_subjects)} subjects")
    
    # Load test data ONCE (1 subject - keep in memory) 
    print(f"Loading test subject {test_subject_id}...")
    test_data, test_y = load_subject_microstate_data(test_subject_id, args, data_path)
    x_test, n_microstates, sequence_length = prepare_model_input(test_data, args.model_name, args.use_embedding)
    y_test = torch.tensor(test_y, dtype=torch.long)
    print(f"Test data shape: {x_test.shape}, labels shape: {y_test.shape}")
    print(f"Number of microstate categories: {n_microstates}")
    
    # 4-Fold Cross Validation on all 49 remaining subjects
    dummy_y = [0] * len(all_remaining_subjects)  # Dummy for StratifiedKFold
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Store results for each fold
    fold_results = []
    cv_balanced_accs = []
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_remaining_subjects, dummy_y)):
        print(f"\n--- {model_name_with_embedding} Fold {fold + 1}/{args.n_folds} ---")
        
        # Get subjects for this fold (~37 train, ~12 val)
        fold_train_subjects = [all_remaining_subjects[i] for i in train_idx]
        fold_val_subjects = [all_remaining_subjects[i] for i in val_idx]
        
        print(f"Loading {len(fold_train_subjects)} training subjects and {len(fold_val_subjects)} validation subjects for fold {fold + 1}")
        
        # Load training and validation data for this fold
        x_train_fold, y_train_fold, _, _ = load_microstate_subjects_batch(fold_train_subjects, args, data_path)
        x_val_fold, y_val_fold, _, _ = load_microstate_subjects_batch(fold_val_subjects, args, data_path)
        
        print(f"Fold {fold + 1} - Train: {x_train_fold.shape}, Val: {x_val_fold.shape}")
        print(f"Memory usage: {len(fold_train_subjects)} train + {len(fold_val_subjects)} val + 1 test = {len(fold_train_subjects) + len(fold_val_subjects) + 1} subjects")
        
        # DataLoaders for this fold
        train_loader = DataLoader(TensorDataset(x_train_fold, y_train_fold), 
                                 batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val_fold, y_val_fold), 
                               batch_size=args.batch_size, shuffle=False)
        
        # Create model for this fold
        n_classes = len(torch.unique(y_train_fold))
        model, criterion, optimizer = create_microstate_model(
            args.model_name, n_microstates, n_classes, sequence_length, args, device
        )
        
        # Training for this fold
        fold_train_losses, fold_val_losses = [], []
        fold_train_balanced_accs, fold_val_balanced_accs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        
        for epoch in range(1, args.epochs + 1):
            # Train - using existing mmf function
            train_loss, train_balanced_acc, train_f1 = train_epoch(
                model, device, train_loader, optimizer, criterion, epoch, 
                args.log_interval if epoch % 10 == 1 else 999)
            
            # Validate - using existing mmf function
            val_loss, val_balanced_acc, val_f1 = validate(model, device, val_loader, criterion)
            
            fold_train_losses.append(train_loss)
            fold_train_balanced_accs.append(train_balanced_acc)
            fold_train_f1s.append(train_f1)
            fold_val_losses.append(val_loss)
            fold_val_balanced_accs.append(val_balanced_acc)
            fold_val_f1s.append(val_f1)
            
            if epoch % 20 == 0 or epoch == args.epochs:  # Print every 20 epochs
                print(f"{model_name_with_embedding} Fold {fold + 1}, Epoch {epoch:02d}/{args.epochs} | "
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
        
        print(f"✅ {model_name_with_embedding} Fold {fold + 1} completed - Val Bal Acc: {val_balanced_acc:.2f}%, F1: {val_f1:.2f}%")
        
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
    print(f"Best {model_name_with_embedding} fold: {best_fold_idx + 1} (Val Bal Acc: {cv_balanced_accs[best_fold_idx]:.2f}%)")
    
    # Final test on held-out test subject - using existing mmf function
    test_loader = DataLoader(TensorDataset(x_test, y_test), 
                            batch_size=args.batch_size, shuffle=False)
    
    test_balanced_acc, test_f1, conf_matrix = test(best_model, device, test_loader)
    print(f"🎯 {model_name_with_embedding} Final Test Results - Balanced Acc: {test_balanced_acc:.2f}%, F1: {test_f1:.2f}%")
    
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



def load_subject_dual_data(subject_id, args, data_path):
    """Load both raw EEG and microstate data for a single subject"""
    
    # Load raw EEG data and labels
    raw_data, y = mf.load_data(subject_id, data_path=data_path)
    
    # Load microstate timeseries using existing function
    ms_data, _ = load_subject_microstate_data(subject_id, args, data_path)
    
    return raw_data, ms_data, y


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


def load_dual_data_subjects_batch(subject_ids, args, data_path, pretrained_models, device):
    """
    Load multiple subjects' dual (raw EEG + microstate) data and extract features efficiently
    
    Args:
        subject_ids: List of subject IDs to load
        args: Arguments containing model configuration
        data_path: Path to data directory
        pretrained_models: Dictionary containing pretrained DCN and MS models
        device: torch.device
    
    Returns:
        raw_features_combined: Concatenated tensor of all subjects' raw EEG features
        ms_features_combined: Concatenated tensor of all subjects' microstate features
        y_combined: Concatenated tensor of all subjects' labels
    """
    if not subject_ids:
        return None, None, None
    
    raw_features_list, ms_features_list, y_list = [], [], []
    
    for subject_id in subject_ids:
        # Load dual data for this subject
        raw_data, ms_data, y = load_subject_dual_data(subject_id, args, data_path)
        
        # Prepare inputs
        raw_x = torch.tensor(raw_data, dtype=torch.float32).squeeze(1)
        ms_x, _, _ = prepare_model_input(ms_data, args.model_name, args.use_embedding)
        
        # Extract features from pretrained models
        print(f"  Extracting features for subject {subject_id}...")
        raw_features = extract_features_from_model(pretrained_models['dcn'][subject_id], raw_x, device)
        ms_features = extract_features_from_model(pretrained_models['ms'][subject_id], ms_x, device)
        
        raw_features_list.append(raw_features)
        ms_features_list.append(ms_features)
        y_list.append(torch.tensor(y, dtype=torch.long))
    
    # Concatenate all subjects
    raw_features_combined = torch.cat(raw_features_list, dim=0)
    ms_features_combined = torch.cat(ms_features_list, dim=0)
    y_combined = torch.cat(y_list, dim=0)
    
    del raw_features_list, ms_features_list, y_list  # Free intermediate memory
    return raw_features_combined, ms_features_combined, y_combined


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


def train_fusion_subject(subject_id, args, device, data_path, pretrained_models):
    """Train fusion model for a single subject with K-fold CV and 10% test split (subject-dependent)"""
    print(f"\n▶ Training Subject {subject_id}")
    
    # Load single subject data for memory efficiency
    raw_data, ms_data, y = load_subject_dual_data(subject_id, args, data_path)
    
    # Prepare inputs for models
    raw_x = torch.tensor(raw_data, dtype=torch.float32).squeeze(1)  # (n_trials, n_channels, timepoints)
    ms_x, n_microstates, sequence_length = prepare_model_input(ms_data, args.model_name, args.use_embedding)
    y = torch.tensor(y, dtype=torch.long)
    
    print(f"Subject {subject_id} raw data shape: {raw_x.shape}")
    print(f"Subject {subject_id} microstate data shape: {ms_x.shape}")
    print(f"Subject {subject_id} labels shape: {y.shape}")
    print(f"Number of classes: {len(torch.unique(y))}")
    
    # Split: 90% for CV, 10% for final test
    indices = np.arange(len(y))
    cv_indices, test_indices = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=y.numpy())
    
    raw_cv, raw_test = raw_x[cv_indices], raw_x[test_indices]
    ms_cv, ms_test = ms_x[cv_indices], ms_x[test_indices]
    y_cv, y_test = y[cv_indices], y[test_indices]
    
    print(f"CV data: Raw {raw_cv.shape}, MS {ms_cv.shape}, Test data: Raw {raw_test.shape}, MS {ms_test.shape}")
    
    # Extract features from pretrained models for all data first
    print(f"Extracting features from pretrained DCN model...")
    raw_features_cv = extract_features_from_model(pretrained_models['dcn'][subject_id], raw_cv, device)
    raw_features_test = extract_features_from_model(pretrained_models['dcn'][subject_id], raw_test, device)
    
    print(f"Extracting features from pretrained MicroStateNet model...")
    ms_features_cv = extract_features_from_model(pretrained_models['ms'][subject_id], ms_cv, device)
    ms_features_test = extract_features_from_model(pretrained_models['ms'][subject_id], ms_test, device)
    
    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    cv_balanced_accs = []
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_features_cv, y_cv)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
        
        # Get fold data using extracted features
        raw_train_fold = raw_features_cv[train_idx]
        ms_train_fold = ms_features_cv[train_idx]
        y_train_fold = y_cv[train_idx]
        raw_val_fold = raw_features_cv[val_idx]
        ms_val_fold = ms_features_cv[val_idx]
        y_val_fold = y_cv[val_idx]
        
        print(f"Fold {fold + 1}: Train Raw {raw_train_fold.shape}, MS {ms_train_fold.shape}")
        print(f"Fold {fold + 1}: Val Raw {raw_val_fold.shape}, MS {ms_val_fold.shape}")
        
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
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        print(f"Using DeepStateNet with DCN + {args.model_name} fusion")
        
        # Training for this fold using dual-modal training
        fold_train_losses, fold_val_losses = [], []
        fold_train_balanced_accs, fold_val_balanced_accs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss, train_balanced_acc, train_f1 = train_dual_epoch(
                model, device, train_loader, optimizer, criterion, epoch,
                args.log_interval if epoch % 10 == 1 else 999)
            
            # Validate
            val_loss, val_balanced_acc, val_f1 = validate_dual(model, device, val_loader, criterion)
            
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
    
    # Cross-validation summary
    mean_cv_bal_acc, std_cv_bal_acc, mean_cv_f1, std_cv_f1 = print_cv_summary(
        cv_balanced_accs, cv_f1_scores, args.n_folds)
    
    # Select best fold model
    best_fold_idx = np.argmax(cv_balanced_accs)
    best_model = fold_results[best_fold_idx]['model']
    print(f"Best fold: {best_fold_idx + 1} (Val Bal Acc: {cv_balanced_accs[best_fold_idx]:.2f}%)")
    
    # Final test on held-out test set
    test_loader = DataLoader(TensorDataset(raw_features_test, ms_features_test, y_test), 
                            batch_size=args.batch_size, shuffle=False)
    
    test_balanced_acc, test_f1, conf_matrix = test_dual(best_model, device, test_loader)
    print(f"🎯 Final Test Results - Balanced Acc: {test_balanced_acc:.2f}%, F1: {test_f1:.2f}%")
    
    # Aggregate training curves
    training_curves = aggregate_fold_training_curves(fold_results)
    
    # Clean up memory
    del raw_data, ms_data, raw_x, ms_x, y, raw_cv, raw_test, ms_cv, ms_test, y_cv, y_test
    del raw_features_cv, raw_features_test, ms_features_cv, ms_features_test
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


def train_fusion_loso_subject(test_subject_id, args, device, data_path, pretrained_models):
    """
    LOSO training for fusion models with 4-fold CV on all 49 remaining subjects
    
    Args:
        test_subject_id: Subject ID to use for testing (0-49)
        args: Argument namespace with training parameters
        device: torch.device
        data_path: Path to data directory
        pretrained_models: Dictionary containing pretrained DCN and MS models
        
    Returns:
        dict: Complete results dictionary with all metrics and models
    """
    embedding_suffix = "_embedded" if args.use_embedding else ""
    model_name_with_embedding = f"dsn_{args.model_name}{embedding_suffix}"
    
    print(f"\n▶ {model_name_with_embedding} LOSO Training - Test Subject {test_subject_id}")
    
    # Get all remaining subjects (49 subjects)
    all_remaining_subjects = [i for i in range(args.n_subjects) if i != test_subject_id]
    
    print(f"Test subject: {test_subject_id}")
    print(f"Remaining subjects for CV: {len(all_remaining_subjects)} subjects")
    
    # Load test data ONCE (1 subject - keep in memory) 
    print(f"Loading test subject {test_subject_id}...")
    test_raw_data, test_ms_data, test_y = load_subject_dual_data(test_subject_id, args, data_path)
    
    # Prepare test inputs and extract features
    test_raw_x = torch.tensor(test_raw_data, dtype=torch.float32).squeeze(1)
    test_ms_x, _, _ = prepare_model_input(test_ms_data, args.model_name, args.use_embedding)
    test_y = torch.tensor(test_y, dtype=torch.long)
    
    print(f"Extracting features from pretrained models for test subject...")
    test_raw_features = extract_features_from_model(pretrained_models['dcn'][test_subject_id], test_raw_x, device)
    test_ms_features = extract_features_from_model(pretrained_models['ms'][test_subject_id], test_ms_x, device)
    
    print(f"Test data - Raw features: {test_raw_features.shape}, MS features: {test_ms_features.shape}, Labels: {test_y.shape}")
    
    # 4-Fold Cross Validation on all 49 remaining subjects
    dummy_y = [0] * len(all_remaining_subjects)  # Dummy for StratifiedKFold
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Store results for each fold
    fold_results = []
    cv_balanced_accs = []
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_remaining_subjects, dummy_y)):
        print(f"\n--- {model_name_with_embedding} Fold {fold + 1}/{args.n_folds} ---")
        
        # Get subjects for this fold (~37 train, ~12 val)
        fold_train_subjects = [all_remaining_subjects[i] for i in train_idx]
        fold_val_subjects = [all_remaining_subjects[i] for i in val_idx]
        
        print(f"Loading {len(fold_train_subjects)} training subjects and {len(fold_val_subjects)} validation subjects for fold {fold + 1}")
        
        # Load training and validation data for this fold
        raw_train_features, ms_train_features, y_train_fold = load_dual_data_subjects_batch(fold_train_subjects, args, data_path, pretrained_models, device)
        raw_val_features, ms_val_features, y_val_fold = load_dual_data_subjects_batch(fold_val_subjects, args, data_path, pretrained_models, device)
        
        print(f"Fold {fold + 1} - Train Raw: {raw_train_features.shape}, MS: {ms_train_features.shape}")
        print(f"Fold {fold + 1} - Val Raw: {raw_val_features.shape}, MS: {ms_val_features.shape}")
        print(f"Memory usage: {len(fold_train_subjects)} train + {len(fold_val_subjects)} val + 1 test = {len(fold_train_subjects) + len(fold_val_subjects) + 1} subjects")
        
        # DataLoaders for this fold
        train_loader = DataLoader(TensorDataset(raw_train_features, ms_train_features, y_train_fold), 
                                 batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(raw_val_features, ms_val_features, y_val_fold), 
                               batch_size=args.batch_size, shuffle=False)
        
        # Create model for this fold
        n_classes = len(torch.unique(y_train_fold))
        raw_feature_dim = raw_train_features.shape[1]
        ms_feature_dim = ms_train_features.shape[1]
        
        model = mm.DeepStateNetClassifier(
            raw_feature_dim, ms_feature_dim, n_classes
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        print(f"Using DeepStateNet with DCN + {args.model_name} fusion")
        
        # Training for this fold
        fold_train_losses, fold_val_losses = [], []
        fold_train_balanced_accs, fold_val_balanced_accs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        
        for epoch in range(1, args.epochs + 1):
            # Train using dual-modal training
            train_loss, train_balanced_acc, train_f1 = train_dual_epoch(
                model, device, train_loader, optimizer, criterion, epoch,
                args.log_interval if epoch % 10 == 1 else 999)
            
            # Validate using dual-modal validation
            val_loss, val_balanced_acc, val_f1 = validate_dual(model, device, val_loader, criterion)
            
            fold_train_losses.append(train_loss)
            fold_train_balanced_accs.append(train_balanced_acc)
            fold_train_f1s.append(train_f1)
            fold_val_losses.append(val_loss)
            fold_val_balanced_accs.append(val_balanced_acc)
            fold_val_f1s.append(val_f1)
            
            if epoch % 20 == 0 or epoch == args.epochs:  # Print every 20 epochs
                print(f"{model_name_with_embedding} Fold {fold + 1}, Epoch {epoch:02d}/{args.epochs} | "
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
        
        print(f"✅ {model_name_with_embedding} Fold {fold + 1} completed - Val Bal Acc: {val_balanced_acc:.2f}%, F1: {val_f1:.2f}%")
        
        # FREE this fold's data immediately
        del raw_train_features, ms_train_features, y_train_fold
        del raw_val_features, ms_val_features, y_val_fold, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"🧹 Fold {fold + 1} memory cleaned")
    
    # Cross-validation summary
    mean_cv_bal_acc, std_cv_bal_acc, mean_cv_f1, std_cv_f1 = print_cv_summary(
        cv_balanced_accs, cv_f1_scores, args.n_folds)
    
    # Select best fold model
    best_fold_idx = np.argmax(cv_balanced_accs)
    best_model = fold_results[best_fold_idx]['model']
    print(f"Best {model_name_with_embedding} fold: {best_fold_idx + 1} (Val Bal Acc: {cv_balanced_accs[best_fold_idx]:.2f}%)")
    
    # Final test on held-out test subject
    test_loader = DataLoader(TensorDataset(test_raw_features, test_ms_features, test_y), 
                            batch_size=args.batch_size, shuffle=False)
    
    test_balanced_acc, test_f1, conf_matrix = test_dual(best_model, device, test_loader)
    print(f"🎯 {model_name_with_embedding} Final Test Results - Balanced Acc: {test_balanced_acc:.2f}%, F1: {test_f1:.2f}%")
    
    # Aggregate training curves
    training_curves = aggregate_fold_training_curves(fold_results)
    
    # Clean up remaining memory
    del test_raw_features, test_ms_features, test_y
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


def train_dual_epoch(model, device, train_loader, optimizer, criterion, epoch, log_interval=10):
    """Train the dual-modal model for one epoch"""
    model.train()
    train_loss = 0
    train_total = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (raw_batch, ms_batch, label_batch) in enumerate(train_loader):
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
        
        # Store predictions and targets for metric computation
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(label_batch.cpu().numpy())
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(raw_batch)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = train_loss / train_total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    
    return avg_loss, balanced_acc, f1_macro


def validate_dual(model, device, val_loader, criterion):
    """Validate the dual-modal model"""
    model.eval()
    val_loss = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for raw_batch, ms_batch, label_batch in val_loader:
            raw_batch = raw_batch.to(device)
            ms_batch = ms_batch.to(device)
            label_batch = label_batch.to(device)
            
            outputs = model(raw_batch, ms_batch)
            val_loss += criterion(outputs, label_batch).item() * raw_batch.size(0)
            preds = outputs.argmax(dim=1)
            total += label_batch.size(0)
            
            # Store predictions and targets for metric computation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(label_batch.cpu().numpy())
    
    avg_loss = val_loss / total
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    
    return avg_loss, balanced_acc, f1_macro


def test_dual(model, device, test_loader, verbose=True):
    """Test the dual-modal model"""
    model.eval()
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for raw_batch, ms_batch, label_batch in test_loader:
            raw_batch = raw_batch.to(device)
            ms_batch = ms_batch.to(device)
            label_batch = label_batch.to(device)
            
            outputs = model(raw_batch, ms_batch)
            preds = outputs.argmax(dim=1)
            
            test_total += label_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(label_batch.cpu().numpy())
    
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    if verbose:
        print(f'Test set: Balanced Accuracy: {balanced_acc:.2f}%, F1 Macro: {f1_macro:.2f}%')
    
    return balanced_acc, f1_macro, conf_matrix