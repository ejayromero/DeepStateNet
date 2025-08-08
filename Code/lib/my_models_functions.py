'''
Reusable training, validation, testing, and plotting functions for neural network models
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


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
            test_loss += F.nll_loss(output, target, reduction='sum').item()
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
    axes[0, 0].plot(all_cv_balanced_accs, marker='s', linestyle='--', color=colors[3], label='CV Mean', linewidth=2)
    axes[0, 0].set_title(f'Test vs CV Balanced Accuracy')
    axes[0, 0].set_xlabel('Subject ID')
    axes[0, 0].set_ylabel('Balanced Accuracy (%)')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend()
    axes[0, 0].set_xticks(range(0, n_subjects, max(1, n_subjects//10)))  # Show every 5th or 10th subject
    axes[0, 0].set_xticklabels([f'S{i}' for i in range(0, n_subjects, max(1, n_subjects//10))], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test F1 Macro
    axes[0, 1].plot(all_test_f1s, marker='o', linestyle='-', color=colors[1], label='Test', linewidth=2)
    axes[0, 1].plot(all_cv_f1s, marker='s', linestyle='--', color=colors[4], label='CV Mean', linewidth=2)
    axes[0, 1].set_title(f'Test vs CV F1 Macro')
    axes[0, 1].set_xlabel('Subject ID')
    axes[0, 1].set_ylabel('F1 Macro (%)')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend()
    axes[0, 1].set_xticks(range(0, n_subjects, max(1, n_subjects//10)))
    axes[0, 1].set_xticklabels([f'S{i}' for i in range(0, n_subjects, max(1, n_subjects//10))], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # CV Validation Scores Distribution
    all_cv_individual_scores = []
    for result in all_results:
        all_cv_individual_scores.extend(result['cv_balanced_accuracies'])
    
    axes[1, 0].hist(all_cv_individual_scores, bins=20, alpha=0.7, color=colors[2])
    axes[1, 0].set_title('Distribution of CV Fold Balanced Accuracies')
    axes[1, 0].set_xlabel('Balanced Accuracy (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test vs CV correlation
    axes[1, 1].scatter(all_cv_balanced_accs, all_test_balanced_accs, alpha=0.6, color=colors[5], s=60)
    axes[1, 1].plot([0, 100], [0, 100], 'r--', alpha=0.5)
    axes[1, 1].set_title('CV vs Test Balanced Accuracy Correlation')
    axes[1, 1].set_xlabel('CV Mean Balanced Accuracy (%)')
    axes[1, 1].set_ylabel('Test Balanced Accuracy (%)')
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.3)
    
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
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, val_bal_accs_mean, color=colors[1], linewidth=2, label='Validation')
    axes[0, 1].fill_between(epochs, val_bal_accs_mean - val_bal_accs_std, 
                           val_bal_accs_mean + val_bal_accs_std, alpha=0.3, color=colors[1])
    axes[0, 1].set_title('Validation Balanced Accuracy (CV)')
    axes[0, 1].set_ylabel('Balanced Accuracy (%)')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Macro
    axes[1, 0].plot(epochs, train_f1s_mean, color=colors[2], linewidth=2, label='Train F1')
    axes[1, 0].fill_between(epochs, train_f1s_mean - train_f1s_std, 
                           train_f1s_mean + train_f1s_std, alpha=0.3, color=colors[2])
    axes[1, 0].set_title('Training F1 Macro')
    axes[1, 0].set_ylabel('F1 Macro (%)')
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, val_f1s_mean, color=colors[3], linewidth=2, label='Val F1')
    axes[1, 1].fill_between(epochs, val_f1s_mean - val_f1s_std, 
                           val_f1s_mean + val_f1s_std, alpha=0.3, color=colors[3])
    axes[1, 1].set_title('Validation F1 Macro (CV)')
    axes[1, 1].set_ylabel('F1 Macro (%)')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Loss - with shared y-axis for better comparison
    loss_min = min(np.min(train_losses_mean), np.min(val_losses_mean))
    loss_max = max(np.max(train_losses_mean), np.max(val_losses_mean))
    
    axes[2, 0].plot(epochs, train_losses_mean, color=colors[4], linewidth=2, label='Train Loss')
    axes[2, 0].fill_between(epochs, train_losses_mean - train_losses_std, 
                           train_losses_mean + train_losses_std, alpha=0.3, color=colors[4])
    axes[2, 0].set_title('Training Loss')
    axes[2, 0].set_ylabel('Loss')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylim(loss_min * 0.9, loss_max * 1.1)
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(epochs, val_losses_mean, color=colors[5], linewidth=2, label='Val Loss')
    axes[2, 1].fill_between(epochs, val_losses_mean - val_losses_std, 
                           val_losses_mean + val_losses_std, alpha=0.3, color=colors[5])
    axes[2, 1].set_title('Validation Loss (CV)')
    axes[2, 1].set_ylabel('Loss')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylim(loss_min * 0.9, loss_max * 1.1)
    axes[2, 1].grid(True, alpha=0.3)
    
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
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_conf_matrix, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(avg_conf_matrix.shape[1])],
                yticklabels=[f'Class {i}' for i in range(avg_conf_matrix.shape[0])])
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