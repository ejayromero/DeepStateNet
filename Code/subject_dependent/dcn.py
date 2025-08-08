'''
Script to train model on 50 subjects, training DeepConvNet on Microstates timeseries
With K-fold cross-validation, balanced accuracy, F1 scores, and single-subject loading
'''
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
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
        
        # if batch_idx % log_interval == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
        #           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
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
    
    # print(f'Validation set: Average loss: {avg_loss:.4f}, '
    #       f'Balanced Accuracy: {balanced_acc:.2f}%, F1 Macro: {f1_macro:.2f}%')
    
    return avg_loss, balanced_acc, f1_macro


def test(model, device, test_loader):
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
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Balanced Accuracy: {balanced_acc:.2f}%, F1 Macro: {f1_macro:.2f}%')
    
    return balanced_acc, f1_macro, conf_matrix


def train_subject(subject_id, args, device, data_path):
    """Train model for a single subject with K-fold CV and 10% test split"""
    print(f"\n▶ Training Subject {subject_id}")
    
    # Load single subject data for memory efficiency
    data, y = mf.load_all_one_subject_data(subject_id, data_path=data_path)
    x = torch.tensor(data, dtype=torch.float32).squeeze(1)  # (n_trials, n_channels, timepoints)
    y = torch.tensor(y, dtype=torch.long)
    
    print(f"Subject {subject_id} data shape: {x.shape}, labels shape: {y.shape}")
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
        base_model = Deep4Net(
            n_chans=x.shape[1],
            n_classes=len(torch.unique(y)),
            input_window_samples=x.shape[2],
            final_conv_length='auto'
        )
        
        model = EEGClassifier(
            base_model,
            criterion=nn.NLLLoss(),
            optimizer=torch.optim.Adam,
            optimizer__lr=args.lr,
            train_split=None,
            device=device
        )
        
        net = model.module.to(device)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        
        # Training for this fold
        fold_train_losses, fold_val_losses = [], []
        fold_train_balanced_accs, fold_val_balanced_accs = [], []
        fold_train_f1s, fold_val_f1s = [], []
        
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss, train_balanced_acc, train_f1 = train_epoch(
                net, device, train_loader, optimizer, criterion, epoch, 
                args.log_interval if epoch % 10 == 1 else 999)  # Reduce logging frequency
            
            # Validate
            val_loss, val_balanced_acc, val_f1 = validate(net, device, val_loader, criterion)
            
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
    mean_cv_bal_acc = np.mean(cv_balanced_accs)
    std_cv_bal_acc = np.std(cv_balanced_accs)
    mean_cv_f1 = np.mean(cv_f1_scores)
    std_cv_f1 = np.std(cv_f1_scores)
    
    print(f"\n📊 {args.n_folds}-Fold CV Results Summary:")
    print(f"CV Balanced Accuracy: {mean_cv_bal_acc:.2f}% ± {std_cv_bal_acc:.2f}%")
    print(f"CV F1 Macro: {mean_cv_f1:.2f}% ± {std_cv_f1:.2f}%")
    
    # Select best fold model (highest validation balanced accuracy)
    best_fold_idx = np.argmax(cv_balanced_accs)
    best_model = fold_results[best_fold_idx]['model']
    print(f"Best fold: {best_fold_idx + 1} (Val Bal Acc: {cv_balanced_accs[best_fold_idx]:.2f}%)")
    
    # Final test on held-out test set using best model
    test_loader = DataLoader(TensorDataset(x_test, y_test), 
                            batch_size=args.batch_size, shuffle=False)
    
    test_balanced_acc, test_f1, conf_matrix = test(best_model.module, device, test_loader)
    print(f"🎯 Final Test Results - Balanced Acc: {test_balanced_acc:.2f}%, F1: {test_f1:.2f}%")
    
    # Aggregate training curves across folds (mean and std)
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
        # Aggregated training curves
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
        'best_model': best_model
    }


def plot_results(all_results, output_path, type_of_subject, n_subjects):
    """Plot training results with CV"""
    all_test_balanced_accs = [result['test_balanced_accuracy'] for result in all_results]
    all_test_f1s = [result['test_f1_macro'] for result in all_results]
    all_cv_balanced_accs = [result['mean_cv_balanced_acc'] for result in all_results]
    all_cv_f1s = [result['mean_cv_f1'] for result in all_results]
    
    # Plot test metrics vs CV metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Test Balanced Accuracy
    axes[0, 0].plot(all_test_balanced_accs, marker='o', linestyle='-', color='blue', label='Test')
    axes[0, 0].plot(all_cv_balanced_accs, marker='s', linestyle='--', color='lightblue', label='CV Mean')
    axes[0, 0].set_title(f'Test vs CV Balanced Accuracy - {type_of_subject}')
    axes[0, 0].set_xlabel('Subject ID')
    axes[0, 0].set_ylabel('Balanced Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].set_xticks(range(n_subjects))
    axes[0, 0].set_xticklabels([f'S{i}' for i in range(n_subjects)], rotation=45)
    
    # Test F1 Macro
    axes[0, 1].plot(all_test_f1s, marker='o', linestyle='-', color='red', label='Test')
    axes[0, 1].plot(all_cv_f1s, marker='s', linestyle='--', color='lightcoral', label='CV Mean')
    axes[0, 1].set_title(f'Test vs CV F1 Macro - {type_of_subject}')
    axes[0, 1].set_xlabel('Subject ID')
    axes[0, 1].set_ylabel('F1 Macro (%)')
    axes[0, 1].legend()
    axes[0, 1].set_xticks(range(n_subjects))
    axes[0, 1].set_xticklabels([f'S{i}' for i in range(n_subjects)], rotation=45)
    
    # CV Validation Scores Distribution
    all_cv_individual_scores = []
    for result in all_results:
        all_cv_individual_scores.extend(result['cv_balanced_accuracies'])
    
    axes[1, 0].hist(all_cv_individual_scores, bins=20, alpha=0.7, color='green')
    axes[1, 0].set_title('Distribution of CV Fold Balanced Accuracies')
    axes[1, 0].set_xlabel('Balanced Accuracy (%)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Test vs CV correlation
    axes[1, 1].scatter(all_cv_balanced_accs, all_test_balanced_accs, alpha=0.6)
    axes[1, 1].plot([min(all_cv_balanced_accs), max(all_cv_balanced_accs)], 
                    [min(all_cv_balanced_accs), max(all_cv_balanced_accs)], 'r--', alpha=0.5)
    axes[1, 1].set_title('CV vs Test Balanced Accuracy Correlation')
    axes[1, 1].set_xlabel('CV Mean Balanced Accuracy (%)')
    axes[1, 1].set_ylabel('Test Balanced Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{type_of_subject}_DCN_CV_test_metrics.png'))
    plt.close()
    
    # Plot aggregated training curves (mean across subjects and folds)
    if len(all_results) > 0:
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
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'{type_of_subject} DeepConvNet - Training Curves (Mean ± STD across subjects and CV folds)', fontsize=16)
        
        # Balanced Accuracy
        axes[0, 0].plot(epochs, train_bal_accs_mean, 'b-', label='Train')
        axes[0, 0].fill_between(epochs, train_bal_accs_mean - train_bal_accs_std, 
                               train_bal_accs_mean + train_bal_accs_std, alpha=0.3, color='blue')
        axes[0, 0].set_title('Training Balanced Accuracy')
        axes[0, 0].set_ylabel('Balanced Accuracy (%)')
        
        axes[0, 1].plot(epochs, val_bal_accs_mean, 'g-', label='Validation')
        axes[0, 1].fill_between(epochs, val_bal_accs_mean - val_bal_accs_std, 
                               val_bal_accs_mean + val_bal_accs_std, alpha=0.3, color='green')
        axes[0, 1].set_title('Validation Balanced Accuracy (CV)')
        axes[0, 1].set_ylabel('Balanced Accuracy (%)')
        
        # F1 Macro
        axes[1, 0].plot(epochs, train_f1s_mean, 'purple', label='Train F1')
        axes[1, 0].fill_between(epochs, train_f1s_mean - train_f1s_std, 
                               train_f1s_mean + train_f1s_std, alpha=0.3, color='purple')
        axes[1, 0].set_title('Training F1 Macro')
        axes[1, 0].set_ylabel('F1 Macro (%)')
        
        axes[1, 1].plot(epochs, val_f1s_mean, 'orange', label='Val F1')
        axes[1, 1].fill_between(epochs, val_f1s_mean - val_f1s_std, 
                               val_f1s_mean + val_f1s_std, alpha=0.3, color='orange')
        axes[1, 1].set_title('Validation F1 Macro (CV)')
        axes[1, 1].set_ylabel('F1 Macro (%)')
        
        # Loss
        axes[2, 0].plot(epochs, train_losses_mean, 'red', label='Train Loss')
        axes[2, 0].fill_between(epochs, train_losses_mean - train_losses_std, 
                               train_losses_mean + train_losses_std, alpha=0.3, color='red')
        axes[2, 0].set_title('Training Loss')
        axes[2, 0].set_ylabel('Loss')
        axes[2, 0].set_xlabel('Epoch')
        
        axes[2, 1].plot(epochs, val_losses_mean, 'brown', label='Val Loss')
        axes[2, 1].fill_between(epochs, val_losses_mean - val_losses_std, 
                               val_losses_mean + val_losses_std, alpha=0.3, color='brown')
        axes[2, 1].set_title('Validation Loss (CV)')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{type_of_subject}_DCN_CV_training_curves.png'))
        plt.subplots_adjust(top=0.93)
        plt.close()
    
    # Plot average confusion matrix across all subjects
    all_conf_matrices = [result['confusion_matrix'] for result in all_results]
    avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_conf_matrix, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(avg_conf_matrix.shape[1])],
                yticklabels=[f'Class {i}' for i in range(avg_conf_matrix.shape[0])])
    plt.title(f'Average Confusion Matrix - {type_of_subject} DeepConvNet (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{type_of_subject}_DCN_avg_confusion_matrix.png'))
    plt.close()


def main():
    """Main training function"""
    # Argument parsing
    parser = argparse.ArgumentParser(description='PyTorch DeepConvNet EEG Classification with K-fold CV')
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
    parser.add_argument('--n-folds', type=int, default=4, metavar='K',
                        help='number of CV folds (default: 4)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
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
    output_path = f'{output_folder}ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_dcn_cv_{args.n_folds}fold_results/'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Training loop
    all_results = []
    output_file = os.path.join(output_path, f'{args.type_of_subject}_dcn_cv_{args.n_folds}fold_results.npy')
    
    # Check for existing results and resume if needed
    if os.path.exists(output_file):
        print(f"Found existing results file: {output_file}")
        existing_results = np.load(output_file, allow_pickle=True).item()
        n_existing = len(existing_results.get('test_balanced_accuracies', []))
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
                'test_balanced_accuracy': existing_results['test_balanced_accuracies'][i],
                'test_f1_macro': existing_results['test_f1_macros'][i],
                'confusion_matrix': existing_results['confusion_matrices'][i],
                'best_fold_idx': existing_results['best_fold_indices'][i],
                'train_losses_mean': existing_results['train_curves_mean']['train_losses_mean'][i],
                'train_balanced_accuracies_mean': existing_results['train_curves_mean']['train_balanced_accuracies_mean'][i],
                'train_f1_macros_mean': existing_results['train_curves_mean']['train_f1_macros_mean'][i],
                'val_losses_mean': existing_results['train_curves_mean']['val_losses_mean'][i],
                'val_balanced_accuracies_mean': existing_results['train_curves_mean']['val_balanced_accuracies_mean'][i],
                'val_f1_macros_mean': existing_results['train_curves_mean']['val_f1_macros_mean'][i],
                'best_model': existing_results['best_models'][i]
            }
            all_results.append(result)
        start_subject = n_existing
    else:
        start_subject = 0
    
    for subject_id in range(start_subject, args.n_subjects):
        mf.print_memory_status(f"- SUBJECT {subject_id} START")
        
        result = train_subject(subject_id, args, device, data_path)
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
            print(f"Results saved to {output_file}")
        
        print(f"✅ Subject {subject_id} processed successfully.\n")
        mf.print_memory_status(f"- SUBJECT {subject_id} END")
    
    # Final summary
    test_bal_accs = [r['test_balanced_accuracy'] for r in all_results]
    test_f1s = [r['test_f1_macro'] for r in all_results]
    cv_bal_accs = [r['mean_cv_balanced_acc'] for r in all_results]
    cv_f1s = [r['mean_cv_f1'] for r in all_results]
    
    print(f"\n🎯 Overall Results Summary:")
    print(f"Configuration: {args.n_folds}-fold CV with 10% test split")
    print(f"Test Results:")
    print(f"  Mean Test Balanced Accuracy: {np.mean(test_bal_accs):.2f}% ± {np.std(test_bal_accs):.2f}%")
    print(f"  Mean Test F1 Macro: {np.mean(test_f1s):.2f}% ± {np.std(test_f1s):.2f}%")
    print(f"  Best Subject Test Bal Acc: {np.max(test_bal_accs):.2f}%")
    print(f"  Worst Subject Test Bal Acc: {np.min(test_bal_accs):.2f}%")
    print(f"\n{args.n_folds}-Fold Cross-Validation Results:")
    print(f"  Mean CV Balanced Accuracy: {np.mean(cv_bal_accs):.2f}% ± {np.std(cv_bal_accs):.2f}%")
    print(f"  Mean CV F1 Macro: {np.mean(cv_f1s):.2f}% ± {np.std(cv_f1s):.2f}%")
    print(f"  Best Subject CV Bal Acc: {np.max(cv_bal_accs):.2f}%")
    print(f"  Worst Subject CV Bal Acc: {np.min(cv_bal_accs):.2f}%")
    
    # Correlation between CV and Test performance
    cv_test_corr = np.corrcoef(cv_bal_accs, test_bal_accs)[0, 1]
    print(f"\nCV-Test Correlation: {cv_test_corr:.3f}")
    
    # Plot results
    plot_results(all_results, output_path, args.type_of_subject, len(all_results))
    
    print('==================== End of script! ====================')


if __name__ == '__main__':
    main()