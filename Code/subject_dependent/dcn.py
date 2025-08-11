'''
Script to train model on 50 subjects, training DeepConvNet on Microstates timeseries
With K-fold cross-validation, balanced accuracy, F1 scores, and single-subject loading
'''
import os
import sys
import argparse
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")

from braindecode.models import Deep4Net
from braindecode.classifier import EEGClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import my_functions as mf
from lib import my_models_functions as mmf 


def train_subject(subject_id, args, device, data_path):
    """Train model for a single subject with K-fold CV and 10% test split"""
    print(f"\n▶ Training Subject {subject_id}")
    
    # Load single subject data for memory efficiency
    data, y = mf.load_data(subject_id, data_path=data_path)
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
            # Train - USING SHARED FUNCTION
            train_loss, train_balanced_acc, train_f1 = mmf.train_epoch(
                net, device, train_loader, optimizer, criterion, epoch, 
                args.log_interval if epoch % 10 == 1 else 999)  # Reduce logging frequency
            
            # Validate - USING SHARED FUNCTION
            val_loss, val_balanced_acc, val_f1 = mmf.validate(net, device, val_loader, criterion)
            
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
    
    # Final test on held-out test set using best model - USING SHARED FUNCTION
    test_loader = DataLoader(TensorDataset(x_test, y_test), 
                            batch_size=args.batch_size, shuffle=False)
    
    test_balanced_acc, test_f1, conf_matrix = mmf.test(best_model.module, device, test_loader)
    print(f"🎯 Final Test Results - Balanced Acc: {test_balanced_acc:.2f}%, F1: {test_f1:.2f}%")
    
    # Aggregate training curves across folds - USING SHARED FUNCTION
    training_curves = mmf.aggregate_fold_training_curves(fold_results)
    
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
    output_path = f'{output_folder}ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_dcn_{args.n_folds}fold_results/'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Training loop
    all_results = []
    output_file = os.path.join(output_path, f'{args.type_of_subject}_dcn_{args.n_folds}fold_results.npy')
    
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
    
    # Final summary - USING SHARED FUNCTION
    mmf.print_final_summary(all_results, "DeepConvNet", args.n_folds)
    
    # Plot results - USING SHARED FUNCTION
    mmf.plot_all_results(all_results, output_path, args.type_of_subject, "DCN", len(all_results))
    
    print('==================== End of script! ====================')


if __name__ == '__main__':
    main()