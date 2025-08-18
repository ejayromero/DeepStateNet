'''
Script to perform adaptive training on MicroStateNet models
Loads pre-trained subject-independent models and fine-tunes them on individual subjects
Uses unified functions from mmf for maximum code reusability
'''
import os
import sys
import argparse
import numpy as np
import seaborn as sns
import torch

sns.set_theme(style="darkgrid")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import my_functions as mf
from lib import my_models as mm
from lib import my_models_functions as mmf 


def main():
    """Main adaptive training function for MicroStateNet models"""
    # Argument parsing (same as msn.py)
    parser = argparse.ArgumentParser(description='PyTorch MicroStateNet Adaptive Training with K-fold CV')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate (will be adapted) (default: 1e-3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-subjects', type=int, default=50, metavar='N',
                        help='number of subjects to process (default: 50)')
    parser.add_argument('--type-of-subject', type=str, default='adaptive',
                        choices=['independent', 'dependent', 'adaptive'],
                        help='type of subject analysis (default: adaptive)')
    parser.add_argument('--n-folds', type=int, default=4, metavar='K',
                        help='number of CV folds (default: 4)')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help='early stopping patience (default: 15)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    # Microstate-specific arguments (same as msn.py)
    parser.add_argument('--model-name', type=str, default='msn',
                        choices=['msn', 'multiscale_msn', 'embedded_msn', 'attention_msn'],
                        help='model architecture to use (default: msn)')
    parser.add_argument('--n-clusters', type=int, default=5, metavar='N',
                        help='number of microstate clusters (default: 5)')
    parser.add_argument('--ms-file-specific', type=str, default='harmonize_overall',
                        choices=['indiv', 'harmonize_overall', 'overall'],
                        help='microstate file specific type (default: harmonize_overall)')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='D',
                        help='dropout rate (default: 0.25)')
    
    # Embedding support
    parser.add_argument('--use-embedding', action='store_true', default=True,
                        help='Use embedding-based input instead of one-hot encoding (default: True)')

    # Attention model specific arguments
    parser.add_argument('--embedding-dim', type=int, default=64, metavar='N',
                        help='embedding dimension for embedding models (default: 64)')
    parser.add_argument('--transformer-layers', type=int, default=4, metavar='N',
                        help='number of transformer layers for attention models (default: 4)')
    parser.add_argument('--transformer-heads', type=int, default=8, metavar='N',
                        help='number of transformer heads for attention models (default: 8)')
    
    args = parser.parse_args()
    
    print(f'==================== Start of script {os.path.basename(__file__)}! ====================')
    
    # Create embedding suffix for output naming
    embedding_suffix = "_embedded" if args.use_embedding else ""
    model_name_with_embedding = f"{args.model_name}{embedding_suffix}"
    
    print(f'{model_name_with_embedding} Adaptive Training with Pre-trained Subject-Independent Models')
    
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
    output_path = f'{output_folder}ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_{model_name_with_embedding}_c{args.n_clusters}_{args.n_folds}fold_results/'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    print(f"Using model: {args.model_name} with embedding: {args.use_embedding}")
    if args.model_name in mm.MODEL_INFO:
        print(f"Model description: {mm.MODEL_INFO[args.model_name]['description']}")

    # Vocabulary scanning for embedding models (same as msn_indep.py)
    if args.use_embedding:
        print("🚀 Embedding mode detected - performing vocabulary scan...")
        
        # Create data loader for scanning
        data_loader = mmf.MicrostateDataLoader()
        
        # Scan all subjects to find max microstate index
        max_microstate_index, actual_vocab_size = mmf.scan_all_subjects_for_max_microstate(
            args, data_path, data_loader
        )
        
        # Store the actual vocabulary size in args for model creation
        args.actual_vocab_size = actual_vocab_size
        
        # Memory cleanup after scanning
        del data_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        mf.print_memory_status("After vocabulary scan")
        
    else:
        print("One-hot mode - no vocabulary scan needed")
        args.actual_vocab_size = args.n_clusters  # Use original parameter

    # Training loop - using unified mmf function
    all_results = []
    output_file = os.path.join(output_path, f'{args.type_of_subject}_{model_name_with_embedding}_c{args.n_clusters}_{args.n_folds}fold_results.npy')
    
    # Check for existing results and resume if needed
    start_subject = 0
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
                'pretrained_performance': existing_results.get('pretrained_performances', [0] * n_existing)[i],
                'improvement': existing_results.get('improvements', [0] * n_existing)[i],
                'adaptive_lr': existing_results.get('adaptive_lrs', [args.lr] * n_existing)[i],
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
    
    for subject_id in range(start_subject, args.n_subjects):
        mf.print_memory_status(f"- SUBJECT {subject_id} START")
        
        # Use unified adaptive training function for microstate models
        result = mmf.train_adaptive_subject(subject_id, args, device, data_path, model_type='msn')
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
                'pretrained_performances': [r['pretrained_performance'] for r in all_results],
                'improvements': [r['improvement'] for r in all_results],
                'adaptive_lrs': [r['adaptive_lr'] for r in all_results],
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
    mmf.print_final_summary(all_results, f"{model_name_with_embedding} Adaptive", args.n_folds)
    
    # Print adaptive-specific summary
    print(f"\n🔄 MSN Adaptive Training Summary:")
    pretrained_accs = [r['pretrained_performance'] for r in all_results]
    improvements = [r['improvement'] for r in all_results]
    adaptive_lrs = [r['adaptive_lr'] for r in all_results]
    
    print(f"Pre-trained Performance: {np.mean(pretrained_accs):.2f}% ± {np.std(pretrained_accs):.2f}%")
    print(f"Average Improvement: {np.mean(improvements):.2f}% ± {np.std(improvements):.2f}%")
    print(f"Subjects with Improvement: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")
    print(f"Best Improvement: {np.max(improvements):.2f}%")
    print(f"Worst Change: {np.min(improvements):.2f}%")
    print(f"High LR used for {sum(1 for lr in adaptive_lrs if lr > 1e-3)}/{len(adaptive_lrs)} subjects")
    
    # Plot results - USING SHARED FUNCTION
    mmf.plot_all_results(all_results, output_path, args.type_of_subject, model_name_with_embedding, len(all_results))
    
    print('==================== End of script! ====================')


if __name__ == '__main__':
    main()
