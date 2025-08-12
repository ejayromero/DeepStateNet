'''
Script to train DeepStateNet (DCN + MicroStateNet fusion) on 50 subjects using LOSO methodology
Subject-independent paradigm with 4-fold cross-validation on all 49 remaining subjects
Uses unified functions from mmf for maximum code reusability and memory efficiency
Combines pretrained DCN and MicroStateNet models through feature fusion
'''
import os
import sys
import argparse
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import my_functions as mf
from lib import my_models as mm
from lib import my_models_functions as mmf 


def main():
    """Main LOSO training function using unified mmf functions for fusion models"""
    # Argument parsing
    parser = argparse.ArgumentParser(description='PyTorch DeepStateNet (DCN + MicroStateNet) EEG Classification with LOSO and 4-fold CV')
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
    parser.add_argument('--type-of-subject', type=str, default='independent',
                        choices=['independent', 'dependent', 'adaptive'],
                        help='type of subject analysis (default: independent)')
    parser.add_argument('--n-folds', type=int, default=4, metavar='K',
                        help='number of CV folds (default: 4)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    # Microstate-specific arguments
    parser.add_argument('--model-name', type=str, default='msn',
                        choices=['msn', 'multiscale_msn', 'embedded_msn', 'attention_msn'],
                        help='microstate model architecture to use (default: msn)')
    parser.add_argument('--n-clusters', type=int, default=5, metavar='N',
                        help='number of microstate clusters (default: 5)')
    parser.add_argument('--ms-file-specific', type=str, default='harmonize_overall',
                        choices=['indiv', 'harmonize_overall'],
                        help='microstate file specific type (default: harmonize_overall)')
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
    
    # Create embedding suffix for output naming
    embedding_suffix = "_embedded" if args.use_embedding else ""
    model_name_with_embedding = f"dsn_{args.model_name}{embedding_suffix}"
    
    print(f'{model_name_with_embedding} LOSO with {args.n_folds}-fold Cross-Validation on all 49 remaining subjects')
    print(f'Using DeepStateNet fusion: DCN + {args.model_name} (embedding: {args.use_embedding})')
    
    # Device setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    if use_cuda:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    mf.print_memory_status("- INITIAL STARTUP")
    
    # Set seed for reproducibility
    mf.set_seed(args.seed)
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, 'Data') + os.sep
    output_folder = os.path.join(project_root, 'Output') + os.sep
    output_path = f'{output_folder}ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_{model_name_with_embedding}_c{args.n_clusters}_{args.n_folds}fold_results/'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    print(f"Using microstate model: {args.model_name} with embedding: {args.use_embedding}")
    if args.model_name in mm.MODEL_INFO:
        print(f"Model description: {mm.MODEL_INFO[args.model_name]['description']}")

    # Load pretrained models for fusion - using dynamic paths from mmf
    print(f"\nLoading pretrained models for fusion...")
    try:
        pretrained_models = mmf.load_pretrained_models(args, project_root)
        print(f"✅ Successfully loaded pretrained DCN and {args.model_name} models")
    except FileNotFoundError as e:
        print(f"❌ Error loading pretrained models: {e}")
        print(f"Make sure you've run both dcn_indep.py and msn_indep.py first to generate the required pretrained models.")
        print(f"Required files:")
        print(f"  - DCN: Output/ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_dcn_{args.n_folds}fold_results/")
        print(f"  - MicroStateNet: Output/ica_rest_all/{args.type_of_subject}/{args.type_of_subject}_{args.model_name}{embedding_suffix}_c{args.n_clusters}_{args.n_folds}fold_results/")
        sys.exit(1)

    # Verify we have enough models for all subjects
    n_dcn_models = len(pretrained_models['dcn'])
    n_ms_models = len(pretrained_models['ms'])
    n_subjects_to_process = min(args.n_subjects, n_dcn_models, n_ms_models)
    
    if n_subjects_to_process < args.n_subjects:
        print(f"Warning: Requested {args.n_subjects} subjects but only have {n_dcn_models} DCN and {n_ms_models} MicroStateNet models.")
        print(f"Processing {n_subjects_to_process} subjects instead.")

    # LOSO Training loop - using unified mmf function for fusion
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
                'test_subject_id': existing_results.get('test_subject_ids', list(range(n_existing)))[i],
                'remaining_subject_ids': existing_results.get('remaining_subject_ids', [list(range(args.n_subjects))] * n_existing)[i],
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
                'best_model': existing_results['best_models'][i] if 'best_models' in existing_results else None
            }
            all_results.append(result)
        start_subject = n_existing
    
    # LOSO loop: each subject becomes test subject once
    for test_subject_id in range(start_subject, n_subjects_to_process):
        mf.print_memory_status(f"- LOSO ITERATION {test_subject_id} START")
        
        # Use the unified LOSO training function for fusion models
        # This will use the pretrained models to extract features and train the fusion classifier
        result = mmf.train_loso(
            test_subject_id=test_subject_id,
            args=args,
            device=device,
            data_path=data_path,
            model_type='fusion'  # Specify fusion model type
        )
        all_results.append(result)
        
        # Save intermediate results using unified save function
        if args.save_model:
            mmf.save_results(all_results, output_file)
            print(f"Results saved to {output_file}")
        
        print(f"✅ LOSO iteration {test_subject_id} processed successfully.\n")
        mf.print_memory_status(f"- LOSO ITERATION {test_subject_id} END")
    
    # Final summary - using unified mmf function
    mmf.print_final_summary(all_results, f"{model_name_with_embedding} LOSO", args.n_folds)
    
    # Plot results - using unified mmf function  
    mmf.plot_all_results(all_results, output_path, args.type_of_subject, f"{model_name_with_embedding}_LOSO", len(all_results))
    
    print('==================== End of script! ====================')


if __name__ == '__main__':
    main()