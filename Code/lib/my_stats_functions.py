"""
Clean Statistical Analysis Functions for EEG Model Performance
Single-line execution for different analysis types
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, levene
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower
from itertools import combinations
import warnings


warnings.filterwarnings('ignore')

# ============================== Loading Data=================================

def make_all_model_variant(model_name, variants):
    """Create all model variants from base model and variant list"""
    all_variants = []
    all_variants.append(model_name)
    variant_1 = f'{model_name}_{variants[1]}'
    variant_2 = f'{variants[0]}_{model_name}'
    variant_3 = f'{variants[0]}_{model_name}_{variants[1]}'
    all_variants.append(variant_1)
    all_variants.append(variant_2)
    all_variants.append(variant_3)
    return all_variants

def generate_all_model_types():
    """Generate all possible model types based on your configuration"""
    subject_types = ['dependent', 'independent', 'adaptive']
    main_models_type = ['dcn', 'msn', 'dsn']
    variants_msn = ['multiscale', 'embedded']  # variants[0] = 'multiscale', variants[1] = 'embedded'
    cluster_sizes = [5, 12]
    k_folds = 4
    
    # Generate models by subject type
    models_by_subject_type = {}
    
    for subject_type in subject_types:
        models_for_this_type = []
        
        for model in main_models_type:
            if model == 'dcn':
                # DCN exists for all subject types
                models_for_this_type.append(model)
            elif model == 'msn':
                if subject_type == 'dependent':
                    # Dependent has all MSN variants: ['msn', 'msn_embedded', 'multiscale_msn', 'multiscale_msn_embedded']
                    msn_names = make_all_model_variant(model, variants_msn)
                    models_for_this_type.extend(msn_names)
                else:
                    # Independent and adaptive only have embedded variants: ['msn_embedded', 'multiscale_msn_embedded']
                    models_for_this_type.extend(['msn_embedded', 'multiscale_msn_embedded'])
            elif model == 'dsn':
                if subject_type == 'dependent':
                    # Dependent has all DSN variants based on all MSN variants
                    msn_names = make_all_model_variant('msn', variants_msn)
                    dsn_names = [f'{model}_{name}' for name in msn_names]
                    models_for_this_type.extend(dsn_names)
                else:
                    # Independent and adaptive only have embedded DSN variants
                    embedded_msn_models = ['msn_embedded', 'multiscale_msn_embedded']
                    dsn_names = [f'{model}_{name}' for name in embedded_msn_models]
                    models_for_this_type.extend(dsn_names)
        
        models_by_subject_type[subject_type] = models_for_this_type
    
    return {
        'subject_types': subject_types,
        'models_by_subject_type': models_by_subject_type,
        'cluster_sizes': cluster_sizes,
        'k_folds': k_folds
    }

def extract_single_result_file(file_path, model_info):
    """Extract results from a single .npy file"""
    try:
        results = np.load(file_path, allow_pickle=True).item()
        
        # Determine if this is LOSO (independent/adaptive) or subject-dependent
        is_loso = 'test_subject_ids' in results or 'test_subject_id' in results
        
        if is_loso:
            # LOSO results structure
            if 'test_subject_ids' in results:
                # Multiple subjects LOSO
                n_subjects = len(results['test_subject_ids'])
                test_bal_accs = results['test_balanced_accuracies']
                test_f1s = results['test_f1_macros']
                cv_bal_accs = results['mean_cv_balanced_accs']
                cv_f1s = results['mean_cv_f1s']
            else:
                # Single subject LOSO (shouldn't happen but handle it)
                n_subjects = 1
                test_bal_accs = [results['test_balanced_accuracy']]
                test_f1s = [results['test_f1_macro']]
                cv_bal_accs = [results['mean_cv_balanced_acc']]
                cv_f1s = [results['mean_cv_f1']]
        else:
            # Subject-dependent results structure
            n_subjects = len(results['test_balanced_accuracies'])
            test_bal_accs = results['test_balanced_accuracies']
            test_f1s = results['test_f1_macros']
            cv_bal_accs = results['mean_cv_balanced_accs']
            cv_f1s = results['mean_cv_f1s']
        
        # Calculate summary statistics
        extracted_data = {
            # File metadata
            'file_path': str(file_path),
            'file_exists': True,
            'n_subjects': n_subjects,
            'is_loso': is_loso,
            
            # Model metadata
            'subject_type': model_info['subject_type'],
            'model_name': model_info['model_name'],
            'cluster_size': model_info.get('cluster_size', 'N/A'),
            'k_folds': model_info['k_folds'],
            
            # Test performance metrics
            'test_bal_acc_mean': np.mean(test_bal_accs),
            'test_bal_acc_std': np.std(test_bal_accs),
            'test_bal_acc_min': np.min(test_bal_accs),
            'test_bal_acc_max': np.max(test_bal_accs),
            'test_bal_acc_all': test_bal_accs,
            
            'test_f1_mean': np.mean(test_f1s),
            'test_f1_std': np.std(test_f1s),
            'test_f1_min': np.min(test_f1s),
            'test_f1_max': np.max(test_f1s),
            'test_f1_all': test_f1s,
            
            # CV performance metrics
            'cv_bal_acc_mean': np.mean(cv_bal_accs),
            'cv_bal_acc_std': np.std(cv_bal_accs),
            'cv_bal_acc_min': np.min(cv_bal_accs),
            'cv_bal_acc_max': np.max(cv_bal_accs),
            'cv_bal_acc_all': cv_bal_accs,
            
            'cv_f1_mean': np.mean(cv_f1s),
            'cv_f1_std': np.std(cv_f1s),
            'cv_f1_min': np.min(cv_f1s),
            'cv_f1_max': np.max(cv_f1s),
            'cv_f1_all': cv_f1s,
        }
        
        # Add early stopping info if available
        if is_loso and 'avg_best_epoch' in results:
            extracted_data['avg_best_epoch'] = results['avg_best_epoch']
            extracted_data['total_epochs_saved'] = results['total_epochs_saved']
        
        return extracted_data
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'file_exists': False,
            'error': str(e),
            'subject_type': model_info['subject_type'],
            'model_name': model_info['model_name'],
            'cluster_size': model_info.get('cluster_size', 'N/A'),
            'k_folds': model_info['k_folds'],
        }

def build_file_path(output_path, subject_type, model_name, cluster_size, k_folds):
    """Build the file path based on naming convention"""
    
    # Handle DCN (no cluster size)
    if model_name == 'dcn':
        folder_name = f"{subject_type}_{model_name}_{k_folds}fold_results"
        file_name = f"{subject_type}_{model_name}_{k_folds}fold_results.npy"
    else:
        # MSN and DSN models (with cluster size)
        folder_name = f"{subject_type}_{model_name}_c{cluster_size}_{k_folds}fold_results"
        file_name = f"{subject_type}_{model_name}_c{cluster_size}_{k_folds}fold_results.npy"
    
    full_path = os.path.join(output_path, subject_type, folder_name, file_name)
    return full_path

def extract_all_results(output_path='../Output/ica_rest_all/', verbose=True):
    """Extract all model results from the output directory"""
    
    config = generate_all_model_types()
    all_extracted_results = []
    
    print(f"🔍 Starting extraction from: {output_path}")
    print(f"📊 Configuration:")
    print(f"   Subject types: {config['subject_types']}")
    print(f"   Cluster sizes: {config['cluster_sizes']}")
    print(f"   K-folds: {config['k_folds']}")
    
    # Show model breakdown by subject type
    for subject_type, models in config['models_by_subject_type'].items():
        print(f"   {subject_type}: {len(models)} models")
        if verbose:
            print(f"      {models}")
    
    total_files = 0
    found_files = 0
    
    for subject_type in config['subject_types']:
        print(f"\n📁 Processing subject type: {subject_type}")
        models_for_this_type = config['models_by_subject_type'][subject_type]
        
        for model_name in models_for_this_type:
            
            if model_name == 'dcn':
                # DCN doesn't use cluster sizes
                cluster_sizes_to_use = [None]
            else:
                cluster_sizes_to_use = config['cluster_sizes']
            
            for cluster_size in cluster_sizes_to_use:
                total_files += 1
                
                # Build file path
                file_path = build_file_path(
                    output_path, subject_type, model_name, 
                    cluster_size, config['k_folds']
                )
                
                # Create model info
                model_info = {
                    'subject_type': subject_type,
                    'model_name': model_name,
                    'cluster_size': cluster_size,
                    'k_folds': config['k_folds']
                }
                
                # Extract results (only if file exists)
                if os.path.exists(file_path):
                    result = extract_single_result_file(file_path, model_info)
                    if result.get('file_exists', False):
                        found_files += 1
                        if verbose:
                            print(f"   ✅ {model_name}" + 
                                  (f"_c{cluster_size}" if cluster_size else "") + 
                                  f": Test Acc {result['test_bal_acc_mean']:.2f}±{result['test_bal_acc_std']:.2f}%")
                else:
                    # File doesn't exist - create placeholder result
                    result = {
                        'file_path': str(file_path),
                        'file_exists': False,
                        'error': 'File not found (may still be running)',
                        'subject_type': model_info['subject_type'],
                        'model_name': model_info['model_name'],
                        'cluster_size': model_info['cluster_size'],
                        'k_folds': model_info['k_folds'],
                    }
                    if verbose:
                        print(f"   ⏳ {model_name}" + 
                              (f"_c{cluster_size}" if cluster_size else "") + 
                              f": File not found (still running?)")
                
                all_extracted_results.append(result)
    
    print(f"\n📈 Extraction Summary:")
    print(f"   Total files expected: {total_files}")
    print(f"   Files found: {found_files}")
    print(f"   Files missing: {total_files - found_files}")
    print(f"   Success rate: {found_files/total_files*100:.1f}%")
    
    if found_files < total_files:
        print(f"   ℹ️  Note: Some files may still be running (independent/adaptive experiments)")
    
    return all_extracted_results

def create_results_dataframe(extracted_results):
    """Convert extracted results to a pandas DataFrame for easy analysis"""
    
    # Filter only successful extractions
    successful_results = [r for r in extracted_results if r.get('file_exists', False)]
    
    if not successful_results:
        print("❌ No successful results found!")
        return pd.DataFrame()
    
    # Create DataFrame with main metrics
    df_data = []
    
    for result in successful_results:
        row = {
            'subject_type': result['subject_type'],
            'model_name': result['model_name'],
            'cluster_size': result['cluster_size'],
            'k_folds': result['k_folds'],
            'n_subjects': result['n_subjects'],
            'is_loso': result['is_loso'],
            
            # Test metrics
            'test_bal_acc_mean': result['test_bal_acc_mean'],
            'test_bal_acc_std': result['test_bal_acc_std'],
            'test_f1_mean': result['test_f1_mean'],
            'test_f1_std': result['test_f1_std'],
            
            # CV metrics
            'cv_bal_acc_mean': result['cv_bal_acc_mean'],
            'cv_bal_acc_std': result['cv_bal_acc_std'],
            'cv_f1_mean': result['cv_f1_mean'],
            'cv_f1_std': result['cv_f1_std'],
            
            # Individual subject scores (for plotting)
            'test_bal_acc_all': result['test_bal_acc_all'],
            'test_f1_all': result['test_f1_all'],
            'cv_bal_acc_all': result['cv_bal_acc_all'],
            'cv_f1_all': result['cv_f1_all'],
        }
        
        # Add early stopping info if available
        if 'avg_best_epoch' in result:
            row['avg_best_epoch'] = result['avg_best_epoch']
            row['total_epochs_saved'] = result['total_epochs_saved']
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df['cluster_size'] = df['cluster_size'].astype('Int64')
    # Create a more readable model identifier
    df['model_id'] = df.apply(lambda row: 
        f"{row['model_name']}" + 
        (f"_c{row['cluster_size']}" if pd.notna(row['cluster_size']) else ""), 
        axis=1)
    
    # Sort by subject type and test performance
    df = df.sort_values(['subject_type', 'test_bal_acc_mean'], ascending=[True, False])
    
    print(f"📊 Created DataFrame with {len(df)} successful results")
    
    return df

def print_results_summary(df):
    """Print a comprehensive summary of all results"""
    
    if df.empty:
        print("❌ No results to summarize!")
        return
    
    print("🏆 COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 60)
    
    for subject_type in df['subject_type'].unique():
        subset = df[df['subject_type'] == subject_type]
        print(f"\n📊 {subject_type.upper()} RESULTS ({len(subset)} models)")
        print("-" * 40)
        
        # Sort by test balanced accuracy
        subset_sorted = subset.sort_values('test_bal_acc_mean', ascending=False)
        
        print(f"{'Model':<30} {'Test Bal Acc':<15} {'Test F1':<15} {'CV Bal Acc':<15} {'N Subj':<8}")
        print("-" * 85)
        
        for _, row in subset_sorted.iterrows():
            print(f"{row['model_id']:<30} "
                  f"{row['test_bal_acc_mean']:.2f}±{row['test_bal_acc_std']:.2f}%"
                  f"{'':<3} "
                  f"{row['test_f1_mean']:.2f}±{row['test_f1_std']:.2f}%"
                  f"{'':<3} "
                  f"{row['cv_bal_acc_mean']:.2f}±{row['cv_bal_acc_std']:.2f}%"
                  f"{'':<3} "
                  f"{row['n_subjects']:<8}")
    
    # Overall best performers
    print(f"\n🥇 OVERALL BEST PERFORMERS")
    print("-" * 40)
    
    best_overall = df.loc[df['test_bal_acc_mean'].idxmax()]
    print(f"Best Test Accuracy: {best_overall['model_id']} ({best_overall['subject_type']}) - "
          f"{best_overall['test_bal_acc_mean']:.2f}±{best_overall['test_bal_acc_std']:.2f}%")
    
    best_f1 = df.loc[df['test_f1_mean'].idxmax()]
    print(f"Best Test F1:       {best_f1['model_id']} ({best_f1['subject_type']}) - "
          f"{best_f1['test_f1_mean']:.2f}±{best_f1['test_f1_std']:.2f}%")
    
    # Model type comparison
    print(f"\n📈 MODEL TYPE COMPARISON (Available Results Only)")
    print("-" * 40)
    
    # Group by base model type
    df['base_model'] = df['model_name'].apply(lambda x: 
        'DCN' if x == 'dcn' else 
        'DSN' if x.startswith('dsn') else 'MSN')
    
    model_summary = df.groupby(['subject_type', 'base_model']).agg({
        'test_bal_acc_mean': ['count', 'mean', 'std', 'max'],
        'test_f1_mean': ['mean', 'std', 'max']
    }).round(2)
    
    print(model_summary)
    
    # Show embedding analysis
    print(f"\n🔗 EMBEDDING vs NON-EMBEDDING ANALYSIS")
    print("-" * 40)
    
    df['is_embedded'] = df['model_name'].str.contains('embedded')
    
    if df['is_embedded'].any():
        embedding_summary = df.groupby(['subject_type', 'is_embedded']).agg({
            'test_bal_acc_mean': ['count', 'mean', 'std'],
        }).round(2)
        print(embedding_summary)
    else:
        print("No embedded models found in current results.")

# Example usage function
# Convenience functions for different user setups
def extract_for_user(user_output_path, custom_save_path=None):
    """
    Convenience function for different users with their own paths
    
    Args:
        user_output_path: Path where user's model results are stored
        custom_save_path: Optional custom save location. If None, uses user_output_path/results_all/
    
    Returns:
        df, extracted_results: DataFrame and raw results
    """
    return run_complete_extraction(user_output_path, custom_save_path)

def get_user_paths(base_path, user_name=None):
    """
    Generate standardized paths for different users
    
    Args:
        base_path: Base directory path
        user_name: Optional user identifier
    
    Returns:
        dict: Dictionary with output_path and save_path
    """
    if user_name:
        user_output_path = os.path.join(base_path, f'user_{user_name}', 'ica_rest_all')
        user_save_path = os.path.join(user_output_path, 'results_all')
    else:
        user_output_path = os.path.join(base_path, 'ica_rest_all')
        user_save_path = os.path.join(user_output_path, 'results_all')
    
    return {
        'output_path': user_output_path,
        'save_path': user_save_path
    }

# Example usage function
def run_complete_extraction(output_path='../Output/ica_rest_all/', save_path=None):
    """Run the complete extraction and analysis pipeline"""
    
    print("🚀 Starting complete model results extraction...")
    print(f"📂 Reading from: {output_path}")
    
    # If no save_path provided, create it based on output_path
    if save_path is None:
        save_path = os.path.join(output_path, 'results_all')
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    print(f"💾 Results will be saved to: {save_path}")
    
    # Extract all results
    extracted_results = extract_all_results(output_path, verbose=True)
    
    # Create DataFrame
    df = create_results_dataframe(extracted_results)
    
    # Print summary
    if not df.empty:
        print_results_summary(df)
        
        # Save DataFrame for further analysis
        df.to_csv(os.path.join(save_path, 'model_results_summary.csv'), index=False)
        print(f"\n💾 Results saved to '{os.path.join(save_path, 'model_results_summary.csv')}'")
        
        return df, extracted_results
    else:
        print("❌ No successful extractions to analyze!")
        return None, extracted_results

# Quick access functions for specific analyses
def get_best_models_by_subject_type(df):
    """Get the best performing model for each subject type"""
    if df.empty:
        return None
    
    best_models = df.loc[df.groupby('subject_type')['test_bal_acc_mean'].idxmax()]
    return best_models[['subject_type', 'model_id', 'test_bal_acc_mean', 'test_bal_acc_std']]

def compare_cluster_sizes(df):
    """Compare performance across different cluster sizes"""
    if df.empty:
        return None
    
    # Filter models that have cluster sizes
    clustered_df = df[df['cluster_size'] != 'N/A']
    
    if clustered_df.empty:
        return None
    
    comparison = clustered_df.groupby(['subject_type', 'model_name', 'cluster_size']).agg({
        'test_bal_acc_mean': 'first',
        'test_bal_acc_std': 'first'
    }).reset_index()
    
    return comparison

def get_missing_files(extracted_results):
    """Get list of missing result files with categorization"""
    missing = [r for r in extracted_results if not r.get('file_exists', False)]
    
    if not missing:
        return pd.DataFrame()
    
    missing_info = []
    for r in missing:
        missing_info.append({
            'subject_type': r['subject_type'],
            'model_name': r['model_name'],
            'cluster_size': r['cluster_size'],
            'model_id': f"{r['model_name']}" + (f"_c{r['cluster_size']}" if r['cluster_size'] != 'N/A' else ""),
            'expected_path': r['file_path'],
            'likely_reason': 'Still running' if r['subject_type'] in ['independent', 'adaptive'] else 'Missing/Error'
        })
    
    missing_df = pd.DataFrame(missing_info)
    return missing_df

def show_experiment_status(extracted_results):
    """Show the status of all experiments"""
    print("🔄 EXPERIMENT STATUS OVERVIEW")
    print("=" * 60)
    
    config = generate_all_model_types()
    
    for subject_type in config['subject_types']:
        print(f"\n📊 {subject_type.upper()} Experiments:")
        
        subject_results = [r for r in extracted_results if r['subject_type'] == subject_type]
        completed = [r for r in subject_results if r.get('file_exists', False)]
        missing = [r for r in subject_results if not r.get('file_exists', False)]
        
        total_expected = len(subject_results)
        
        print(f"   ✅ Completed: {len(completed)}/{total_expected} ({len(completed)/total_expected*100:.1f}%)")
        print(f"   ⏳ Missing:   {len(missing)}/{total_expected} ({len(missing)/total_expected*100:.1f}%)")
        
        if missing:
            print(f"   📝 Missing models:")
            for r in missing:
                model_id = f"{r['model_name']}" + (f"_c{r['cluster_size']}" if r['cluster_size'] != 'N/A' else "")
                print(f"      - {model_id}")
    
    print(f"\n🔄 Overall Progress:")
    total_results = len(extracted_results)
    total_completed = len([r for r in extracted_results if r.get('file_exists', False)])
    print(f"   {total_completed}/{total_results} experiments completed ({total_completed/total_results*100:.1f}%)")
    
    return {
        'total_expected': total_results,
        'total_completed': total_completed,
        'completion_rate': total_completed/total_results*100
    }
# ============================== Plotting Functions =================================
def get_model_colors(df):
    """
    Generate color palettes for each subject type based on available models
    Creates unique colors for each model+cluster combination
    
    Args:
        df: DataFrame with results
        
    Returns:
        dict: {subject_type: {model_key: color}} where model_key includes cluster info
    """
    # Define color palettes for each subject type
    palette_map = {
        'dependent': 'flare_r',
        'independent': 'summer_r', 
        'adaptive': 'Blues_d'
    }
    
    color_dict = {}
    
    for subject_type in ['dependent', 'independent', 'adaptive']:
        # Get all unique model+cluster combinations for this subject type
        subject_df = df[df['subject_type'] == subject_type]
        
        # Create unique keys that include cluster info
        model_keys = []
        for _, row in subject_df.iterrows():
            if row['model_name'] == 'dcn':
                model_key = 'dcn'  # DCN doesn't have clusters
            else:
                model_key = f"{row['model_name']}_c{row['cluster_size']}"
            model_keys.append(model_key)
        
        # Get unique model keys
        unique_model_keys = list(dict.fromkeys(model_keys))  # Preserves order
        n_models = len(unique_model_keys)
        
        if n_models > 0:
            # Generate color palette
            colors = sns.color_palette(palette_map[subject_type], n_models)
            color_dict[subject_type] = dict(zip(unique_model_keys, colors))
        else:
            color_dict[subject_type] = {}
    
    return color_dict

def get_model_key(row):
    """Helper function to create consistent model keys"""
    if row['model_name'] == 'dcn':
        return 'dcn'
    else:
        return f"{row['model_name']}_c{row['cluster_size']}"

def plot_model_comparison(df, score_types=['test_bal_acc', 'test_f1'], metric_type='test'):
    """
    Main function that creates 2x2 subplots (top row: balanced accuracy, bottom row: f1)
    
    Args:
        df: DataFrame with model results
        score_types: List of score types to plot (first will be top row, second bottom row)
    """
    if df is None:
        return
    
    # Get the color dictionary
    color_dict = get_model_colors(df)
    subject_types = df['subject_type'].unique()
    n_subjects = len(subject_types)
    
    with sns.plotting_context('talk'):
        fig, axes = plt.subplots(2, n_subjects, figsize=(15, 12))
        fig.suptitle(f'Model Performance Comparison in {metric_type}', fontsize=16, fontweight='bold')
        # If only one subject type, make axes 2D
        if n_subjects == 1:
            axes = axes.reshape(2, 1)
        
        # Plot balanced accuracy on top row
        for j, subject_type in enumerate(subject_types):
            plot_single_metric(df, axes[0, j], score_types[0], color_dict, subject_type)
        
        # Plot F1 on bottom row  
        for j, subject_type in enumerate(subject_types):
            plot_single_metric(df, axes[1, j], score_types[1], color_dict, subject_type)
        
        plt.tight_layout()
        plt.show()


def plot_single_metric(df, ax, score_type, color_dict, subject_type):
    """
    Plot a single metric for a single subject type on given axis
    
    Args:
        df: DataFrame with model results
        ax: Matplotlib axis to plot on
        score_type: The score type to plot (e.g., 'test_f1', 'test_bal_acc')
        color_dict: Color dictionary from get_model_colors()
        subject_type: Which subject type to plot ('dependent', 'independent', 'adaptive')
    """
    # Filter data for this subject type
    subset = df[df['subject_type'] == subject_type]
    
    # Create model keys and get corresponding colors
    model_keys = [get_model_key(row) for _, row in subset.iterrows()]
    colors = [color_dict[subject_type][key] for key in model_keys]
    
    # Get mean and std columns
    mean_col = f'{score_type}_mean'
    std_col = f'{score_type}_std'
    
    # Create bar plot with colors
    x_pos = range(len(subset))
    bars = ax.bar(x_pos, subset[mean_col], 
                  yerr=subset[std_col], 
                  capsize=3,
                  color=colors)
    
    # Set labels based on score type
    if 'f1' in score_type.lower():
        ylabel = 'Test F1 Score (%)'
    elif 'bal_acc' in score_type.lower():
        ylabel = 'Test Balanced Accuracy (%)'
    else:
        ylabel = f'{score_type.replace("_", " ").title()} (%)'
    
    ax.set_title(f'{subject_type.title()} Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subset['model_id'], rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='x', bottom=True)

    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, subset[mean_col], subset[std_col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.5,
                f'{mean_val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(df[mean_col] + df[std_col]) + 5)

def plot_model_violin_comparison(df, transpose=False):
    """
    Create violin plots comparing all models across subject types
    
    Args:
        df: DataFrame with results
        
    Returns:
        matplotlib figure
    """
    # Model name mapping for shorter labels (including cluster info)
    def get_model_short_name(row):
        model_mapping = {
            'dcn': 'DCN',
            'msn': 'MSN',
            'msn_embedded': 'MSN(emb)',
            'multiscale_msn': 'MSN(multi)',
            'multiscale_msn_embedded': 'MSN(multi+emb)',
            'dsn_msn': 'DSN',
            'dsn_msn_embedded': 'DSN(emb)',
            'dsn_multiscale_msn': 'DSN(multi)',
            'dsn_multiscale_msn_embedded': 'DSN(multi+emb)'
        }
        
        base_name = model_mapping.get(row['model_name'], row['model_name'])
        
        # Add cluster info for non-DCN models
        if row['model_name'] != 'dcn':
            return f"{base_name}_C{row['cluster_size']}"
        else:
            return base_name
    
    # Get color mappings using the dedicated function
    color_dict = get_model_colors(df)
    
    # Prepare data for violin plots - include ALL models (both C5 and C12)
    plot_data = []
    
    for _, row in df.iterrows():
        model_short = get_model_short_name(row)
        model_key = get_model_key(row)  # Use consistent model key
        
        # Add balanced accuracy data
        for score in row['test_bal_acc_all']:
            plot_data.append({
                'subject_type': row['subject_type'],
                'model': model_short,
                'model_full': row['model_name'],
                'model_key': model_key,  # Add model key for color mapping
                'metric': 'Balanced Accuracy',
                'score': score,
                'cluster_size': row['cluster_size']
            })
        
        # Add F1 data
        for score in row['test_f1_all']:
            plot_data.append({
                'subject_type': row['subject_type'],
                'model': model_short,
                'model_full': row['model_name'],
                'model_key': model_key,  # Add model key for color mapping
                'metric': 'F1 Macro',
                'score': score,
                'cluster_size': row['cluster_size']
            })
    
    plot_df = pd.DataFrame(plot_data)
    # subject_types = ['dependent']
    # subject_types = ['dependent', 'independent']
    subject_types = ['dependent', 'independent', 'adaptive']
    metrics = ['Balanced Accuracy', 'F1 Macro']
    # Create subplots: 2 rows (metrics) x 1 col (subject types)
    if transpose:
        n_rows = len(subject_types)
        n_cols = len(metrics)
        fig_size = (20, 10)
    else:
        n_rows = len(metrics)
        n_cols = len(subject_types)
        fig_size = (30, 20)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharey=True)
    fig.suptitle('Model Performance Comparison Across Subject Types', fontsize=16, y=0.98)
    
    for row, metric in enumerate(metrics):
        for col, subject_type in enumerate(subject_types):
            if n_cols == 1:
                ax = axes[row]
            elif n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Filter data for this combination
            subset = plot_df[(plot_df['subject_type'] == subject_type) & 
                           (plot_df['metric'] == metric)]
            
            if not subset.empty:
                # Get unique models for this subject type in order
                unique_models = subset['model'].unique()
                unique_model_keys = []
                
                # Create mapping from model short name to model key for color lookup
                model_to_key = {}
                for _, row_data in subset.iterrows():
                    if row_data['model'] not in model_to_key:
                        model_to_key[row_data['model']] = row_data['model_key']
                
                # Get colors using the get_model_colors function
                subject_colors = color_dict.get(subject_type, {})
                colors = []
                for model in unique_models:
                    model_key = model_to_key[model]
                    color = subject_colors.get(model_key, 'gray')  # Default to gray if not found
                    colors.append(color)
                
                # Create violin plot with no inner elements and cut=0
                sns.violinplot(data=subset, x='model', y='score', ax=ax, 
                            palette=colors,
                            inner_kws=dict(box_width=10, whis_width=2, color = 'dimgray'), 
                            cut=0)
                # Add mean markers
                # means = subset.groupby('model')['score'].mean()
                # for i, (model, mean_val) in enumerate(means.items()):
                #     ax.scatter(i, mean_val, color='red', s=100, zorder=3, marker='_')
            # Formatting
            ax.set_ylim(-3, 103)
            ax.set_title(f'{subject_type.title()}' if row == 0 else '')
            ax.set_xlabel('Model Type' if row == 1 else '')
            ax.set_ylabel(f'{metric} (%)' if col == 0 else '')
            if not subset.empty:
                ax.set_xticks(range(len(unique_models)))
                ax.set_xticklabels(unique_models, rotation=45, ha='right')
            ax.tick_params(axis='x', rotation=45, colors='black', bottom=True)
            ax.grid(True, axis='x')
            
            # Remove x-axis labels for top row
            if row == 0:
                # ax.set_xticklabels([])
                ax.set_xlabel('')
    
    plt.tight_layout()
    return fig

def plot_violin_comparison(df, save_path=None):
    """Create and optionally save violin comparison plot"""
    with sns.plotting_context("talk"):
        fig = plot_model_violin_comparison(df)
        
        if save_path:
            fig.savefig(os.path.join(save_path, 'model_violin_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"💾 Violin plot saved to {save_path}")
        
        plt.close()
        return fig
    
def plot_grouped_violin_comparison(df, save_path=None):
    """
    Create grouped violin plots by model family (DCN, MSN, DSN)
    Uses same visual style and colors as the main violin plot
    
    Args:
        df: DataFrame with results
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib figure
    """
    
    # Model name mapping for shorter labels (same as main function)
    def get_model_short_name(row):
        model_mapping = {
            'dcn': 'DCN',
            'msn': 'MSN',
            'msn_embedded': 'MSN(emb)',
            'multiscale_msn': 'MSN(multi)',
            'multiscale_msn_embedded': 'MSN(multi+emb)',
            'dsn_msn': 'DSN',
            'dsn_msn_embedded': 'DSN(emb)',
            'dsn_multiscale_msn': 'DSN(multi)',
            'dsn_multiscale_msn_embedded': 'DSN(multi+emb)'
        }
        
        base_name = model_mapping.get(row['model_name'], row['model_name'])
        
        # Add cluster info for non-DCN models
        if row['model_name'] != 'dcn':
            return f"{base_name}_C{row['cluster_size']}"
        else:
            return base_name
    
    # Get the same color mappings as main function
    color_dict = get_model_colors(df)
    
    # Prepare data with model family grouping
    plot_data = []
    
    for _, row in df.iterrows():
        model_short = get_model_short_name(row)
        model_key = get_model_key(row)  # Use consistent model key
        
        # Determine model family
        if row['model_name'] == 'dcn':
            model_family = 'DCN'
        elif row['model_name'].startswith('dsn_'):
            model_family = 'DSN'
        else:
            model_family = 'MSN'
        
        # Add balanced accuracy data
        for score in row['test_bal_acc_all']:
            plot_data.append({
                'subject_type': row['subject_type'],
                'model_family': model_family,
                'model': model_short,
                'model_full': row['model_name'],
                'model_key': model_key,  # Add model key for color mapping
                'metric': 'Balanced Accuracy',
                'score': score,
                'cluster_size': row['cluster_size']
            })
        
        # Add F1 data
        for score in row['test_f1_all']:
            plot_data.append({
                'subject_type': row['subject_type'],
                'model_family': model_family,
                'model': model_short,
                'model_full': row['model_name'],
                'model_key': model_key,  # Add model key for color mapping
                'metric': 'F1 Macro',
                'score': score,
                'cluster_size': row['cluster_size']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    with sns.plotting_context("talk"):
        # Create subplots: 2 rows (metrics) x 3 cols (model families)
        fig, axes = plt.subplots(2, 3, figsize=(30, 20), sharey=True)
        fig.suptitle('Model Performance by Family (DCN, MSN, DSN)', fontsize=16, y=0.98)
        
        metrics = ['Balanced Accuracy', 'F1 Macro']
        model_families = ['DCN', 'MSN', 'DSN']
        subject_types = ['dependent', 'independent', 'adaptive']
        
        for row, metric in enumerate(metrics):
            for col, model_family in enumerate(model_families):
                ax = axes[row, col]
                
                # Filter data for this model family and metric
                family_data = plot_df[(plot_df['model_family'] == model_family) & 
                                    (plot_df['metric'] == metric)]
                
                if not family_data.empty:
                    # For DCN: simple plot by subject_type
                    if model_family == 'DCN':
                        # Create violin plot with subject types on x-axis
                        subset_dcn = family_data.copy()
                        
                        # Get unique subject types and their colors from color_dict
                        unique_subjects = subset_dcn['subject_type'].unique()
                        colors = []
                        
                        for subject_type in unique_subjects:
                            # Get DCN color for this subject type from color_dict
                            subject_colors = color_dict.get(subject_type, {})
                            dcn_color = subject_colors.get('dcn', 'gray')
                            colors.append(dcn_color)
                        
                        sns.violinplot(data=subset_dcn, x='subject_type', y='score', 
                                    ax=ax, palette=colors,
                                    inner_kws=dict(box_width=10, whis_width=2, color = 'dimgray'), 
                                    cut=0)
                        
                        unique_models = unique_subjects
                        
                    else:
                        # For MSN/DSN: grouped by subject type, models within each group
                        # Create a combined x-axis label
                        family_data['x_label'] = family_data['subject_type'] + '_' + family_data['model']
                        
                        # Get unique combinations and sort them
                        unique_combos = []
                        colors = []
                        
                        # Create mapping from x_label to model_key for color lookup
                        label_to_key = {}
                        for _, row_data in family_data.iterrows():
                            x_label = row_data['x_label']
                            if x_label not in label_to_key:
                                label_to_key[x_label] = row_data['model_key']
                        
                        for subject_type in subject_types:
                            subject_data = family_data[family_data['subject_type'] == subject_type]
                            if not subject_data.empty:
                                subject_models = subject_data['model'].unique()
                                
                                # Get colors from color_dict using model keys
                                subject_colors = color_dict.get(subject_type, {})
                                
                                for model in subject_models:
                                    x_label = f"{subject_type}_{model}"
                                    unique_combos.append(x_label)
                                    
                                    # Look up color using model key
                                    model_key = label_to_key.get(x_label, '')
                                    color = subject_colors.get(model_key, 'gray')
                                    colors.append(color)
                        
                        # Create violin plot
                        sns.violinplot(data=family_data, x='x_label', y='score', 
                                    ax=ax, palette=colors,
                                    inner_kws=dict(box_width=10, whis_width=2, color = 'dimgray'), 
                                    cut=0, order=unique_combos)
                        
                        unique_models = unique_combos
                
                # Formatting (same style as main function)
                ax.set_ylim(-3, 103)
                ax.set_title(f'{model_family}' if row == 0 else '')
                ax.set_xlabel('Models' if row == 1 else '')
                ax.set_ylabel(f'{metric} (%)' if col == 0 else '')
                
                if not family_data.empty:
                    n_models = len(unique_models)
                    ax.set_xticks(range(n_models))
                    
                    # Create cleaner x-tick labels
                    if model_family == 'DCN':
                        clean_labels = unique_models
                    else:
                        # For MSN/DSN, clean up the labels
                        clean_labels = []
                        for combo in unique_models:
                            subject, model = combo.split('_', 1)
                            clean_labels.append(f"{subject}\n{model}")
                    
                    ax.set_xticklabels(clean_labels, rotation=45, ha='right')
                    ax.tick_params(axis='x', rotation=45, colors='black', bottom=True)
                    ax.grid(True, axis='x')
                
                # Remove x-axis labels for top row
                if row == 0:
                    ax.set_xlabel('')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(os.path.join(save_path, 'grouped_model_violin_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"💾 Grouped violin plot saved to {save_path}")
        
        plt.close()
        return fig

def plot_grouped_comparison(df, save_path=None):
    """Create grouped violin comparison plot with same style as main plot"""
    return plot_grouped_violin_comparison(df, save_path)






# Default plot style dictionary - modify this to change all plot styles at once
# Default plot style dictionary - modify this to change all plot styles at once
DEFAULT_PLOT_STYLE = {
    'line': {
        'marker': 'o',
        'markersize': 4,
        'linewidth': 2,
        'alpha': 0.8,
        'linestyle': '--'  # dashed lines
    },
    'scatter': {
        'marker': 'o',
        's': 40,
        'alpha': 0.7
    },
    'figure': {
        'context': 'talk',
        'figsize_2x3': (24, 16),
        'figsize_1x3': (24, 8),
        'suptitle_fontsize': 16,
        'title_fontsize': 14,
        'legend_fontsize': 10,
        'grid_alpha': 0.3
    }
}

def filter_models_by_focus(df, focus_on=None):
    """
    Filter DataFrame based on focus parameter(s)
    
    Args:
        df: DataFrame with results
        focus_on: str, list of str, or None. Options:
            - 'c5': Only cluster size 5 models (+ DCN)
            - 'c12': Only cluster size 12 models (+ DCN)
            - 'embedded': Only embedded models (+ DCN)
            - 'not_embedded': Only non-embedded models (+ DCN)
            - 'multiscale': Only multiscale models (+ DCN)
            - 'not_multiscale': Only non-multiscale models (+ DCN)
            - List: Multiple filters combined with AND logic
            - None: All models
    
    Returns:
        Filtered DataFrame
    """
    if focus_on is None:
        return df.copy()
    
    # Convert single string to list for uniform processing
    if isinstance(focus_on, str):
        focus_list = [focus_on]
    else:
        focus_list = focus_on
    
    # Always keep DCN
    dcn_mask = df['model_name'] == 'dcn'
    
    # Start with all True for MSN/DSN models, then apply filters with AND logic
    msn_dsn_mask = df['model_name'] != 'dcn'  # Start with all non-DCN models
    
    for focus in focus_list:
        if focus == 'c5':
            filter_mask = df['cluster_size'] == 5
        elif focus == 'c12':
            filter_mask = df['cluster_size'] == 12
        elif focus == 'embedded':
            filter_mask = df['model_name'].str.contains('embedded', na=False)
        elif focus == 'not_embedded':
            filter_mask = ~df['model_name'].str.contains('embedded', na=False)
        elif focus == 'multiscale':
            filter_mask = df['model_name'].str.contains('multiscale', na=False)
        elif focus == 'not_multiscale':
            filter_mask = ~df['model_name'].str.contains('multiscale', na=False)
        else:
            raise ValueError(f"Invalid focus_on value: {focus}. "
                            f"Valid options: 'c5', 'c12', 'embedded', 'not_embedded', "
                            f"'multiscale', 'not_multiscale'")
        
        # Apply this filter to MSN/DSN models only (AND logic)
        msn_dsn_mask = msn_dsn_mask & filter_mask
    
    # Combine DCN mask with filtered MSN/DSN mask
    final_mask = dcn_mask | msn_dsn_mask
    
    return df[final_mask].copy()

def plot_subjects_comparison(df, metric='Balanced Accuracy', plot_type='line', 
                           focus_on=None, plot_style=None, save_path=None):
    """
    Create line or scatter plots showing all models across all 50 subjects
    
    Args:
        df: DataFrame with results
        metric: 'Balanced Accuracy' or 'F1 Macro'
        plot_type: 'line' or 'scatter'
        focus_on: str, list of str, or None - filter models. Options:
                 'c5', 'c12', 'embedded', 'not_embedded', 'multiscale', 'not_multiscale'
                 Can be single string or list for multiple filters (AND logic)
        plot_style: dict, plotting style parameters (uses DEFAULT_PLOT_STYLE if None)
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib figure
    """
    
    # Use default style if none provided
    if plot_style is None:
        plot_style = DEFAULT_PLOT_STYLE
    
    # Filter models based on focus
    filtered_df = filter_models_by_focus(df, focus_on)
    
    # Model name mapping (same as violin plot)
    def get_model_short_name(row):
        model_mapping = {
            'dcn': 'DCN',
            'msn': 'MSN',
            'msn_embedded': 'MSN(emb)',
            'multiscale_msn': 'MSN(multi)',
            'multiscale_msn_embedded': 'MSN(multi+emb)',
            'dsn_msn': 'DSN',
            'dsn_msn_embedded': 'DSN(emb)',
            'dsn_multiscale_msn': 'DSN(multi)',
            'dsn_multiscale_msn_embedded': 'DSN(multi+emb)'
        }
        
        base_name = model_mapping.get(row['model_name'], row['model_name'])
        
        # Add cluster info for non-DCN models
        if row['model_name'] != 'dcn':
            return f"{base_name}_C{row['cluster_size']}"
        else:
            return base_name
    
    # Get color mappings (same as violin plot)
    color_dict = get_model_colors(filtered_df)
    
    # Prepare data - expand all individual subject scores
    plot_data = []
    
    for _, row in filtered_df.iterrows():
        model_short = get_model_short_name(row)
        model_key = get_model_key(row)
        
        # Choose the right score column based on metric
        if metric == 'Balanced Accuracy':
            all_scores = row['test_bal_acc_all']
        else:  # F1 Macro
            all_scores = row['test_f1_all']
        
        # Add each subject's score as a separate data point
        for subject_idx, score in enumerate(all_scores):
            plot_data.append({
                'subject_type': row['subject_type'],
                'model': model_short,
                'model_key': model_key,
                'subject_id': subject_idx + 1,  # 1-indexed subjects
                'score': score
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create focus suffix for title
    if focus_on:
        if isinstance(focus_on, list):
            focus_suffix = f" (Focus: {', '.join(focus_on)})"
        else:
            focus_suffix = f" (Focus: {focus_on})"
    else:
        focus_suffix = ""
    
    with sns.plotting_context(plot_style['figure']['context']):
        # Create subplots: 2 rows x 3 cols
        fig, axes = plt.subplots(2, 3, figsize=plot_style['figure']['figsize_2x3'])
        fig.suptitle(f'All Subjects {metric} Comparison ({plot_type.title()} Plot){focus_suffix}', 
                     fontsize=plot_style['figure']['suptitle_fontsize'], y=0.98)
        
        subject_types = ['dependent', 'independent', 'adaptive']
        
        # Row 0: By Subject Type
        for col, subject_type in enumerate(subject_types):
            ax = axes[0, col]
            
            # Filter data for this subject type
            subject_data = plot_df[plot_df['subject_type'] == subject_type]
            
            if not subject_data.empty:
                # Get unique models for this subject type
                unique_models = subject_data['model'].unique()
                
                # Get colors for this subject type
                subject_colors = color_dict.get(subject_type, {})
                
                for model in unique_models:
                    model_data = subject_data[subject_data['model'] == model]
                    if not model_data.empty:
                        # Get model key for color
                        model_key = model_data.iloc[0]['model_key']
                        color = subject_colors.get(model_key, 'gray')
                        
                        # Sort by subject_id for proper line connection
                        model_data_sorted = model_data.sort_values('subject_id')
                        
                        if plot_type == 'line':
                            ax.plot(model_data_sorted['subject_id'], model_data_sorted['score'],
                                   color=color, label=model, **plot_style['line'])
                        else:  # scatter
                            ax.scatter(model_data_sorted['subject_id'], model_data_sorted['score'],
                                     color=color, label=model, **plot_style['scatter'])
            
            # Formatting
            ax.set_ylim(0, 100)
            ax.set_xlim(0.5, 50.5)
            ax.set_title(f'{subject_type.title()} Models', 
                        fontsize=plot_style['figure']['title_fontsize'], fontweight='bold')
            ax.set_ylabel(f'{metric} (%)' if col == 0 else '')
            ax.set_xlabel('Subject ID')
            ax.grid(True, alpha=plot_style['figure']['grid_alpha'])
            ax.legend(fontsize=plot_style['figure']['legend_fontsize'], loc='best')
        
        # Row 1: By Model Family
        model_families = ['DCN', 'MSN', 'DSN']
        
        for col, model_family in enumerate(model_families):
            ax = axes[1, col]
            
            # Filter data for this model family
            if model_family == 'DCN':
                family_data = plot_df[plot_df['model'] == 'DCN']
            elif model_family == 'MSN':
                family_data = plot_df[plot_df['model'].str.startswith('MSN')]
            else:  # DSN
                family_data = plot_df[plot_df['model'].str.startswith('DSN')]
            
            if not family_data.empty:
                # Get unique models for this family
                unique_models = family_data['model'].unique()
                
                for model in unique_models:
                    model_data = family_data[family_data['model'] == model]
                    if not model_data.empty:
                        # Group by subject type to get appropriate colors
                        for subject_type in subject_types:
                            subject_model_data = model_data[model_data['subject_type'] == subject_type]
                            if not subject_model_data.empty:
                                # Get color for this subject type and model
                                model_key = subject_model_data.iloc[0]['model_key']
                                subject_colors = color_dict.get(subject_type, {})
                                color = subject_colors.get(model_key, 'gray')
                                
                                # Sort by subject_id for proper line connection
                                subject_model_sorted = subject_model_data.sort_values('subject_id')
                                
                                # Create label combining model and subject type
                                label = f"{model} ({subject_type})"
                                
                                if plot_type == 'line':
                                    ax.plot(subject_model_sorted['subject_id'], subject_model_sorted['score'],
                                           color=color, label=label, **plot_style['line'])
                                else:  # scatter
                                    ax.scatter(subject_model_sorted['subject_id'], subject_model_sorted['score'],
                                             color=color, label=label, **plot_style['scatter'])
            
            # Formatting
            ax.set_ylim(0, 100)
            ax.set_xlim(0.5, 50.5)
            ax.set_title(f'{model_family} Models', 
                        fontsize=plot_style['figure']['title_fontsize'], fontweight='bold')
            ax.set_ylabel(f'{metric} (%)' if col == 0 else '')
            ax.set_xlabel('Subject ID')
            ax.grid(True, alpha=plot_style['figure']['grid_alpha'])
            ax.legend(fontsize=plot_style['figure']['legend_fontsize'], loc='best')
        
        plt.tight_layout()
        
        if save_path:
            if focus_on:
                if isinstance(focus_on, list):
                    focus_str = f"_{'_'.join(focus_on)}"
                else:
                    focus_str = f"_{focus_on}"
            else:
                focus_str = ""
            plot_name = f'subjects_{plot_type}_{metric.lower().replace(" ", "_")}_comparison{focus_str}.png'
            fig.savefig(os.path.join(save_path, plot_name), 
                       dpi=300, bbox_inches='tight')
            print(f"💾 Subjects {plot_type} plot saved to {save_path}")
        
        return fig

def plot_subjects_by_type_only(df, metric='Balanced Accuracy', plot_type='line', 
                             focus_on=None, plot_style=None, save_path=None):
    """
    Create plots showing all models across all subjects, organized by subject type only
    One subplot per subject type (3 subplots total)
    
    Args:
        df: DataFrame with results
        metric: 'Balanced Accuracy' or 'F1 Macro'
        plot_type: 'line' or 'scatter'
        focus_on: str, list of str, or None - filter models. Options:
                 'c5', 'c12', 'embedded', 'not_embedded', 'multiscale', 'not_multiscale'
                 Can be single string or list for multiple filters (AND logic)
        plot_style: dict, plotting style parameters (uses DEFAULT_PLOT_STYLE if None)
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib figure
    """
    
    # Use default style if none provided
    if plot_style is None:
        plot_style = DEFAULT_PLOT_STYLE
    
    # Filter models based on focus
    filtered_df = filter_models_by_focus(df, focus_on)
    
    # Model name mapping (same as violin plot)
    def get_model_short_name(row):
        model_mapping = {
            'dcn': 'DCN',
            'msn': 'MSN',
            'msn_embedded': 'MSN(emb)',
            'multiscale_msn': 'MSN(multi)',
            'multiscale_msn_embedded': 'MSN(multi+emb)',
            'dsn_msn': 'DSN',
            'dsn_msn_embedded': 'DSN(emb)',
            'dsn_multiscale_msn': 'DSN(multi)',
            'dsn_multiscale_msn_embedded': 'DSN(multi+emb)'
        }
        
        base_name = model_mapping.get(row['model_name'], row['model_name'])
        
        # Add cluster info for non-DCN models
        if row['model_name'] != 'dcn':
            return f"{base_name}_C{row['cluster_size']}"
        else:
            return base_name
    
    # Get color mappings (same as violin plot)
    color_dict = get_model_colors(filtered_df)
    
    # Prepare data - expand all individual subject scores
    plot_data = []
    
    for _, row in filtered_df.iterrows():
        model_short = get_model_short_name(row)
        model_key = get_model_key(row)
        
        # Choose the right score column based on metric
        if metric == 'Balanced Accuracy':
            all_scores = row['test_bal_acc_all']
        else:  # F1 Macro
            all_scores = row['test_f1_all']
        
        # Add each subject's score as a separate data point
        for subject_idx, score in enumerate(all_scores):
            plot_data.append({
                'subject_type': row['subject_type'],
                'model': model_short,
                'model_key': model_key,
                'subject_id': subject_idx + 1,  # 1-indexed subjects
                'score': score
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create focus suffix for title
    if focus_on:
        if isinstance(focus_on, list):
            focus_suffix = f" (Focus: {', '.join(focus_on)})"
        else:
            focus_suffix = f" (Focus: {focus_on})"
    else:
        focus_suffix = ""
    
    with sns.plotting_context(plot_style['figure']['context']):
        # Create subplots: 1 row x 3 cols (one per subject type)
        fig, axes = plt.subplots(1, 3, figsize=plot_style['figure']['figsize_1x3'])
        fig.suptitle(f'All Subjects {metric} Comparison by Subject Type ({plot_type.title()} Plot){focus_suffix}', 
                     fontsize=plot_style['figure']['suptitle_fontsize'], y=0.98)
        
        subject_types = ['dependent', 'independent', 'adaptive']
        
        for col, subject_type in enumerate(subject_types):
            ax = axes[col]
            
            # Filter data for this subject type
            subject_data = plot_df[plot_df['subject_type'] == subject_type]
            
            if not subject_data.empty:
                # Get unique models for this subject type
                unique_models = subject_data['model'].unique()
                
                # Get colors for this subject type
                subject_colors = color_dict.get(subject_type, {})
                
                for model in unique_models:
                    model_data = subject_data[subject_data['model'] == model]
                    if not model_data.empty:
                        # Get model key for color
                        model_key = model_data.iloc[0]['model_key']
                        color = subject_colors.get(model_key, 'gray')
                        
                        # Sort by subject_id for proper line connection
                        model_data_sorted = model_data.sort_values('subject_id')
                        
                        if plot_type == 'line':
                            ax.plot(model_data_sorted['subject_id'], model_data_sorted['score'],
                                   color=color, label=model, **plot_style['line'])
                        else:  # scatter
                            ax.scatter(model_data_sorted['subject_id'], model_data_sorted['score'],
                                     color=color, label=model, **plot_style['scatter'])
            
            # Formatting
            ax.set_ylim(0, 100)
            ax.set_xlim(0.5, 50.5)
            ax.set_title(f'{subject_type.title()} Models', 
                        fontsize=plot_style['figure']['title_fontsize'], fontweight='bold')
            ax.set_ylabel(f'{metric} (%)' if col == 0 else '')
            ax.set_xlabel('Subject ID')
            ax.grid(True, alpha=plot_style['figure']['grid_alpha'])
            ax.legend(fontsize=plot_style['figure']['legend_fontsize'], loc='best')
        
        plt.tight_layout()
        
        if save_path:
            if focus_on:
                if isinstance(focus_on, list):
                    focus_str = f"_{'_'.join(focus_on)}"
                else:
                    focus_str = f"_{focus_on}"
            else:
                focus_str = ""
            plot_name = f'subjects_by_type_{plot_type}_{metric.lower().replace(" ", "_")}_comparison{focus_str}.png'
            fig.savefig(os.path.join(save_path, plot_name), 
                       dpi=300, bbox_inches='tight')
            print(f"💾 Subjects by type {plot_type} plot saved to {save_path}")
        
        return fig

# Convenience functions
def plot_subjects_scatter_comparison(df, metric='Balanced Accuracy', focus_on=None, 
                                   plot_style=None, save_path=None):
    """Convenience function for scatter plots of all subjects"""
    return plot_subjects_comparison(df, metric=metric, plot_type='scatter', 
                                  focus_on=focus_on, plot_style=plot_style, save_path=save_path)

def plot_subjects_line_comparison(df, metric='Balanced Accuracy', focus_on=None, 
                                plot_style=None, save_path=None):
    """Convenience function for line plots of all subjects"""
    return plot_subjects_comparison(df, metric=metric, plot_type='line', 
                                  focus_on=focus_on, plot_style=plot_style, save_path=save_path)

def plot_subjects_by_type_scatter(df, metric='Balanced Accuracy', focus_on=None, 
                                plot_style=None, save_path=None):
    """Convenience function for scatter plots by subject type only"""
    return plot_subjects_by_type_only(df, metric=metric, plot_type='scatter', 
                                    focus_on=focus_on, plot_style=plot_style, save_path=save_path)

def plot_subjects_by_type_line(df, metric='Balanced Accuracy', focus_on=None, 
                             plot_style=None, save_path=None):
    """Convenience function for line plots by subject type only"""
    return plot_subjects_by_type_only(df, metric=metric, plot_type='line', 
                                    focus_on=focus_on, plot_style=plot_style, save_path=save_path)

# ================================== Statistics =====================================
"""
my_stats_functions.py
Comprehensive statistical analysis functions for ML model comparison
Author: Your Name
Date: 2025

This module provides functions for:
1. Cluster size comparison (5 vs 12)
2. Encoding type comparison (one-hot vs embedded) for dependent models
3. Kernel scale comparison (single vs multiscale)

Each analysis includes ANOVA, effect sizes, and model-specific pairwise comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size between two groups
    
    Parameters:
    -----------
    group1, group2 : array-like
        Data for comparison groups
    
    Returns:
    --------
    cohens_d : float
        Cohen's d effect size
    interpretation : str
        Interpretation of effect size magnitude
    """
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                         (len(group2) - 1) * np.var(group2, ddof=1)) / 
                        (len(group1) + len(group2) - 2))
    
    if pooled_std == 0:
        return 0, "No effect"
    
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # Interpretation for Cohen's d
    abs_effect = abs(cohens_d)
    if abs_effect < 0.2:
        interpretation = "Negligible"
    elif abs_effect < 0.5:
        interpretation = "Small"
    elif abs_effect < 0.8:
        interpretation = "Medium"
    else:
        interpretation = "Large"
        
    return cohens_d, interpretation

def calculate_eta_squared(groups_data):
    """
    Calculate eta squared (effect size for ANOVA)
    
    Parameters:
    -----------
    groups_data : list of arrays
        List containing data for each group
    
    Returns:
    --------
    eta_squared : float
        Eta squared effect size
    interpretation : str
        Interpretation of effect size
    """
    all_data = np.concatenate(groups_data)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups_data)
    ss_total = np.sum((all_data - grand_mean)**2)
    
    if ss_total == 0:
        return 0, "No effect"
        
    eta_squared = ss_between / ss_total
    
    # Interpretation for eta squared
    if eta_squared < 0.01:
        interpretation = "Negligible"
    elif eta_squared < 0.06:
        interpretation = "Small"
    elif eta_squared < 0.14:
        interpretation = "Medium"
    else:
        interpretation = "Large"
    
    return eta_squared, interpretation

def perform_anova_analysis(data, dv_col, iv_col):
    """
    Perform one-way ANOVA analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing dependent and independent variables
    dv_col : str
        Column name for dependent variable (performance metric)
    iv_col : str
        Column name for independent variable (factor to test)
    
    Returns:
    --------
    results : dict
        Dictionary containing ANOVA results and effect sizes
    """
    # Remove any missing values
    clean_data = data[[dv_col, iv_col]].dropna()
    
    if len(clean_data) == 0:
        return {"error": "No valid data for analysis"}
    
    # Get unique groups
    groups = clean_data[iv_col].unique()
    
    if len(groups) < 2:
        return {"error": "Need at least 2 groups for comparison"}
    
    # Prepare data for analysis
    group_data = [clean_data[clean_data[iv_col] == group][dv_col].values 
                  for group in groups]
    
    # Perform one-way ANOVA
    f_stat, p_value = f_oneway(*group_data)
    
    # Calculate eta squared
    eta_squared, eta_interpretation = calculate_eta_squared(group_data)
    
    # Calculate descriptive statistics
    descriptives = clean_data.groupby(iv_col)[dv_col].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    results = {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'eta_interpretation': eta_interpretation,
        'significant': p_value < 0.05,
        'groups': groups,
        'descriptives': descriptives,
        'n_total': len(clean_data),
        'dv_col': dv_col,
        'iv_col': iv_col
    }
    
    return results

def perform_model_specific_pairwise_comparisons(data, dv_col, comparison_type='cluster', alpha=0.05):
    """
    Perform model-specific pairwise comparisons based on comparison type
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data for analysis
    dv_col : str
        Dependent variable column
    comparison_type : str
        Type of comparison ('cluster', 'embedding', 'scale')
    alpha : float
        Significance level
    
    Returns:
    --------
    comparisons : pd.DataFrame
        DataFrame with model-specific pairwise comparison results
    """
    comparisons = []
    
    if comparison_type == 'cluster':
        # Compare same models with different cluster sizes: msn_c5 vs msn_c12, etc.
        model_bases = set()
        for model_name in data['model_name'].unique():
            if 'dcn' not in model_name.lower():
                # Remove cluster size suffix to get base model
                base_model = model_name.replace('_c5', '').replace('_c12', '')
                model_bases.add(base_model)
        
        for base_model in model_bases:
            # Get models with cluster 5 and 12 for this base model
            model_c5_data = data[(data['model_name'].str.contains(base_model, case=False, regex=False)) & 
                               (data['cluster_size'] == 5)]
            model_c12_data = data[(data['model_name'].str.contains(base_model, case=False, regex=False)) & 
                                (data['cluster_size'] == 12)]
            
            if len(model_c5_data) > 0 and len(model_c12_data) > 0:
                data1 = model_c5_data[dv_col].values
                data2 = model_c12_data[dv_col].values
                
                # Get actual model names
                model_c5_name = model_c5_data['model_id'].iloc[0]
                model_c12_name = model_c12_data['model_id'].iloc[0]
                
                # Perform t-test
                t_stat, p_val = ttest_ind(data1, data2)
                
                # Calculate Cohen's d
                cohen_d, d_interpretation = calculate_cohens_d(data1, data2)
                
                # Calculate means and differences
                mean1, mean2 = np.mean(data1), np.mean(data2)
                mean_diff = mean1 - mean2
                
                comparisons.append({
                    'group1': model_c5_name,
                    'group2': model_c12_name,
                    'mean1': mean1,
                    'mean2': mean2,
                    'mean_difference': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohen_d': cohen_d,
                    'effect_interpretation': d_interpretation,
                    'n1': len(data1),
                    'n2': len(data2)
                })
    
    elif comparison_type == 'embedding':
        # Compare embedded vs non-embedded for same base models: msn_c5 vs msn_embedded_c5, etc.
        # Group by base model pattern and cluster size
        grouped_models = {}
        
        for _, row in data.iterrows():
            model_name = row['model_name']
            cluster_size = row.get('cluster_size', 'N/A')
            is_embedded = row['is_embedded']
            
            if 'dcn' in model_name.lower():
                continue
                
            # Extract base model pattern
            if 'embedded' in model_name:
                base_pattern = model_name.replace('_embedded', '')
            else:
                base_pattern = model_name
            
            # Create grouping key
            group_key = f"{base_pattern}_{cluster_size}"
            
            if group_key not in grouped_models:
                grouped_models[group_key] = {'embedded': [], 'non_embedded': []}
            
            if is_embedded:
                grouped_models[group_key]['embedded'].append(row)
            else:
                grouped_models[group_key]['non_embedded'].append(row)
        
        # Compare within each group
        for group_key, group_data in grouped_models.items():
            embedded_rows = group_data['embedded']
            non_embedded_rows = group_data['non_embedded']
            
            if len(embedded_rows) > 0 and len(non_embedded_rows) > 0:
                data1 = [row[dv_col] for row in non_embedded_rows]  # One-hot
                data2 = [row[dv_col] for row in embedded_rows]      # Embedded
                
                # Get model names
                model1_name = non_embedded_rows[0]['model_id'] + "_onehot"
                model2_name = embedded_rows[0]['model_id'] + "_embedded"
                
                # Perform t-test
                t_stat, p_val = ttest_ind(data1, data2)
                
                # Calculate Cohen's d
                cohen_d, d_interpretation = calculate_cohens_d(data1, data2)
                
                # Calculate means and differences
                mean1, mean2 = np.mean(data1), np.mean(data2)
                mean_diff = mean1 - mean2
                
                comparisons.append({
                    'group1': model1_name,
                    'group2': model2_name,
                    'mean1': mean1,
                    'mean2': mean2,
                    'mean_difference': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohen_d': cohen_d,
                    'effect_interpretation': d_interpretation,
                    'n1': len(data1),
                    'n2': len(data2)
                })
    
    elif comparison_type == 'scale':
        # Compare single-scale vs multiscale for same base models: msn_c5 vs multiscale_msn_c5, etc.
        grouped_models = {}
        
        for _, row in data.iterrows():
            model_name = row['model_name']
            cluster_size = row.get('cluster_size', 'N/A')
            
            if 'dcn' in model_name.lower():
                continue
            
            # Extract base model pattern
            if 'multiscale' in model_name:
                base_pattern = model_name.replace('multiscale_', '')
                is_multiscale = True
            else:
                base_pattern = model_name
                is_multiscale = False
            
            # Create grouping key
            group_key = f"{base_pattern}_{cluster_size}_{row.get('is_embedded', False)}"
            
            if group_key not in grouped_models:
                grouped_models[group_key] = {'single': [], 'multi': []}
            
            if is_multiscale:
                grouped_models[group_key]['multi'].append(row)
            else:
                grouped_models[group_key]['single'].append(row)
        
        # Compare within each group
        for group_key, group_data in grouped_models.items():
            single_rows = group_data['single']
            multi_rows = group_data['multi']
            
            if len(single_rows) > 0 and len(multi_rows) > 0:
                data1 = [row[dv_col] for row in single_rows]   # Single-scale
                data2 = [row[dv_col] for row in multi_rows]    # Multiscale
                
                # Get model names
                model1_name = single_rows[0]['model_id']
                model2_name = multi_rows[0]['model_id']
                
                # Perform t-test
                t_stat, p_val = ttest_ind(data1, data2)
                
                # Calculate Cohen's d
                cohen_d, d_interpretation = calculate_cohens_d(data1, data2)
                
                # Calculate means and differences
                mean1, mean2 = np.mean(data1), np.mean(data2)
                mean_diff = mean1 - mean2
                
                comparisons.append({
                    'group1': model1_name,
                    'group2': model2_name,
                    'mean1': mean1,
                    'mean2': mean2,
                    'mean_difference': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohen_d': cohen_d,
                    'effect_interpretation': d_interpretation,
                    'n1': len(data1),
                    'n2': len(data2)
                })
    
    if not comparisons:
        return pd.DataFrame()
    
    comp_df = pd.DataFrame(comparisons)
    
    # Apply Bonferroni correction
    comp_df['p_corrected'] = comp_df['p_value'] * len(comp_df)
    comp_df['p_corrected'] = comp_df['p_corrected'].clip(upper=1.0)
    
    comp_df['significant_uncorrected'] = comp_df['p_value'] < alpha
    comp_df['significant_corrected'] = comp_df['p_corrected'] < alpha
    
    return comp_df.round(4)

def plot_anova_results(anova_results, data, dv_col, iv_col, title_prefix="", save_path=None):
    """
    Create horizontal visualization for ANOVA results
    
    Parameters:
    -----------
    anova_results : dict
        Results from perform_anova_analysis
    data : pd.DataFrame
        Original data
    dv_col : str
        Dependent variable column
    iv_col : str
        Independent variable column
    title_prefix : str
        Prefix for plot title
    save_path : str, optional
        Full path (including filename) to save the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object containing the plots
    """
    clean_data = data[[dv_col, iv_col]].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix} ANOVA Results: {dv_col} by {iv_col}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Horizontal box plot
    sns.boxplot(data=clean_data, y=iv_col, x=dv_col, ax=axes[0,0])
    axes[0,0].set_title('Distribution by Group')
    
    # 2. Horizontal violin plot
    sns.violinplot(data=clean_data, y=iv_col, x=dv_col, ax=axes[0,1])
    axes[0,1].set_title('Density Distribution by Group')
    
    # 3. Horizontal bar plot with error bars
    means = clean_data.groupby(iv_col)[dv_col].mean()
    stds = clean_data.groupby(iv_col)[dv_col].std()
    
    y_pos = range(len(means))
    bars = axes[1,0].barh(y_pos, means.values, xerr=stds.values, 
                         capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[1,0].set_yticks(y_pos)
    axes[1,0].set_yticklabels(means.index)
    axes[1,0].set_title('Means with Standard Deviation')
    axes[1,0].set_xlabel(dv_col)
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, means.values, stds.values)):
        axes[1,0].text(bar.get_width() + std_val + max(means.values) * 0.01, 
                      bar.get_y() + bar.get_height()/2, 
                      f'{mean_val:.2f}±{std_val:.2f}', 
                      ha='left', va='center', fontweight='bold')
    
    # 4. Results summary
    axes[1,1].axis('off')
    
    # Create summary text
    summary_text = f"""
ANOVA Results Summary:

F-statistic: {anova_results['f_statistic']:.4f}
p-value: {anova_results['p_value']:.4f}
Eta-squared: {anova_results['eta_squared']:.4f}
Effect size: {anova_results['eta_interpretation']}

Significance: {'Yes' if anova_results['significant'] else 'No'} (α = 0.05)

Sample size: {anova_results['n_total']}
Groups: {len(anova_results['groups'])}
"""
    
    axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ANOVA plot saved to: {save_path}")
    
    return fig

def plot_pairwise_comparisons(comparisons_df, title_prefix="", alpha=0.05, save_path=None):
    """
    Create horizontal visualization for pairwise comparisons
    
    Parameters:
    -----------
    comparisons_df : pd.DataFrame
        Results from perform_model_specific_pairwise_comparisons
    title_prefix : str
        Prefix for plot title
    alpha : float
        Significance level
    save_path : str, optional
        Full path (including filename) to save the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object containing the plots
    """
    if len(comparisons_df) == 0:
        print("No comparisons to plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(comparisons_df) * 0.8)))
    fig.suptitle(f'{title_prefix} Model-Specific Pairwise Comparisons', fontsize=16, fontweight='bold')
    
    # Create comparison labels
    comparisons_df['comparison'] = (comparisons_df['group1'].astype(str) + 
                                   ' vs ' + 
                                   comparisons_df['group2'].astype(str))
    
    # 1. Horizontal -log10(p-values) plot
    neg_log_p_uncorrected = -np.log10(comparisons_df['p_value'])
    neg_log_p_corrected = -np.log10(comparisons_df['p_corrected'])
    
    y_pos = range(len(comparisons_df))
    
    # Plot uncorrected p-values
    bars1 = axes[0].barh([y - 0.2 for y in y_pos], neg_log_p_uncorrected, 
                        height=0.4, label='Uncorrected', alpha=0.7, color='lightcoral')
    
    # Plot corrected p-values
    bars2 = axes[0].barh([y + 0.2 for y in y_pos], neg_log_p_corrected, 
                        height=0.4, label='Bonferroni Corrected', alpha=0.7, color='skyblue')
    
    # Add significance lines
    axes[0].axvline(x=-np.log10(alpha), color='red', linestyle='--', 
                   label=f'α = {alpha}', linewidth=2)
    
    axes[0].set_ylabel('Model Comparison')
    axes[0].set_xlabel('-log₁₀(p-value)')
    axes[0].set_title('Statistical Significance')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(comparisons_df['comparison'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add Cohen's d values on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        cohen_d = comparisons_df.iloc[i]['cohen_d']
        axes[0].text(bar1.get_width() + 0.05, bar1.get_y() + bar1.get_height()/2,
                    f'd={cohen_d:.2f}', ha='left', va='center', fontsize=9)
    
    # 2. Horizontal effect sizes plot
    cohen_d_values = comparisons_df['cohen_d'].abs()
    colors = ['green' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'red' 
              for d in comparisons_df['cohen_d']]
    
    bars3 = axes[1].barh(y_pos, cohen_d_values, color=colors, alpha=0.7)
    
    # Add effect size interpretation lines
    axes[1].axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='Small (0.2)')
    axes[1].axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium (0.5)')
    axes[1].axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Large (0.8)')
    
    axes[1].set_ylabel('Model Comparison')
    axes[1].set_xlabel('|Cohen\'s d|')
    axes[1].set_title('Effect Sizes')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(comparisons_df['comparison'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add effect size values on bars
    for i, bar in enumerate(bars3):
        effect_interp = comparisons_df.iloc[i]['effect_interpretation']
        axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    effect_interp, ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Pairwise comparison plot saved to: {save_path}")
    
    return fig

def cluster_comparison_analysis(df, metric='test_bal_acc_mean', subject_types=None, save_plots=False, plot_dir=None):
    """
    Compare cluster sizes (5 vs 12) across subject types with model-specific pairwise comparisons
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with model results
    metric : str
        Performance metric to analyze
    subject_types : list, optional
        Subject types to include (default: all)
    save_plots : bool
        Whether to save plots
    plot_dir : str, optional
        Directory path to save plots
    
    Returns:
    --------
    results : dict
        Complete analysis results including ANOVA, comparisons, and recommendations
    """
    print("🔍 CLUSTER SIZE COMPARISON ANALYSIS (5 vs 12)")
    print("=" * 60)
    
    # Filter data for models with cluster sizes
    cluster_data = df[df['cluster_size'].notna()].copy()
    
    if subject_types:
        cluster_data = cluster_data[cluster_data['subject_type'].isin(subject_types)]
    
    if len(cluster_data) == 0:
        return {"error": "No data with cluster sizes found"}
    
    results = {}
    
    # Overall analysis
    print(f"\n📊 Overall Analysis (All Subject Types)")
    print("-" * 40)
    
    overall_anova = perform_anova_analysis(cluster_data, metric, 'cluster_size')
    overall_comparisons = perform_model_specific_pairwise_comparisons(cluster_data, metric, 'cluster')
    
    results['overall'] = {
        'anova': overall_anova,
        'comparisons': overall_comparisons
    }
    
    print(f"ANOVA: F={overall_anova['f_statistic']:.4f}, p={overall_anova['p_value']:.4f}")
    print(f"Effect size (η²): {overall_anova['eta_squared']:.4f} ({overall_anova['eta_interpretation']})")
    print(f"Significant: {'Yes' if overall_anova['significant'] else 'No'}")
    
    # Print model-specific comparisons
    if len(overall_comparisons) > 0:
        print(f"\nModel-Specific Comparisons:")
        for _, row in overall_comparisons.iterrows():
            sig_status = "✓" if row['significant_corrected'] else "✗"
            print(f"  {sig_status} {row['group1']} vs {row['group2']}: "
                  f"p={row['p_corrected']:.4f}, d={row['cohen_d']:.3f} ({row['effect_interpretation']})")
    
    # Analysis by subject type
    results['by_subject_type'] = {}
    
    for subject_type in cluster_data['subject_type'].unique():
        print(f"\n📈 Analysis for {subject_type.upper()}")
        print("-" * 40)
        
        subset = cluster_data[cluster_data['subject_type'] == subject_type]
        
        if len(subset['cluster_size'].unique()) < 2:
            print(f"  ⚠️ Only one cluster size available for {subject_type}")
            continue
        
        anova_result = perform_anova_analysis(subset, metric, 'cluster_size')
        comparisons_result = perform_model_specific_pairwise_comparisons(subset, metric, 'cluster')
        
        results['by_subject_type'][subject_type] = {
            'anova': anova_result,
            'comparisons': comparisons_result
        }
        
        print(f"  ANOVA: F={anova_result['f_statistic']:.4f}, p={anova_result['p_value']:.4f}")
        print(f"  Effect size (η²): {anova_result['eta_squared']:.4f} ({anova_result['eta_interpretation']})")
        print(f"  Significant: {'Yes' if anova_result['significant'] else 'No'}")
        
        # Show means
        means = subset.groupby('cluster_size')[metric].agg(['mean', 'std', 'count'])
        print(f"  Descriptive statistics:")
        for cluster_size in means.index:
            mean_val = means.loc[cluster_size, 'mean']
            std_val = means.loc[cluster_size, 'std']
            n_val = means.loc[cluster_size, 'count']
            print(f"    Cluster {cluster_size}: {mean_val:.2f} ± {std_val:.2f} (n={n_val})")
    
    # Generate recommendation
    recommendation = generate_cluster_recommendation(results, metric)
    results['recommendation'] = recommendation
    
    print(f"\n🏆 RECOMMENDATION")
    print("-" * 40)
    print(recommendation)
    
    # Generate plots if requested
    if save_plots and plot_dir:
        # ANOVA plot
        anova_path = f"{plot_dir}/cluster_anova_{metric}.png"
        fig1 = plot_anova_results(
            overall_anova, cluster_data, metric, 'cluster_size',
            f"Cluster Size Comparison - {metric}", save_path=anova_path
        )
        plt.close(fig1)
        
        # Pairwise comparison plot
        if len(overall_comparisons) > 0:
            pairwise_path = f"{plot_dir}/cluster_pairwise_{metric}.png"
            fig2 = plot_pairwise_comparisons(
                overall_comparisons, f"Cluster Size - {metric}", save_path=pairwise_path
            )
            plt.close(fig2)
    
    return results

def embedding_comparison_analysis(df, metric='test_bal_acc_mean', save_plots=False, plot_dir=None):
    """
    Compare embedding vs one-hot encoding for dependent subject type only
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with model results
    metric : str
        Performance metric to analyze
    save_plots : bool
        Whether to save plots
    plot_dir : str, optional
        Directory path to save plots
    
    Returns:
    --------
    results : dict
        Complete analysis results
    """
    print("🔗 EMBEDDING vs ONE-HOT COMPARISON ANALYSIS (Dependent Only)")
    print("=" * 60)
    
    # Filter for dependent subject type only
    dependent_data = df[df['subject_type'] == 'dependent'].copy()
    
    if len(dependent_data) == 0:
        return {"error": "No dependent subject type data found"}
    
    # Check if we have both embedded and non-embedded models
    embedding_types = dependent_data['is_embedded'].unique()
    
    if len(embedding_types) < 2:
        return {"error": "Need both embedded and non-embedded models for comparison"}
    
    print(f"📊 Analyzing {len(dependent_data)} dependent models")
    print(f"Embedding types available: {embedding_types}")
    
    # Create readable labels
    dependent_data['encoding_type'] = dependent_data['is_embedded'].map({
        True: 'Embedded', 
        False: 'One-Hot'
    })
    
    # Perform analysis
    anova_result = perform_anova_analysis(dependent_data, metric, 'encoding_type')
    comparisons_result = perform_model_specific_pairwise_comparisons(dependent_data, metric, 'embedding')
    
    results = {
        'anova': anova_result,
        'comparisons': comparisons_result,
        'data_used': dependent_data
    }
    
    print(f"\nANOVA Results:")
    print(f"  F-statistic: {anova_result['f_statistic']:.4f}")
    print(f"  p-value: {anova_result['p_value']:.4f}")
    print(f"  Effect size (η²): {anova_result['eta_squared']:.4f} ({anova_result['eta_interpretation']})")
    print(f"  Significant: {'Yes' if anova_result['significant'] else 'No'}")
    
    # Print model-specific comparisons
    if len(comparisons_result) > 0:
        print(f"\nModel-Specific Comparisons:")
        for _, row in comparisons_result.iterrows():
            sig_status = "✓" if row['significant_corrected'] else "✗"
            print(f"  {sig_status} {row['group1']} vs {row['group2']}: "
                  f"p={row['p_corrected']:.4f}, d={row['cohen_d']:.3f} ({row['effect_interpretation']})")
    
    # Show descriptive statistics
    print(f"\nDescriptive Statistics:")
    means = dependent_data.groupby('encoding_type')[metric].agg(['mean', 'std', 'count'])
    for encoding_type in means.index:
        mean_val = means.loc[encoding_type, 'mean']
        std_val = means.loc[encoding_type, 'std']
        n_val = means.loc[encoding_type, 'count']
        print(f"  {encoding_type}: {mean_val:.2f} ± {std_val:.2f} (n={n_val})")
    
    # Generate recommendation
    recommendation = generate_embedding_recommendation(results, metric)
    results['recommendation'] = recommendation
    
    print(f"\n🏆 RECOMMENDATION")
    print("-" * 40)
    print(recommendation)
    
    # Generate plots if requested
    if save_plots and plot_dir:
        # ANOVA plot
        anova_path = f"{plot_dir}/encoding_anova_{metric}.png"
        fig1 = plot_anova_results(
            anova_result, dependent_data, metric, 'encoding_type',
            f"Encoding Type Comparison - {metric}", save_path=anova_path
        )
        plt.close(fig1)
        
        # Pairwise comparison plot
        if len(comparisons_result) > 0:
            pairwise_path = f"{plot_dir}/encoding_pairwise_{metric}.png"
            fig2 = plot_pairwise_comparisons(
                comparisons_result, f"Encoding Type - {metric}", save_path=pairwise_path
            )
            plt.close(fig2)
    
    return results

def kernel_scale_comparison_analysis(df, metric='test_bal_acc_mean', subject_types=None, save_plots=False, plot_dir=None):
    """
    Compare single-scale vs multiscale models
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with model results
    metric : str
        Performance metric to analyze
    subject_types : list, optional
        Subject types to include
    save_plots : bool
        Whether to save plots
    plot_dir : str, optional
        Directory path to save plots
    
    Returns:
    --------
    results : dict
        Complete analysis results
    """
    print("📏 KERNEL SCALE COMPARISON ANALYSIS (Single vs Multiscale)")
    print("=" * 60)
    
    # Create scale type column
    df_analysis = df.copy()
    df_analysis['scale_type'] = df_analysis['model_name'].apply(
        lambda x: 'Multiscale' if 'multiscale' in x.lower() else 'Single-scale'
    )
    
    # Filter for MSN and DSN models only (exclude DCN)
    scale_data = df_analysis[~df_analysis['model_name'].str.contains('dcn', case=False)].copy()
    
    if subject_types:
        scale_data = scale_data[scale_data['subject_type'].isin(subject_types)]
    
    if len(scale_data) == 0:
        return {"error": "No MSN/DSN models found for scale comparison"}
    
    # Check if we have both scale types
    scale_types = scale_data['scale_type'].unique()
    
    if len(scale_types) < 2:
        return {"error": "Need both single-scale and multiscale models for comparison"}
    
    results = {}
    
    # Overall analysis
    print(f"\n📊 Overall Analysis (All Subject Types)")
    print("-" * 40)
    
    overall_anova = perform_anova_analysis(scale_data, metric, 'scale_type')
    overall_comparisons = perform_model_specific_pairwise_comparisons(scale_data, metric, 'scale')
    
    results['overall'] = {
        'anova': overall_anova,
        'comparisons': overall_comparisons
    }
    
    print(f"ANOVA: F={overall_anova['f_statistic']:.4f}, p={overall_anova['p_value']:.4f}")
    print(f"Effect size (η²): {overall_anova['eta_squared']:.4f} ({overall_anova['eta_interpretation']})")
    print(f"Significant: {'Yes' if overall_anova['significant'] else 'No'}")
    
    # Print model-specific comparisons
    if len(overall_comparisons) > 0:
        print(f"\nModel-Specific Comparisons:")
        for _, row in overall_comparisons.iterrows():
            sig_status = "✓" if row['significant_corrected'] else "✗"
            print(f"  {sig_status} {row['group1']} vs {row['group2']}: "
                  f"p={row['p_corrected']:.4f}, d={row['cohen_d']:.3f} ({row['effect_interpretation']})")
    
    # Analysis by subject type
    results['by_subject_type'] = {}
    
    for subject_type in scale_data['subject_type'].unique():
        print(f"\n📈 Analysis for {subject_type.upper()}")
        print("-" * 40)
        
        subset = scale_data[scale_data['subject_type'] == subject_type]
        
        if len(subset['scale_type'].unique()) < 2:
            print(f"  ⚠️ Only one scale type available for {subject_type}")
            continue
        
        anova_result = perform_anova_analysis(subset, metric, 'scale_type')
        comparisons_result = perform_model_specific_pairwise_comparisons(subset, metric, 'scale')
        
        results['by_subject_type'][subject_type] = {
            'anova': anova_result,
            'comparisons': comparisons_result
        }
        
        print(f"  ANOVA: F={anova_result['f_statistic']:.4f}, p={anova_result['p_value']:.4f}")
        print(f"  Effect size (η²): {anova_result['eta_squared']:.4f} ({anova_result['eta_interpretation']})")
        print(f"  Significant: {'Yes' if anova_result['significant'] else 'No'}")
        
        # Show means
        means = subset.groupby('scale_type')[metric].agg(['mean', 'std', 'count'])
        print(f"  Descriptive statistics:")
        for scale_type in means.index:
            mean_val = means.loc[scale_type, 'mean']
            std_val = means.loc[scale_type, 'std']
            n_val = means.loc[scale_type, 'count']
            print(f"    {scale_type}: {mean_val:.2f} ± {std_val:.2f} (n={n_val})")
    
    # Generate recommendation
    recommendation = generate_scale_recommendation(results, metric)
    results['recommendation'] = recommendation
    
    print(f"\n🏆 RECOMMENDATION")
    print("-" * 40)
    print(recommendation)
    
    # Generate plots if requested
    if save_plots and plot_dir:
        # ANOVA plot
        anova_path = f"{plot_dir}/scale_anova_{metric}.png"
        fig1 = plot_anova_results(
            overall_anova, scale_data, metric, 'scale_type',
            f"Kernel Scale Comparison - {metric}", save_path=anova_path
        )
        plt.close(fig1)
        
        # Pairwise comparison plot
        if len(overall_comparisons) > 0:
            pairwise_path = f"{plot_dir}/scale_pairwise_{metric}.png"
            fig2 = plot_pairwise_comparisons(
                overall_comparisons, f"Kernel Scale - {metric}", save_path=pairwise_path
            )
            plt.close(fig2)
    
    return results

def generate_cluster_recommendation(results, metric):
    """Generate recommendation for cluster size comparison"""
    
    overall_significant = results['overall']['anova']['significant']
    
    if not overall_significant:
        return f"No significant difference found between cluster sizes 5 and 12 for {metric}. Choice can be based on computational efficiency (cluster 5 is faster)."
    
    # Check individual subject types and model-specific comparisons
    recommendations = []
    
    # Check overall model-specific comparisons
    overall_comparisons = results['overall']['comparisons']
    if len(overall_comparisons) > 0:
        significant_comparisons = overall_comparisons[overall_comparisons['significant_corrected']]
        if len(significant_comparisons) > 0:
            recommendations.append("Significant model-specific differences found:")
            for _, row in significant_comparisons.iterrows():
                better_model = row['group1'] if row['mean1'] > row['mean2'] else row['group2']
                effect_size = abs(row['cohen_d'])
                recommendations.append(f"  • {better_model} performs significantly better (d = {effect_size:.3f}, {row['effect_interpretation']})")
    
    if recommendations:
        return "\n".join(recommendations)
    else:
        return f"Overall significant difference found but no specific model pairs show significant differences after correction."

def generate_embedding_recommendation(results, metric):
    """Generate recommendation for embedding vs one-hot comparison"""
    
    significant = results['anova']['significant']
    
    if not significant:
        return f"No significant difference found between Embedded and One-Hot encoding for {metric} in dependent models. Choice can be based on other factors like interpretability or computational requirements."
    
    # Check model-specific comparisons
    comparisons = results['comparisons']
    if len(comparisons) > 0:
        significant_comparisons = comparisons[comparisons['significant_corrected']]
        if len(significant_comparisons) > 0:
            recommendations = ["Significant model-specific differences found:"]
            for _, row in significant_comparisons.iterrows():
                better_model = row['group1'] if row['mean1'] > row['mean2'] else row['group2']
                encoding_type = "Embedded" if "embedded" in better_model else "One-Hot"
                effect_size = abs(row['cohen_d'])
                recommendations.append(f"  • {encoding_type} encoding performs better for this model (d = {effect_size:.3f}, {row['effect_interpretation']})")
            return "\n".join(recommendations)
    
    return "Significant overall difference found but no specific model pairs show significant differences after correction."

def generate_scale_recommendation(results, metric):
    """Generate recommendation for scale comparison"""
    
    overall_significant = results['overall']['anova']['significant']
    
    if not overall_significant:
        return f"No significant difference found between Single-scale and Multiscale approaches for {metric}. Choice can be based on computational complexity preferences."
    
    # Check model-specific comparisons
    overall_comparisons = results['overall']['comparisons']
    if len(overall_comparisons) > 0:
        significant_comparisons = overall_comparisons[overall_comparisons['significant_corrected']]
        if len(significant_comparisons) > 0:
            recommendations = ["Significant model-specific differences found:"]
            for _, row in significant_comparisons.iterrows():
                better_model = row['group1'] if row['mean1'] > row['mean2'] else row['group2']
                scale_type = "Multiscale" if "multiscale" in better_model else "Single-scale"
                effect_size = abs(row['cohen_d'])
                recommendations.append(f"  • {scale_type} performs better for this model (d = {effect_size:.3f}, {row['effect_interpretation']})")
            return "\n".join(recommendations)
    
    return f"Overall significant difference found but no specific model pairs show significant differences after correction."

# Convenience functions for easy usage
def analyze_clusters(df, metric='test_bal_acc_mean', save_plots=False, plot_dir=None):
    """Convenience function for cluster analysis with optional plot saving"""
    return cluster_comparison_analysis(df, metric, save_plots=save_plots, plot_dir=plot_dir)

def analyze_embedding(df, metric='test_bal_acc_mean', save_plots=False, plot_dir=None):
    """Convenience function for embedding analysis with optional plot saving"""  
    return embedding_comparison_analysis(df, metric, save_plots=save_plots, plot_dir=plot_dir)

def analyze_scales(df, metric='test_bal_acc_mean', save_plots=False, plot_dir=None):
    """Convenience function for scale analysis with optional plot saving"""
    return kernel_scale_comparison_analysis(df, metric, save_plots=save_plots, plot_dir=plot_dir)

def run_comprehensive_analysis(df, metrics=['test_bal_acc_mean', 'test_f1_mean'], 
                              save_plots=True, plot_dir='analysis_plots'):
    """
    Run all three analyses for multiple metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with model results
    metrics : list
        List of metrics to analyze
    save_plots : bool
        Whether to save plots to files
    plot_dir : str
        Directory to save plots
    
    Returns:
    --------
    full_results : dict
        Complete results for all analyses
    """
    print("🚀 COMPREHENSIVE ML MODEL COMPARISON ANALYSIS")
    print("=" * 80)
    
    import os
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"📁 Plots will be saved to: {plot_dir}")
    
    full_results = {}
    
    for metric in metrics:
        print(f"\n\n🎯 ANALYZING METRIC: {metric.upper()}")
        print("=" * 80)
        
        metric_results = {}
        
        # 1. Cluster Size Comparison
        print(f"\n{'='*20} RESEARCH QUESTION 1: CLUSTER SIZE COMPARISON {'='*20}")
        cluster_results = analyze_clusters(df, metric, save_plots, f"{plot_dir}/cluster")
        metric_results['cluster_comparison'] = cluster_results
        
        # 2. Embedding vs One-Hot Comparison
        print(f"\n{'='*20} RESEARCH QUESTION 2: ENCODING TYPE COMPARISON {'='*20}")
        embedding_results = analyze_embedding(df, metric, save_plots, f"{plot_dir}/embedding")
        metric_results['embedding_comparison'] = embedding_results
        
        # 3. Kernel Scale Comparison
        print(f"\n{'='*20} RESEARCH QUESTION 3: KERNEL SCALE COMPARISON {'='*20}")
        scale_results = analyze_scales(df, metric, save_plots, f"{plot_dir}/scale")
        metric_results['scale_comparison'] = scale_results
        
        full_results[metric] = metric_results
    
    # Generate summary
    create_summary_table(full_results, metrics)
    
    return full_results

def create_summary_table(full_results, metrics):
    """
    Create a summary table of all analyses
    
    Parameters:
    -----------
    full_results : dict
        Results from comprehensive analysis
    metrics : list
        List of analyzed metrics
    """
    print(f"\n\n📋 COMPREHENSIVE SUMMARY TABLE")
    print("=" * 80)
    
    summary_data = []
    
    for metric in metrics:
        if metric not in full_results:
            continue
            
        metric_results = full_results[metric]
        
        # Cluster comparison
        if 'cluster_comparison' in metric_results and 'error' not in metric_results['cluster_comparison']:
            cluster_res = metric_results['cluster_comparison']['overall']['anova']
            summary_data.append({
                'Analysis': 'Cluster Size (5 vs 12)',
                'Metric': metric,
                'F-statistic': f"{cluster_res['f_statistic']:.4f}",
                'p-value': f"{cluster_res['p_value']:.4f}",
                'Effect Size (η²)': f"{cluster_res['eta_squared']:.4f}",
                'Interpretation': cluster_res['eta_interpretation'],
                'Significant': 'Yes' if cluster_res['significant'] else 'No'
            })
        
        # Embedding comparison
        if 'embedding_comparison' in metric_results and 'error' not in metric_results['embedding_comparison']:
            embed_res = metric_results['embedding_comparison']['anova']
            summary_data.append({
                'Analysis': 'Encoding (Embedded vs One-Hot)',
                'Metric': metric,
                'F-statistic': f"{embed_res['f_statistic']:.4f}",
                'p-value': f"{embed_res['p_value']:.4f}",
                'Effect Size (η²)': f"{embed_res['eta_squared']:.4f}",
                'Interpretation': embed_res['eta_interpretation'],
                'Significant': 'Yes' if embed_res['significant'] else 'No'
            })
        
        # Scale comparison
        if 'scale_comparison' in metric_results and 'error' not in metric_results['scale_comparison']:
            scale_res = metric_results['scale_comparison']['overall']['anova']
            summary_data.append({
                'Analysis': 'Kernel Scale (Single vs Multi)',
                'Metric': metric,
                'F-statistic': f"{scale_res['f_statistic']:.4f}",
                'p-value': f"{scale_res['p_value']:.4f}",
                'Effect Size (η²)': f"{scale_res['eta_squared']:.4f}",
                'Interpretation': scale_res['eta_interpretation'],
                'Significant': 'Yes' if scale_res['significant'] else 'No'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Count significant results
        significant_count = summary_df['Significant'].value_counts().get('Yes', 0)
        total_count = len(summary_df)
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   Total analyses: {total_count}")
        print(f"   Significant results: {significant_count} ({significant_count/total_count*100:.1f}%)")
        print(f"   Non-significant results: {total_count - significant_count} ({(total_count - significant_count)/total_count*100:.1f}%)")
    else:
        print("No valid results to summarize.")

# Example usage and testing
def test_functions():
    """Test the functions with sample data"""
    print("🧪 Testing functions...")
    
    # Create sample data similar to your structure
    np.random.seed(42)
    
    sample_data = []
    models = ['msn', 'msn_embedded', 'multiscale_msn', 'multiscale_msn_embedded', 
              'dsn_msn', 'dsn_msn_embedded', 'dsn_multiscale_msn', 'dsn_multiscale_msn_embedded']
    subject_types = ['dependent', 'independent']
    cluster_sizes = [5, 12]
    
    for subject_type in subject_types:
        for model in models:
            for cluster_size in cluster_sizes:
                if 'dcn' in model:
                    continue
                    
                is_embedded = 'embedded' in model
                performance = np.random.normal(75, 10)
                
                model_id = f"{model}_c{cluster_size}"
                
                sample_data.append({
                    'subject_type': subject_type,
                    'model_name': model,
                    'model_id': model_id,
                    'cluster_size': cluster_size,
                    'is_embedded': is_embedded,
                    'test_bal_acc_mean': performance,
                    'test_f1_mean': performance - 3
                })
    
    df = pd.DataFrame(sample_data)
    
    try:
        print("Testing cluster analysis...")
        cluster_results = analyze_clusters(df)
        print("✅ Cluster analysis passed")
        
        print("Testing embedding analysis...")
        embedding_results = analyze_embedding(df)
        print("✅ Embedding analysis passed")
        
        print("Testing scale analysis...")
        scale_results = analyze_scales(df)
        print("✅ Scale analysis passed")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

