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
    
    # Create a more readable model identifier
    df['model_id'] = df.apply(lambda row: 
        f"{row['model_name']}" + 
        (f"_c{row['cluster_size']}" if row['cluster_size'] != 'N/A' else ""), 
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
                             palette=colors, cut=0)
            
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
                                     ax=ax, palette=colors, cut=0)
                        
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
                                     ax=ax, palette=colors, cut=0, order=unique_combos)
                        
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

# ================================== Statistics =====================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, friedmanchisquare, shapiro, levene, kruskal
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.multitest import multipletests
import warnings
from itertools import combinations
import os

def get_model_components(model_name, cluster_size):
    """Extract model family, architecture type, and cluster info"""
    
    # Model family
    if model_name == 'dcn':
        family = 'DCN'
        architecture = 'base'
        has_clusters = False
    elif model_name.startswith('dsn_'):
        family = 'DSN'
        architecture = model_name.replace('dsn_', '')
        has_clusters = True
    else:  # MSN variants
        family = 'MSN'
        architecture = model_name
        has_clusters = True
    
    # Architecture details
    is_multiscale = 'multiscale' in architecture
    is_embedded = 'embedded' in architecture
    
    return {
        'family': family,
        'architecture': architecture,
        'is_multiscale': is_multiscale,
        'is_embedded': is_embedded,
        'has_clusters': has_clusters,
        'cluster_size': cluster_size if has_clusters else None
    }

def get_model_short_name(model_name, cluster_size):
    """Get shortened model names for better plot readability"""
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
    
    base_name = model_mapping.get(model_name, model_name)
    
    if model_name != 'dcn':
        return f"{base_name}_C{cluster_size}"
    return base_name

def prepare_analysis_data(df):
    """Convert DataFrame to analysis-ready format with proper grouping"""
    analysis_data = []
    
    for _, row in df.iterrows():
        model_short = get_model_short_name(row['model_name'], row['cluster_size'])
        model_components = get_model_components(row['model_name'], row['cluster_size'])
        
        n_subjects = len(row['test_bal_acc_all'])
        
        # Balanced accuracy data
        for i, score in enumerate(row['test_bal_acc_all']):
            analysis_data.append({
                'subject_id': f'S{i:03d}',
                'subject_type': row['subject_type'],
                'model_name': row['model_name'],
                'model_short': model_short,
                'cluster_size': row['cluster_size'],
                'metric': 'balanced_accuracy',
                'score': score * 100,  # Convert to percentage
                **model_components
            })
        
        # F1 score data
        for i, score in enumerate(row['test_f1_all']):
            analysis_data.append({
                'subject_id': f'S{i:03d}',
                'subject_type': row['subject_type'],
                'model_name': row['model_name'],
                'model_short': model_short,
                'cluster_size': row['cluster_size'],
                'metric': 'f1_score',
                'score': score * 100,  # Convert to percentage
                **model_components
            })
    
    return pd.DataFrame(analysis_data)

class RigorousEEGAnalyzer:
    """Statistically rigorous EEG model analyzer with proper corrections"""
    
    def __init__(self, data_df, base_alpha=0.05):
        self.data = data_df
        self.base_alpha = base_alpha
        
        # Define research questions and their alpha corrections
        self.research_questions = [
            'models_within_subjects',
            'subjects_within_models', 
            'model_families',
            'cluster_effects',
            'architecture_effects'
        ]
        
        # Family-wise error correction across research questions
        self.n_questions = len(self.research_questions)
        self.alpha_per_family = self.base_alpha / self.n_questions
        
        print(f"🎯 STATISTICAL RIGOR SETTINGS:")
        print(f"   Base α = {self.base_alpha}")
        print(f"   Number of research question families = {self.n_questions}")
        print(f"   Corrected α per family = {self.alpha_per_family:.4f}")
        print(f"   Post-hoc correction: False Discovery Rate (FDR)")
        
        self.results = {}
        self.print_data_summary()
    
    def print_data_summary(self):
        """Print detailed data availability summary"""
        print(f"\n📊 DATA AVAILABILITY SUMMARY")
        print("="*80)
        
        # Create comprehensive availability matrix
        all_models = sorted(self.data['model_short'].unique())
        all_subjects = sorted(self.data['subject_type'].unique())
        
        print(f"\n📋 Model Availability Matrix:")
        header = f"{'Model':<25}"
        for subject in all_subjects:
            header += f" {subject.capitalize():<12}"
        print(header)
        print("-" * len(header))
        
        availability_matrix = {}
        for model in all_models:
            row = f"{model:<25}"
            availability_matrix[model] = {}
            for subject in all_subjects:
                has_model = len(self.data[(self.data['model_short'] == model) & 
                                        (self.data['subject_type'] == subject)]) > 0
                status = "✅" if has_model else "❌"
                row += f" {status:<12}"
                availability_matrix[model][subject] = has_model
            print(row)
        
        # Summary statistics
        print(f"\n📊 Data Summary:")
        for subject in all_subjects:
            subject_data = self.data[self.data['subject_type'] == subject]
            n_models = len(subject_data['model_short'].unique())
            n_families = len(subject_data['family'].unique())
            print(f"   {subject}: {n_models} models, {n_families} families")
        
        # Training completeness analysis
        complete_models = [m for m in all_models 
                          if all(availability_matrix[m].get(s, False) for s in all_subjects)]
        partial_models = [m for m in all_models if m not in complete_models 
                         and any(availability_matrix[m].get(s, False) for s in all_subjects)]
        
        print(f"\n🔄 Training Status:")
        print(f"   ✅ Complete across all subjects: {len(complete_models)} models")
        if complete_models:
            print(f"      {complete_models}")
        print(f"   🔄 Partial training: {len(partial_models)} models")
        print(f"   📊 Analysis power: {len(complete_models)}/{len(all_models)} models ready for cross-subject analysis")
    
    def perform_robust_statistical_test(self, data_subset, comparison_name, alpha):
        """Perform robust statistical test with proper effect size calculation"""
        
        groups = data_subset['group'].unique()
        n_groups = len(groups)
        
        if n_groups < 2:
            return {'success': False, 'reason': 'insufficient_groups', 'n_groups': n_groups}
        
        # Prepare group data
        group_data = []
        group_names = []
        group_stats = {}
        
        for group in groups:
            group_scores = data_subset[data_subset['group'] == group]['score'].values
            if len(group_scores) > 0:
                group_data.append(group_scores)
                group_names.append(group)
                group_stats[group] = {
                    'n': len(group_scores),
                    'mean': np.mean(group_scores),
                    'median': np.median(group_scores),
                    'std': np.std(group_scores, ddof=1),
                    'q25': np.percentile(group_scores, 25),
                    'q75': np.percentile(group_scores, 75)
                }
        
        if len(group_data) < 2:
            return {'success': False, 'reason': 'insufficient_data'}
        
        # Check if all groups have same number of subjects
        group_lengths = [len(g) for g in group_data]
        all_same_length = len(set(group_lengths)) == 1
        
        result = {
            'comparison_name': comparison_name,
            'n_groups': len(group_data),
            'group_names': group_names,
            'group_stats': group_stats,
            'all_same_length': all_same_length,
            'alpha_used': alpha
        }
        
        try:
            if all_same_length and len(group_data) >= 3:
                # Friedman test (non-parametric repeated measures)
                stat, p_value = friedmanchisquare(*group_data)
                test_type = 'Friedman Test'
                
                # Effect size (Kendall's W)
                n = len(group_data[0])  # number of subjects
                k = len(group_data)     # number of groups
                w = stat / (n * (k - 1))  # Kendall's W
                effect_size = w
                effect_size_name = "Kendall's W"
                
            elif len(group_data) >= 3:
                # Kruskal-Wallis test (non-parametric independent samples)
                stat, p_value = kruskal(*group_data)
                test_type = 'Kruskal-Wallis Test'
                
                # Effect size (eta-squared approximation)
                total_n = sum(group_lengths)
                eta_squared = (stat - len(group_data) + 1) / (total_n - len(group_data))
                effect_size = max(0, min(eta_squared, 1))  # Bound between 0 and 1
                effect_size_name = "η² (approx)"
                
            elif len(group_data) == 2:
                # Two groups comparison
                if all_same_length:
                    # Wilcoxon signed-rank test
                    stat, p_value = stats.wilcoxon(group_data[0], group_data[1])
                    test_type = 'Wilcoxon Signed-Rank Test'
                    
                    # Effect size (rank-biserial correlation)
                    n = len(group_data[0])
                    r = 1 - (2 * stat) / (n * (n + 1))
                    effect_size = abs(r)
                    effect_size_name = "r (rank-biserial)"
                    
                else:
                    # Mann-Whitney U test
                    stat, p_value = stats.mannwhitneyu(group_data[0], group_data[1], 
                                                      alternative='two-sided')
                    test_type = 'Mann-Whitney U Test'
                    
                    # Effect size (rank-biserial correlation)
                    n1, n2 = len(group_data[0]), len(group_data[1])
                    r = 1 - (2 * stat) / (n1 * n2)
                    effect_size = abs(r)
                    effect_size_name = "r (rank-biserial)"
            
            result.update({
                'test_type': test_type,
                'statistic': stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'effect_size_name': effect_size_name,
                'success': True
            })
            
        except Exception as e:
            result.update({
                'success': False,
                'reason': f'Statistical test failed: {str(e)}'
            })
            return result
        
        # Interpret effect size
        result['effect_interpretation'] = self.interpret_effect_size(effect_size, effect_size_name)
        
        # Post-hoc analysis if significant and multiple groups
        if result['success'] and result['p_value'] < alpha and len(group_data) > 2:
            result['posthoc'] = self.perform_fdr_posthoc(group_data, group_names, alpha)
        
        return result
    
    def interpret_effect_size(self, effect_size, effect_size_name):
        """Interpret effect size magnitude"""
        if pd.isna(effect_size):
            return 'unknown'
        
        if 'W' in effect_size_name or 'η²' in effect_size_name:
            # For eta-squared and Kendall's W
            if effect_size < 0.01:
                return 'negligible'
            elif effect_size < 0.06:
                return 'small'
            elif effect_size < 0.14:
                return 'medium'
            else:
                return 'large'
        else:
            # For correlation-based measures (r)
            if effect_size < 0.1:
                return 'negligible'
            elif effect_size < 0.3:
                return 'small'
            elif effect_size < 0.5:
                return 'medium'
            else:
                return 'large'
    
    def perform_fdr_posthoc(self, group_data, group_names, alpha):
        """Perform post-hoc tests with FDR correction"""
        
        n_comparisons = len(list(combinations(range(len(group_data)), 2)))
        pairwise_results = []
        p_values = []
        
        # Perform all pairwise comparisons
        for i, j in combinations(range(len(group_data)), 2):
            group1_data = group_data[i]
            group2_data = group_data[j]
            group1_name = group_names[i]
            group2_name = group_names[j]
            
            # Choose appropriate test
            if len(group1_data) == len(group2_data):
                # Paired comparison
                try:
                    stat, p_val = stats.wilcoxon(group1_data, group2_data)
                    test_type = 'Wilcoxon'
                    
                    # Effect size
                    n = len(group1_data)
                    r = 1 - (2 * stat) / (n * (n + 1))
                    effect_size = abs(r)
                    
                except:
                    # Fallback to Mann-Whitney if Wilcoxon fails
                    stat, p_val = stats.mannwhitneyu(group1_data, group2_data, 
                                                   alternative='two-sided')
                    test_type = 'Mann-Whitney'
                    n1, n2 = len(group1_data), len(group2_data)
                    r = 1 - (2 * stat) / (n1 * n2)
                    effect_size = abs(r)
            else:
                # Independent comparison
                stat, p_val = stats.mannwhitneyu(group1_data, group2_data, 
                                               alternative='two-sided')
                test_type = 'Mann-Whitney'
                n1, n2 = len(group1_data), len(group2_data)
                r = 1 - (2 * stat) / (n1 * n2)
                effect_size = abs(r)
            
            # Store results
            median_diff = np.median(group1_data) - np.median(group2_data)
            
            pairwise_results.append({
                'comparison': f'{group1_name} vs {group2_name}',
                'test_type': test_type,
                'statistic': stat,
                'p_value': p_val,
                'effect_size': effect_size,
                'effect_interpretation': self.interpret_effect_size(effect_size, 'r'),
                'group1_median': np.median(group1_data),
                'group2_median': np.median(group2_data),
                'median_diff': median_diff,
                'group1_name': group1_name,
                'group2_name': group2_name
            })
            
            p_values.append(p_val)
        
        # Apply FDR correction
        if p_values:
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method='fdr_bh'
            )
            
            # Update results with FDR correction
            for i, result in enumerate(pairwise_results):
                result['p_fdr'] = p_corrected[i]
                result['significant_fdr'] = rejected[i]
                result['alpha_fdr'] = alpha  # FDR controls this automatically
        
        return {
            'pairwise_comparisons': pairwise_results,
            'n_comparisons': n_comparisons,
            'fdr_alpha': alpha,
            'n_significant': sum(rejected) if p_values else 0
        }
    
    def analyze_models_within_subjects(self, metric='balanced_accuracy'):
        """Q1: Compare all models within each subject type"""
        print(f"\n{'='*80}")
        print(f"Q1: WHICH MODEL PERFORMS BEST WITHIN EACH SUBJECT TYPE? ({metric.upper()})")
        print(f"Alpha (family-corrected) = {self.alpha_per_family:.4f}")
        print(f"{'='*80}")
        
        metric_data = self.data[self.data['metric'] == metric]
        results = {}
        
        for subject_type in metric_data['subject_type'].unique():
            print(f"\n🔍 Analyzing {subject_type} subjects...")
            
            subset = metric_data[metric_data['subject_type'] == subject_type].copy()
            subset['group'] = subset['model_short']
            
            available_models = subset['model_short'].unique()
            print(f"   Models available: {len(available_models)}")
            print(f"   {list(available_models)}")
            
            if len(available_models) < 2:
                print(f"   ⚠️  Insufficient models for comparison (need ≥2)")
                results[subject_type] = {
                    'success': False, 
                    'reason': 'insufficient_models',
                    'available_models': list(available_models)
                }
                continue
            
            result = self.perform_robust_statistical_test(
                subset, f"{subject_type}_models", self.alpha_per_family
            )
            results[subject_type] = result
            
            if result['success']:
                significance = "SIGNIFICANT" if result['p_value'] < self.alpha_per_family else "NOT SIGNIFICANT"
                print(f"   ✅ {result['test_type']}")
                print(f"      Statistic = {result['statistic']:.3f}")
                print(f"      p-value = {result['p_value']:.6f}")
                print(f"      Result: {significance}")
                print(f"      Effect size: {result['effect_size_name']} = {result['effect_size']:.3f} ({result['effect_interpretation']})")
                
                if 'posthoc' in result:
                    n_sig = result['posthoc']['n_significant']
                    n_total = result['posthoc']['n_comparisons']
                    print(f"   🔍 Post-hoc (FDR): {n_sig}/{n_total} significant pairwise differences")
                    
                    # Show top significant differences
                    significant_pairs = [p for p in result['posthoc']['pairwise_comparisons'] 
                                       if p['significant_fdr']]
                    if significant_pairs:
                        # Sort by effect size
                        significant_pairs.sort(key=lambda x: x['effect_size'], reverse=True)
                        print(f"   🏆 Top significant differences:")
                        for pair in significant_pairs[:5]:  # Show top 5
                            print(f"      • {pair['comparison']}: "
                                  f"Δ={pair['median_diff']:.2f}%, "
                                  f"r={pair['effect_size']:.3f} ({pair['effect_interpretation']}), "
                                  f"p_FDR={pair['p_fdr']:.4f}")
            else:
                print(f"   ❌ Analysis failed: {result.get('reason', 'Unknown error')}")
        
        return results
    
    def analyze_subjects_within_models(self, metric='balanced_accuracy'):
        """Q2: Compare subject types within each model"""
        print(f"\n{'='*80}")
        print(f"Q2: DO MODELS PERFORM DIFFERENTLY ACROSS SUBJECT TYPES? ({metric.upper()})")
        print(f"Alpha (family-corrected) = {self.alpha_per_family:.4f}")
        print(f"{'='*80}")
        
        metric_data = self.data[self.data['metric'] == metric]
        results = {}
        
        # Analyze each model across subject types
        for model_short in sorted(metric_data['model_short'].unique()):
            print(f"\n🔍 Analyzing {model_short} across subject types...")
            
            subset = metric_data[metric_data['model_short'] == model_short].copy()
            subset['group'] = subset['subject_type']
            
            available_subjects = subset['subject_type'].unique()
            print(f"   Subject types available: {list(available_subjects)}")
            
            if len(available_subjects) < 2:
                print(f"   ⚠️  Insufficient subject types (need ≥2) - training in progress")
                results[model_short] = {
                    'success': False,
                    'reason': 'insufficient_subjects',
                    'available_subjects': list(available_subjects)
                }
                continue
            
            result = self.perform_robust_statistical_test(
                subset, f"{model_short}_subjects", self.alpha_per_family
            )
            results[model_short] = result
            
            if result['success']:
                significance = "SIGNIFICANT" if result['p_value'] < self.alpha_per_family else "NOT SIGNIFICANT"
                print(f"   ✅ {result['test_type']}")
                print(f"      Statistic = {result['statistic']:.3f}")
                print(f"      p-value = {result['p_value']:.6f}")
                print(f"      Result: {significance}")
                print(f"      Effect size: {result['effect_size_name']} = {result['effect_size']:.3f} ({result['effect_interpretation']})")
                
                if 'posthoc' in result:
                    significant_pairs = [p for p in result['posthoc']['pairwise_comparisons'] 
                                       if p['significant_fdr']]
                    if significant_pairs:
                        print(f"   🏆 Significant subject type differences:")
                        for pair in significant_pairs:
                            print(f"      • {pair['comparison']}: "
                                  f"Δ={pair['median_diff']:.2f}%, p_FDR={pair['p_fdr']:.4f}")
            else:
                print(f"   ❌ Analysis failed: {result.get('reason', 'Unknown error')}")
        
        return results
    
    def analyze_model_families(self, metric='balanced_accuracy'):
        """Q3: Compare DCN vs MSN vs DSN families overall"""
        print(f"\n{'='*80}")
        print(f"Q3: WHICH MODEL FAMILY PERFORMS BEST OVERALL? ({metric.upper()})")
        print("Pooling across all subject types and model variants")
        print(f"Alpha (family-corrected) = {self.alpha_per_family:.4f}")
        print(f"{'='*80}")
        
        metric_data = self.data[self.data['metric'] == metric].copy()
        metric_data['group'] = metric_data['family']
        
        families = metric_data['family'].unique()
        print(f"\nAvailable families: {list(families)}")
        
        # Check data balance
        for family in families:
            family_data = metric_data[metric_data['family'] == family]
            n_obs = len(family_data)
            n_subjects = len(family_data['subject_type'].unique())
            print(f"   {family}: {n_obs} observations, {n_subjects} subject types")
        
        if len(families) < 2:
            print("⚠️  Insufficient model families for comparison")
            return {'model_families': {'success': False, 'reason': 'insufficient_families'}}
        
        result = self.perform_robust_statistical_test(
            metric_data, "model_families", self.alpha_per_family
        )
        
        print(f"\n🔍 Family comparison results:")
        if result['success']:
            significance = "SIGNIFICANT" if result['p_value'] < self.alpha_per_family else "NOT SIGNIFICANT"
            print(f"   ✅ {result['test_type']}")
            print(f"      Statistic = {result['statistic']:.3f}")
            print(f"      p-value = {result['p_value']:.6f}")
            print(f"      Result: {significance}")
            print(f"      Effect size: {result['effect_size_name']} = {result['effect_size']:.3f} ({result['effect_interpretation']})")
            
            if 'posthoc' in result:
                significant_pairs = [p for p in result['posthoc']['pairwise_comparisons'] 
                                   if p['significant_fdr']]
                if significant_pairs:
                    print(f"   🏆 Significant family differences:")
                    for pair in significant_pairs:
                        better_family = pair['group1_name'] if pair['median_diff'] > 0 else pair['group2_name']
                        print(f"      • {pair['comparison']}: "
                              f"Δ={abs(pair['median_diff']):.2f}% ({better_family} better), "
                              f"p_FDR={pair['p_fdr']:.4f}")
        else:
            print(f"   ❌ Analysis failed: {result.get('reason', 'Unknown error')}")
        
        return {'model_families': result}
    
    def analyze_cluster_effects(self, metric='balanced_accuracy'):
        """Q4: Compare 5 vs 12 clusters within MSN and DSN families"""
        print(f"\n{'='*80}")
        print(f"Q4: WHICH CLUSTER SIZE PERFORMS BETTER? ({metric.upper()})")
        print(f"Alpha (family-corrected) = {self.alpha_per_family:.4f}")
        print(f"{'='*80}")
        
        metric_data = self.data[self.data['metric'] == metric]
        results = {}
        
        for family in ['MSN', 'DSN']:
            print(f"\n🔍 Analyzing {family} cluster size effects...")
            
            family_data = metric_data[metric_data['family'] == family].copy()
            
            if len(family_data) == 0:
                print(f"   ⚠️  No {family} models available yet")
                results[f'{family}_clusters'] = {'success': False, 'reason': 'no_family_data'}
                continue
            
            family_data['group'] = 'C' + family_data['cluster_size'].astype(str)
            cluster_sizes = family_data['cluster_size'].unique()
            
            print(f"   Available cluster sizes: {list(cluster_sizes)}")
            
            for cluster in cluster_sizes:
                cluster_subset = family_data[family_data['cluster_size'] == cluster]
                n_models = len(cluster_subset['model_short'].unique())
                n_subjects = len(cluster_subset['subject_type'].unique())
                print(f"      C{cluster}: {n_models} models, {n_subjects} subject types")
            
            if len(cluster_sizes) < 2:
                print(f"   ⚠️  Need both C5 and C12 for comparison")
                results[f'{family}_clusters'] = {'success': False, 'reason': 'insufficient_clusters'}
                continue
            
            result = self.perform_robust_statistical_test(
                family_data, f"{family}_clusters", self.alpha_per_family
            )
            results[f'{family}_clusters'] = result
            
            if result['success']:
                significance = "SIGNIFICANT" if result['p_value'] < self.alpha_per_family else "NOT SIGNIFICANT"
                print(f"   ✅ {result['test_type']}")
                print(f"      p-value = {result['p_value']:.6f} ({significance})")
                print(f"      Effect size: {result['effect_size_name']} = {result['effect_size']:.3f} ({result['effect_interpretation']})")
                
                # Determine which cluster size is better
                if significance == "SIGNIFICANT" and 'posthoc' in result:
                    for pair in result['posthoc']['pairwise_comparisons']:
                        if 'C5' in pair['comparison'] and 'C12' in pair['comparison']:
                            better_cluster = pair['group1_name'] if pair['median_diff'] > 0 else pair['group2_name']
                            print(f"   🏆 {better_cluster} performs better (Δ={abs(pair['median_diff']):.2f}%)")
            else:
                print(f"   ❌ Analysis failed: {result.get('reason', 'Unknown error')}")
        
        return results
    
    def analyze_architecture_effects(self, metric='balanced_accuracy'):
        """Q5: Compare architectural variants"""
        print(f"\n{'='*80}")
        print(f"Q5: WHICH ARCHITECTURAL VARIANTS PERFORM BETTER? ({metric.upper()})")
        print(f"Alpha (family-corrected) = {self.alpha_per_family:.4f}")
        print(f"{'='*80}")
        
        metric_data = self.data[self.data['metric'] == metric]
        results = {}
        
        # Multiscale effect analysis
        print(f"\n🔍 Analyzing multiscale effects...")
        multiscale_data = metric_data[metric_data['family'].isin(['MSN', 'DSN'])].copy()
        
        if len(multiscale_data) > 0:
            multiscale_data['group'] = multiscale_data['is_multiscale'].map({
                True: 'multiscale', False: 'base'
            })
            
            multiscale_counts = multiscale_data.groupby('group').size()
            print(f"   Data: {dict(multiscale_counts)}")
            
            if len(multiscale_data['group'].unique()) >= 2:
                result = self.perform_robust_statistical_test(
                    multiscale_data, "multiscale_effect", self.alpha_per_family
                )
                results['multiscale'] = result
                
                if result['success']:
                    significance = "SIGNIFICANT" if result['p_value'] < self.alpha_per_family else "NOT SIGNIFICANT"
                    print(f"   ✅ Multiscale vs Base: {result['test_type']}")
                    print(f"      p-value = {result['p_value']:.6f} ({significance})")
                    print(f"      Effect size: {result['effect_size_name']} = {result['effect_size']:.3f} ({result['effect_interpretation']})")
                    
                    if 'posthoc' in result:
                        for pair in result['posthoc']['pairwise_comparisons']:
                            better_arch = pair['group1_name'] if pair['median_diff'] > 0 else pair['group2_name']
                            print(f"   🏆 {better_arch} performs better (Δ={abs(pair['median_diff']):.2f}%)")
            else:
                print(f"   ⚠️  Insufficient multiscale data for comparison")
                results['multiscale'] = {'success': False, 'reason': 'insufficient_data'}
        
        # Embedding effect analysis
        print(f"\n🔍 Analyzing embedding effects...")
        embedding_data = metric_data[metric_data['family'].isin(['MSN', 'DSN'])].copy()
        
        if len(embedding_data) > 0:
            embedding_data['group'] = embedding_data['is_embedded'].map({
                True: 'embedded', False: 'base'
            })
            
            embedding_counts = embedding_data.groupby('group').size()
            print(f"   Data: {dict(embedding_counts)}")
            
            if len(embedding_data['group'].unique()) >= 2:
                result = self.perform_robust_statistical_test(
                    embedding_data, "embedding_effect", self.alpha_per_family
                )
                results['embedding'] = result
                
                if result['success']:
                    significance = "SIGNIFICANT" if result['p_value'] < self.alpha_per_family else "NOT SIGNIFICANT"
                    print(f"   ✅ Embedded vs Base: {result['test_type']}")
                    print(f"      p-value = {result['p_value']:.6f} ({significance})")
                    print(f"      Effect size: {result['effect_size_name']} = {result['effect_size']:.3f} ({result['effect_interpretation']})")
                    
                    if 'posthoc' in result:
                        for pair in result['posthoc']['pairwise_comparisons']:
                            better_arch = pair['group1_name'] if pair['median_diff'] > 0 else pair['group2_name']
                            print(f"   🏆 {better_arch} performs better (Δ={abs(pair['median_diff']):.2f}%)")
            else:
                print(f"   ⚠️  Insufficient embedding data for comparison")
                results['embedding'] = {'success': False, 'reason': 'insufficient_data'}
        
        return results
    
    def run_comprehensive_analysis(self):
        """Run all analyses with proper statistical rigor"""
        print("\n🚀 RUNNING STATISTICALLY RIGOROUS EEG MODEL ANALYSIS")
        print("="*80)
        print("📊 Statistical Framework:")
        print(f"   • Family-wise error rate control across {self.n_questions} research questions")
        print(f"   • Base α = {self.base_alpha}, Corrected α per family = {self.alpha_per_family:.4f}")
        print(f"   • Post-hoc correction: False Discovery Rate (Benjamini-Hochberg)")
        print(f"   • Effect sizes: Context-appropriate measures with interpretations")
        print("="*80)
        
        self.results = {}
        
        for metric in ['balanced_accuracy', 'f1_score']:
            print(f"\n🎯 ANALYZING {metric.upper().replace('_', ' ')}")
            print("="*80)
            
            self.results[metric] = {
                'models_within_subjects': self.analyze_models_within_subjects(metric),
                'subjects_within_models': self.analyze_subjects_within_models(metric),
                'model_families': self.analyze_model_families(metric),
                'cluster_effects': self.analyze_cluster_effects(metric),
                'architecture_effects': self.analyze_architecture_effects(metric)
            }
        
        return self.results

def create_comprehensive_horizontal_plots(analyzer, output_path=None):
    """Create horizontal bar plots for all analysis results"""
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    
    results = analyzer.results
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 5 subplots (2x3 grid, with last subplot spanning)
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    fig.delaxes(axes[2, 1])  # Remove the bottom-right subplot
    
    # Color scheme
    colors = {
        'significant': '#2E8B57',      # Dark green
        'not_significant': '#DC143C',  # Crimson  
        'no_data': '#808080',          # Gray
        'pending': '#FFA500'           # Orange
    }
    
    # Analysis configurations with subplot positions
    analysis_configs = [
        ('models_within_subjects', 'Q1: Best Models within Subject Types'),
        ('subjects_within_models', 'Q2: Subject Effects within Models'),
        ('model_families', 'Q3: Model Family Comparison'),
        ('cluster_effects', 'Q4: Cluster Size Effects'), 
        ('architecture_effects', 'Q5: Architecture Effects')
    ]
    
    # Subplot positions
    subplot_positions = [(0,0), (0,1), (1,0), (1,1), (2,0)]
    
    for analysis_idx, (analysis_key, analysis_title) in enumerate(analysis_configs):
        row, col = subplot_positions[analysis_idx]
        ax = axes[row, col]
        
        # Combine data from both metrics for plotting
        combined_plot_data = []
        
        for metric in ['balanced_accuracy', 'f1_score']:
            metric_title = metric.replace('_', ' ').title()
            
            if analysis_key in results[metric]:
                analysis_results = results[metric][analysis_key]
                
                # Extract data for plotting
                for key, result in analysis_results.items():
                    if isinstance(result, dict):
                        if result.get('success', False):
                            p_val = result.get('p_value', 1.0)
                            effect_size = result.get('effect_size', 0)
                            effect_size_name = result.get('effect_size_name', '')
                            is_significant = p_val < analyzer.alpha_per_family
                            
                            # Handle very small p-values and calculate -log10(p)
                            if p_val <= 0:
                                neg_log_p = 50  # Cap for extremely small p-values
                            elif p_val < 1e-50:
                                neg_log_p = 50
                            else:
                                neg_log_p = -np.log10(p_val)
                            
                            combined_plot_data.append({
                                'name': f"{key} ({metric_title})",
                                'neg_log_p': neg_log_p,
                                'effect_size': effect_size,
                                'effect_size_name': effect_size_name,
                                'is_significant': is_significant,
                                'p_value': p_val,
                                'metric': metric,
                                'comparison': key,
                                'status': 'significant' if is_significant else 'not_significant'
                            })
                        else:
                            reason = result.get('reason', 'unknown')
                            if 'insufficient' in reason or 'training' in reason:
                                status = 'pending'
                            else:
                                status = 'no_data'
                            
                            combined_plot_data.append({
                                'name': f"{key} ({metric_title})",
                                'neg_log_p': 0.05,  # Very small value for visibility
                                'effect_size': 0,
                                'effect_size_name': '',
                                'is_significant': False,
                                'p_value': 1.0,
                                'metric': metric,
                                'comparison': key,
                                'status': status
                            })
        
        if combined_plot_data:
            # Sort by significance and effect size
            combined_plot_data.sort(key=lambda x: (x['is_significant'], x['effect_size']), reverse=True)
            
            names = [item['name'] for item in combined_plot_data]
            neg_log_p_values = [item['neg_log_p'] for item in combined_plot_data]
            statuses = [item['status'] for item in combined_plot_data]
            effect_sizes = [item['effect_size'] for item in combined_plot_data]
            effect_size_names = [item['effect_size_name'] for item in combined_plot_data]
            
            # Adjust figure height for Q2 (many models) - increase spacing
            if analysis_key == 'subjects_within_models' and len(names) > 8:
                # Increase y-axis spacing for crowded plots
                ax.set_ylim(-0.5, len(names) - 0.5)
            
            # Create horizontal bar plot
            bar_colors = [colors[status] for status in statuses]
            
            bars = ax.barh(range(len(names)), neg_log_p_values, 
                          color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Calculate appropriate x-axis limits BEFORE adding annotations
            data_values = [x for x in neg_log_p_values if x > 0.1]  # Exclude training bars
            if data_values:
                max_data_val = max(data_values)
                x_max = min(max_data_val * 1.3, 50)  # 30% padding, cap at 50
            else:
                x_max = 5
            
            # Ensure minimum reasonable range
            if x_max < 3:
                x_max = 5
            
            ax.set_xlim(0, x_max)
            
            # Add significance threshold line
            alpha_line = -np.log10(analyzer.alpha_per_family)
            if alpha_line <= x_max:
                ax.axvline(x=alpha_line, color='red', linestyle='--', linewidth=2,
                          label=f'α = {analyzer.alpha_per_family:.4f}')
            
            # Add effect size annotations with proper positioning
            for i, (bar, effect, effect_name, item) in enumerate(zip(bars, effect_sizes, effect_size_names, combined_plot_data)):
                if item['status'] in ['significant', 'not_significant'] and effect > 0:
                    # Position annotation based on bar width and plot limits
                    bar_end = bar.get_width()
                    text_x = min(bar_end + x_max * 0.02, x_max * 0.95)  # 2% padding or 95% of plot width
                    
                    # Create effect size label with name
                    if effect_name:
                        effect_label = f"{effect_name}={effect:.3f}"
                    else:
                        effect_label = f"ES={effect:.3f}"
                    
                    ax.text(text_x, bar.get_y() + bar.get_height()/2,
                           effect_label, 
                           va='center', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                           
                elif item['status'] == 'pending':
                    # Position training text better
                    text_x = x_max * 0.3  # Place at 30% of plot width
                    ax.text(text_x, bar.get_y() + bar.get_height()/2,
                           'Training...', 
                           va='center', ha='center', fontsize=8, style='italic', color='orange',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Formatting
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('-log₁₀(p-value)', fontsize=11)
            ax.set_title(analysis_title, fontsize=12, fontweight='bold')
            
            # Create meaningful x-ticks based on the range
            if x_max <= 5:
                x_ticks = [0, 1, 2, 3, 4, 5]
                x_labels = ['1.0', '0.1', '0.01', '0.001', '0.0001', '0.00001']
            elif x_max <= 10:
                x_ticks = [0, 2, 4, 6, 8, 10]
                x_labels = ['1.0', '0.01', '0.0001', '1e-6', '1e-8', '1e-10']
            elif x_max <= 20:
                x_ticks = [0, 5, 10, 15, 20]
                x_labels = ['1.0', '1e-5', '1e-10', '1e-15', '1e-20']
            else:
                x_ticks = [0, 10, 20, 30, 40, 50]
                x_labels = ['1.0', '1e-10', '1e-20', '1e-30', '1e-40', '1e-50']
            
            # Only show ticks that are within our range
            valid_ticks = [tick for tick in x_ticks if tick <= x_max]
            valid_labels = [x_labels[i] for i, tick in enumerate(x_ticks) if tick <= x_max]
            
            ax.set_xticks(valid_ticks)
            ax.set_xticklabels(valid_labels, fontsize=9)
            
            # Add legend in upper right corner for plots with significance threshold
            if alpha_line <= x_max and any(item['status'] in ['significant', 'not_significant'] for item in combined_plot_data):
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            
            ax.grid(axis='x', alpha=0.3)
            
        else:
            # No data available
            ax.text(0.5, 0.5, 'No data available\n(Training in progress)', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, style='italic')
            ax.set_title(analysis_title, fontsize=12, fontweight='bold')
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_xticklabels(['1.0', '0.1', '0.01', '0.001', '0.0001', '0.00001'])
    
    # Add overall title
    fig.suptitle('EEG Model Performance Analysis - Statistical Results\n' + 
                f'Family-wise α = {analyzer.base_alpha}, Per-family α = {analyzer.alpha_per_family:.4f}, Post-hoc: FDR',
                fontsize=16, fontweight='bold', y=0.96)
    
    # Create custom legend at bottom
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors['significant'], label='Significant'),
        plt.Rectangle((0,0),1,1, facecolor=colors['not_significant'], label='Not Significant'),
        plt.Rectangle((0,0),1,1, facecolor=colors['pending'], label='Training in Progress'),
        plt.Rectangle((0,0),1,1, facecolor=colors['no_data'], label='No Data')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, 0.02), fontsize=11)
    
    # Add explanation text
    explanation_text = (
        'X-axis: -log₁₀(p-value) - Higher bars = more significant results\n'
        'Effect sizes: η² = eta-squared, W = Kendall\'s W, r = rank-biserial correlation\n'
        'Interpretation: <0.1=negligible, 0.1-0.3=small, 0.3-0.5=medium, >0.5=large'
    )
    
    fig.text(0.02, 0.08, explanation_text, fontsize=10, style='italic',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92)
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'rigorous_eeg_analysis_horizontal_final.png'),
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(output_path, 'rigorous_eeg_analysis_horizontal_final.pdf'),
                   bbox_inches='tight', facecolor='white')
    
    plt.show()
    return fig

def print_final_summary(analyzer):
    """Print publication-ready summary of results"""
    print("\n" + "="*80)
    print("📊 PUBLICATION-READY SUMMARY")
    print("="*80)
    
    print(f"Statistical Framework:")
    print(f"  • Family-wise error rate: α = {analyzer.base_alpha}")
    print(f"  • Number of research question families: {analyzer.n_questions}")
    print(f"  • Bonferroni-corrected α per family: {analyzer.alpha_per_family:.4f}")
    print(f"  • Post-hoc correction: False Discovery Rate (Benjamini-Hochberg)")
    print(f"  • Effect size measures: Context-appropriate (Kendall's W, η², rank-biserial r)")
    
    for metric in ['balanced_accuracy', 'f1_score']:
        print(f"\n{metric.upper().replace('_', ' ')} RESULTS:")
        
        for analysis_type, analysis_name in [
            ('models_within_subjects', 'Q1: Model Comparison within Subject Types'),
            ('subjects_within_models', 'Q2: Subject Type Effects within Models'),
            ('model_families', 'Q3: Model Family Comparison'),
            ('cluster_effects', 'Q4: Cluster Size Effects'),
            ('architecture_effects', 'Q5: Architecture Effects')
        ]:
            
            if analysis_type in analyzer.results[metric]:
                results = analyzer.results[metric][analysis_type]
                
                significant_count = 0
                total_count = 0
                pending_count = 0
                
                for key, result in results.items():
                    if isinstance(result, dict):
                        if result.get('success', False):
                            total_count += 1
                            if result.get('p_value', 1) < analyzer.alpha_per_family:
                                significant_count += 1
                        elif 'insufficient' in result.get('reason', '') or 'training' in result.get('reason', ''):
                            pending_count += 1
                
                status = ""
                if total_count > 0:
                    status += f"{significant_count}/{total_count} significant"
                if pending_count > 0:
                    status += f", {pending_count} pending training"
                if not status:
                    status = "No data"
                
                print(f"  {analysis_name}: {status}")
    
    print(f"\n💡 Training Recommendations:")
    print(f"  • Priority: Train MSN/DSN models for independent/adaptive subjects")
    print(f"  • This will enable cross-subject analysis for all model families")
    print(f"  • Full architectural analysis requires complete training matrix")

def run_rigorous_eeg_analysis(df, output_path=None, base_alpha=0.05):
    """
    Run statistically rigorous EEG model analysis
    
    Args:
        df: DataFrame with model results
        output_path: Optional path to save results
        base_alpha: Base significance level (default: 0.05)
    
    Returns:
        dict: Complete analysis results with proper corrections
    """
    
    print("📊 Preparing EEG model data for rigorous statistical analysis...")
    analysis_data = prepare_analysis_data(df)
    
    print(f"✅ Prepared {len(analysis_data)} observations")
    print(f"📈 Subjects: {len(analysis_data['subject_id'].unique())}")
    print(f"🧠 Subject types: {analysis_data['subject_type'].unique()}")
    print(f"🤖 Models: {len(analysis_data['model_short'].unique())}")
    
    # Initialize rigorous analyzer
    analyzer = RigorousEEGAnalyzer(analysis_data, base_alpha=base_alpha)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Create horizontal visualizations
    print("\n📊 Creating comprehensive horizontal plots...")
    fig = create_comprehensive_horizontal_plots(analyzer, output_path)
    
    # Print final summary
    print_final_summary(analyzer)
    
    if output_path:
        # Save detailed results
        import pickle
        results_package = {
            'results': results,
            'analyzer_settings': {
                'base_alpha': base_alpha,
                'alpha_per_family': analyzer.alpha_per_family,
                'n_questions': analyzer.n_questions,
                'correction_method': 'Bonferroni (family-wise) + FDR (post-hoc)'
            },
            'data_summary': {
                'n_observations': len(analysis_data),
                'n_subjects': len(analysis_data['subject_id'].unique()),
                'subject_types': list(analysis_data['subject_type'].unique()),
                'n_models': len(analysis_data['model_short'].unique()),
                'models': list(analysis_data['model_short'].unique())
            }
        }
        
        with open(os.path.join(output_path, 'rigorous_eeg_analysis_complete.pkl'), 'wb') as f:
            pickle.dump(results_package, f)
        
        # Save summary tables
        summary_data = []
        for metric in ['balanced_accuracy', 'f1_score']:
            for analysis_type, analysis_results in results[metric].items():
                for key, result in analysis_results.items():
                    if isinstance(result, dict) and result.get('success', False):
                        summary_data.append({
                            'Metric': metric,
                            'Analysis': analysis_type,
                            'Comparison': key,
                            'Test': result.get('test_type', 'Unknown'),
                            'Statistic': result.get('statistic', np.nan),
                            'P_Value': result.get('p_value', np.nan),
                            'Alpha_Used': result.get('alpha_used', analyzer.alpha_per_family),
                            'Significant': result.get('p_value', 1) < result.get('alpha_used', analyzer.alpha_per_family),
                            'Effect_Size': result.get('effect_size', np.nan),
                            'Effect_Size_Name': result.get('effect_size_name', ''),
                            'Effect_Interpretation': result.get('effect_interpretation', ''),
                            'N_Groups': result.get('n_groups', np.nan)
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(output_path, 'rigorous_analysis_summary.csv'), index=False)
        
        print(f"\n💾 Complete results saved to {output_path}")
    
    return {
        'results': results,
        'analyzer': analyzer,
        'figure': fig,
        'data': analysis_data
    }

# Example usage:
# results = run_rigorous_eeg_analysis(df, output_path='./rigorous_analysis_results/')