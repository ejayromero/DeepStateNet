"""
Clean Statistical Analysis Functions for EEG Model Performance
Single-line execution for different analysis types
Updated to handle CV-based results instead of test-based results
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import colorsys
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
    subject_types = ['dependent']
    main_models_type = ['dcn', 'msn', 'dsn']
    variants_msn = ['multiscale', 'embedded']  # variants[0] = 'multiscale', variants[1] = 'embedded'
    cluster_sizes = [5, 12]
    k_folds = 5
    
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
    """Extract results from a single .npy file - Updated for CV-based results"""
    try:
        results = np.load(file_path, allow_pickle=True).item()
        
        # Updated for CV-based results structure
        # Check if we have the expected CV results format
        if 'mean_cv_balanced_accs' in results:
            n_subjects = len(results['mean_cv_balanced_accs'])
            cv_bal_accs = results['mean_cv_balanced_accs']  # These are mean CV accuracies per subject
            cv_f1s = results['mean_cv_f1s']  # These are mean CV F1 scores per subject
            
            # Extract individual fold results if available (for more detailed analysis)
            if 'cv_balanced_accuracies' in results:
                # cv_balanced_accuracies contains lists of fold results for each subject
                all_fold_bal_accs = []
                all_fold_f1s = []
                for subject_folds in results['cv_balanced_accuracies']:
                    all_fold_bal_accs.extend(subject_folds)
                for subject_folds in results['cv_f1_scores']:
                    all_fold_f1s.extend(subject_folds)
            else:
                # If fold details not available, use mean values
                all_fold_bal_accs = cv_bal_accs
                all_fold_f1s = cv_f1s
                
        else:
            # Fallback for older format or different structure
            print(f"Warning: Expected CV results format not found in {file_path}")
            print(f"Available keys: {list(results.keys())}")
            return {
                'file_path': str(file_path),
                'file_exists': False,
                'error': 'CV results format not found',
                'subject_type': model_info['subject_type'],
                'model_name': model_info['model_name'],
                'cluster_size': model_info.get('cluster_size', 'N/A'),
                'k_folds': model_info['k_folds'],
            }
        
        # Calculate summary statistics
        extracted_data = {
            # File metadata
            'file_path': str(file_path),
            'file_exists': True,
            'n_subjects': n_subjects,
            'is_cv_based': True,  # Flag to indicate this is CV-based data
            
            # Model metadata
            'subject_type': model_info['subject_type'],
            'model_name': model_info['model_name'],
            'cluster_size': model_info.get('cluster_size', 'N/A'),
            'k_folds': model_info['k_folds'],
            
            # CV performance metrics (renamed from test_ to cv_ for clarity)
            'cv_bal_acc_mean': np.mean(cv_bal_accs),
            'cv_bal_acc_std': np.std(cv_bal_accs),
            'cv_bal_acc_min': np.min(cv_bal_accs),
            'cv_bal_acc_max': np.max(cv_bal_accs),
            'cv_bal_acc_all': cv_bal_accs,  # Mean CV accuracy for each subject
            
            'cv_f1_mean': np.mean(cv_f1s),
            'cv_f1_std': np.std(cv_f1s),
            'cv_f1_min': np.min(cv_f1s),
            'cv_f1_max': np.max(cv_f1s),
            'cv_f1_all': cv_f1s,  # Mean CV F1 for each subject
            
            # All fold results (if available) - for deeper analysis
            'all_fold_bal_acc': all_fold_bal_accs,
            'all_fold_f1': all_fold_f1s,
            
            # Keep old naming for backward compatibility
            'test_bal_acc_mean': np.mean(cv_bal_accs),
            'test_bal_acc_std': np.std(cv_bal_accs),
            'test_bal_acc_min': np.min(cv_bal_accs),
            'test_bal_acc_max': np.max(cv_bal_accs),
            'test_bal_acc_all': cv_bal_accs,
            
            'test_f1_mean': np.mean(cv_f1s),
            'test_f1_std': np.std(cv_f1s),
            'test_f1_min': np.min(cv_f1s),
            'test_f1_max': np.max(cv_f1s),
            'test_f1_all': cv_f1s,
        }
        
        # Add standard deviation info if available
        if 'std_cv_balanced_accs' in results:
            extracted_data['cv_bal_acc_std_per_subject'] = results['std_cv_balanced_accs']
            extracted_data['cv_f1_std_per_subject'] = results['std_cv_f1s']
        
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
        print(f"\n🔍 Processing subject type: {subject_type}")
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
                                  f": CV Acc {result['cv_bal_acc_mean']:.2f}±{result['cv_bal_acc_std']:.2f}%")
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
        # Determine if we have embedding info
        is_embedded = 'embedded' in result['model_name']
        
        row = {
            'subject_type': result['subject_type'],
            'model_name': result['model_name'],
            'cluster_size': result['cluster_size'],
            'k_folds': result['k_folds'],
            'n_subjects': result['n_subjects'],
            'is_cv_based': result.get('is_cv_based', True),
            'is_embedded': is_embedded,
            
            # CV metrics (primary)
            'cv_bal_acc_mean': result['cv_bal_acc_mean'],
            'cv_bal_acc_std': result['cv_bal_acc_std'],
            'cv_f1_mean': result['cv_f1_mean'],
            'cv_f1_std': result['cv_f1_std'],
            
            # Keep test_ naming for backward compatibility
            'test_bal_acc_mean': result['cv_bal_acc_mean'],
            'test_bal_acc_std': result['cv_bal_acc_std'],
            'test_f1_mean': result['cv_f1_mean'],
            'test_f1_std': result['cv_f1_std'],
            
            # Individual subject scores (for plotting)
            'cv_bal_acc_all': result['cv_bal_acc_all'],
            'cv_f1_all': result['cv_f1_all'],
            'test_bal_acc_all': result['cv_bal_acc_all'],  # Backward compatibility
            'test_f1_all': result['cv_f1_all'],  # Backward compatibility
        }
        
        # Add per-subject standard deviations if available
        if 'cv_bal_acc_std_per_subject' in result:
            row['cv_bal_acc_std_per_subject'] = result['cv_bal_acc_std_per_subject']
            row['cv_f1_std_per_subject'] = result['cv_f1_std_per_subject']
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df['cluster_size'] = df['cluster_size'].astype('Int64')
    
    # Create a more readable model identifier
    df['model_id'] = df.apply(lambda row: 
        f"{row['model_name']}" + 
        (f"_c{row['cluster_size']}" if pd.notna(row['cluster_size']) else ""), 
        axis=1)
    
    # Sort by subject type and CV performance
    df = df.sort_values(['subject_type', 'cv_bal_acc_mean'], ascending=[True, False])
    
    print(f"📊 Created DataFrame with {len(df)} successful results (CV-based)")
    
    return df

def print_results_summary(df):
    """Print a comprehensive summary of all results - Updated for CV metrics"""
    
    if df.empty:
        print("❌ No results to summarize!")
        return
    
    print("🏆 COMPREHENSIVE RESULTS SUMMARY (CV-based)")
    print("=" * 60)
    
    for subject_type in df['subject_type'].unique():
        subset = df[df['subject_type'] == subject_type]
        print(f"\n📊 {subject_type.upper()} RESULTS ({len(subset)} models)")
        print("-" * 40)
        
        # Sort by CV balanced accuracy
        subset_sorted = subset.sort_values('cv_bal_acc_mean', ascending=False)
        
        print(f"{'Model':<30} {'CV Bal Acc':<15} {'CV F1':<15} {'N Subj':<8}")
        print("-" * 70)
        
        for _, row in subset_sorted.iterrows():
            print(f"{row['model_id']:<30} "
                  f"{row['cv_bal_acc_mean']:.2f}±{row['cv_bal_acc_std']:.2f}%"
                  f"{'   ':<3} "
                  f"{row['cv_f1_mean']:.2f}±{row['cv_f1_std']:.2f}%"
                  f"{'   ':<3} "
                  f"{row['n_subjects']:<8}")
    
    # Overall best performers
    print(f"\n🥇 OVERALL BEST PERFORMERS")
    print("-" * 40)
    
    best_overall = df.loc[df['cv_bal_acc_mean'].idxmax()]
    print(f"Best CV Accuracy: {best_overall['model_id']} ({best_overall['subject_type']}) - "
          f"{best_overall['cv_bal_acc_mean']:.2f}±{best_overall['cv_bal_acc_std']:.2f}%")
    
    best_f1 = df.loc[df['cv_f1_mean'].idxmax()]
    print(f"Best CV F1:       {best_f1['model_id']} ({best_f1['subject_type']}) - "
          f"{best_f1['cv_f1_mean']:.2f}±{best_f1['cv_f1_std']:.2f}%")
    
    # Model type comparison
    print(f"\n📈 MODEL TYPE COMPARISON (Available Results Only)")
    print("-" * 40)
    
    # Group by base model type
    df['base_model'] = df['model_name'].apply(lambda x: 
        'DCN' if x == 'dcn' else 
        'DSN' if x.startswith('dsn') else 'MSN')
    
    model_summary = df.groupby(['subject_type', 'base_model']).agg({
        'cv_bal_acc_mean': ['count', 'mean', 'std', 'max'],
        'cv_f1_mean': ['mean', 'std', 'max']
    }).round(2)
    
    print(model_summary)
    
    # Show embedding analysis
    print(f"\n🔗 EMBEDDING vs NON-EMBEDDING ANALYSIS")
    print("-" * 40)
    
    if df['is_embedded'].any():
        embedding_summary = df.groupby(['subject_type', 'is_embedded']).agg({
            'cv_bal_acc_mean': ['count', 'mean', 'std'],
        }).round(2)
        print(embedding_summary)
    else:
        print("No embedded models found in current results.")

# Keep all the convenience functions but update the default metrics
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

def run_complete_extraction(output_path='../Output/ica_rest_all/', save_path=None):
    """Run the complete extraction and analysis pipeline"""
    
    print("🚀 Starting complete model results extraction (CV-based)...")
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
        df.to_csv(os.path.join(save_path, 'model_results_summary_cv.csv'), index=False)
        print(f"\n💾 Results saved to '{os.path.join(save_path, 'model_results_summary_cv.csv')}'")
        
        return df, extracted_results
    else:
        print("❌ No successful extractions to analyze!")
        return None, extracted_results

# Quick access functions for specific analyses
def get_best_models_by_subject_type(df):
    """Get the best performing model for each subject type"""
    if df.empty:
        return None
    
    best_models = df.loc[df.groupby('subject_type')['cv_bal_acc_mean'].idxmax()]
    return best_models[['subject_type', 'model_id', 'cv_bal_acc_mean', 'cv_bal_acc_std']]

def compare_cluster_sizes(df):
    """Compare performance across different cluster sizes"""
    if df.empty:
        return None
    
    # Filter models that have cluster sizes
    clustered_df = df[df['cluster_size'] != 'N/A']
    
    if clustered_df.empty:
        return None
    
    comparison = clustered_df.groupby(['subject_type', 'model_name', 'cluster_size']).agg({
        'cv_bal_acc_mean': 'first',
        'cv_bal_acc_std': 'first'
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
    print("📄 EXPERIMENT STATUS OVERVIEW")
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
            print(f"   🔍 Missing models:")
            for r in missing:
                model_id = f"{r['model_name']}" + (f"_c{r['cluster_size']}" if r['cluster_size'] != 'N/A' else "")
                print(f"      - {model_id}")
    
    print(f"\n📄 Overall Progress:")
    total_results = len(extracted_results)
    total_completed = len([r for r in extracted_results if r.get('file_exists', False)])
    print(f"   {total_completed}/{total_results} experiments completed ({total_completed/total_results*100:.1f}%)")
    
    return {
        'total_expected': total_results,
        'total_completed': total_completed,
        'completion_rate': total_completed/total_results*100
    }

# ============================== Plotting Functions =================================
# All plotting functions remain the same since they use test_bal_acc_mean which we map to cv_bal_acc_mean


def get_model_order_key(model_key):
    """
    Generate a sorting key for consistent model ordering:
    1. DCN first
    2. MSN models (by cluster size, then by type)
    3. DSN models (by cluster size, then by type)
    
    Args:
        model_key: String like 'dcn', 'msn_c5', 'dsn_msn_c12', etc.
    
    Returns:
        Tuple for sorting
    """
    if model_key == 'dcn':
        return (0, 0, 0, '')  # DCN always first
    
    # Extract parts from model key
    parts = model_key.split('_')
    
    # Get cluster size (always at the end after 'c')
    cluster_size = 0
    if parts[-1].startswith('c'):
        try:
            cluster_size = int(parts[-1][1:])  # Extract number after 'c'
        except ValueError:
            cluster_size = 0
    
    # Determine model family and type
    model_name = '_'.join(parts[:-1])  # Everything except cluster size
    
    # MSN models (group 1)
    if model_name == 'msn':
        return (1, cluster_size, 0, 'msn')
    elif model_name == 'msn_embedded':
        return (1, cluster_size, 1, 'msn_embedded')
    elif model_name == 'multiscale_msn':
        return (1, cluster_size, 2, 'multiscale_msn')
    elif model_name == 'multiscale_msn_embedded':
        return (1, cluster_size, 3, 'multiscale_msn_embedded')
    
    # DSN models (group 2)
    elif model_name == 'dsn_msn':
        return (2, cluster_size, 0, 'dsn_msn')
    elif model_name == 'dsn_msn_embedded':
        return (2, cluster_size, 1, 'dsn_msn_embedded')
    elif model_name == 'dsn_multiscale_msn':
        return (2, cluster_size, 2, 'dsn_multiscale_msn')
    elif model_name == 'dsn_multiscale_msn_embedded':
        return (2, cluster_size, 3, 'dsn_multiscale_msn_embedded')
    
    # Fallback for unknown models
    else:
        return (3, cluster_size, 0, model_name)


def sort_models_consistently(model_keys):
    """
    Sort model keys in consistent order: DCN, MSN variants, DSN variants
    
    Args:
        model_keys: List of model key strings
    
    Returns:
        Sorted list of model keys
    """
    return sorted(model_keys, key=get_model_order_key)


def generate_n_spectrum_palettes(n_palettes, n_color_shades=3, 
                                 light_sat=0.45, light_val=0.87,
                                 medium_sat=0.72, medium_val=0.73,
                                 dark_sat=0.82, dark_val=0.53):
    """Generate n distinct color palettes using full spectrum division"""
    
    # Split full spectrum into n_palettes+1 parts, take first n_palettes
    hue_step = 1.0 / (n_palettes + 1)
    base_hues = [i * hue_step for i in range(n_palettes)]
    
    palettes = {}
    
    for i, base_hue in enumerate(base_hues):
        colors = []
        
        if n_color_shades == 3:
            # Light, medium, dark
            light_rgb = colorsys.hsv_to_rgb(base_hue, light_sat, light_val)
            medium_rgb = colorsys.hsv_to_rgb(base_hue, medium_sat, medium_val)
            dark_rgb = colorsys.hsv_to_rgb(base_hue, dark_sat, dark_val)
            colors = [light_rgb, medium_rgb, dark_rgb]
        else:
            # Handle other cases if needed
            sat_values = np.linspace(light_sat, dark_sat, n_color_shades)
            val_values = np.linspace(light_val, dark_val, n_color_shades)
            
            for sat, val in zip(sat_values, val_values):
                rgb = colorsys.hsv_to_rgb(base_hue, sat, val)
                colors.append(rgb)
        
        palettes[f'model_{i+1}'] = colors
    
    return palettes

def get_model_key(row):
    """Helper function to create consistent model keys"""
    if row['model_name'] == 'dcn':
        return 'dcn'
    else:
        return f"{row['model_name']}_c{row['cluster_size']}"

def get_all_unique_models(df):
    """Get all unique models across all subject types in consistent order"""
    all_model_keys = []
    
    for _, row in df.iterrows():
        model_key = get_model_key(row)
        all_model_keys.append(model_key)
    
    # Get unique model keys preserving order, then sort consistently
    unique_model_keys = list(dict.fromkeys(all_model_keys))
    return sort_models_consistently(unique_model_keys)

def get_model_colors(df):
    """
    Generate color palettes for each subject type based on available models
    Creates consistent colors across subject types with different shades
    
    Args:
        df: DataFrame with results containing 'model_name', 'cluster_size', 'subject_type'
        
    Returns:
        dict: {subject_type: {model_key: color}}
    """
    
    # Get all unique models across all subject types
    all_unique_models = get_all_unique_models(df)
    n_models = len(all_unique_models)
    
    print(f"Found {n_models} unique models: {all_unique_models}")
    
    # Generate n_models color palettes with 3 shades each
    color_palettes = generate_n_spectrum_palettes(n_palettes=n_models, n_color_shades=3)
    
    # Map each model to its palette
    model_to_palette = {}
    for i, model_key in enumerate(all_unique_models):
        palette_key = f'model_{i+1}'
        model_to_palette[model_key] = color_palettes[palette_key]
    
    # Create the output in the SAME FORMAT as your original function
    color_dict = {
        'independent': {},    # Light shade (index 0)
        'dependent': {},  # Medium shade (index 1)
        'adaptive': {}      # Dark shade (index 2)
    }
    
    # For each subject type, add colors for models that exist in that subject type
    for subject_type in ['independent', 'dependent', 'adaptive']:
        subject_df = df[df['subject_type'] == subject_type]
        
        # Get model keys that exist in this subject type
        existing_model_keys = []
        for _, row in subject_df.iterrows():
            model_key = get_model_key(row)
            existing_model_keys.append(model_key)
        
        unique_existing = list(dict.fromkeys(existing_model_keys))
        
        # Assign appropriate shade based on subject type
        shade_index = {'dependent': 1, 'independent': 0, 'adaptive': 2}[subject_type]
        
        for model_key in unique_existing:
            if model_key in model_to_palette:
                color_dict[subject_type][model_key] = model_to_palette[model_key][shade_index]
    
    return color_dict

def get_color_for_row(row, color_dict):
    """Get the appropriate color for a dataframe row - UPDATED to match new format"""
    model_key = get_model_key(row)
    subject_type = row['subject_type']
    
    # Return the color for this model+subject combination
    return color_dict[subject_type].get(model_key, 'gray')

def merge_plot_styles(custom_style=None):
    """
    Merge custom plot style with default style
    
    Args:
        custom_style: Dictionary with custom style overrides
        
    Returns:
        Merged style dictionary
    """
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
        'bar': {
            'capsize': 3,
            'alpha': 0.8,
            'edgecolor': 'black',
            'linewidth': 0.5
        },
        'violin': {
            'inner_kws': {'box_width': 10, 'whis_width': 2, 'color': 'dimgray'},
            'cut': 0,
            'alpha': 0.8
        },
        'figure': {
            'context': 'talk',
            'figsize_2x3': (24, 16),
            'figsize_1x3': (24, 8),
            'figsize_comparison': (15, 12),
            'figsize_violin': (30, 20),
            'figsize_violin_grouped': (30, 20),
            'suptitle_fontsize': 16,
            'title_fontsize': 14,
            'label_fontsize': 12,
            'legend_fontsize': 10,
            'ticks_fontsize': 10,
            'bar_label_fontsize': 9,
            'grid_alpha': 0.3
        },
        'colors': {
            'text_color': 'black',
            'grid_color': 'gray',
            'fallback_color': 'gray'
        }
    }
    
    if custom_style is None:
        return DEFAULT_PLOT_STYLE
    
    # Deep merge custom style with default
    merged_style = DEFAULT_PLOT_STYLE.copy()
    for key, value in custom_style.items():
        if key in merged_style and isinstance(value, dict):
            merged_style[key].update(value)
        else:
            merged_style[key] = value
    
    return merged_style


def plot_model_comparison(df, score_types=['test_bal_acc', 'test_f1'], metric_type='CV', my_plot_style=None):
    """
    Main function that creates 2x2 subplots (top row: balanced accuracy, bottom row: f1)
    Updated to use CV results but keep same interface for backward compatibility
    
    Args:
        df: DataFrame with model results
        score_types: List of score types to plot (mapped to CV results)
        metric_type: Type of metric being plotted (default changed to 'CV')
        my_plot_style: Custom plot style dictionary to override defaults
    """
    if df is None:
        return
    
    # Merge custom style with defaults
    style = merge_plot_styles(my_plot_style)
    
    # Get the color dictionary
    color_dict = get_model_colors(df)
    subject_types = df['subject_type'].unique()
    n_subjects = len(subject_types)
    
    with sns.plotting_context(style['figure']['context']):
        fig, axes = plt.subplots(2, n_subjects, figsize=style['figure']['figsize_comparison'])
        fig.suptitle(f'Model Performance Comparison in {metric_type}', 
                    fontsize=style['figure']['suptitle_fontsize'], fontweight='bold')
        
        # If only one subject type, make axes 2D
        if n_subjects == 1:
            axes = axes.reshape(2, 1)
        
        # Plot balanced accuracy on top row (use CV results)
        for j, subject_type in enumerate(subject_types):
            plot_single_metric(df, axes[0, j], score_types[0], color_dict, subject_type, style)
        
        # Plot F1 on bottom row (use CV results)
        for j, subject_type in enumerate(subject_types):
            plot_single_metric(df, axes[1, j], score_types[1], color_dict, subject_type, style)
        
        plt.tight_layout()
        plt.show()


def plot_single_metric(df, ax, score_type, color_dict, subject_type, style):
    """
    Plot a single metric for a single subject type on given axis
    Updated to handle CV results and maintain consistent ordering
    """
    # Filter data for this subject type
    subset = df[df['subject_type'] == subject_type].copy()
    
    # Create model keys and sort them consistently
    model_keys = [get_model_key(row) for _, row in subset.iterrows()]
    unique_model_keys = list(dict.fromkeys(model_keys))  # Preserve order
    sorted_model_keys = sort_models_consistently(unique_model_keys)
    
    # Reorder subset DataFrame to match sorted model keys
    subset_list = []
    for model_key in sorted_model_keys:
        # Find the row with this model key
        for _, row in subset.iterrows():
            if get_model_key(row) == model_key:
                subset_list.append(row)
                break
    
    subset_ordered = pd.DataFrame(subset_list)
    colors = [color_dict[subject_type][key] for key in sorted_model_keys]
    
    # Map test_ metrics to cv_ metrics (backward compatibility)
    metric_mapping = {
        'test_bal_acc': 'cv_bal_acc',
        'test_f1': 'cv_f1'
    }
    
    # Get the base metric name
    if score_type in metric_mapping:
        base_metric = metric_mapping[score_type]
    else:
        base_metric = score_type
    
    # Get mean and std columns
    mean_col = f'{base_metric}_mean'
    std_col = f'{base_metric}_std'
    
    # Create bar plot with colors and style
    x_pos = range(len(subset_ordered))
    bar_style = style['bar'].copy()
    bars = ax.bar(x_pos, subset_ordered[mean_col], 
                  yerr=subset_ordered[std_col], 
                  color=colors,
                  **bar_style)
    
    # Set labels based on score type
    if 'f1' in score_type.lower():
        ylabel = 'CV F1 Score (%)'
    elif 'bal_acc' in score_type.lower():
        ylabel = 'CV Balanced Accuracy (%)'
    else:
        ylabel = f'{score_type.replace("_", " ").title()} (%)'
    
    ax.set_title(f'{subject_type.title()} Models', 
                fontsize=style['figure']['title_fontsize'], fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=style['figure']['label_fontsize'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subset_ordered['model_id'], rotation=45, ha='right', 
                      fontsize=style['figure']['ticks_fontsize'])
    ax.tick_params(axis='x', bottom=True)

    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, subset_ordered[mean_col], subset_ordered[std_col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.5,
                f'{mean_val:.1f}%', ha='center', va='bottom', 
                fontsize=style['figure']['bar_label_fontsize'])

    ax.set_ylim(0, max(df[mean_col] + df[std_col]) + 5)


def plot_model_violin_comparison(df, transpose=False, my_plot_style=None):
    """
    Create violin plots comparing all models across subject types
    Updated to use consistent ordering and CV results
    """
    # Merge custom style with defaults
    style = merge_plot_styles(my_plot_style)
    
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
        
        # Add balanced accuracy data (using CV results)
        for score in row['cv_bal_acc_all']:
            plot_data.append({
                'subject_type': row['subject_type'],
                'model': model_short,
                'model_full': row['model_name'],
                'model_key': model_key,  # Add model key for color mapping
                'metric': 'Balanced Accuracy',
                'score': score,
                'cluster_size': row['cluster_size']
            })
        
        # Add F1 data (using CV results)
        for score in row['cv_f1_all']:
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
    subject_types = ['dependent']
    metrics = ['Balanced Accuracy', 'F1 Macro']
    
    # Create subplots based on transpose setting
    if transpose:
        n_rows = len(subject_types)
        n_cols = len(metrics)
        fig_size = style['figure']['figsize_violin']
    else:
        n_rows = len(metrics)
        n_cols = len(subject_types)
        fig_size = style['figure']['figsize_violin']
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharey=True)
    fig.suptitle('Model Performance Comparison Across Subject Types (CV Results)', 
                fontsize=style['figure']['suptitle_fontsize'], y=0.98)
    
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
                # Get unique models for this subject type and sort consistently
                unique_model_keys = subset['model_key'].unique()
                sorted_model_keys = sort_models_consistently(unique_model_keys)
                
                # Create mapping from model key to short name
                key_to_short = {}
                for _, row_data in subset.iterrows():
                    if row_data['model_key'] not in key_to_short:
                        key_to_short[row_data['model_key']] = row_data['model']
                
                # Get ordered model short names
                ordered_model_names = [key_to_short[key] for key in sorted_model_keys]
                
                # Reorder the subset data to match the sorted order
                subset_ordered = []
                for model_key in sorted_model_keys:
                    model_data = subset[subset['model_key'] == model_key]
                    subset_ordered.append(model_data)
                subset_ordered = pd.concat(subset_ordered, ignore_index=True)
                
                # Get colors using the get_model_colors function
                subject_colors = color_dict.get(subject_type, {})
                colors = []
                for model_key in sorted_model_keys:
                    color = subject_colors.get(model_key, style['colors']['fallback_color'])
                    colors.append(color)
                
                # Create violin plot with custom style and ordered data
                violin_style = style['violin'].copy()
                sns.violinplot(data=subset_ordered, x='model', y='score', ax=ax, 
                            palette=colors, order=ordered_model_names, **violin_style)
            
            # Formatting with custom style
            ax.set_ylim(-3, 103)
            ax.set_title(f'{subject_type.title()}' if row == 0 else '', 
                        fontsize=style['figure']['title_fontsize'])
            ax.set_xlabel('Model Type' if row == 1 else '', 
                         fontsize=style['figure']['label_fontsize'])
            ax.set_ylabel(f'{metric} (%)' if col == 0 else '', 
                         fontsize=style['figure']['label_fontsize'])
            
            if not subset.empty:
                ax.set_xticks(range(len(ordered_model_names)))
                ax.set_xticklabels(ordered_model_names, rotation=45, ha='right', 
                                 fontsize=style['figure']['ticks_fontsize'])
            
            ax.tick_params(axis='x', rotation=45, colors=style['colors']['text_color'], bottom=True)
            ax.grid(True, axis='x', alpha=style['figure']['grid_alpha'])
            
            # Remove x-axis labels for top row
            if row == 0:
                ax.set_xlabel('')
    
    plt.tight_layout()
    return fig


def plot_violin_comparison(df, save_path=None, my_plot_style=None):
    """Create and optionally save violin comparison plot"""
    # Merge custom style with defaults
    style = merge_plot_styles(my_plot_style)
    
    with sns.plotting_context(style['figure']['context']):
        fig = plot_model_violin_comparison(df, my_plot_style=my_plot_style)
        
        if save_path:
            fig.savefig(os.path.join(save_path, 'model_violin_comparison_cv.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"💾 Violin plot saved to {save_path}")
        
        plt.close()
        return fig


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