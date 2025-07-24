"""
Clean Statistical Analysis Functions for EEG Model Performance
Single-line execution for different analysis types
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, ttest_rel, shapiro, levene
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def extract_results_from_test_data(all_test_results, metric='accuracy'):
    """
    Extract and organize results from comprehensive test results structure
    FIXED for your actual data structure
    
    Parameters:
    -----------
    all_test_results : dict
        The comprehensive test results dictionary from your testing
    metric : str, default='accuracy'
        Metric to extract: 'accuracy' or 'f1_macro'
    
    Returns:
    --------
    dict
        Organized results in format expected by statistical functions
    """
    
    # Initialize results dictionary
    organized_results = {}
    
    # FIXED: Use the actual keys from your data structure
    metric_mapping = {
        'accuracy': 'test_accuracy',  # YOUR DATA USES 'test_accuracy' (singular)
        'f1_macro': 'f1_macro'        # YOUR DATA USES 'f1_macro' (correct)
    }
    
    if metric not in metric_mapping:
        raise ValueError(f"Metric must be one of {list(metric_mapping.keys())}")
    
    metric_key = metric_mapping[metric]
    
    # Process each subject type
    for subject_type, models in all_test_results.items():
        # Clean subject type name (remove 'subject_' prefix)
        clean_subject_type = subject_type.replace('subject_', '')
        organized_results[clean_subject_type] = {}
        
        # Process each model type
        for model_name, model_results in models.items():
            if model_results is None:
                continue
                
            # Extract test results
            test_results = model_results.get('test_results', [])
            if not test_results:
                continue
            
            # Handle different model types
            if model_name == 'Multimodal' and clean_subject_type == 'adaptive':
                # For adaptive multimodal, ONLY use fine-tuned results, label as "Multimodal"
                finetuned_scores = []
                
                for subject_result in test_results:
                    # Look for fine-tuned scores only
                    if metric == 'accuracy':
                        if 'finetuned_test_accuracy' in subject_result:
                            finetuned_scores.append(subject_result['finetuned_test_accuracy'])
                    else:  # f1_macro
                        if 'finetuned_f1_macro' in subject_result:
                            finetuned_scores.append(subject_result['finetuned_f1_macro'])
                
                # Store as just "Multimodal" (no base, no "(Fine-tuned)" label)
                if finetuned_scores:
                    organized_results[clean_subject_type]['Multimodal'] = finetuned_scores
                    
            else:
                # Standard model extraction - use the correct key
                scores = []
                for subject_result in test_results:
                    if metric_key in subject_result:
                        scores.append(subject_result[metric_key])
                
                if scores:
                    # Use raw for DeepConvNet to match original naming
                    clean_model_name = 'raw' if model_name == 'DeepConvNet' else model_name
                    organized_results[clean_subject_type][clean_model_name] = scores
    
    return organized_results

def extract_results_for_subject_comparison(all_test_results, metric='accuracy'):
    """
    Extract data organized by model type for subject comparisons
    INCLUDES adaptive multimodal using BASE scores for fair comparison
    """
    
    # Initialize results organized by model type
    subject_comparison_data = {}
    
    # FIXED: Use the actual keys from your data structure
    metric_mapping = {
        'accuracy': 'test_accuracy',
        'f1_macro': 'f1_macro'
    }
    
    metric_key = metric_mapping[metric]
    
    # Process each subject type
    for subject_type, models in all_test_results.items():
        clean_subject_type = subject_type.replace('subject_', '')
        
        for model_name, model_results in models.items():
            if model_results is None:
                continue
                
            test_results = model_results.get('test_results', [])
            if not test_results:
                continue
            
            # Initialize model in results if not exists
            clean_model_name = 'raw' if model_name == 'DeepConvNet' else model_name
            if clean_model_name not in subject_comparison_data:
                subject_comparison_data[clean_model_name] = {}
            
            # Extract scores based on model and subject type
            scores = []
            
            if model_name == 'Multimodal' and clean_subject_type == 'adaptive':
                # For adaptive multimodal, use BASE scores for subject comparison
                for subject_result in test_results:
                    if metric == 'accuracy':
                        if 'base_test_accuracy' in subject_result:
                            scores.append(subject_result['base_test_accuracy'])
                    else:  # f1_macro
                        if 'base_f1_macro' in subject_result:
                            scores.append(subject_result['base_f1_macro'])
            else:
                # Standard extraction for all other cases
                for subject_result in test_results:
                    if metric_key in subject_result:
                        scores.append(subject_result[metric_key])
            
            if scores:
                subject_comparison_data[clean_model_name][clean_subject_type] = scores
    
    # Remove any models that don't have data from multiple subject types
    subject_comparison_data = {
        model: subjects for model, subjects in subject_comparison_data.items() 
        if len(subjects) > 1
    }
    
    return subject_comparison_data

# ============================================================================
# UPDATED ANALYSIS FUNCTIONS
# ============================================================================

def analyze_accuracy_models(all_test_results, output_path=None):
    """
    ONE LINE: Analyze accuracy differences between models within each subject type
    """
    if output_path:
        full_path = os.path.join(output_path, 'accuracy_models')
        os.makedirs(full_path, exist_ok=True)
    else:
        full_path = None
    
    organized_data = extract_results_from_test_data(all_test_results, 'accuracy')
    
    print("🎯 ACCURACY ANALYSIS: Comparing models within each subject type")
    statistical_results = perform_statistical_tests(organized_data, type_wise="model", output_path=full_path)
    plot_results = create_statistical_plots(organized_data, type_wise="model", output_path=full_path)
    
    return {
        'statistical_results': statistical_results,
        'organized_data': organized_data,
        'plot_results': plot_results
    }

def analyze_f1_models(all_test_results, output_path=None):
    """
    ONE LINE: Analyze F1-macro differences between models within each subject type
    """
    if output_path:
        full_path = os.path.join(output_path, 'f1_models')
        os.makedirs(full_path, exist_ok=True)
    else:
        full_path = None
    
    organized_data = extract_results_from_test_data(all_test_results, 'f1_macro')
    
    print("🎯 F1-MACRO ANALYSIS: Comparing models within each subject type")
    statistical_results = perform_statistical_tests(organized_data, type_wise="model", output_path=full_path)
    plot_results = create_statistical_plots(organized_data, type_wise="model", output_path=full_path)
    
    return {
        'statistical_results': statistical_results,
        'organized_data': organized_data,
        'plot_results': plot_results
    }

def analyze_accuracy_subjects(all_test_results, output_path=None):
    """
    ONE LINE: Analyze accuracy differences between subject types within each model
    INCLUDES adaptive multimodal using base scores for fair comparison
    """
    if output_path:
        full_path = os.path.join(output_path, 'accuracy_subjects')
        os.makedirs(full_path, exist_ok=True)
    else:
        full_path = None
    
    organized_data = extract_results_for_subject_comparison(all_test_results, 'accuracy')
    
    print("🎯 ACCURACY ANALYSIS: Comparing subject types within each model")
    print("📝 Note: Adaptive multimodal uses base scores for fair comparison")
    statistical_results = perform_statistical_tests(organized_data, type_wise="subject", output_path=full_path)
    plot_results = create_statistical_plots(organized_data, type_wise="subject", output_path=full_path)
    
    return {
        'statistical_results': statistical_results,
        'organized_data': organized_data,
        'plot_results': plot_results
    }

def analyze_f1_subjects(all_test_results, output_path=None):
    """
    ONE LINE: Analyze F1-macro differences between subject types within each model
    INCLUDES adaptive multimodal using base scores for fair comparison
    """
    if output_path:
        full_path = os.path.join(output_path, 'f1_subjects')
        os.makedirs(full_path, exist_ok=True)
    else:
        full_path = None
    
    organized_data = extract_results_for_subject_comparison(all_test_results, 'f1_macro')
    
    print("🎯 F1-MACRO ANALYSIS: Comparing subject types within each model")
    print("📝 Note: Adaptive multimodal uses base scores for fair comparison")
    statistical_results = perform_statistical_tests(organized_data, type_wise="subject", output_path=full_path)
    plot_results = create_statistical_plots(organized_data, type_wise="subject", output_path=full_path)
    
    return {
        'statistical_results': statistical_results,
        'organized_data': organized_data,
        'plot_results': plot_results
    }

def analyze_all_combinations(all_test_results, output_path=None):
    """
    ONE LINE: Run all four analysis combinations
    """
    print("🚀 Running all statistical analyses...")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    
    results = {}
    
    # Run all four analyses
    results['accuracy_models'] = analyze_accuracy_models(all_test_results, output_path)
    results['f1_models'] = analyze_f1_models(all_test_results, output_path)
    results['accuracy_subjects'] = analyze_accuracy_subjects(all_test_results, output_path)
    results['f1_subjects'] = analyze_f1_subjects(all_test_results, output_path)
    
    # Create comprehensive summary
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE SUMMARY")
    print("="*80)
    
    summary_data = []
    for analysis_name, analysis_results in results.items():
        if analysis_results and analysis_results['statistical_results']:
            for key, stats in analysis_results['statistical_results'].items():
                summary_data.append({
                    'Analysis': analysis_name,
                    'Comparison': key,
                    'F_Statistic': stats['anova_f_stat'],
                    'P_Value': stats['anova_p_value'],
                    'Significant': stats['significant'],
                    'Effect_Size': stats.get('eta_squared', np.nan)
                })
    
    if summary_data:
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.round(4).to_string(index=False))
        
        if output_path:
            summary_df.to_csv(os.path.join(output_path, 'all_analyses_summary.csv'), index=False)
    
    return results

def analyze_custom(all_test_results, metric='accuracy', comparison_type='model', output_path=None):
    """
    ONE LINE: Custom analysis with specified metric and comparison type
    
    Parameters:
    -----------
    metric : str
        'accuracy' or 'f1_macro'
    comparison_type : str
        'model' or 'subject'
    """
    if output_path:
        full_path = os.path.join(output_path, f'{metric}_{comparison_type}')
        os.makedirs(full_path, exist_ok=True)
    else:
        full_path = None
    
    # Use appropriate extraction based on comparison type
    if comparison_type == 'subject':
        organized_data = extract_results_for_subject_comparison(all_test_results, metric)
        print("📝 Note: Adaptive multimodal uses base scores for fair comparison")
    else:
        organized_data = extract_results_from_test_data(all_test_results, metric)
    
    print(f"🎯 CUSTOM ANALYSIS: {metric.upper()} - {comparison_type.upper()} comparisons")
    statistical_results = perform_statistical_tests(organized_data, type_wise=comparison_type, output_path=full_path)
    plot_results = create_statistical_plots(organized_data, type_wise=comparison_type, output_path=full_path)
    
    return {
        'statistical_results': statistical_results,
        'organized_data': organized_data,
        'plot_results': plot_results
    }

def print_quick_summary(results):
    """Print a quick summary of analysis results"""
    
    if not results or not results.get('statistical_results'):
        print("❌ No valid results to summarize")
        return
    
    print("\n📊 QUICK SUMMARY:")
    print("-" * 40)
    
    for key, stats in results['statistical_results'].items():
        significance = "✅ SIGNIFICANT" if stats['significant'] else "❌ Not significant"
        effect_size = stats.get('eta_squared', 0)
        
        if effect_size > 0.14:
            effect_desc = "(large effect)"
        elif effect_size > 0.06:
            effect_desc = "(medium effect)"
        else:
            effect_desc = "(small effect)"
        
        print(f"{key}: p={stats['anova_p_value']:.4f} {significance} {effect_desc}")

def load_and_analyze(results_file=None, analysis_type='all', output_path=None):
    """
    ONE LINE: Load results and run specified analysis
    
    Parameters:
    -----------
    results_file : str, optional
        Path to test results file. If None, uses default path.
    analysis_type : str, default='all'
        Type of analysis: 'accuracy_models', 'f1_models', 'accuracy_subjects', 'f1_subjects', 'all'
    output_path : str, optional
        Output path for results
    """
    
    # Load results
    if results_file is None:
        results_file = '../Output/ica_rest_all/results_all/all_models_test_results.npy'
    
    try:
        import numpy as np
        all_test_results = np.load(results_file, allow_pickle=True).item()
        print(f"✅ Loaded results from: {results_file}")
    except FileNotFoundError:
        print(f"❌ Could not find results file: {results_file}")
        return None
    
    # Set default output path
    if output_path is None:
        output_path = '../Output/ica_rest_all/statistical_analysis/'
    
    # Run specified analysis
    if analysis_type == 'accuracy_models':
        return analyze_accuracy_models(all_test_results, output_path)
    elif analysis_type == 'f1_models':
        return analyze_f1_models(all_test_results, output_path)
    elif analysis_type == 'accuracy_subjects':
        return analyze_accuracy_subjects(all_test_results, output_path)
    elif analysis_type == 'f1_subjects':
        return analyze_f1_subjects(all_test_results, output_path)
    elif analysis_type == 'all':
        return analyze_all_combinations(all_test_results, output_path)
    else:
        print(f"❌ Unknown analysis type: {analysis_type}")
        return None

print("✅ FIXED: Adaptive multimodal NOW INCLUDED in subject comparisons!")
print("📊 Model comparisons: Adaptive uses fine-tuned scores")  
print("📊 Subject comparisons: Adaptive uses base scores for fair comparison")
print("🎯 All three subject types compared for all models including Multimodal")

from scipy import stats
from scipy.stats import f_oneway, ttest_rel, shapiro, levene
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
import os
from itertools import combinations

def perform_statistical_tests(results, type_wise="model", output_path=None):
    """
    Perform statistical tests with flexible comparison types
    
    Parameters:
    -----------
    results : dict
        Dictionary containing the results data
    type_wise : str, default="model"
        Type of comparison to perform:
        - "model": Compare model types within each study type
        - "subject": Compare study types within each model type
    output_path : str, optional
        Path to save results (creates directory if needed)
    
    Returns:
    --------
    dict
        Statistical results for each comparison
    """
    
    # Create output directory if specified
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Store results
    statistical_results = {}
    
    if type_wise == "model":
        # Compare model types within each study type
        print("STATISTICAL ANALYSIS OF MODEL TYPE COMPARISONS")
        print("=" * 60)
        print("Testing whether there are significant differences between model types")
        print("within each study type using repeated measures ANOVA.")
        print()
        
        for study_type, model_accuracies in results.items():
            if study_type == 'subject':
                study_type_name = 'subject_dependent'
            else:
                study_type_name = study_type
                
            print(f"\n{'='*60}")
            print(f"STATISTICAL ANALYSIS FOR: {study_type_name.upper()}")
            print(f"{'='*60}")
            
            # Prepare data for statistical tests
            model_types = list(model_accuracies.keys())
            n_subjects = len(model_accuracies[model_types[0]])
            
            # Create DataFrame for repeated measures ANOVA
            data_for_anova = []
            for subject_id in range(n_subjects):
                for model_type in model_types:
                    data_for_anova.append({
                        'Subject': f'S{subject_id}',
                        'Comparison_Type': model_type,
                        'Accuracy': model_accuracies[model_type][subject_id]
                    })
            
            df_anova = pd.DataFrame(data_for_anova)
            comparison_types = model_types
            comparison_accuracies = model_accuracies
            result_key = study_type_name
            
    elif type_wise == "subject":
        # Compare study types within each model type
        print("STATISTICAL ANALYSIS OF STUDY TYPE COMPARISONS")
        print("=" * 60)
        print("Testing whether there are significant differences between study types")
        print("within each model type using repeated measures ANOVA.")
        print()
        
        # First, reorganize data by model type
        model_types = set()
        study_types = list(results.keys())
        
        # Get all model types
        for study_type, model_accuracies in results.items():
            model_types.update(model_accuracies.keys())
        
        model_types = list(model_types)
        
        # For each model type, compare study types
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"STATISTICAL ANALYSIS FOR: {model_type.upper()}")
            print(f"{'='*60}")
            
            # Prepare data for this model type across study types
            study_accuracies = {}
            n_subjects = None
            
            for study_type in study_types:
                if model_type in results[study_type]:
                    study_name = 'subject_dependent' if study_type == 'subject' else study_type
                    study_accuracies[study_name] = results[study_type][model_type]
                    if n_subjects is None:
                        n_subjects = len(results[study_type][model_type])
            
            # Skip if not enough study types have this model
            if len(study_accuracies) < 2:
                print(f"Insufficient data for {model_type} (found in {len(study_accuracies)} study types)")
                continue
                
            # Create DataFrame for repeated measures ANOVA
            data_for_anova = []
            for subject_id in range(n_subjects):
                for study_name in study_accuracies.keys():
                    data_for_anova.append({
                        'Subject': f'S{subject_id}',
                        'Comparison_Type': study_name,
                        'Accuracy': study_accuracies[study_name][subject_id]
                    })
            
            df_anova = pd.DataFrame(data_for_anova)
            comparison_types = list(study_accuracies.keys())
            comparison_accuracies = study_accuracies
            result_key = model_type
    
    else:
        raise ValueError("type_wise must be either 'model' or 'subject'")
    
    # Common analysis function for both cases
    def analyze_comparison(df_anova, comparison_types, comparison_accuracies, result_key, n_subjects):
        """Perform the statistical analysis for a given comparison"""
        
        # 1. Descriptive Statistics
        print(f"\n1. DESCRIPTIVE STATISTICS:")
        print("-" * 40)
        desc_stats = df_anova.groupby('Comparison_Type')['Accuracy'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        print(desc_stats)
        
        # 2. Check assumptions
        print(f"\n2. ASSUMPTION CHECKS:")
        print("-" * 40)
        
        # Normality test for each comparison type
        normality_results = {}
        for comp_type in comparison_types:
            comp_data = df_anova[df_anova['Comparison_Type'] == comp_type]['Accuracy']
            stat, p_value = shapiro(comp_data)
            normality_results[comp_type] = {'statistic': stat, 'p_value': p_value}
            print(f"Normality test ({comp_type}): W={stat:.4f}, p={p_value:.4f}")
        
        # Homogeneity of variance (Levene's test)
        comp_data_list = [df_anova[df_anova['Comparison_Type'] == ct]['Accuracy'] 
                          for ct in comparison_types]
        levene_stat, levene_p = levene(*comp_data_list)
        print(f"Levene's test (homogeneity): W={levene_stat:.4f}, p={levene_p:.4f}")
        
        # 3. Repeated Measures ANOVA
        print(f"\n3. REPEATED MEASURES ANOVA:")
        print("-" * 40)
        
        try:
            # Perform repeated measures ANOVA
            anova_results = AnovaRM(df_anova, 'Accuracy', 'Subject', within=['Comparison_Type'])
            anova_table = anova_results.fit()
            print(anova_table.summary())
            
            # Extract F-statistic and p-value
            f_stat = anova_table.anova_table['F Value']['Comparison_Type']
            p_value = anova_table.anova_table['Pr > F']['Comparison_Type']
            
            # Effect size (eta-squared)
            ss_between = anova_table.anova_table['SS']['Comparison_Type']
            ss_total = anova_table.anova_table['SS'].sum()
            eta_squared = ss_between / ss_total
            
            print(f"\nANOVA Results:")
            print(f"F({len(comparison_types)-1}, {(len(comparison_types)-1)*(n_subjects-1)}) = {f_stat:.4f}")
            print(f"p-value = {p_value:.4f}")
            print(f"Effect size (η²) = {eta_squared:.4f}")
            
            # Interpretation
            if p_value < 0.05:
                print(f"✓ SIGNIFICANT difference between {type_wise} types!")
            else:
                print(f"✗ No significant difference between {type_wise} types")
                
        except Exception as e:
            print(f"ANOVA failed: {e}")
            print("Falling back to alternative tests...")
            
            # Alternative: One-way ANOVA (less appropriate but workable)
            f_stat, p_value = f_oneway(*comp_data_list)
            print(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
        
        # 4. Post-hoc tests (if significant)
        if p_value < 0.05:
            print(f"\n4. POST-HOC TESTS (Pairwise Comparisons):")
            print("-" * 40)
            
            # Tukey's HSD
            tukey_results = pairwise_tukeyhsd(df_anova['Accuracy'], 
                                            df_anova['Comparison_Type'], 
                                            alpha=0.05)
            print("Tukey's HSD Results:")
            print(tukey_results.summary())
            
            # Pairwise t-tests with Bonferroni correction
            print(f"\nPairwise t-tests with Bonferroni correction:")
            
            n_comparisons = len(list(combinations(comparison_types, 2)))
            alpha_bonferroni = 0.05 / n_comparisons
            
            for i, (comp1, comp2) in enumerate(combinations(comparison_types, 2)):
                data1 = comparison_accuracies[comp1]
                data2 = comparison_accuracies[comp2]
                
                t_stat, p_val = ttest_rel(data1, data2)
                
                # Cohen's d for effect size
                pooled_std = np.sqrt(((np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2))
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                
                print(f"{comp1} vs {comp2}:")
                print(f"  t({n_subjects-1}) = {t_stat:.4f}, p = {p_val:.4f}")
                print(f"  Bonferroni-corrected α = {alpha_bonferroni:.4f}")
                print(f"  Cohen's d = {cohens_d:.4f}")
                
                if p_val < alpha_bonferroni:
                    print(f"  ✓ SIGNIFICANT after Bonferroni correction")
                else:
                    print(f"  ✗ Not significant after correction")
                print()
        
        # Store results
        return {
            'descriptive_stats': desc_stats,
            'normality_tests': normality_results,
            'levene_test': {'statistic': levene_stat, 'p_value': levene_p},
            'anova_f_stat': f_stat,
            'anova_p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Execute analysis based on type_wise
    if type_wise == "model":
        for study_type, model_accuracies in results.items():
            if study_type == 'subject':
                study_type_name = 'subject_dependent'
            else:
                study_type_name = study_type
                
            # Setup data
            model_types = list(model_accuracies.keys())
            n_subjects = len(model_accuracies[model_types[0]])
            
            data_for_anova = []
            for subject_id in range(n_subjects):
                for model_type in model_types:
                    data_for_anova.append({
                        'Subject': f'S{subject_id}',
                        'Comparison_Type': model_type,
                        'Accuracy': model_accuracies[model_type][subject_id]
                    })
            
            df_anova = pd.DataFrame(data_for_anova)
            
            # Perform analysis
            statistical_results[study_type_name] = analyze_comparison(
                df_anova, model_types, model_accuracies, study_type_name, n_subjects
            )
    
    else:  # type_wise == "subject"
        model_types = set()
        study_types = list(results.keys())
        
        # Get all model types
        for study_type, model_accuracies in results.items():
            model_types.update(model_accuracies.keys())
        
        model_types = list(model_types)
        
        for model_type in model_types:
            # Prepare data for this model type across study types
            study_accuracies = {}
            n_subjects = None
            
            for study_type in study_types:
                if model_type in results[study_type]:
                    study_name = 'subject_dependent' if study_type == 'subject' else study_type
                    study_accuracies[study_name] = results[study_type][model_type]
                    if n_subjects is None:
                        n_subjects = len(results[study_type][model_type])
            
            # Skip if not enough study types have this model
            if len(study_accuracies) < 2:
                continue
                
            # Setup data
            comparison_types = list(study_accuracies.keys())
            data_for_anova = []
            for subject_id in range(n_subjects):
                for study_name in study_accuracies.keys():
                    data_for_anova.append({
                        'Subject': f'S{subject_id}',
                        'Comparison_Type': study_name,
                        'Accuracy': study_accuracies[study_name][subject_id]
                    })
            
            df_anova = pd.DataFrame(data_for_anova)
            
            # Perform analysis
            statistical_results[model_type] = analyze_comparison(
                df_anova, comparison_types, study_accuracies, model_type, n_subjects
            )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF STATISTICAL TESTS")
    print(f"{'='*60}")

    for key, stats in statistical_results.items():
        significance = "SIGNIFICANT" if stats['significant'] else "NOT SIGNIFICANT"
        print(f"{key}: F = {stats['anova_f_stat']:.4f}, p = {stats['anova_p_value']:.4f} ({significance})")

    print(f"\n{'='*60}")
    print("INTERPRETATION GUIDE:")
    if type_wise == "model":
        print("- p < 0.05: Significant difference between model types")
    else:
        print("- p < 0.05: Significant difference between study types")
    print("- Effect size (η²): 0.01=small, 0.06=medium, 0.14=large")
    print("- Cohen's d: 0.2=small, 0.5=medium, 0.8=large effect")
    print(f"{'='*60}")
    
    return statistical_results

def create_statistical_plots(results, type_wise="model", output_path=None):
    """
    Create 3 separate plots for statistical analysis results with flexible comparison types
    
    Parameters:
    -----------
    results : dict
        Dictionary containing the results data
    type_wise : str, default="model"
        Type of comparison to perform:
        - "model": Compare model types within each study type
        - "subject": Compare study types within each model type
    output_path : str, optional
        Path to save plots (creates directory if needed)
    
    Returns:
    --------
    dict
        Statistical results for each comparison
    """
    
    # Create output directory if specified
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Define color schemes
    colors_stat = {
        'significant': '#117733',     # green
        'not_significant': '#d81b60',  # Magenta
        'neutral': '#4682B4',         # Steel blue
        'background': '#F5F5F5'       # White smoke
    }
    
    # Store all results
    all_results = {}
    
    if type_wise == "model":
        # Compare model types within each study type
        print("Creating statistical plots for MODEL TYPE comparisons...")
        comparison_name = "model types"
        file_suffix = ""
        title_suffix = "model types within each study type"
        
        for study_type, model_accuracies in results.items():
            if study_type == 'subject':
                study_type_name = 'subject_dependent'
            else:
                study_type_name = study_type
                
            model_types = list(model_accuracies.keys())
            n_subjects = len(model_accuracies[model_types[0]])
            
            # Create DataFrame for ANOVA
            data_for_anova = []
            for subject_id in range(n_subjects):
                for model_type in model_types:
                    data_for_anova.append({
                        'Subject': f'S{subject_id}',
                        'Comparison_Type': model_type,
                        'Accuracy': model_accuracies[model_type][subject_id]
                    })
            
            df_anova = pd.DataFrame(data_for_anova)
            comparison_types = model_types
            comparison_accuracies = model_accuracies
            result_key = study_type_name
            
            # Process this study type
            all_results[result_key] = process_comparison_group(
                df_anova, comparison_types, comparison_accuracies, n_subjects
            )
    
    elif type_wise == "subject":
        # Compare study types within each model type
        print("Creating statistical plots for STUDY TYPE comparisons...")
        comparison_name = "study types"
        file_suffix = "_study_types"
        title_suffix = "study types within each model type"
        
        # First, reorganize data by model type
        model_types = set()
        study_types = list(results.keys())
        
        # Get all model types
        for study_type, model_accuracies in results.items():
            model_types.update(model_accuracies.keys())
        
        model_types = list(model_types)
        
        # Process each model type
        for model_type in model_types:
            # Prepare data for this model type across study types
            study_accuracies = {}
            n_subjects = None
            
            for study_type in study_types:
                if model_type in results[study_type]:
                    study_name = 'subject_dependent' if study_type == 'subject' else study_type
                    # Clean study names for display
                    study_name_clean = study_name.replace('subject_', '').replace('_subject', '').replace('clean', 'c')
                    study_accuracies[study_name_clean] = results[study_type][model_type]
                    if n_subjects is None:
                        n_subjects = len(results[study_type][model_type])
            
            # Skip if not enough study types have this model
            if len(study_accuracies) < 2:
                print(f"Insufficient data for {model_type} (found in {len(study_accuracies)} study types)")
                continue
                
            # Create DataFrame for ANOVA
            data_for_anova = []
            comparison_types = list(study_accuracies.keys())
            for subject_id in range(n_subjects):
                for study_name in study_accuracies.keys():
                    data_for_anova.append({
                        'Subject': f'S{subject_id}',
                        'Comparison_Type': study_name,
                        'Accuracy': study_accuracies[study_name][subject_id]
                    })
            
            df_anova = pd.DataFrame(data_for_anova)
            comparison_accuracies = study_accuracies
            result_key = model_type
            
            # Process this model type
            all_results[result_key] = process_comparison_group(
                df_anova, comparison_types, comparison_accuracies, n_subjects
            )
    
    else:
        raise ValueError("type_wise must be either 'model' or 'subject'")
    
    # =============================================================================
    # PLOT 1: ANOVA RESULTS TABLE
    # =============================================================================
    
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    ax1.axis('off')
    
    # Prepare ANOVA summary table
    anova_summary = []
    for key, res in all_results.items():
        significance = "Significant" if res['anova_p'] < 0.05 else "Not Significant"
        
        # Handle effect size display
        if np.isnan(res['eta_squared']) or res['eta_squared'] is None:
            eta_squared_str = "N/A"
            effect_size = "N/A"
        else:
            eta_squared_str = f"{res['eta_squared']:.3f}"
            if res['eta_squared'] > 0.14:
                effect_size = "Large"
            elif res['eta_squared'] > 0.06:
                effect_size = "Medium"
            else:
                effect_size = "Small"
        
        # Clean display names
        display_key = key.replace('raw', 'DeepConvNet') if key == 'raw' else key
        
        anova_summary.append([
            display_key,
            f"{res['anova_f']:.3f}",
            f"{res['anova_p']:.4f}",
            eta_squared_str,
            effect_size,
            significance
        ])
    
    # Create ANOVA table with appropriate column header
    first_col_name = 'Study Type' if type_wise == "model" else 'Model Type'
    anova_df = pd.DataFrame(anova_summary, columns=[
        first_col_name, 'F-Statistic', 'P-Value', 'η² (Effect Size)', 'Effect Magnitude', 'Result'
    ])
    
    # Plot ANOVA table
    table1 = ax1.table(cellText=anova_df.values, colLabels=anova_df.columns,
                      cellLoc='center', loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(14)
    table1.scale(1, 2.5)
    
    # Color code the results
    try:
        for i in range(len(anova_summary)):
            result_col = len(anova_df.columns) - 1
            if "Significant" in anova_summary[i][-1] and "Not" not in anova_summary[i][-1]:
                table1[(i+1, result_col)].set_facecolor(colors_stat['significant'])
                table1[(i+1, result_col)].set_text_props(weight='bold', color='white')
            else:
                table1[(i+1, result_col)].set_facecolor(colors_stat['not_significant'])
                table1[(i+1, result_col)].set_text_props(weight='bold', color='white')
        
        # Header styling
        for j in range(len(anova_df.columns)):
            table1[(0, j)].set_facecolor(colors_stat['neutral'])
            table1[(0, j)].set_text_props(weight='bold', color='white')
    except Exception as e:
        print(f"Warning: ANOVA table formatting error: {e}")
    
    ax1.set_title(f'REPEATED MEASURES ANOVA RESULTS\nTesting differences between {title_suffix}', 
                  fontsize=18, fontweight='bold')
    
    # Add interpretation
    interpretation_text = f"""
    INTERPRETATION:
    • Green: Significant difference between {comparison_name} (p < 0.05)
    • Red: No significant difference between {comparison_name} (p >= 0.05)
    • Effect Size (eta-squared): 0.01=Small, 0.06=Medium, 0.14=Large
    """
    
    plt.figtext(0.02, 0.2, interpretation_text, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors_stat['background']))
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    if output_path:
        plt.savefig(os.path.join(output_path, f'statistical_analysis_1_anova{file_suffix}.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # =============================================================================
    # PLOT 2: PAIRWISE COMPARISONS
    # =============================================================================
    
    keys_list = list(all_results.keys())
    n_keys = len(keys_list)
    
    # Create subplots for pairwise comparisons - always use 3 columns
    ncols = 3
    nrows = (n_keys + ncols - 1) // ncols  # Ceiling division to get required rows
    
    fig2, axes = plt.subplots(nrows, ncols, figsize=(18, 6*nrows))
    
    # Handle different subplot configurations
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # Already a 1D array
    else:
        axes = axes.flatten()  # Flatten to 1D array
    
    for idx, key in enumerate(keys_list):
        ax = axes[idx]
        
        pairwise_data = all_results[key]['pairwise']
        display_key = key.replace('raw', 'DeepConvNet') if key == 'raw' else key
        
        if not pairwise_data:
            ax.text(0.5, 0.5, f'No pairwise comparisons\navailable for {display_key}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{display_key}: Pairwise Comparisons', fontsize=16, fontweight='bold')
            continue
        
        # Get the Bonferroni-corrected alpha
        alpha_bonferroni = pairwise_data[0]['Alpha_Bonferroni']
        
        # Create pairwise comparison visualization
        comparisons = [p['Comparison'] for p in pairwise_data]
        # Clean comparison names for model comparisons
        if type_wise == "model":
            comparisons = [c.replace('raw', 'DeepConvNet') for c in comparisons]
        
        p_values = [p['P_Value'] for p in pairwise_data]
        effect_sizes = [abs(p['Cohens_D']) for p in pairwise_data]
        significant = [p['Significant'] for p in pairwise_data]
        
        # Create horizontal bar plot
        colors_bars = [colors_stat['significant'] if sig else colors_stat['not_significant'] 
                      for sig in significant]
        
        bars = ax.barh(range(len(comparisons)), [-np.log10(p) for p in p_values], 
                      color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add significance lines
        ax.axvline(x=-np.log10(0.05), color='orange', linestyle='--', alpha=0.8, 
                  label='α = 0.05 (uncorrected)', linewidth=2)
        ax.axvline(x=-np.log10(alpha_bonferroni), color='red', linestyle='-', alpha=0.8, 
                  label=f'α = {alpha_bonferroni:.3f} (Bonferroni)', linewidth=2)
        
        # Add effect size annotations
        for i, (bar, effect) in enumerate(zip(bars, effect_sizes)):
            if bar.get_width() > 0:
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                       f'd={effect:.2f}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_yticks(range(len(comparisons)))
        # Format y-labels differently for different comparison types
        if type_wise == "model":
            ax.set_yticklabels([c.replace(' vs ', '\nvs\n') for c in comparisons], fontsize=10)
        else:
            ax.set_yticklabels(comparisons, fontsize=12)
        
        ax.set_xlabel('-log10(P-Value)', fontsize=13)
        ax.set_title(f'{display_key}\nPairwise Comparisons', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_keys, len(axes)):
        axes[idx].set_visible(False)
    
    fig2.suptitle(f'PAIRWISE COMPARISONS BETWEEN {comparison_name.upper()}\nBonferroni-corrected p-values', 
                  fontsize=16, fontweight='bold')
    
    # Add interpretation
    interpretation_text2 = f"""
    INTERPRETATION:
    • Green bars: Significant after Bonferroni correction
    • Red bars: Not significant after correction
    • Orange line: Uncorrected α = 0.05
    • Red line: Bonferroni-corrected α threshold
    • Cohen's d: 0.2=Small, 0.5=Medium, 0.8=Large effect
    """
    
    plt.figtext(0.02, -0.02, interpretation_text2, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor=colors_stat['background'], alpha=0.8))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, f'statistical_analysis_2_pairwise{file_suffix}.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # =============================================================================
    # PLOT 3: DESCRIPTIVE STATISTICS
    # =============================================================================
    
    fig3, ax3 = plt.subplots(figsize=(20, 12))
    ax3.axis('off')
    
    # Create descriptive statistics table
    desc_summary = []
    for key, res in all_results.items():
        for comp_type, stats in res['descriptive'].iterrows():
            # Clean display names
            display_key = key.replace('raw', 'DeepConvNet') if key == 'raw' else key
            display_comp_type = comp_type.replace('raw', 'DeepConvNet') if comp_type == 'raw' else comp_type
            
            if type_wise == "model":
                # Study Type, Model Type format
                desc_summary.append([
                    display_key,  # Study type
                    display_comp_type,  # Model type
                    f"{stats['mean']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['median']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    f"{stats['range']:.2f}"
                ])
            else:
                # Model Type, Study Type format
                desc_summary.append([
                    display_key,  # Model type
                    display_comp_type,  # Study type
                    f"{stats['mean']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['median']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    f"{stats['range']:.2f}"
                ])
    
    # Column names based on comparison type
    if type_wise == "model":
        columns = ['Study Type', 'Model Type', 'Mean (%)', 'Std Dev (%)', 'Median (%)', 'Min (%)', 'Max (%)', 'Range (%)']
    else:
        columns = ['Model Type', 'Study Type', 'Mean (%)', 'Std Dev (%)', 'Median (%)', 'Min (%)', 'Max (%)', 'Range (%)']
    
    desc_df = pd.DataFrame(desc_summary, columns=columns)
    
    # Plot descriptive table
    table3 = ax3.table(cellText=desc_df.values, colLabels=desc_df.columns,
                      cellLoc='center', loc='center')
    table3.auto_set_font_size(False)
    table3.set_fontsize(12)
    table3.scale(1, 2)
    
    # Color coding
    try:
        keys_list = list(all_results.keys())
        colors = ['#FFB6C1', '#ADD8E6', '#90EE90', '#FFDEAD', '#DDA0DD']  # Pastel colors
        for i, row in desc_df.iterrows():
            if type_wise == "model":
                # For model comparisons, use the first column (Study Type)
                lookup_key = row[columns[0]]
            else:
                # For subject comparisons, use the first column (Model Type)
                # The data should already have the correct key names (DeepConvNet, etc.)
                lookup_key = row[columns[0]]
                # If display shows 'DeepConvNet' but data key is 'raw', map back
                if lookup_key == 'DeepConvNet' and 'raw' in keys_list and 'DeepConvNet' not in keys_list:
                    lookup_key = 'raw'
                
            # Find the key in our results, handling case where key might not exist
            if lookup_key in keys_list:
                key_idx = keys_list.index(lookup_key)
            else:
                # Fallback: use index based on position
                key_idx = i % len(keys_list)
                print(f"Debug: Could not find '{lookup_key}' in keys_list: {keys_list}")
                
            color = colors[key_idx % len(colors)]
            for j in range(len(desc_df.columns)):
                table3[(i+1, j)].set_facecolor(color)
        
        # Header styling
        for j in range(len(desc_df.columns)):
            table3[(0, j)].set_facecolor(colors_stat['neutral'])
            table3[(0, j)].set_text_props(weight='bold', color='white')
    except Exception as e:
        print(f"Warning: Descriptive table formatting error: {e}")
        print(f"Debug info - keys_list: {list(all_results.keys())}")
        print(f"Debug info - first few rows of desc_df:")
        print(desc_df.head())
    
    if type_wise == "model":
        title_desc = 'DESCRIPTIVE STATISTICS BY MODEL TYPE AND STUDY TYPE\nTest Accuracy Results Summary'
    else:
        title_desc = 'DESCRIPTIVE STATISTICS BY STUDY TYPE AND MODEL TYPE\nTest Accuracy Results Summary'
    
    ax3.set_title(title_desc, fontsize=18, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if output_path:
        plt.savefig(os.path.join(output_path, f'statistical_analysis_3_descriptive{file_suffix}.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return all_results


def process_comparison_group(df_anova, comparison_types, comparison_accuracies, n_subjects):
    """
    Process statistical analysis for a single comparison group
    
    Returns statistical results dictionary for the group
    """
    
    # Descriptive statistics
    desc_stats = df_anova.groupby('Comparison_Type')['Accuracy'].agg([
        'mean', 'std', 'median', 'min', 'max'
    ]).round(2)
    
    # Calculate range
    desc_stats['range'] = (desc_stats['max'] - desc_stats['min']).round(2)
    
    # Perform ANOVA
    try:
        anova_results = AnovaRM(df_anova, 'Accuracy', 'Subject', within=['Comparison_Type'])
        anova_table = anova_results.fit()
        f_stat = anova_table.anova_table['F Value']['Comparison_Type']
        p_value = anova_table.anova_table['Pr > F']['Comparison_Type']
        
        # Effect size
        ss_between = anova_table.anova_table['SS']['Comparison_Type']
        ss_total = anova_table.anova_table['SS'].sum()
        eta_squared = ss_between / ss_total
        
    except Exception as e:
        print(f"Repeated measures ANOVA failed: {e}")
        print("Falling back to one-way ANOVA...")
        
        # Fallback to one-way ANOVA
        comp_data_list = [df_anova[df_anova['Comparison_Type'] == ct]['Accuracy'] 
                         for ct in comparison_types]
        f_stat, p_value = f_oneway(*comp_data_list)
        
        # Calculate eta-squared for one-way ANOVA
        grand_mean = df_anova['Accuracy'].mean()
        
        # Calculate SS_between
        ss_between = 0
        for ct in comparison_types:
            group_data = df_anova[df_anova['Comparison_Type'] == ct]['Accuracy']
            group_mean = group_data.mean()
            n_group = len(group_data)
            ss_between += n_group * (group_mean - grand_mean) ** 2
        
        # Calculate SS_total
        ss_total = ((df_anova['Accuracy'] - grand_mean) ** 2).sum()
        
        # Calculate eta-squared
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Pairwise comparisons
    pairwise_results = []
    n_comparisons = len(list(combinations(comparison_types, 2)))
    alpha_bonferroni = 0.05 / n_comparisons
    
    for comp1, comp2 in combinations(comparison_types, 2):
        data1 = comparison_accuracies[comp1]
        data2 = comparison_accuracies[comp2]
        
        t_stat, p_val = ttest_rel(data1, data2)
        
        # Cohen's d
        pooled_std = np.sqrt(((np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2))
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        pairwise_results.append({
            'Comparison': f'{comp1} vs {comp2}',
            'Mean_Diff': np.mean(data1) - np.mean(data2),
            'T_Stat': t_stat,
            'P_Value': p_val,
            'Cohens_D': cohens_d,
            'Significant': p_val < alpha_bonferroni,
            'Alpha_Bonferroni': alpha_bonferroni
        })
    
    return {
        'descriptive': desc_stats,
        'anova_f': f_stat,
        'anova_p': p_value,
        'eta_squared': eta_squared,
        'pairwise': pairwise_results
    }

