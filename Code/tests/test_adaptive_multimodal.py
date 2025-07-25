'''
Script to test existing adaptive combined EEG and Microstates multimodal models
Tests on the same 10% split used for final testing after fine-tuning
Uses saved split indices for consistency
'''

print('==================== Start of script test_existing_adaptive_multimodal_models.py! ===================')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

# Change directory to Notebooks folder (same as original)
if os.path.basename(os.getcwd()) != 'Notebooks':
    if os.path.basename(os.getcwd()) == 'lib':
        os.chdir(os.path.join(os.getcwd(), '..', 'Notebooks'))
    else:
        os.chdir(os.path.join(os.getcwd(), 'Notebooks'))

from lib import my_functions as mf
from lib.my_models import FeatureExtractor, MultiModalClassifier, MicroSNet

# ---------------------------# Parameters (same as original) ---------------------------
excluded_from_training = [-1]  # For testing purposes, exclude no subjects
batch_size = 32
type_of_subject = 'adaptive'  # 'independent' or 'adaptive'

# ---------------------------# Load files (same paths as original) ---------------------------
data_path = '../Data/'
input_path = f'../Output/ica_rest_all/'
output_path = f'../Output/ica_rest_all/{type_of_subject}/'
do_all = False
n_subjects = 50

def extract_features_from_single_model(model, data, device):
    """Extract features from a single model for one subject's data (same as original)"""
    
    # Convert to tensor if needed
    if not isinstance(data, torch.Tensor):
        x = torch.tensor(data, dtype=torch.float32)
    else:
        x = data.clone()
    
    print(f"  Original data shape: {x.shape}")
    
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
        print(f"  3D format (likely microstate), shape: {x.shape}")
    else:
        print(f"  Unexpected shape: {x.shape}")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()
    
    # Extract features
    features_list = []
    
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            batch_features = feature_extractor(batch_x)
            features_list.append(batch_features.cpu())
    
    subject_features = torch.cat(features_list, dim=0)
    print(f"  Extracted features shape: {subject_features.shape}")
    
    return subject_features

def test_adaptive_multimodal_models(base_model, finetuned_model, results, ms_results, 
                                  all_data, finals_ls, all_y, subject_id, 
                                  final_test_indices, device='cuda'):
    """Test both base and fine-tuned models on the 10% final test split"""
    
    print(f"\nTesting adaptive multimodal models for Subject {subject_id}")
    
    # Extract features for the full test subject data
    raw_test_full_feats = extract_features_from_single_model(
        results['models'][subject_id], all_data[subject_id], device
    )
    ms_test_full_feats = extract_features_from_single_model(
        ms_results['models'][subject_id], finals_ls[subject_id], device
    )
    y_test_full = torch.tensor(all_y[subject_id], dtype=torch.long)
    
    # Use only the final test indices (10% split)
    raw_final_test_feats = raw_test_full_feats[final_test_indices]
    ms_final_test_feats = ms_test_full_feats[final_test_indices]
    y_final_test = y_test_full[final_test_indices]
    
    print(f"Final test data shape: {raw_final_test_feats.shape[0]} samples")
    print(f"Test labels shape: {y_final_test.shape}")
    
    # Test base model
    print("Testing base model...")
    base_model.eval()
    base_preds = []
    base_probs = []
    
    with torch.no_grad():
        for i in range(0, len(y_final_test), 32):
            raw_batch = raw_final_test_feats[i:i+32].to(device)
            ms_batch = ms_final_test_feats[i:i+32].to(device)
            
            outputs = base_model(raw_batch, ms_batch)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            base_preds.extend(preds.cpu().numpy())
            base_probs.extend(probs.cpu().numpy())
    
    # Test fine-tuned model
    print("Testing fine-tuned model...")
    finetuned_model.eval()
    finetuned_preds = []
    finetuned_probs = []
    
    with torch.no_grad():
        for i in range(0, len(y_final_test), 32):
            raw_batch = raw_final_test_feats[i:i+32].to(device)
            ms_batch = ms_final_test_feats[i:i+32].to(device)
            
            outputs = finetuned_model(raw_batch, ms_batch)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            finetuned_preds.extend(preds.cpu().numpy())
            finetuned_probs.extend(probs.cpu().numpy())
    
    # Calculate accuracies
    base_acc = accuracy_score(y_final_test.numpy(), base_preds) * 100
    finetuned_acc = accuracy_score(y_final_test.numpy(), finetuned_preds) * 100
    improvement = finetuned_acc - base_acc
    
    # Calculate per-class metrics for both models
    n_classes = len(np.unique(y_final_test.numpy()))
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Base model classification report
    base_clf_report = classification_report(
        y_final_test.numpy(), base_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    base_f1_macro = base_clf_report['macro avg']['f1-score'] if 'macro avg' in base_clf_report else 0
    
    # Fine-tuned model classification report
    finetuned_clf_report = classification_report(
        y_final_test.numpy(), finetuned_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    finetuned_f1_macro = finetuned_clf_report['macro avg']['f1-score'] if 'macro avg' in finetuned_clf_report else 0
    
    # Confusion matrices
    base_conf_matrix = confusion_matrix(y_final_test.numpy(), base_preds)
    finetuned_conf_matrix = confusion_matrix(y_final_test.numpy(), finetuned_preds)
    
    print(f"  Subject {subject_id} Test Results:")
    print(f"    Base Model Accuracy: {base_acc:.2f}%")
    print(f"    Base Model F1 Macro: {base_f1_macro*100:.2f}%")
    print(f"    Fine-tuned Model Accuracy: {finetuned_acc:.2f}%")
    print(f"    Fine-tuned Model F1 Macro: {finetuned_f1_macro*100:.2f}%")
    print(f"    Improvement: {improvement:+.2f}%")
    print(f"    Total test samples: {len(y_final_test)}")
    
    # Print per-class accuracy for both models
    print(f"    Base Model Per-Class Performance:")
    for i in range(n_classes):
        class_acc = base_clf_report[f'Class_{i}']['precision'] * 100 if f'Class_{i}' in base_clf_report else 0
        class_recall = base_clf_report[f'Class_{i}']['recall'] * 100 if f'Class_{i}' in base_clf_report else 0
        class_f1 = base_clf_report[f'Class_{i}']['f1-score'] * 100 if f'Class_{i}' in base_clf_report else 0
        print(f"      Class {i}: Precision={class_acc:.2f}%, Recall={class_recall:.2f}%, F1={class_f1:.2f}%")
    
    print(f"    Fine-tuned Model Per-Class Performance:")
    for i in range(n_classes):
        class_acc = finetuned_clf_report[f'Class_{i}']['precision'] * 100 if f'Class_{i}' in finetuned_clf_report else 0
        class_recall = finetuned_clf_report[f'Class_{i}']['recall'] * 100 if f'Class_{i}' in finetuned_clf_report else 0
        class_f1 = finetuned_clf_report[f'Class_{i}']['f1-score'] * 100 if f'Class_{i}' in finetuned_clf_report else 0
        print(f"      Class {i}: Precision={class_acc:.2f}%, Recall={class_recall:.2f}%, F1={class_f1:.2f}%")
    
    return {
        'subject_id': subject_id,
        'base_test_accuracy': base_acc,
        'finetuned_test_accuracy': finetuned_acc,
        'base_f1_macro': base_f1_macro * 100,
        'finetuned_f1_macro': finetuned_f1_macro * 100,
        'improvement': improvement,
        'total_samples': len(y_final_test),
        'base_predictions': base_preds,
        'finetuned_predictions': finetuned_preds,
        'true_labels': y_final_test.numpy(),
        'base_probabilities': base_probs,
        'finetuned_probabilities': finetuned_probs,
        'base_classification_report': base_clf_report,
        'finetuned_classification_report': finetuned_clf_report,
        'base_confusion_matrix': base_conf_matrix,
        'finetuned_confusion_matrix': finetuned_conf_matrix,
        'n_classes': n_classes
    }

def test_all_adaptive_multimodal_models(device='cuda'):
    """
    Load and test all adaptive multimodal models from saved results
    Uses exact same file names and paths as the original training script
    Tests on the same 10% final test split used during training
    """
    
    print("Loading adaptive multimodal results...")
    
    # Use exact same file name as original script
    results_file = os.path.join(output_path, f'{type_of_subject}_multimodal_results_ica_rest_all.npy')
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please check if the training has been completed.")
        return None
    
    # Load the multimodal results
    multimodal_results = np.load(results_file, allow_pickle=True).item()
    
    print(f"Found {len(multimodal_results['subject_results'])} subjects in results")
    
    # Load the original data and models (same as original script)
    print("Loading original data and models...")
    
    all_data, all_y = mf.load_all_data(subjects_list=None, do_all=do_all)
    
    ms_timeseries_path = os.path.join(input_path, 'ms_timeseries.npy')
    with open(ms_timeseries_path, 'rb') as f:
        finals_ls = pickle.load(f)
    
    # Load feature extraction models (same file names as original)
    results = np.load(os.path.join(output_path, f'{type_of_subject}_results_ica_rest_all.npy'), allow_pickle=True).item()
    ms_results = np.load(os.path.join(output_path, f'{type_of_subject}_ms_results_ica_rest_all.npy'), allow_pickle=True).item()
    
    print(f'N_models in results: {len(results["models"])}')
    print(f'N_models in ms_results: {len(ms_results["models"])}')
    
    # Load the saved split indices (CRITICAL for consistency)
    print("Loading saved split indices for consistency...")
    all_train_indices, all_test_indices = mf.load_split_indices(
        output_path, filename=f'{type_of_subject}_split_indices.pkl'
    )
    
    if all_train_indices is None or all_test_indices is None:
        print(f"ERROR: Split indices not found at {output_path}/{type_of_subject}_split_indices.pkl")
        print("Cannot proceed without consistent split indices.")
        print("Please make sure the training script has been run and saved the split indices.")
        return None
    
    print("Split indices loaded successfully!")
    
    n_subjects = len(multimodal_results['subject_results'])
    all_test_results = []
    
    print(f"\nTesting {n_subjects} adaptive multimodal models...")
    
    for subject_id in range(n_subjects):
        print(f"\n{'='*60}")
        print(f"Testing Subject {subject_id}")
        print(f"{'='*60}")
        
        # Get the trained models (both base and fine-tuned)
        subject_result = multimodal_results['subject_results'][subject_id]
        base_model = subject_result['model']  # Base model
        finetuned_model = subject_result['finetuned_model']  # Fine-tuned model
        
        # Move models to device
        base_model.to(device)
        finetuned_model.to(device)
        
        # Get the saved test indices for this subject
        final_test_indices = all_test_indices[subject_id]
        print(f"Using saved test indices: {len(final_test_indices)} samples")
        
        # Test both models on the final test split
        test_result = test_adaptive_multimodal_models(
            base_model, finetuned_model, results, ms_results, 
            all_data, finals_ls, all_y, subject_id, 
            final_test_indices, device=device
        )
        
        all_test_results.append(test_result)
    
    # Calculate overall statistics
    print(f"\n{'='*60}")
    print("OVERALL TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    base_accuracies = [res['base_test_accuracy'] for res in all_test_results]
    finetuned_accuracies = [res['finetuned_test_accuracy'] for res in all_test_results]
    base_f1_macros = [res['base_f1_macro'] for res in all_test_results]
    finetuned_f1_macros = [res['finetuned_f1_macro'] for res in all_test_results]
    improvements = [res['improvement'] for res in all_test_results]
    total_samples = [res['total_samples'] for res in all_test_results]
    
    print(f"Individual Subject Results:")
    for i, res in enumerate(all_test_results):
        print(f"  Subject {i}: Base={res['base_test_accuracy']:.2f}% -> "
              f"Fine-tuned={res['finetuned_test_accuracy']:.2f}% "
              f"(improvement: {res['improvement']:+.2f}%) "
              f"({res['total_samples']} samples)")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Base Accuracy: {np.mean(base_accuracies):.2f}% ± {np.std(base_accuracies):.2f}%")
    print(f"  Mean Fine-tuned Accuracy: {np.mean(finetuned_accuracies):.2f}% ± {np.std(finetuned_accuracies):.2f}%")
    print(f"  Mean Base F1-Macro: {np.mean(base_f1_macros):.2f}% ± {np.std(base_f1_macros):.2f}%")
    print(f"  Mean Fine-tuned F1-Macro: {np.mean(finetuned_f1_macros):.2f}% ± {np.std(finetuned_f1_macros):.2f}%")
    print(f"  Mean Improvement: {np.mean(improvements):+.2f}% ± {np.std(improvements):.2f}%")
    print(f"  Median Base Accuracy: {np.median(base_accuracies):.2f}%")
    print(f"  Median Fine-tuned Accuracy: {np.median(finetuned_accuracies):.2f}%")
    print(f"  Min Base Accuracy: {np.min(base_accuracies):.2f}%")
    print(f"  Max Fine-tuned Accuracy: {np.max(finetuned_accuracies):.2f}%")
    print(f"  Total Test Samples: {np.sum(total_samples)}")
    
    # Count improvements
    positive_improvements = [imp for imp in improvements if imp > 0]
    negative_improvements = [imp for imp in improvements if imp < 0]
    
    print(f"\nImprovement Analysis:")
    print(f"  Subjects with improvement: {len(positive_improvements)}/{len(improvements)}")
    print(f"  Subjects with degradation: {len(negative_improvements)}/{len(improvements)}")
    print(f"  Best improvement: {max(improvements):+.2f}%")
    print(f"  Worst degradation: {min(improvements):+.2f}%")
    
    # Calculate per-class statistics across all subjects (for base and fine-tuned models)
    print(f"\nPer-Class Performance (Base Model, averaged across subjects):")
    all_base_class_reports = [res['base_classification_report'] for res in all_test_results]
    n_classes_per_subject = [res['n_classes'] for res in all_test_results]
    max_classes = max(n_classes_per_subject)
    
    for class_id in range(max_classes):
        class_key = f'Class_{class_id}'
        precisions = []
        recalls = []
        f1_scores = []
        
        for report in all_base_class_reports:
            if class_key in report:
                precisions.append(report[class_key]['precision'])
                recalls.append(report[class_key]['recall'])
                f1_scores.append(report[class_key]['f1-score'])
        
        if precisions:
            print(f"  Class {class_id}:")
            print(f"    Precision: {np.mean(precisions)*100:.2f}% ± {np.std(precisions)*100:.2f}%")
            print(f"    Recall: {np.mean(recalls)*100:.2f}% ± {np.std(recalls)*100:.2f}%")
            print(f"    F1-Score: {np.mean(f1_scores)*100:.2f}% ± {np.std(f1_scores)*100:.2f}%")
    
    print(f"\nPer-Class Performance (Fine-tuned Model, averaged across subjects):")
    all_finetuned_class_reports = [res['finetuned_classification_report'] for res in all_test_results]
    
    for class_id in range(max_classes):
        class_key = f'Class_{class_id}'
        precisions = []
        recalls = []
        f1_scores = []
        
        for report in all_finetuned_class_reports:
            if class_key in report:
                precisions.append(report[class_key]['precision'])
                recalls.append(report[class_key]['recall'])
                f1_scores.append(report[class_key]['f1-score'])
        
        if precisions:
            print(f"  Class {class_id}:")
            print(f"    Precision: {np.mean(precisions)*100:.2f}% ± {np.std(precisions)*100:.2f}%")
            print(f"    Recall: {np.mean(recalls)*100:.2f}% ± {np.std(recalls)*100:.2f}%")
            print(f"    F1-Score: {np.mean(f1_scores)*100:.2f}% ± {np.std(f1_scores)*100:.2f}%")
    
    # Save detailed results (same naming convention as original)
    detailed_results = {
        'test_results': all_test_results,
        'summary': {
            'mean_base_accuracy': np.mean(base_accuracies),
            'std_base_accuracy': np.std(base_accuracies),
            'mean_finetuned_accuracy': np.mean(finetuned_accuracies),
            'std_finetuned_accuracy': np.std(finetuned_accuracies),
            'mean_base_f1_macro': np.mean(base_f1_macros),
            'std_base_f1_macro': np.std(base_f1_macros),
            'mean_finetuned_f1_macro': np.mean(finetuned_f1_macros),
            'std_finetuned_f1_macro': np.std(finetuned_f1_macros),
            'mean_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements),
            'median_base_accuracy': np.median(base_accuracies),
            'median_finetuned_accuracy': np.median(finetuned_accuracies),
            'min_base_accuracy': np.min(base_accuracies),
            'max_finetuned_accuracy': np.max(finetuned_accuracies),
            'best_improvement': max(improvements),
            'worst_degradation': min(improvements),
            'subjects_with_improvement': len(positive_improvements),
            'subjects_with_degradation': len(negative_improvements),
            'total_samples': np.sum(total_samples),
            'individual_base_accuracies': base_accuracies,
            'individual_finetuned_accuracies': finetuned_accuracies,
            'individual_base_f1_macros': base_f1_macros,
            'individual_finetuned_f1_macros': finetuned_f1_macros,
            'individual_improvements': improvements,
            'individual_sample_counts': total_samples
        }
    }
    
    return detailed_results

# Main execution
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run the testing
    test_results = test_all_adaptive_multimodal_models(device=device)
    
    if test_results is not None:
        # Save the test results (same naming convention as original)
        test_output_file = os.path.join(output_path, f'{type_of_subject}_multimodal_test_results_ica_rest_all.npy')
        np.save(test_output_file, test_results)
        print(f"\nDetailed test results saved to: {test_output_file}")
        
        # Create a summary CSV
        summary_df = pd.DataFrame({
            'Subject': range(len(test_results['test_results'])),
            'Base_Test_Accuracy': [res['base_test_accuracy'] for res in test_results['test_results']],
            'Finetuned_Test_Accuracy': [res['finetuned_test_accuracy'] for res in test_results['test_results']],
            'Base_F1_Macro': [res['base_f1_macro'] for res in test_results['test_results']],
            'Finetuned_F1_Macro': [res['finetuned_f1_macro'] for res in test_results['test_results']],
            'Improvement': [res['improvement'] for res in test_results['test_results']],
            'Sample_Count': [res['total_samples'] for res in test_results['test_results']],
            'N_Classes': [res['n_classes'] for res in test_results['test_results']]
        })
        
        summary_csv_file = os.path.join(output_path, f'{type_of_subject}_multimodal_test_summary.csv')
        summary_df.to_csv(summary_csv_file, index=False)
        print(f"Summary CSV saved to: {summary_csv_file}")
        
        print("\nTesting completed successfully!")
        print(f"Average base performance: {test_results['summary']['mean_base_accuracy']:.2f}%")
        print(f"Average fine-tuned performance: {test_results['summary']['mean_finetuned_accuracy']:.2f}%")
        print(f"Average improvement: {test_results['summary']['mean_improvement']:+.2f}%")
    
print('==================== End of script test_existing_adaptive_multimodal_models.py! ===================')