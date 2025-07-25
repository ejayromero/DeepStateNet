'''
Script to test existing independent combined EEG and Microstates multimodal models
Loads pre-trained models and evaluates them on test data
Uses the exact same file names and paths as the original training script
'''

print('==================== Start of script test_existing_independent_multimodal_models.py! ===================')

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
excluded_from_training = [-1]  # No exclusions for independent clean
batch_size = 32
type_of_subject = 'independent'  # 'independent' or 'adaptive'

# ---------------------------# Load files (same paths as original) ---------------------------
data_path = '../Data/'
output_path = f'../Output/ica_rest_all/{type_of_subject}/'
input_path = '../Output/ica_rest_all/'
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

def test_single_multimodal_model_simple(multimodal_model, results, ms_results, all_data, finals_ls, all_y,
                                       subject_id, device='cuda'):
    """Test a single multimodal model directly on the subject's data"""
    
    print(f"\nTesting multimodal model for Subject {subject_id}")
    
    # Extract features for the test subject only
    raw_test_feats = extract_features_from_single_model(results['models'][subject_id], all_data[subject_id], device)
    ms_test_feats = extract_features_from_single_model(ms_results['models'][subject_id], finals_ls[subject_id], device)
    y_test = torch.tensor(all_y[subject_id], dtype=torch.long)
    
    print(f"Raw test features shape: {raw_test_feats.shape}")
    print(f"MS test features shape: {ms_test_feats.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Test the model
    multimodal_model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(y_test), 32):
            batch_raw = raw_test_feats[i:i+32].to(device)
            batch_ms = ms_test_feats[i:i+32].to(device)
            
            outputs = multimodal_model(batch_raw, batch_ms)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_acc = accuracy_score(y_test.numpy(), all_preds) * 100
    
    # Calculate per-class metrics
    n_classes = len(np.unique(y_test.numpy()))
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Classification report
    clf_report = classification_report(
        y_test.numpy(), all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Calculate F1 macro
    f1_macro = clf_report['macro avg']['f1-score'] if 'macro avg' in clf_report else 0
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test.numpy(), all_preds)
    
    print(f"  Subject {subject_id} Test Results:")
    print(f"    Overall Accuracy: {test_acc:.2f}%")
    print(f"    F1 Macro: {f1_macro*100:.2f}%")
    print(f"    Total test samples: {len(y_test)}")
    
    # Print per-class accuracy
    for i in range(n_classes):
        class_acc = clf_report[f'Class_{i}']['precision'] * 100 if f'Class_{i}' in clf_report else 0
        class_recall = clf_report[f'Class_{i}']['recall'] * 100 if f'Class_{i}' in clf_report else 0
        class_f1 = clf_report[f'Class_{i}']['f1-score'] * 100 if f'Class_{i}' in clf_report else 0
        print(f"    Class {i}: Precision={class_acc:.2f}%, Recall={class_recall:.2f}%, F1={class_f1:.2f}%")
    
    return {
        'subject_id': subject_id,
        'test_accuracy': test_acc,
        'f1_macro': f1_macro * 100,  # Convert to percentage
        'total_samples': len(y_test),
        'predictions': all_preds,
        'true_labels': y_test.numpy(),
        'probabilities': all_probs,
        'classification_report': clf_report,
        'confusion_matrix': conf_matrix,
        'n_classes': n_classes
    }

def test_all_independent_multimodal_models(device='cuda'):
    """
    Load and test all independent multimodal models from saved results
    Uses exact same file names and paths as the original training script
    """
    
    print("Loading independent multimodal results...")
    
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
    
    n_subjects = len(multimodal_results['subject_results'])
    all_test_results = []
    
    print(f"\nTesting {n_subjects} independent multimodal models...")
    
    for subject_id in range(n_subjects):
        print(f"\n{'='*60}")
        print(f"Testing Subject {subject_id}")
        print(f"{'='*60}")
        
        # Get the trained multimodal model
        multimodal_model = multimodal_results['subject_results'][subject_id]['model']
        multimodal_model.to(device)
        
        # Test the multimodal model directly on the subject
        test_result = test_single_multimodal_model_simple(
            multimodal_model, results, ms_results, all_data, finals_ls, all_y,
            subject_id, device=device
        )
        
        all_test_results.append(test_result)
    
    # Calculate overall statistics
    print(f"\n{'='*60}")
    print("OVERALL TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    test_accuracies = [res['test_accuracy'] for res in all_test_results]
    f1_macros = [res['f1_macro'] for res in all_test_results]
    total_samples = [res['total_samples'] for res in all_test_results]
    
    print(f"Individual Subject Results:")
    for i, (acc, f1) in enumerate(zip(test_accuracies, f1_macros)):
        print(f"  Subject {i}: Accuracy={acc:.2f}%, F1-Macro={f1:.2f}% ({total_samples[i]} samples)")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Accuracy: {np.mean(test_accuracies):.2f}% ± {np.std(test_accuracies):.2f}%")
    print(f"  Mean F1-Macro: {np.mean(f1_macros):.2f}% ± {np.std(f1_macros):.2f}%")
    print(f"  Median Accuracy: {np.median(test_accuracies):.2f}%")
    print(f"  Median F1-Macro: {np.median(f1_macros):.2f}%")
    print(f"  Min Accuracy: {np.min(test_accuracies):.2f}%")
    print(f"  Max Accuracy: {np.max(test_accuracies):.2f}%")
    print(f"  Total Test Samples: {np.sum(total_samples)}")
    
    # Calculate per-class statistics across all subjects
    all_class_reports = []
    for res in all_test_results:
        all_class_reports.append(res['classification_report'])
    
    # Get unique number of classes
    n_classes_per_subject = [res['n_classes'] for res in all_test_results]
    max_classes = max(n_classes_per_subject)
    
    print(f"\nPer-Class Performance (averaged across subjects):")
    for class_id in range(max_classes):
        class_key = f'Class_{class_id}'
        precisions = []
        recalls = []
        f1_scores = []
        
        for report in all_class_reports:
            if class_key in report:
                precisions.append(report[class_key]['precision'])
                recalls.append(report[class_key]['recall'])
                f1_scores.append(report[class_key]['f1-score'])
        
        if precisions:  # Only if we have data for this class
            print(f"  Class {class_id}:")
            print(f"    Precision: {np.mean(precisions)*100:.2f}% ± {np.std(precisions)*100:.2f}%")
            print(f"    Recall: {np.mean(recalls)*100:.2f}% ± {np.std(recalls)*100:.2f}%")
            print(f"    F1-Score: {np.mean(f1_scores)*100:.2f}% ± {np.std(f1_scores)*100:.2f}%")
    
    # Save detailed results (same naming convention as original)
    detailed_results = {
        'test_results': all_test_results,
        'summary': {
            'mean_accuracy': np.mean(test_accuracies),
            'std_accuracy': np.std(test_accuracies),
            'mean_f1_macro': np.mean(f1_macros),
            'std_f1_macro': np.std(f1_macros),
            'median_accuracy': np.median(test_accuracies),
            'median_f1_macro': np.median(f1_macros),
            'min_accuracy': np.min(test_accuracies),
            'max_accuracy': np.max(test_accuracies),
            'total_samples': np.sum(total_samples),
            'individual_accuracies': test_accuracies,
            'individual_f1_macros': f1_macros,
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
    test_results = test_all_independent_multimodal_models(device=device)
    
    if test_results is not None:
        # Save the test results (same naming convention as original)
        test_output_file = os.path.join(output_path, f'{type_of_subject}_multimodal_test_results_ica_rest_all.npy')
        np.save(test_output_file, test_results)
        print(f"\nDetailed test results saved to: {test_output_file}")
        
        # Create a summary CSV
        summary_df = pd.DataFrame({
            'Subject': range(len(test_results['test_results'])),
            'Test_Accuracy': [res['test_accuracy'] for res in test_results['test_results']],
            'F1_Macro': [res['f1_macro'] for res in test_results['test_results']],
            'Sample_Count': [res['total_samples'] for res in test_results['test_results']],
            'N_Classes': [res['n_classes'] for res in test_results['test_results']]
        })
        
        summary_csv_file = os.path.join(output_path, f'{type_of_subject}_multimodal_test_summary.csv')
        summary_df.to_csv(summary_csv_file, index=False)
        print(f"Summary CSV saved to: {summary_csv_file}")
        
        print("\nTesting completed successfully!")
    
print('==================== End of script test_existing_independent_multimodal_models.py! ===================')