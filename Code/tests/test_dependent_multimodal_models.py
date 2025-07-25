'''
Script to test existing combined EEG and Microstates multimodal models
Loads pre-trained models and evaluates them on test data
'''
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

# Change directory to Notebooks folder
if os.path.basename(os.getcwd()) != 'Notebooks':
    if os.path.basename(os.getcwd()) == 'lib':
        os.chdir(os.path.join(os.getcwd(), '..', 'Notebooks'))
    else:
        os.chdir(os.path.join(os.getcwd(), 'Notebooks'))

from lib import my_functions as mf
from lib.my_models import MicroSNet, FeatureExtractor, MultiModalClassifier

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features_from_single_model(model, data, device):
    """Extract features from a single model for one subject's data"""
    
    # Convert to tensor if needed
    if not isinstance(data, torch.Tensor):
        x = torch.tensor(data, dtype=torch.float32)
    else:
        x = data.clone()
    
    print(f"  Original data shape: {x.shape}")
    
    # Handle different data formats
    if len(x.shape) == 4:
        if x.shape[1] == 1:  # (n_trials, 1, n_channels, timepoints) - raw EEG format
            x = x.squeeze(1)
            print(f"  Raw EEG format detected, shape after squeeze: {x.shape}")
        elif x.shape[2] == 1:  # (n_trials, n_channels, 1, timepoints)
            x = x.squeeze(2)
            print(f"  Format with singleton at dim 2, shape after squeeze: {x.shape}")
    elif len(x.shape) == 3:
        print(f"  3D format (likely microstate), shape: {x.shape}")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()
    
    # Extract features
    features_list = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            batch_features = feature_extractor(batch_x)
            features_list.append(batch_features.cpu())
    
    subject_features = torch.cat(features_list, dim=0)
    print(f"  Extracted features shape: {subject_features.shape}")
    
    return subject_features

def test_single_multimodal_model(multimodal_model, raw_features, ms_features, labels, 
                                subject_id, test_size=0.2, device='cuda'):
    """Test a single multimodal model and return detailed metrics"""
    
    print(f"\nTesting multimodal model for Subject {subject_id}")
    print(f"Raw features shape: {raw_features.shape}")
    print(f"MS features shape: {ms_features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    set_seed(42)  # Use same seed as training for consistent splits
    
    # Split data (same way as in training)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42,
        stratify=labels.numpy() if len(np.unique(labels.numpy())) > 1 else None
    )
    
    # Create test dataset
    test_dataset = TensorDataset(
        raw_features[test_idx], ms_features[test_idx], labels[test_idx]
    )
    
    batch_size = min(32, len(test_idx))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test the model
    multimodal_model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for raw_batch, ms_batch, label_batch in test_loader:
            raw_batch = raw_batch.to(device)
            ms_batch = ms_batch.to(device)
            label_batch = label_batch.to(device)
            
            outputs = multimodal_model(raw_batch, ms_batch)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            test_correct += (preds == label_batch).sum().item()
            test_total += label_batch.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_acc = test_correct / test_total * 100
    
    # Calculate per-class metrics
    n_classes = len(np.unique(all_labels))
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Classification report
    clf_report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Calculate F1 macro manually
    f1_macro = clf_report['macro avg']['f1-score'] if 'macro avg' in clf_report else 0
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"  Subject {subject_id} Test Results:")
    print(f"    Overall Accuracy: {test_acc:.2f}%")
    print(f"    F1 Macro: {f1_macro*100:.2f}%")
    print(f"    Total test samples: {test_total}")
    
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
        'total_samples': test_total,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs,
        'classification_report': clf_report,
        'confusion_matrix': conf_matrix,
        'n_classes': n_classes
    }

def test_all_multimodal_models(results_file_path, device='cuda'):
    """
    Load and test all multimodal models from saved results
    
    Args:
        results_file_path: Path to the saved multimodal results file
        device: Device to run computations on
    """
    
    print("Loading multimodal results...")
    
    # Load the multimodal results
    multimodal_results = np.load(results_file_path, allow_pickle=True).item()
    
    print(f"Found {len(multimodal_results['subject_results'])} subjects in results")
    
    # Load the original data and models needed for feature extraction
    print("Loading original data and models...")
    
    # Set paths
    type_of_subject = 'dependent'  # or 'independent', adjust as needed
    data_path = '../Data/'
    output_path = f'../Output/ica_rest_all/{type_of_subject}/'
    input_path = '../Output/ica_rest_all/'
    
    # Load data
    all_data, all_y = mf.load_all_data(subjects_list=None, do_all=False)
    
    ms_timeseries_path = os.path.join(input_path, 'ms_timeseries.npy')
    with open(ms_timeseries_path, 'rb') as f:
        finals_ls = pickle.load(f)
    
    # Load feature extraction models
    results = np.load(os.path.join(output_path, f'{type_of_subject}_results_ica_rest_all.npy'), allow_pickle=True).item()
    ms_results = np.load(os.path.join(output_path, f'{type_of_subject}_ms_results_ica_rest_all.npy'), allow_pickle=True).item()
    
    n_subjects = len(multimodal_results['subject_results'])
    all_test_results = []
    
    print(f"\nTesting {n_subjects} multimodal models...")
    
    for subject_id in range(n_subjects):
        print(f"\n{'='*60}")
        print(f"Testing Subject {subject_id}")
        print(f"{'='*60}")
        
        # Get the trained multimodal model
        multimodal_model = multimodal_results['subject_results'][subject_id]['model']
        multimodal_model.to(device)
        
        # Get data for this subject
        raw_data = all_data[subject_id]
        ms_data = finals_ls[subject_id]
        labels = torch.tensor(all_y[subject_id], dtype=torch.long)
        
        # Extract features (same as in training)
        print(f"Extracting features from raw EEG model...")
        raw_features = extract_features_from_single_model(
            results['models'][subject_id], raw_data, device
        )
        
        print(f"Extracting features from microstate model...")
        ms_features = extract_features_from_single_model(
            ms_results['models'][subject_id], ms_data, device
        )
        
        # Test the multimodal model
        test_result = test_single_multimodal_model(
            multimodal_model, raw_features, ms_features, labels, subject_id, device=device
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
    
    # Save detailed results
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
    
    # Set the path to your multimodal results file
    type_of_subject = 'dependent'  # Change this if needed
    output_path = f'../Output/ica_rest_all/{type_of_subject}/'
    results_file = os.path.join(output_path, f'{type_of_subject}_multimodal_results_ica_rest_all.npy')
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please check the path and file name.")
    else:
        # Run the testing
        test_results = test_all_multimodal_models(results_file, device=device)
        
        # Save the test results
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