import psutil
import os

import numpy as np
import pandas as pd

import random
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def test_lib():
    """
    Test function to check if the library is working correctly.
    """
    print("Library is working correctly.")

def get_file_path(data_path, folder_path, file_name):
    """
    Get the path to the data file.

    Parameters:
        data_path (str): Base path to the data directory.
        folder_path (str): Subfolder path within the data directory.
        file_name (str): Name of the data file to load.
    Returns:
        numpy.ndarray: Loaded data array.   
    """

    if not os.path.exists(data_path + folder_path):
        raise FileNotFoundError(f"Data folder {data_path + folder_path} does not exist.")

    file_path = os.path.join(data_path, folder_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} does not exist.")

    data = np.load(file_path)
    # print('Data shape:', data.shape)
    return data

def get_gfp(data_array):
    """
    Calculate the Global Field Power (GFP) for the given data array.
    
    Parameters:
        data_array (numpy.ndarray): 2D array of shape (n_channels, n_timepoints)
    
    Returns:
        numpy.ndarray: 1D array of GFP values
    """
    return np.std(data_array, axis=0)

def get_gmd(data_array, gfp, u_ind, v_ind):
    """
    Calculate the Global Mean Dissimilarity (GMD) between two maps for the given data array.
    
    Parameters:
        data_array (numpy.ndarray): 2D array of shape (n_channels, n_timepoints)
        gfp (numpy.ndarray): 1D array of GFP values
        u_ind (int): Index of the first map
        v_ind (int): Index of the second map
        
    Returns:
        double: GMD value   
    """
    u = data_array[:, u_ind]
    v = data_array[:, v_ind]
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    u_GFP = gfp[u_ind]
    v_GFP = gfp[v_ind]
    return np.sqrt(np.sum(((u - u_mean)/u_GFP - (v - v_mean)/v_GFP)**2)/data_array.shape[0])

# Extract the trial data from the wide-format DataFrame
def extract_all_gfp(df, sampling_rate):
    """
    Extract Global Field Power (GFP) for each trial from the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with MultiIndex (trial, channel) and timepoint columns.
        sampling_rate (int): Sampling rate in Hz.
        
    Returns:
        list: List of GFP arrays for each trial.
    """
    all_gfp = []
    for nt_trial in range(df.index.get_level_values('trial').nunique()):
        df_trial = df.xs(nt_trial, level='trial')  # shape: (61 channels, 1000 time points)

        # Convert timepoint labels ('t0', 't1', ...) to integers
        timepoints = [int(col[1:]) for col in df_trial.columns]
        # get timepoints know the sampling rate
        timepoints = np.arange(0, len(timepoints)) / sampling_rate  # Convert to seconds
        data_array = df_trial.values  # shape: (61, 1000)
        gfp = get_gfp(data_array)
        all_gfp.append(gfp)
    return all_gfp

def get_all_peaks(all_gfp, timepoints, distance_10):
    """
    Extract peaks from all trials' GFP data.
    
    Parameters:
        all_gfp (list): List of GFP arrays for each trial.
        timepoints (np.ndarray): Array of timepoints corresponding to the GFP data.
        distance_10 (int): Minimum distance between peaks in samples.
        
    Returns:
        tuple: Lists of peaks, local maxima, and their corresponding timepoints for each trial.
    """
    all_peaks = []
    all_shapes = []
    all_gfp_local_max = []
    all_gfp_timepoints = []
    
    for nt_trial in range(len(all_gfp)):
        gfp = all_gfp[nt_trial]
        peaks, _ = find_peaks(gfp, distance=distance_10)  # Adjust height as needed
        gfp_local_max = gfp[peaks]
        gfp_timepoints = timepoints[peaks]
        
        all_peaks.append(peaks)
        all_shapes.append(peaks.shape)
        all_gfp_local_max.append(gfp_local_max)
        all_gfp_timepoints.append(gfp_timepoints)
    
    return all_peaks, all_shapes, all_gfp_local_max, all_gfp_timepoints

def get_all_topographies(df, all_peaks):
    """
    Extract topographies for each trial based on the peaks found in the GFP data.
    
    Parameters:
        df (pd.DataFrame): DataFrame with MultiIndex (trial, channel) and timepoint columns.
        all_peaks (list): List of peak indices for each trial.
        
    Returns:
        list: List of topographies for each trial.
    """
    all_topographies = []
    for nt_trial in range(df.index.get_level_values('trial').nunique()):
        df_trial = df.xs(nt_trial, level='trial')  # shape: (61 channels, 1000 time points)
        data_array = df_trial.values  # shape: (61, 1000)
        topographies = data_array[:, all_peaks[nt_trial]].T  # shape: (n_peaks, n_channels)
        all_topographies.append(topographies)
    return all_topographies

def get_microstate_labels(all_topographies, cluster_templates, all_gfp_combine, position=0):
    """
    Get microstate labels for each topography.      
    
    Parameters:
    all_topographies (numpy.ndarray): 2D array of shape (n_peaks, n_channels)
    cluster_templates (numpy.ndarray): 2D array of shape (n_clusters, n_channels)
    all_gfp_combine (numpy.ndarray): 1D array of GFP values for all datapoints

    Returns:
    numpy.ndarray: Array of microstate labels
    """
    # get cluster templates
    gmd_values = []
    for i in range(cluster_templates.shape[0]):
        gmd = get_gmd(all_topographies.T, all_gfp_combine, position, i)
        gmd_values.append(gmd)
    gmd_values = np.array(gmd_values)
    gmd_labels = np.argmin(gmd_values, axis=0)  # shape: (n_peaks,)

    return gmd_labels, gmd_values

# -------------------------------- Find best K cluster --------------------------------
def get_best_k_clusters(all_topographies, max_k=10):
    """
    Find the best K clusters using silhouette score.
    
    Parameters:
        all_topographies (np.ndarray): Array of topographies.
        max_k (int): Maximum number of clusters to consider.
    
    Returns:
        int: Best number of clusters based on silhouette score.
    """
    all_scores = []
    best_k = 2
    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(all_topographies)
    min_score = silhouette_score(all_topographies, kmeans.labels_)
    all_scores.append(min_score)
    for k in range(3, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_topographies)
        score = silhouette_score(all_topographies, kmeans.labels_)
        all_scores.append(score)
        if score < min_score:
            min_score = score
            best_k = k
        
    return best_k, all_scores

def load_all_data(subjects_list=None, do_all=False, data_path='../../Data/',output_folder='../../Output/'):
    """
    Load all data from the specified subjects.
    Parameters:
        subjects_list (list): List of subject indices to load data for. If None, loads all subjects.
        do_all (bool): If True, loads data for all subjects as a single array.
        data_path (str): Path to the data directory.
        output_folder (str): Path to the output directory where results will be saved.
    Returns:
        tuple: A tuple containing two lists:
            - all_data: List of data arrays for each subject.
            - all_y: List of labels for each subject. {0 = rest, 1 = open, 2 = close}
    """
    
    # load all data from rest_close
    folder_path = 'ica_rest_close/'
    output_path = os.path.join(output_folder, folder_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    n_subjects = 50
    if subjects_list is None and not do_all:
        subjects_list = list(range(n_subjects))
    elif do_all:
        subjects_list = [0]
    all_data_close = []
    all_y_close = []
    for i in subjects_list:
        if do_all:
            id_name = '_all'  # Use '_all' for all subjects
        else:
            id_name = f'{i:03d}'  # Format id as three digits
        file_name = f's{id_name}.npy'
        data = get_file_path(data_path, folder_path, file_name)
        all_data_close.append(data)
        file_name_y = f'y{id_name}.npy'
        data_y = get_file_path(data_path, folder_path, file_name_y)
        for i in range(len(data_y)):
            if data_y[i] == 1:
                data_y[i] = 2
        all_y_close.append(data_y)

    # load all data from rest_open
    folder_path = 'ica_rest_open/'
    output_path = '../Output/' + folder_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_data_open = []
    all_y_open = []
    for i in subjects_list:
        if do_all:
            id_name = '_all'
        else:
            id_name = f'{i:03d}'  # Format id as three digits
        file_name = f's{id_name}.npy'
        data = get_file_path(data_path, folder_path, file_name)
        all_data_open.append(data)
        file_name_y = f'y{id_name}.npy'
        data_y = get_file_path(data_path, folder_path, file_name_y)
        all_y_open.append(data_y)

    # concatenate all data
    # into one list
    all_data = []
    all_y = []
    for i in subjects_list:
        data_close = all_data_close[i]
        data_open = all_data_open[i]
        data = np.concatenate((data_close, data_open), axis=0)
        all_data.append(data)
        data_y_close = all_y_close[i]
        data_y_open = all_y_open[i]
        data_y = np.concatenate((data_y_close, data_y_open), axis=0)
        all_y.append(data_y)

    return all_data, all_y

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import random

def get_val_ids(seed, test_id, excluded_from_training):
    rng = random.Random(seed)  # local RNG, avoids global randomness issues
    val_candidates = [i for i in range(50) if i != test_id and i not in excluded_from_training]
    val_ids = rng.sample(val_candidates, 4)
    return val_candidates, val_ids

def get_consistent_split_indices(data_length, test_id, train_ratio=0.9, base_seed=123):
    """
    Generate consistent train/test split indices for a given subject.
    
    Parameters:
    -----------
    data_length : int
        Total number of samples in the dataset
    test_id : int
        Subject ID (used to generate subject-specific but consistent seed)
    train_ratio : float, default=0.9
        Proportion of data to use for training/fine-tuning
    base_seed : int, default=123
        Base seed for reproducibility
    
    Returns:
    --------
    train_indices : torch.Tensor
        Indices for training/fine-tuning data
    test_indices : torch.Tensor
        Indices for testing data
    """
    import torch
    import numpy as np
    
    # Set seed for consistent splitting
    split_seed = base_seed + test_id
    torch.manual_seed(split_seed)
    np.random.seed(split_seed)
    
    # Generate random permutation
    indices = torch.randperm(data_length)
    
    # Split indices
    train_size = int(train_ratio * data_length)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    return train_indices, test_indices

# Alternative function that returns the actual split data
def split_subject_data_consistently(x_data, y_data, test_id, train_ratio=0.9, base_seed=123):
    """
    Split subject data consistently for fine-tuning experiments.
    
    Parameters:
    -----------
    x_data : torch.Tensor
        Feature data
    y_data : torch.Tensor
        Label data
    test_id : int
        Subject ID
    train_ratio : float, default=0.9
        Proportion of data for training
    base_seed : int, default=123
        Base seed for reproducibility
    
    Returns:
    --------
    x_train, y_train, x_test, y_test : torch.Tensor
        Split data tensors
    """
    train_indices, test_indices = get_consistent_split_indices(
        len(x_data), test_id, train_ratio, base_seed
    )
    
    x_train = x_data[train_indices]
    y_train = y_data[train_indices]
    x_test = x_data[test_indices]
    y_test = y_data[test_indices]
    
    return x_train, y_train, x_test, y_test

# Function to save/load split indices for ultimate consistency
def save_split_indices(output_path, all_train_indices, all_test_indices, filename='split_indices.pkl'):
    """Save split indices to file for reuse across scripts."""
    import pickle
    import os
    
    split_data = {
        'train_indices': all_train_indices,
        'test_indices': all_test_indices
    }
    
    filepath = os.path.join(output_path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"Split indices saved to {filepath}")

def load_split_indices(output_path, filename='split_indices.pkl'):
    """Load split indices from file."""
    import pickle
    import os
    
    filepath = os.path.join(output_path, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            split_data = pickle.load(f)
        print(f"Split indices loaded from {filepath}")
        return split_data['train_indices'], split_data['test_indices']
    else:
        print(f"No split indices file found at {filepath}")
        return None, None



def test_model_results(model_results, data, labels, type_of_subject, output_path, 
                      model_type='single', feature_extraction_models=None, device='cuda', 
                      batch_size=32):
    """
    Unified function to test any models for all subject types
    
    Args:
        model_results: Dictionary with 'models' (single) or 'subject_results' (multimodal)
        data: List of data for all subjects (raw EEG, microstate, or None for multimodal)
        labels: List of labels for all subjects  
        type_of_subject: 'dependent', 'independent', or 'adaptive'
        output_path: Path to save test results
        model_type: 'single' (DeepConvNet/MicroSNet) or 'multimodal' (combined models)
        feature_extraction_models: For multimodal: {'raw': results, 'ms': ms_results, 'raw_data': all_data, 'ms_data': finals_ls}
        device: Device to run computations on
        batch_size: Batch size for testing
        
    Returns:
        Dictionary with comprehensive test results and summary statistics
    """
    
    print(f"Starting unified model testing for {type_of_subject} {model_type} models...")
    
    def test_single_model_on_data(model, data_input, test_labels, subject_id):
        """Test single model (DeepConvNet/MicroSNet) on data"""
        
        # Handle EEGClassifier objects (from braindecode)
        if hasattr(model, 'module'):
            # EEGClassifier case - get the underlying network
            net = model.module
        else:
            # Regular PyTorch model
            net = model
        
        net.eval()
        all_preds = []
        all_probs = []
        
        # Create data loader
        dataset = TensorDataset(data_input, test_labels)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = net(batch_data)
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return calculate_metrics(test_labels.numpy(), all_preds, all_probs, subject_id)
    
    def test_multimodal_model_on_data(model, raw_features, ms_features, test_labels, subject_id):
        """Test multimodal model on extracted features"""
        
        model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(test_labels), batch_size):
                raw_batch = raw_features[i:i+batch_size].to(device)
                ms_batch = ms_features[i:i+batch_size].to(device)
                
                outputs = model(raw_batch, ms_batch)
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return calculate_metrics(test_labels.numpy(), all_preds, all_probs, subject_id)
    
    def extract_features_for_multimodal(subject_id):
        """Extract features from raw EEG and microstate models for multimodal testing"""
        
        from .my_models import FeatureExtractor
        
        # Extract raw EEG features
        raw_model = feature_extraction_models['raw']['models'][subject_id]
        raw_data = feature_extraction_models['raw_data'][subject_id]
        
        if not isinstance(raw_data, torch.Tensor):
            raw_tensor = torch.tensor(raw_data, dtype=torch.float32)
        else:
            raw_tensor = raw_data.clone()
            
        if len(raw_tensor.shape) == 4 and raw_tensor.shape[1] == 1:
            raw_tensor = raw_tensor.squeeze(1)
        
        # Handle EEGClassifier for feature extraction
        if hasattr(raw_model, 'module'):
            raw_net = raw_model.module
        else:
            raw_net = raw_model
            
        raw_extractor = FeatureExtractor(raw_net).to(device)
        raw_extractor.eval()
        
        raw_features_list = []
        with torch.no_grad():
            for i in range(0, len(raw_tensor), batch_size):
                batch_x = raw_tensor[i:i+batch_size].to(device)
                batch_features = raw_extractor(batch_x)
                raw_features_list.append(batch_features.cpu())
        raw_features = torch.cat(raw_features_list, dim=0)
        
        # Extract microstate features
        ms_model = feature_extraction_models['ms']['models'][subject_id]
        ms_data = feature_extraction_models['ms_data'][subject_id]
        
        if not isinstance(ms_data, torch.Tensor):
            ms_tensor = torch.tensor(ms_data, dtype=torch.float32)
        else:
            ms_tensor = ms_data.clone()
            
        if len(ms_tensor.shape) == 4 and ms_tensor.shape[1] == 1:
            ms_tensor = ms_tensor.squeeze(1)
        
        # Handle EEGClassifier for microstate models too (if applicable)
        if hasattr(ms_model, 'module'):
            ms_net = ms_model.module
        else:
            ms_net = ms_model
            
        ms_extractor = FeatureExtractor(ms_net).to(device)
        ms_extractor.eval()
        
        ms_features_list = []
        with torch.no_grad():
            for i in range(0, len(ms_tensor), batch_size):
                batch_x = ms_tensor[i:i+batch_size].to(device)
                batch_features = ms_extractor(batch_x)
                ms_features_list.append(batch_features.cpu())
        ms_features = torch.cat(ms_features_list, dim=0)
        
        return raw_features, ms_features
    
    def calculate_metrics(true_labels, predictions, probabilities, subject_id):
        """Calculate comprehensive test metrics"""
        
        test_acc = accuracy_score(true_labels, predictions) * 100
        
        n_classes = len(np.unique(true_labels))
        class_names = [f'Class_{i}' for i in range(n_classes)]
        
        clf_report = classification_report(
            true_labels, predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        f1_macro = clf_report['macro avg']['f1-score'] if 'macro avg' in clf_report else 0
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        print(f"  Subject {subject_id} Test Results:")
        print(f"    Overall Accuracy: {test_acc:.2f}%")
        print(f"    F1 Macro: {f1_macro*100:.2f}%")
        print(f"    Total test samples: {len(true_labels)}")
        
        # Print per-class metrics
        for i in range(n_classes):
            if f'Class_{i}' in clf_report:
                precision = clf_report[f'Class_{i}']['precision'] * 100
                recall = clf_report[f'Class_{i}']['recall'] * 100
                f1 = clf_report[f'Class_{i}']['f1-score'] * 100
                print(f"    Class {i}: Precision={precision:.2f}%, Recall={recall:.2f}%, F1={f1:.2f}%")
        
        return {
            'subject_id': subject_id,
            'test_accuracy': test_acc,
            'f1_macro': f1_macro * 100,
            'total_samples': len(true_labels),
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'classification_report': clf_report,
            'confusion_matrix': conf_matrix,
            'n_classes': n_classes
        }
    
    # Determine number of subjects and models source
    if model_type == 'multimodal':
        n_subjects = len(model_results['subject_results'])
        models_source = model_results['subject_results']
    else:
        n_subjects = len(model_results['models'])
        models_source = model_results['models']
    
    all_test_results = []
    
    # Main testing loop
    for subject_id in range(n_subjects):
        print(f"\n{'='*60}")
        print(f"Testing Subject {subject_id} ({type_of_subject} {model_type})")
        print(f"{'='*60}")
        
        subject_labels = torch.tensor(labels[subject_id], dtype=torch.long)
        
        if model_type == 'single':
            # Prepare data for single models
            subject_data = torch.tensor(data[subject_id], dtype=torch.float32)
            if len(subject_data.shape) == 4 and subject_data.shape[1] == 1:
                subject_data = subject_data.squeeze(1)
            
            current_model = models_source[subject_id]
            
            # Test based on subject type
            if type_of_subject == 'dependent':
                # Use train/test split (same as training)
                set_seed(42)
                indices = np.arange(len(subject_labels))
                train_idx, test_idx = train_test_split(
                    indices, test_size=0.2, random_state=42,
                    stratify=subject_labels.numpy() if len(np.unique(subject_labels.numpy())) > 1 else None
                )
                test_data = subject_data[test_idx]
                test_labels = subject_labels[test_idx]
                
            elif type_of_subject == 'independent':
                # Test on entire subject (LOSO was used in training)
                test_data = subject_data
                test_labels = subject_labels
                
            elif type_of_subject == 'adaptive':
                # Test on 10% final split
                all_train_indices, all_test_indices = load_split_indices(
                    output_path, filename=f'{type_of_subject}_split_indices.pkl'
                )
                if all_train_indices is not None:
                    test_idx = all_test_indices[subject_id]
                else:
                    # Fallback
                    indices = np.arange(len(subject_labels))
                    _, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
                
                test_data = subject_data[test_idx]
                test_labels = subject_labels[test_idx]
            
            test_result = test_single_model_on_data(current_model, test_data, test_labels, subject_id)
            
        else:  # multimodal
            print(f"Extracting features...")
            raw_features, ms_features = extract_features_for_multimodal(subject_id)
            
            if type_of_subject == 'dependent':
                # Use train/test split (same as training)
                set_seed(42)
                indices = np.arange(len(subject_labels))
                train_idx, test_idx = train_test_split(
                    indices, test_size=0.2, random_state=42,
                    stratify=subject_labels.numpy() if len(np.unique(subject_labels.numpy())) > 1 else None
                )
                test_raw = raw_features[test_idx]
                test_ms = ms_features[test_idx]
                test_labels = subject_labels[test_idx]
                
                model = models_source[subject_id]['model']
                test_result = test_multimodal_model_on_data(model, test_raw, test_ms, test_labels, subject_id)
                
            elif type_of_subject == 'independent':
                # Test on entire subject
                model = models_source[subject_id]['model']
                test_result = test_multimodal_model_on_data(model, raw_features, ms_features, subject_labels, subject_id)
                
            elif type_of_subject == 'adaptive':
                # Test on 10% final split for both base and fine-tuned models
                all_train_indices, all_test_indices = load_split_indices(
                    output_path, filename=f'{type_of_subject}_split_indices.pkl'
                )
                if all_train_indices is not None:
                    test_idx = all_test_indices[subject_id]
                else:
                    indices = np.arange(len(subject_labels))
                    _, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
                
                test_raw = raw_features[test_idx]
                test_ms = ms_features[test_idx]
                test_labels = subject_labels[test_idx]
                
                # Test both base and fine-tuned models
                base_model = models_source[subject_id]['model']
                finetuned_model = models_source[subject_id].get('finetuned_model', base_model)
                
                base_result = test_multimodal_model_on_data(base_model, test_raw, test_ms, test_labels, subject_id)
                finetuned_result = test_multimodal_model_on_data(finetuned_model, test_raw, test_ms, test_labels, subject_id)
                
                # Combine results for adaptive
                test_result = {
                    'subject_id': subject_id,
                    'base_test_accuracy': base_result['test_accuracy'],
                    'finetuned_test_accuracy': finetuned_result['test_accuracy'],
                    'base_f1_macro': base_result['f1_macro'],
                    'finetuned_f1_macro': finetuned_result['f1_macro'],
                    'improvement': finetuned_result['test_accuracy'] - base_result['test_accuracy'],
                    'total_samples': base_result['total_samples'],
                    'base_predictions': base_result['predictions'],
                    'finetuned_predictions': finetuned_result['predictions'],
                    'true_labels': base_result['true_labels'],
                    'base_probabilities': base_result['probabilities'],
                    'finetuned_probabilities': finetuned_result['probabilities'],
                    'base_classification_report': base_result['classification_report'],
                    'finetuned_classification_report': finetuned_result['classification_report'],
                    'base_confusion_matrix': base_result['confusion_matrix'],
                    'finetuned_confusion_matrix': finetuned_result['confusion_matrix'],
                    'n_classes': base_result['n_classes']
                }
        
        all_test_results.append(test_result)
    
    # Calculate and print summary
    summary = calculate_summary_stats(all_test_results, type_of_subject, model_type)
    
    # Save results
    model_name = model_type
    test_output_file = os.path.join(output_path, f'{type_of_subject}_{model_name}_test_results_ica_rest_all.npy')
    detailed_results = {
        'test_results': all_test_results,
        'summary': summary,
        'type_of_subject': type_of_subject,
        'model_type': model_type
    }
    np.save(test_output_file, detailed_results)
    
    # Create and save CSV
    summary_df = create_summary_csv(all_test_results, type_of_subject, model_type)
    summary_csv_file = os.path.join(output_path, f'{type_of_subject}_{model_name}_test_summary.csv')
    summary_df.to_csv(summary_csv_file, index=False)
    
    print(f"\nTest results saved:")
    print(f"  Detailed: {test_output_file}")
    print(f"  Summary CSV: {summary_csv_file}")
    
    return detailed_results


def calculate_summary_stats(all_test_results, type_of_subject, model_type):
    """Calculate and print summary statistics"""
    
    print(f"\n{'='*60}")
    print(f"OVERALL TEST RESULTS SUMMARY ({type_of_subject.upper()} {model_type.upper()})")
    print(f"{'='*60}")
    
    if type_of_subject == 'adaptive' and model_type == 'multimodal':
        # Adaptive multimodal: base + fine-tuned
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
                  f"(improvement: {res['improvement']:+.2f}%) ({res['total_samples']} samples)")
        
        summary = {
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
            'total_samples': np.sum(total_samples),
            'individual_base_accuracies': base_accuracies,
            'individual_finetuned_accuracies': finetuned_accuracies,
            'individual_base_f1_macros': base_f1_macros,
            'individual_finetuned_f1_macros': finetuned_f1_macros,
            'individual_improvements': improvements,
            'individual_sample_counts': total_samples
        }
        
        print(f"\nOverall Statistics:")
        print(f"  Mean Base Accuracy: {summary['mean_base_accuracy']:.2f}% ± {summary['std_base_accuracy']:.2f}%")
        print(f"  Mean Fine-tuned Accuracy: {summary['mean_finetuned_accuracy']:.2f}% ± {summary['std_finetuned_accuracy']:.2f}%")
        print(f"  Mean Improvement: {summary['mean_improvement']:+.2f}% ± {summary['std_improvement']:.2f}%")
        
    else:
        # Standard: single models or non-adaptive multimodal
        test_accuracies = [res['test_accuracy'] for res in all_test_results]
        f1_macros = [res['f1_macro'] for res in all_test_results]
        total_samples = [res['total_samples'] for res in all_test_results]
        
        print(f"Individual Subject Results:")
        for i, (acc, f1) in enumerate(zip(test_accuracies, f1_macros)):
            print(f"  Subject {i}: Accuracy={acc:.2f}%, F1-Macro={f1:.2f}% ({total_samples[i]} samples)")
        
        summary = {
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
        
        print(f"\nOverall Statistics:")
        print(f"  Mean Accuracy: {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}%")
        print(f"  Mean F1-Macro: {summary['mean_f1_macro']:.2f}% ± {summary['std_f1_macro']:.2f}%")
    
    return summary


def create_summary_csv(all_test_results, type_of_subject, model_type):
    """Create summary DataFrame"""
    
    if type_of_subject == 'adaptive' and model_type == 'multimodal':
        return pd.DataFrame({
            'Subject': range(len(all_test_results)),
            'Base_Test_Accuracy': [res['base_test_accuracy'] for res in all_test_results],
            'Finetuned_Test_Accuracy': [res['finetuned_test_accuracy'] for res in all_test_results],
            'Base_F1_Macro': [res['base_f1_macro'] for res in all_test_results],
            'Finetuned_F1_Macro': [res['finetuned_f1_macro'] for res in all_test_results],
            'Improvement': [res['improvement'] for res in all_test_results],
            'Sample_Count': [res['total_samples'] for res in all_test_results],
            'N_Classes': [res['n_classes'] for res in all_test_results]
        })
    else:
        return pd.DataFrame({
            'Subject': range(len(all_test_results)),
            'Test_Accuracy': [res['test_accuracy'] for res in all_test_results],
            'F1_Macro': [res['f1_macro'] for res in all_test_results],
            'Sample_Count': [res['total_samples'] for res in all_test_results],
            'N_Classes': [res['n_classes'] for res in all_test_results]
        })


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_memory_usage():
    """Get current memory usage information"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    return {
        'process_rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'process_vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'system_total_gb': system_memory.total / 1024 / 1024 / 1024,  # Total system RAM in GB
        'system_available_gb': system_memory.available / 1024 / 1024 / 1024,  # Available RAM in GB
        'system_used_percent': system_memory.percent  # Used RAM percentage
    }

def print_memory_status(stage=""):
    """Print current memory status"""
    mem = get_memory_usage()
    print(f"\n{'='*50}")
    print(f"MEMORY STATUS {stage}")
    print(f"{'='*50}")
    print(f"Process Memory (RSS): {mem['process_rss_mb']:.1f} MB")
    print(f"Process Memory (VMS): {mem['process_vms_mb']:.1f} MB")
    print(f"System Total RAM: {mem['system_total_gb']:.1f} GB")
    print(f"System Available RAM: {mem['system_available_gb']:.1f} GB")
    print(f"System RAM Usage: {mem['system_used_percent']:.1f}%")
    
    # GPU memory if CUDA is available
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
        print(f"GPU Memory Allocated: {gpu_memory_allocated:.1f} MB")
        print(f"GPU Memory Reserved: {gpu_memory_reserved:.1f} MB")
    
    print(f"{'='*50}\n")