import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

import pickle
from sklearn.preprocessing import normalize
from pycrostates.cluster import ModKMeans
from pycrostates.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    dunn_score,
    davies_bouldin_score,
)


n_clusters = 10
max_k_clusters = 10  # Maximum number of clusters to evaluate
cluster_numbers = range(4, max_k_clusters + 1)  # Range of cluster numbers to evaluate
n_subjects = 50  # Number of subjects
n_init = 50
max_iter =200
do_all = False  # If True, process all subjects together
random_state = 42

find_best_k = True
input_path = os.path.abspath('Output/ica_rest_all')
if not os.path.exists(input_path):
    os.makedirs(input_path)
gfp_path = os.path.abspath(os.path.join(input_path, 'gfp_peaks'))
#results_path = os.path.join(input_path, 'kmeans_results.npy')
epochs_path = os.path.abspath(os.path.join(input_path, 'epochs'))
if find_best_k:
    models_path = f'find_{max_k_clusters}_clusters'
else:
    models_path = f'{n_clusters}_clusters'
modkmeans_path = os.path.abspath(os.path.join(input_path, 'modkmeans_results', 'models', models_path))
if not os.path.exists(modkmeans_path):
    os.makedirs(modkmeans_path)

i =0
id_name = f'{i:03d}'

def modkmeans_compatible_spatial_score(cluster):
    """
    Compute spatial correlation using the same method as ModKMeans.
    This matches the polarity-invariant dot product approach.
    """
    
    cluster_centers = cluster.cluster_centers_
    labels = cluster.labels_
    fitted_data = cluster.fitted_data
    n_clusters = cluster_centers.shape[0]
    
    all_correlations = []
    
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_timepoints = fitted_data[:, cluster_mask]
        
        if cluster_timepoints.shape[1] == 0:
            continue
            
        cluster_center = cluster_centers[i]
        
        # Method 1: Simple dot product (normalized)
        for j in range(cluster_timepoints.shape[1]):
            timepoint = cluster_timepoints[:, j]
            
            # Normalize both vectors (like correlation)
            norm_timepoint = timepoint / np.linalg.norm(timepoint)
            norm_center = cluster_center / np.linalg.norm(cluster_center)
            
            # Dot product of normalized vectors = cosine similarity ≈ correlation
            dot_product = np.dot(norm_timepoint, norm_center)
            all_correlations.append(abs(dot_product))  # Absolute for polarity invariance
    
    return np.mean(all_correlations)

def modkmeans_exact_spatial_score(cluster):
    """
    Use the EXACT same computation as ModKMeans polarity alignment.
    """
    
    cluster_centers = cluster.cluster_centers_
    labels = cluster.labels_
    fitted_data = cluster.fitted_data
    
    # Replicate the exact ModKMeans polarity alignment
    x = cluster_centers[labels].T  # Expanded cluster centers
    
    # This is the exact computation from ModKMeans
    correlations = (x.T * fitted_data.T).sum(axis=1)  # Dot products
    
    # Take absolute values for polarity invariance
    abs_correlations = np.abs(correlations)
    
    # Normalize by vector norms to get true correlations
    data_norms = np.linalg.norm(fitted_data.T, axis=1)
    center_norms = np.linalg.norm(x.T, axis=1)
    
    # Avoid division by zero
    valid_mask = (data_norms > 0) & (center_norms > 0)
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    # Normalized correlations (this is what ModKMeans actually computes)
    normalized_correlations = abs_correlations[valid_mask] / (data_norms[valid_mask] * center_norms[valid_mask])
    
    return np.mean(normalized_correlations)

def spatial_consistency_score(cluster):
    """
    Simplified spatial score focusing on internal consistency only.
    Often more stable and easier to interpret.
    
    Returns:
    --------
    score : float
        Average correlation between timepoints and their cluster centers
    """
    
    cluster_centers = cluster.cluster_centers_
    labels = cluster.labels_
    fitted_data = cluster.fitted_data
    n_clusters = cluster_centers.shape[0]
    
    all_correlations = []
    
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_timepoints = fitted_data[:, cluster_mask]
        
        for j in range(cluster_timepoints.shape[1]):
            corr = np.corrcoef(cluster_timepoints[:, j], cluster_centers[i])[0, 1]
            all_correlations.append(abs(corr))
    
    return np.mean(all_correlations)

def analyze_memory_usage(cluster):
    """Analyze memory requirements for different metrics."""
    
    n_timepoints = len(cluster.labels_)
    n_channels = cluster.fitted_data.shape[0]
    n_clusters = cluster.cluster_centers_.shape[0]
    
    print(f"Dataset size: {n_channels} channels × {n_timepoints} timepoints")
    
    # Estimate memory usage (in MB)
    base_data_mb = (n_channels * n_timepoints * 8) / (1024**2)  # 8 bytes per float64
    
    print(f"Base data size: {base_data_mb:.1f} MB")
    print(f"Number of clusters: {n_clusters}")
    
    # Memory estimates for different operations
    estimates = {
        'GEV': 0,  # Already computed
        'ModKMeans_exact_spatial': base_data_mb * 0.1,  # Minimal additional memory
        'Standard_correlation': base_data_mb * 0.2,  # Small overhead
        'Standard_sklearn_metrics': base_data_mb * 0.3,  # Moderate overhead
        'Silhouette': base_data_mb * n_timepoints / 1000,  # O(n²) - dangerous!
        'Dunn': base_data_mb * n_timepoints / 1000,  # O(n²) - dangerous!
    }
    
    print("\nEstimated additional memory usage:")
    for method, mb in estimates.items():
        if mb > 1000:
            status = "🔥 DANGER"
        elif mb > 100:
            status = "⚠️ RISKY"
        else:
            status = "✅ SAFE"
        print(f"  {method}: {mb:.1f} MB {status}")
    
    return estimates

def compare_correlation_methods(cluster):
    """Compare different correlation computation methods."""
    
    # Method 1: Standard Pearson correlation
    standard_score = spatial_consistency_score(cluster)
    
    # Method 2: ModKMeans-compatible (normalized dot product)
    compatible_score = modkmeans_compatible_spatial_score(cluster)
    
    # Method 3: Exact ModKMeans computation
    exact_score = modkmeans_exact_spatial_score(cluster)
    
    print("Correlation Method Comparison:")
    print(f"  Standard Pearson: {standard_score:.4f}")
    print(f"  ModKMeans-compatible: {compatible_score:.4f}")
    print(f"  ModKMeans-exact: {exact_score:.4f}")
    
    return {
        'standard': standard_score,
        'compatible': compatible_score,
        'exact': exact_score
    }



print(f"Loading data")
with open(os.path.join(gfp_path, f'gfp_peaks_s{id_name}.pkl'), 'rb') as f:
    gfp_peaks = pickle.load(f)
with open(os.path.join(epochs_path, f'epochs_s{id_name}.pkl'), 'rb') as f:
    epochs = pickle.load(f)
scores = {
    "GEV": np.zeros(len(cluster_numbers)),
    # "Silhouette": np.zeros(len(cluster_numbers)),
    "Calinski-Harabasaz": np.zeros(len(cluster_numbers)),
    # "Dunn": np.zeros(len(cluster_numbers)),
    "Davies-Bouldin": np.zeros(len(cluster_numbers)),
}

for k, n_clusters in enumerate(cluster_numbers):
    print(f'Processing subject {id_name}, number of clusters: {n_clusters}')
    # fit K-means algorithm with a set number of cluster centers
    ModK = ModKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=random_state)
    ModK.fit(gfp_peaks, n_jobs=-1, verbose="WARNING")
    # Check your dataset
    # memory_analysis = analyze_memory_usage(ModK)
    # # compute scores
    # print(f"Scores for subject {id_name}, number of clusters {n_clusters}:")

    # scores["GEV"][k] = ModK.GEV_
    # print(f"    GEV: {scores['GEV'][k]:.3f}")
    # Test the differences
    correlation_comparison = compare_correlation_methods(ModK)
    # # scores["Silhouette"][k] = silhouette_score(ModK)
    # # print(f"    Silhouette: {scores['Silhouette'][k]:.3f}")

    # scores["Calinski-Harabasaz"][k] = calinski_harabasz_score(ModK)
    # print(f"    Calinski-Harabasz: {scores['Calinski-Harabasaz'][k]:.3f}")

    # # scores["Dunn"][k] = dunn_score(ModK)
    # # print(f"    Dunn: {scores['Dunn'][k]:.3f}")

    # scores["Davies-Bouldin"][k] = davies_bouldin_score(ModK)
    # print(f"    Davies-Bouldin: {scores['Davies-Bouldin'][k]:.3f}")
    # #invert
    # invert_db = 1 / (1 + scores["Davies-Bouldin"][k])
    # print(f"    Invert Davies-Bouldin: {invert_db:.3f}")




print("All subjects processed and results saved successfully.")