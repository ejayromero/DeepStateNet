import os
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")

import pickle

from pycrostates.cluster import ModKMeans
from pycrostates.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    dunn_score,
    davies_bouldin_score,
)


n_clusters = 5 
n_subjects = 50  # Number of subjects
do_all = False  # If True, process all subjects together
input_path = 'Output/ica_rest_all'

gfp_path = os.path.join(input_path, 'gfp_peaks')
results_path = os.path.join(input_path, 'kmeans_results.npy')
epochs_path = os.path.join(input_path, 'epochs')


for i in range(50):
    print(f"Processing subject {i}")
    if do_all:
        id_name = '_all'
    else:
        id_name = f'{i:03d}'
    results_path = os.path.join(input_path, f'modkmeans_results/modkmeans_s{id_name}.npy')
    if os.path.exists(results_path):
        print(f"Results for subject {i} already exist at {results_path}. Skipping.")
        continue
    # if not os.path.exists(results_path):
    #     kmeans_results = {
    #         'subjects_list': [],
    #         'scores': [],
    #         'kmeans_models': [],
    #         'microstate_sequences': []
    #     }
    #     np.save(results_path, kmeans_results)
    #     print(f"Created new results file at {results_path}")
    # print(f"Loading results")
    # kmeans_results = np.load(results_path, allow_pickle=True).item()
    # if kmeans_results['subjects_list']:
    #     if kmeans_results['subjects_list'][-1] >= i:
    #         print(f"Skipping subject {i} as it has already been processed.")
    #         continue
    # all_subjects = kmeans_results['subjects_list']
    # all_scores = kmeans_results['scores']
    # all_kmeans_models = kmeans_results['kmeans_models']
    # all_microstate_sequences = kmeans_results['microstate_sequences']

    

    print(f"Loading data")
    with open(os.path.join(gfp_path, f'gfp_peaks_s{id_name}.pkl'), 'rb') as f:
        gfp_peaks = pickle.load(f)
    with open(os.path.join(epochs_path, f'epochs_s{id_name}.pkl'), 'rb') as f:
        epochs = pickle.load(f)

    scores = {
        "GEV": [],
        "Silhouette": [],
        "Calinski-Harabasaz": [],
        "Dunn": [],
        "Davies-Bouldin": []
    }

    # modkmeans = ModKMeans(n_clusters=n_clusters, n_init=50, max_iter=200, random_state=42)
    modkmeans = ModKMeans(n_clusters=n_clusters, random_state=42)
    # Fit the model to the data
    print(f"Fitting ModKMeans for subject {i} with {n_clusters} clusters")
    modkmeans.fit(gfp_peaks, n_jobs=-1, verbose='INFO')
    print(f"Model fitted for subject {i}")
    # Predict microstate sequence for all epochs
    microstate_sequence = modkmeans.predict(epochs, reject_by_annotation=True, factor=10,
                            half_window_size=10, min_segment_length=5,
                            reject_edges=True)

    # compute scores
    # print(f"Computing scores for subject {i}")
    # scores["GEV"] = modkmeans.GEV_
    # scores["Silhouette"] = silhouette_score(modkmeans)
    # scores["Calinski-Harabasaz"] = calinski_harabasz_score(modkmeans)
    # scores["Dunn"] = dunn_score(modkmeans)
    # scores["Davies-Bouldin"] = davies_bouldin_score(modkmeans)
    
    # Save results in a dictionary
    kmeans_results = {
        # 'scores': scores,
        'kmeans_model': modkmeans,
        'microstate_sequence': microstate_sequence
    }
    # all_subjects.append(i)
    # all_scores.append(scores)
    # all_kmeans_models.append(modkmeans)
    # all_microstate_sequences.append(microstate_sequence)

    # kmeans_results = {
    #     'subjects_list': all_subjects,
    #     'scores': all_scores,
    #     'kmeans_models': all_kmeans_models,
    #     'microstate_sequences': all_microstate_sequences
    # }
    np.save(results_path, kmeans_results)
    print(f"Results saved to {results_path} at subject {i}")
print("All subjects processed and results saved successfully.")