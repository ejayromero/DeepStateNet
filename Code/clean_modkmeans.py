import os
import sys

import pickle
import numpy as np

n_clusters = 10
max_k_clusters = 50  # Maximum number of clusters to evaluate
n_subjects = 50  # Number of subjects

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

for i in range(n_subjects):
    print(f"Zipping modkmeans for subject {i}")
    id_name = f'{i:03d}'
    doc_name = f'modkmeans_s{id_name}.npy'
    results_path = os.path.join(input_path, modkmeans_path, doc_name)
    # pickle_name = f'modkmeans_s{id_name}.pkl'
    # pickle_path = os.path.join(input_path, modkmeans_path, pickle_path)
    if os.path.exists(results_path):
        results = np.load(results_path, allow_pickle=True).item()
        # print(results)
        print(f"Succesfully loaded file of s{id_name}")
        cluster_numbers = results['cluster_number']
        # print(cluster_numbers)
        scores = results['scores']
        # print(scores)
        kmeans_results = {
            'cluster_number': cluster_numbers,
            'scores': scores,
            # 'kmeans_model': modkmeans,
            # 'microstate_sequence': microstate_sequences
        }

        print(f"The file '{doc_name}' has been modified successfully.")
        np.save(results_path, kmeans_results)
        print(f'The file {doc_name} has been saved successfully.')
        
print('End of cleaning')