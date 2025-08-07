import os
import gc
import seaborn as sns
import numpy as np
sns.set_theme(style="darkgrid")

from pycrostates.cluster import ModKMeans

n_clusters = 12
max_k_clusters = 50  # Maximum number of clusters to evaluate
cluster_numbers = range(4, max_k_clusters + 1)  # Range of cluster numbers to evaluate
n_subjects = 50  # Number of subjects
n_init = 100
max_iter = 300
do_all = False  # If True, process all subjects together
random_state = 42

find_best_k = False
input_path = os.path.abspath('Output/ica_rest_all')
if not os.path.exists(input_path):
    os.makedirs(input_path)
    
gfp_path = os.path.abspath(os.path.join(input_path, 'gfp_peaks'))
epochs_path = os.path.abspath(os.path.join(input_path, 'epochs'))

if find_best_k:
    models_path = f'find_{max_k_clusters}_clusters'
else:
    models_path = f'{n_clusters}_clusters'
    
modkmeans_path = os.path.abspath(os.path.join(input_path, 'modkmeans_results', 'models', models_path))
if not os.path.exists(modkmeans_path):
    os.makedirs(modkmeans_path)

backup_path = "D:/modkmeans_backup/"

# Target cluster number to extract
target_k = 12

for i in range(n_subjects):
    print(f"Processing subject {i}")
    
    if do_all:
        id_name = '_all'
    else:
        id_name = f'{i:03d}'
    
    # Load original results
    results_path = os.path.join(backup_path, f'modkmeans_s{id_name}.npy')
    new_results_path = os.path.join(input_path, modkmeans_path, f'modkmeans_s{id_name}.npy')
    if os.path.exists(new_results_path):
        print(f"Results for subject {i} already exist at {results_path}. Skipping.")
        continue
    
    if os.path.exists(results_path):
        print(f"Loading results for subject {i} from {results_path}")
        results = np.load(results_path, allow_pickle=True).item()
        
        # Get position of target cluster number
        pos = list(results['cluster_number']).index(target_k)
        print(f"  Cluster {target_k} found at position {pos}")
        
        # Extract data for the specific cluster number
        new_results = {
            'cluster_number': target_k,  # Single value instead of range
            'scores': {},
            'kmeans_model': results['kmeans_model'][pos],  # Extract model at position pos
            'microstate_sequence': results['microstate_sequence'][pos]  # Extract sequence at position pos
        }
        
        # Extract scores for the specific cluster number
        for metric_name, metric_values in results['scores'].items():
            new_results['scores'][metric_name] = metric_values[pos]  # Single value instead of array
        
        print(f"  Extracted scores: {new_results['scores']}")
        
        # Save new results
        new_results_path = os.path.join(input_path, modkmeans_path, f'modkmeans_s{id_name}.npy')
        np.save(new_results_path, new_results)
        print(f"  Saved extracted results to {new_results_path}")

        # memory cleanup
        del results
        del new_results
        gc.collect()
    else:
        print(f"Results for subject {i} not found at {results_path}. Skipping.")
    
    print()

print("Extraction complete!")