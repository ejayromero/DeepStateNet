import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

import pickle
from sklearn.preprocessing import normalize
from pycrostates.cluster import ModKMeans
from pycrostates.metrics import calinski_harabasz_score, davies_bouldin_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.my_stats_functions import modkmeans_exact_spatial_score

n_clusters = 10
max_k_clusters = 50  # Maximum number of clusters to evaluate
cluster_numbers = range(4, max_k_clusters + 1)  # Range of cluster numbers to evaluate
n_subjects = 50  # Number of subjects
n_init = 100
max_iter =300
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

for i in range(n_subjects):
    print(f"Processing subject {i}")
    if do_all:
        id_name = '_all'
    else:
        id_name = f'{i:03d}'
    results_path = os.path.join(input_path, modkmeans_path,f'modkmeans_s{id_name}.npy')
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

    # scores = {
    #     "GEV": [],
    #     "Silhouette": [],
    #     "Calinski-Harabasaz": [],
    #     "Dunn": [],
    #     "Davies-Bouldin": []
    # }

    if find_best_k:
        scores = {
            "GEV": np.zeros(len(cluster_numbers)),
            # "Silhouette": np.zeros(len(cluster_numbers)),
            "Calinski-Harabasaz": np.zeros(len(cluster_numbers)),
            # "Dunn": np.zeros(len(cluster_numbers)),
            "Davies-Bouldin": np.zeros(len(cluster_numbers)),
            "Inv-Davies-Bouldin": np.zeros(len(cluster_numbers)),
            "Spatial_correlation": np.zeros(len(cluster_numbers))
        }

        for k, n_clusters in enumerate(cluster_numbers):
            print(f'Processing subject {id_name}, number of clusters: {n_clusters}')
            # fit K-means algorithm with a set number of cluster centers
            ModK = ModKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=random_state)
            ModK.fit(gfp_peaks, n_jobs=-1, verbose="WARNING")

            # compute scores
            print(f"Scores for subject {id_name}, number of clusters {n_clusters}:")

            scores["GEV"][k] = ModK.GEV_
            print(f"    GEV: {scores['GEV'][k]:.3f}")

            # scores["Silhouette"][k] = silhouette_score(ModK)
            # print(f"    Silhouette: {scores['Silhouette'][k]:.3f}")

            scores["Calinski-Harabasaz"][k] = calinski_harabasz_score(ModK)
            print(f"    Calinski-Harabasz: {scores['Calinski-Harabasaz'][k]:.3f}")

            # scores["Dunn"][k] = dunn_score(ModK)
            # print(f"    Dunn: {scores['Dunn'][k]:.3f}")

            scores["Davies-Bouldin"][k] = davies_bouldin_score(ModK)
            print(f"    Davies-Bouldin: {scores['Davies-Bouldin'][k]:.3f}")
            scores["Inv-Davies-Bouldin"][k] = 1 / (1 + scores["Davies-Bouldin"][k])
            print(f"    Invert Davies-Bouldin: {scores['Inv-Davies-Bouldin'][k]:.3f}")

            scores["Spatial_correlation"][k] = modkmeans_exact_spatial_score(ModK)
            print(f"    Spatial Correlation Consistency: {scores['Spatial_correlation'][k]:.3f}")

        # invert davies-bouldin scores
        # scores["Davies-Bouldin"] = 1 / (1 + scores["Davies-Bouldin"])

        kmeans_results = {
            'scores': scores,
            'kmeans_model': modkmeans,
            'microstate_sequence': microstate_sequence
        }
        # normalize scores using sklearn

        normalized_scores = {score: normalize(value[:, np.newaxis], axis=0).ravel()
                for score, value in scores.items()
                if score != "GEV"}  # Normalize all scores except GEV



        # set width of a bar and define colors
        barWidth = 0.18
        colors = sns.color_palette('colorblind', len(normalized_scores))  # Microstate colors
        print('Making Score Summary plot')
        # create figure
        plt.figure(figsize=(10, 8))
        # create the position of the bars on the X-axis
        x = [[elt + k * barWidth for elt in np.arange(len(cluster_numbers))]
            for k in range(len(normalized_scores))]
        # create plots
        for k, (score, values) in enumerate(normalized_scores.items()):
            plt.bar(
                x=x[k],
                height=values,
                width=barWidth,
                edgecolor="grey",
                color=colors[k],
                label=score,
            )
        # add labels and legend
        plt.title(f's{id}: Clustering scores for different number of clusters')
        plt.xlabel("Number of clusters")
        plt.ylabel("Score normalize to unit norm")
        plt.xticks(
            [pos + 1.5 * barWidth for pos in range(len(cluster_numbers))],
            [str(k) for k in cluster_numbers],
        )
        plt.ylim(0, 1.1)  # Set y-axis limit to 0-1
        plt.legend()
        plt.tight_layout()
        # save the model
        id_name = f'{id:03d}'  # Format id as three digits
        output_im_score = os.path.join(modkmeans_path, 'scores', f'Cluster_score_s{id_name}.png')
        if not os.path.exists(os.path.join(modkmeans_path, 'scores')):
            os.makedirs(os.path.join(modkmeans_path, 'scores'))
        plt.savefig(output_im_score, dpi=300, bbox_inches='tight')
        plt.close()

    else:
        # modkmeans = ModKMeans(n_clusters=n_clusters, n_init=50, max_iter=200, random_state=42)
        modkmeans = ModKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=random_state)
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