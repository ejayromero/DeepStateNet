import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

import mne

from pycrostates.cluster import ModKMeans
from pycrostates.preprocessing import extract_gfp_peaks
from sklearn.preprocessing import normalize

from pycrostates.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    dunn_score,
    davies_bouldin_score,
)

# change directory go into Notebooks folder
if os.path.basename(os.getcwd()) != 'Notebooks':
    if os.path.basename(os.getcwd()) == 'lib':
        os.chdir(os.path.join(os.getcwd(), '..', 'Notebooks'))
    else:
        os.chdir(os.path.join(os.getcwd(), 'Notebooks'))
else:
    # if already in Notebooks folder, do nothing
    pass

from lib import my_functions as mf

id = 0

data_path = '../Data/'
folder_path = 'ica_rest_close/'
output_path = '../Output/' + folder_path
if not os.path.exists(output_path):
    os.makedirs(output_path)

id_name = f'{id:03d}'  # Format id as three digits
# id_name ='_all'
file_name = f's{id_name}.npy'

# load all data from rest_close
start_subject = 20
n_subjects = 50
random_state = 42  # Set a random state for reproducibility
max_k_clusters = 20  # Maximum number of clusters to evaluate
cluster_numbers = range(2, max_k_clusters + 1)  # Range of cluster numbers to evaluate

all_data_close = []
all_y_close = []
for i in range(0, n_subjects):
    id_name = f'{i:03d}'  # Format id as three digits
    file_name = f's{id_name}.npy'
    data = mf.get_file_path(data_path, folder_path, file_name)
    all_data_close.append(data)
    file_name_y = f'y{id_name}.npy'
    data_y = mf.get_file_path(data_path, folder_path, file_name_y)
    for i in range(len(data_y)):
        if data_y[i] == 1:
            data_y[i] = 2
    all_y_close.append(data_y)

rest_ch_ls = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3',
            'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6',
            'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5',
            'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz',
            'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
sampling_rate = 250 # Hz

for id in range(start_subject, n_subjects):
    print(f'Extracting microstates for subject {id}...')
    
    # Example values, adjust according to your data
    eeg_data = np.array(all_data_close[id])
    n_trials, _, n_channels, n_times = eeg_data.shape

    info = mne.create_info(ch_names=rest_ch_ls, sfreq=sampling_rate, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Remove singleton dimension (if shape is (n_trials, 1, n_channels, n_times))
    eeg_data = eeg_data[:, 0, :, :]  # shape becomes (n_trials, n_channels, n_times)

    # Create MNE Epochs object
    epochs = mne.EpochsArray(eeg_data, info)
    # Extract Global Field Power (GFP) peaks to reduce number of samples
    gfp_peaks = extract_gfp_peaks(epochs)

    scores = {
        "GEV": np.zeros(len(cluster_numbers)),
        "Silhouette": np.zeros(len(cluster_numbers)),
        "Calinski-Harabasaz": np.zeros(len(cluster_numbers)),
        "Dunn": np.zeros(len(cluster_numbers)),
        "Davies-Bouldin": np.zeros(len(cluster_numbers)),
    }

    for k, n_clusters in enumerate(cluster_numbers):
        print(f'Processing subject {id}, number of clusters: {n_clusters}')
        # fit K-means algorithm with a set number of cluster centers
        ModK = ModKMeans(n_clusters=n_clusters, random_state=random_state)
        ModK.fit(gfp_peaks, n_jobs=-1, verbose="WARNING")

        # compute scores
        scores["GEV"][k] = ModK.GEV_
        scores["Silhouette"][k] = silhouette_score(ModK)
        scores["Calinski-Harabasaz"][k] = calinski_harabasz_score(ModK)
        scores["Dunn"][k] = dunn_score(ModK)
        scores["Davies-Bouldin"][k] = davies_bouldin_score(ModK)

        print(f"Scores for subject {id}, number of clusters {n_clusters}:")
        print(f"    GEV: {scores['GEV'][k]:.3f}")
        print(f"    Silhouette: {scores['Silhouette'][k]:.3f}")
        print(f"    Calinski-Harabasz: {scores['Calinski-Harabasaz'][k]:.3f}")
        print(f"    Dunn: {scores['Dunn'][k]:.3f}")
        print(f"    Davies-Bouldin: {scores['Davies-Bouldin'][k]:.3f}")

    # invert davies-bouldin scores
    scores["Davies-Bouldin"] = 1 / (1 + scores["Davies-Bouldin"])

    # normalize scores using sklearn

    normalized_scores = {score: normalize(value[:, np.newaxis], axis=0).ravel()
            for score, value in scores.items()
            if score != "GEV"}  # Normalize all scores except GEV
    scores.update(normalized_scores)


    # set width of a bar and define colors
    barWidth = 0.18
    colors = sns.color_palette('colorblind', len(scores))  # Microstate colors

    # create figure
    plt.figure(figsize=(10, 8))
    # create the position of the bars on the X-axis
    x = [[elt + k * barWidth for elt in np.arange(len(cluster_numbers))]
        for k in range(len(scores))]
    # create plots
    for k, (score, values) in enumerate(scores.items()):
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
    output_im_score = os.path.join(output_path, 'scores', f'Cluster_score_s{id_name}.png')
    if not os.path.exists(os.path.join(output_path, 'scores')):
        os.makedirs(os.path.join(output_path, 'scores'))
    plt.savefig(output_im_score, dpi=300, bbox_inches='tight')
    plt.close()

    output_scores = os.path.join(output_path, 'scores', f'Cluster_score_s{id_name}.npy')
    np.save(output_scores, scores)
    print(f'Scores saved to {output_scores}.')
    print(f'Finished extracting microstates for subject {id}.')

# End of the script
print("All subjects processed.")