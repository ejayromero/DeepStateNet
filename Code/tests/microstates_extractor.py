import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.signal import find_peaks

sns.set_theme(style="darkgrid")

# change directory go into Notebooks folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from lib import my_functions as mf

mf.test_lib()

# Check if the data folder and file exist
print('Current working directory:', os.getcwd())

# -------------------------------- Parameters --------------------------------
id = 1

n_subjects = 50
extract_best_k = True  # Extract best K clusters or not
n_clusters = 4  # Number of clusters
max_k_clusters = 40
# -------------------------------- Paths --------------------------------

data_path = '../Data/'
folder_path = 'ica_rest_close/'
output_path = '../Output/' + folder_path
if not os.path.exists(output_path):
    os.makedirs(output_path)


rest_ch_ls = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3',
                'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6',
                'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5',
                'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz',
                'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
sampling_rate = 250 # Hz

best_Ks_clusters = []

print(f'Extracting microstates for {n_subjects} subjects...')

for id in range(0, n_subjects):
    # -------------------------------- Load data --------------------------------
    id_name = f'{id:03d}'  # Format id as three digits
    # id_name ='_all'
    file_name = f's{id_name}.npy'

    print(f'Extracting microstates for subject {id_name}...')

    data = mf.get_file_path(data_path, folder_path, file_name)

    # -------------------------------- Reshape data ------------------------------
    # Remove the singleton dimension (channel)
    data_squeeze = data.squeeze(1)  # Shape becomes (200, 61, 1000)

    # Reshape to long format: (trial, electrode, time)
    n_trials, n_channel, n_timepoints = data_squeeze.shape
    reshaped = data_squeeze.reshape(n_trials * n_channel, n_timepoints)

    # Create a MultiIndex for the rows
    trials = np.repeat(np.arange(n_trials), n_channel)
    channel = np.tile(np.arange(n_channel), n_trials)
    index = pd.MultiIndex.from_arrays([trials, channel+1], names=['trial', 'channel'])

    # Create DataFrame
    df = pd.DataFrame(reshaped, index=index, columns=[f't{t}' for t in range(n_timepoints)])

    # -------------------------------- Extract Global Field Power (GFP) ------------------------------

    all_gfp = mf.extract_all_gfp(df, sampling_rate)
    # print('GFP shape:', np.array(all_gfp).shape)  # (200, 1000)

    # -------------------------------- Extracting all Peaks (Global maxima) --------------------------------
    timepoints = np.arange(0, np.array(all_gfp).shape[1]) / sampling_rate  # Convert to seconds
    distance_10 = 0.01 * sampling_rate  # 10ms in samples

    # get peaks for all trials


    all_peaks, all_shapes, all_gfp_local_max, all_gfp_timepoints = mf.get_all_peaks(all_gfp, timepoints, distance_10)
    # print('All peaks shape:', np.array(all_shapes).shape)  # (200, n_peaks)

    # -------------------------------- Extracting all Topographies --------------------------------

    all_peaks_topographies = np.concatenate(mf.get_all_topographies(df, all_peaks), axis=0)  # shape: (n_trials * n_peaks, n_channels)
    all_peaks_topographies = normalize(all_peaks_topographies, norm='l2', axis=1)
    # print('Topographies shape:', np.array(all_peaks_topographies).shape)  # (200, n_peaks, 61)

    # -------------------------------- Clustering Topographies --------------------------------
    # -------------------------------- Find best K cluster --------------------------------
    if extract_best_k:
        best_k, silhouette_scores = mf.get_best_k_clusters(all_peaks_topographies, max_k=max_k_clusters)
        best_Ks_clusters.append(best_k)

        output_silhouette_file = os.path.join(output_path, f'k{id_name}.npy')
        # Save silhouette scores if best K clusters are extracted
        np.save(output_silhouette_file, silhouette_scores)
        print(f'Silhouette scores saved to {output_silhouette_file}.')
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_peaks_topographies)
        labels = kmeans.labels_  # microstate label for each GFP peak

        all_gfp_combine = np.concatenate(all_gfp, axis=0)
        # print('All GFP shape:', all_gfp_combine.shape)  # (200, 1000)
        all_peaks_combine, _ = find_peaks(all_gfp_combine, distance= distance_10)  # Adjust height as needed
        # print('All peaks shape:', np.array(all_peaks_combine).shape)  # (200, n_peaks)

        all_timepoints = np.arange(0, len(all_gfp_combine)) / sampling_rate  # Convert to seconds

        # -------------------------------- Combine all topographies for each trial ------------------------------
        all_topographies = []
        all_combined_peaks = []
        for nt_trial in range(data.shape[0]):
            df_trial = df.loc[nt_trial]  # shape: (61 channels, 1000 time points)
            data_array = df_trial.values  # shape: (61, 1000)
            topographies = data_array.T  # shape: (n_peaks, n_channels)
            all_topographies.append(topographies)
        # fuse all topographies
        all_topographies = np.concatenate(all_topographies, axis=0)  # shape: (n_trials * n_peaks, n_channels)
        # print('Topographies shape:', np.array(all_topographies).shape)  # (200, n_peaks, 61)

        # -------------------------------- Cluster Templates --------------------------------
        cluster_templates = kmeans.cluster_centers_
        # print('Cluster templates shape:', cluster_templates.shape)

        # -------------------------------- Extract Microstate Labels and GMD Values --------------------------------
        all_microstate_labels = []
        all_gmd_values = []
        for i in range(len(all_topographies)):
            gmd_labels, gmd_values = mf.get_microstate_labels(all_topographies, cluster_templates, all_gfp_combine, position=i)
            all_microstate_labels.append(gmd_labels)
            all_gmd_values.append(gmd_values)

        all_microstate_labels = np.array(all_microstate_labels) # (n_timepoints,)
        # print('All microstate labels shape:', all_microstate_labels.shape) 
        all_gmd_values = np.array(all_gmd_values) # (all_timepoints, n_clusters)
        # print('All GMD values shape:', all_gmd_values.shape)  

        # -------------------------------- Reshape results --------------------------------
        all_microstate_labels_reshaped = all_microstate_labels.reshape(data.shape[0], -1)
        all_microstate_labels_reshaped.shape 
        all_gmd_values_reshaped = reshaped = all_gmd_values.reshape(n_trials, 1000, n_clusters).transpose(0, 2, 1)

        # print('All GMD values reshaped shape:', all_gmd_values_reshaped.shape)  # (n_trials, n_clusters, timepoints)
        # reshape and add singleton dimension
        # to match the original data shape (n_trials, 1, n_clusters, timepoints)
        all_gmd_values_reshaped = all_gmd_values_reshaped[:, np.newaxis, :, :]  # (n_trials, 1, n_clusters, timepoints)
        # print('All GMD values reshaped with singleton dimension shape:', all_gmd_values_reshaped.shape)  # (n_trials, 1, n_clusters, timepoints)

        all_gfp = np.array(all_gfp)
        # print('All GFP shape:', all_gfp.shape)  # (n_trials, timepoints)



        # -------------------------------- Save results --------------------------------
        output_file = os.path.join(output_path, f'm{id_name}.npy')
        # Save the only microstates_labels_reshaped in a .npy file
        np.save(output_file, all_microstate_labels_reshaped)

        # Save the all_gmd_values_reshaped in a .npy file
        output_gmd_file = os.path.join(output_path, f'g{id_name}.npy')
        np.save(output_gmd_file, all_gmd_values_reshaped)

        # save cluster templates
        output_templates_file = os.path.join(output_path, f't{id_name}.npy')
        np.save(output_templates_file, cluster_templates)
        print(f'Extraction done for subject {id_name}. Results saved to {output_file} and {output_gmd_file}.')

if extract_best_k:
    output_best_ks = os.path.join(output_path, 'best_k_clusters.npy')
    # Save the best K clusters to a .npy file
    np.save(output_best_ks, best_Ks_clusters)
    print(f'Best K clusters saved to {output_best_ks}.')

print('Microstates extraction completed for all subjects.')
