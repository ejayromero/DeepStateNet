import os
import numpy as np
# # fix name of image files in Output folder
# list_of_files_to_change = np.arange(20, 30)
# output_path = 'Output/ica_rest_close/scores/'
# if not os.path.exists(output_path):
#     print(f"Output path {output_path} does not exist.")
# else:
#     for i in list_of_files_to_change:
#         old_file_name = f'Cluster_score_s{i:03d}.npy'
#         new_file_name = f'Cluser_score_s{i+20:03d}.npy'
#         old_file_path = os.path.join(output_path, old_file_name)
#         new_file_path = os.path.join(output_path, new_file_name)


#         if os.path.exists(old_file_path):
#             os.rename(old_file_path, new_file_path)
#             print(f"Renamed {old_file_name} to {new_file_name}")
#         else:
#             print(f"File {old_file_name} does not exist in {output_path}.")

kmeans_results_path = 'Output/ica_rest_all/kmeans_results.npy'
if os.path.exists(kmeans_results_path):
    kmeans_results = np.load(kmeans_results_path, allow_pickle=True).item()
    for i in kmeans_results['subjects_list']:
        print(f"Processing subject {i}")
        results_i = {
            'scores': kmeans_results['scores'][i],
            'kmeans_model': kmeans_results['kmeans_models'][i],
            'microstate_sequence': kmeans_results['microstate_sequences'][i]
        }
        id_name = f'{i:03d}'
        output_path = os.path.join('Output/ica_rest_all/modkmeans_results', f'modkmeans_s{id_name}.npy')
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        np.save(output_path, results_i)
        print(f"Saved results for subject {id_name} to {output_path}")
else:
    print(f"KMeans results file {kmeans_results_path} does not exist.")
print("Processing completed for all subjects in kmeans_results.")