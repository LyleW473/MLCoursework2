import numpy as np
import torch
import random
import pickle
import os

from src.simclr import get_embeddings
from src.plot_functions import plot_by_cluster_assignment, plot_by_true_labels, plot_by_log_density

if __name__ == "__main__":

    model_names = ["simclr", "dino"]
    settings = {
            "top": {"B": 10, "dataset_sizes": [10, 20, 30, 40, 50, 60]},
            "bottom": {"B": 50, "dataset_sizes": [50, 100, 150, 200, 250, 300]}
            } # B = Number of new samples to query (active learning batch size)
    
    MAX_CLUSTERS = 500

    for setting in settings.keys():
        B = settings[setting]["B"]
        dataset_sizes = settings[setting]["dataset_sizes"]
        num_iterations_for_sizes = [int(dataset_sizes[i] / B) for i in range(len(dataset_sizes))] # Number of iterations for each dataset size
        print(num_iterations_for_sizes)

        for x in range(len(dataset_sizes)):

            num_iterations = num_iterations_for_sizes[x]

            for model_name in model_names:
                np.random.seed(2004)
                torch.manual_seed(2004)
                random.seed(2004)


                base_path = f"embeddings/{model_name}/random/{setting}/{num_iterations}_iterations_B{B}"


                if not os.path.exists(base_path):

                    # Select random embeddings
                    os.makedirs(base_path, exist_ok=True)

                    # Total number of samples at the end = NUM_ITERATIONS * B
                    embedding_dict = get_embeddings(model_name=model_name)

                    num_samples_to_select = dataset_sizes[x]
                    total_samples = len(embedding_dict)
                    random_indices = np.random.choice(total_samples, num_samples_to_select, replace=False)

                    for i, rand_idx in enumerate(random_indices):
                        random_embedding = embedding_dict[rand_idx]
                        with open(f"{base_path}/embedding_{i}.pkl", "wb") as f:
                            pickle.dump(random_embedding, f)
                else:
                    print(f"Plotting for: Model: {model_name} | Setting: {setting} | Dataset Size: {dataset_sizes[x]}")
                    # Load the active learning embeddings
                    num_active_learning_embeddings = 0
                    active_learning_embeddings = {}
                    for i in range(num_iterations * B):
                        with open(f"{base_path}/embedding_{i}.pkl", "rb") as f:
                            embedding = pickle.load(f)
                            num_active_learning_embeddings += 1
                            active_learning_embeddings[i] = embedding
                    
                    print(f"No. of embeddings/images for new dataset: {num_active_learning_embeddings}")
                    print("Done!")
                    
                    plot_by_cluster_assignment(active_learning_embeddings)
                    plot_by_true_labels(active_learning_embeddings)
                    plot_by_log_density(active_learning_embeddings)