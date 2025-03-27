import torch
import numpy as np
import pickle
import random
import os

from src.plot_functions import plot_by_cluster_assignment, plot_by_true_labels, plot_by_log_density
from src.simclr import get_embeddings
from src.typiclust import perform_typiclust

"""
This script performs active learning using the TypiClust algorithm on the CIFAR-10 dataset. 

1. **Representation Learning**:
    - Embeddings are generated using a SimCLR model pre-trained on CIFAR-10.
    - The embeddings are extracted from the penultimate layer of the model for each image in the dataset.

2. **Active Learning**:
    - K-Means clustering is applied to the embeddings (of all images) to create clusters.
    - The most typical image from the largest cluster is selected based on the computed typicality scores
      (computed using k-nearest neighbors).
    - The selected image is added to the labeled dataset.

3. **Iterative Process**:
    - The process is repeated for `NUM_ITERATIONS` until the active learning budget is exhausted.
    - The total number of selected samples at the end is `NUM_ITERATIONS`.
    - The number of clusters is adjusted dynamically using `K = min(L + B, MAX_CLUSTERS)`, 
      where `L` is the number of already labeled samples and `B` is the batch size.

"""


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
        num_iterations_for_sizes = [int(dataset_sizes[i] / B) for i in range(len(dataset_sizes))]
        
        for x in range(len(dataset_sizes)):

            num_iterations = num_iterations_for_sizes[x]

            for model_name in model_names:

                np.random.seed(2004)
                torch.manual_seed(2004)
                random.seed(2004)

                base_path = f"embeddings/{model_name}/typiclust/{setting}/{num_iterations}_iterations_B{B}"#

                # Total number of samples at the end = NUM_ITERATIONS * B
                if not os.path.exists(base_path):
                    os.makedirs(base_path, exist_ok=True)
                    embedding_dict = get_embeddings(model_name=model_name)

                    perform_typiclust(
                                    embedding_dict=embedding_dict,
                                    num_iterations=num_iterations,
                                    B=B,
                                    base_path=base_path,
                                    max_clusters=MAX_CLUSTERS
                                    )
                    del embedding_dict
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