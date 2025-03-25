import torch
import numpy as np
import pickle
import os
import random

from src.plot_functions import plot_by_cluster_assignment, plot_by_true_labels, plot_by_log_density
from src.simclr import get_simclr_embeddings
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

Output:
- The selected embeddings are saved to `embeddings/active_learning_embeddings.pkl` for further evaluation.
"""


if __name__ == "__main__":

    np.random.seed(2004)
    torch.manual_seed(2004)
    random.seed(2004)

    B = 50 # Number of new samples to query (active learning batch size)
    NUM_ITERATIONS = 10
    MAX_CLUSTERS = 500

    # Total number of samples at the end = NUM_ITERATIONS * B
    
    embedding_dict = get_simclr_embeddings()

    perform_typiclust(
                    embedding_dict=embedding_dict,
                    num_iterations=NUM_ITERATIONS,
                    B=B,
                    max_clusters=MAX_CLUSTERS
                    )
    del embedding_dict

    # Load the active learning embeddings
    num_active_learning_embeddings = 0
    active_learning_embeddings = {}
    for i in range(NUM_ITERATIONS * B):
        with open(f"embeddings/{NUM_ITERATIONS}_iterations/embedding_{i}.pkl", "rb") as f:
            embedding = pickle.load(f)
            num_active_learning_embeddings += 1
            active_learning_embeddings[i] = embedding
    
    print(f"No. of embeddings/images for new dataset: {num_active_learning_embeddings}")
    print("Done!")


    plot_by_cluster_assignment(
                                active_learning_embeddings=active_learning_embeddings,
                                )
    plot_by_true_labels(
                        active_learning_embeddings=active_learning_embeddings,
                        )
    
    plot_by_log_density(
                        active_learning_embeddings=active_learning_embeddings,
                        )