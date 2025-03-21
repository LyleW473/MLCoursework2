import torch
import numpy as np
import torchvision
import pickle
import os

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from SCAN.utils.config import create_config
from SCAN.utils.common_config import get_model, get_val_transformations

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

def select_most_typical(embedding_dict, cluster_labels, num_clusters, k_neighbours=20):
    """
    Selects the most typical image from each cluster based on the typicality scores
    computed using the k-nearest neighbours algorithm.

    Args:
        embedding_dict (Dict[int, Dict[str, Any]]): A dictionary containing the embeddings of the images.
        cluster_labels (np.ndarray): The cluster labels assigned by the clustering algorithm.
        num_clusters (int): The number of clusters.
        k_neighbours (int): The number of neighbours to consider when computing the typicality scores.
                            Set to 20 by default, same as in the paper.
    """

    # Find the largest cluster
    largest_cluster_size = 0
    largest_cluster_indices = []

    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        if len(cluster_indices) < 5:  # Drop clusters with fewer than 5 images
            continue
        if len(cluster_indices) > largest_cluster_size:
            largest_cluster_size = len(cluster_indices)
            largest_cluster_indices = cluster_indices

    # Select the largest cluster out of all clusters with more than 5 images
    cluster_embeddings = np.array([embedding_dict[i]["embedding"] for i in largest_cluster_indices])

    # Compute typicality scores for each image in the cluster
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbours, len(cluster_embeddings)), algorithm="auto").fit(cluster_embeddings)
    distances, _ = nbrs.kneighbors(cluster_embeddings)
    typicality_scores = 1 / np.mean(distances, axis=1)

    # Select the most typical image from the cluster
    most_typical_idx = largest_cluster_indices[np.argmax(typicality_scores)]
    return most_typical_idx

if __name__ == "__main__":

    if not os.path.exists("embeddings/simclr_cifar10_embeddings.pkl"):
        print("Hello")

        state_dict = torch.load("simclr_cifar-10.pth", map_location='cpu')
        print(state_dict.keys())

        config = create_config(config_file_env="SCAN/configs/env.yml", config_file_exp="SCAN/configs/pretext/simclr_cifar10.yml")
        print(config)

        model = get_model(config)
        print('Model is {}'.format(model.__class__.__name__))
        print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        print(model)
        model = model.cuda()

        val_transforms = get_val_transformations(config)
        print(val_transforms)
        print(type(val_transforms))
        
        def standard_transform(image):
            return np.array(image)
        
        val_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=standard_transform) # Using the train set for creating the embeddings
        val_dl = torch.utils.data.DataLoader(val_set, batch_size=2, shuffle=True)
        
        embedding_dict = {}
        for i, (images, labels) in enumerate(val_dl):

            # Use transformed images for creating the embeddings
            transformed_images = torch.stack([val_transforms(Image.fromarray(np.array(image))) for image in images])
            
            embeddings = model(transformed_images.cuda()).cpu().detach().numpy()

            for j in range(embeddings.shape[0]):
                embedding_dict[i * val_dl.batch_size + j] = {
                                                            "embedding": embeddings[j],
                                                            "image": np.array(images[j]), # Store original image
                                                            "label": np.array(labels[j])
                                                            }
                
            print(f"Batch: {i+1}/{len(val_dl)}")

        # Save the embeddings
        os.makedirs("embeddings", exist_ok=True)
        with open("embeddings/simclr_cifar10_embeddings.pkl", "wb") as f:
            pickle.dump(embedding_dict, f)

    else:
        with open("embeddings/simclr_cifar10_embeddings.pkl", "rb") as f:
            embedding_dict = pickle.load(f)

        print(f"Number of embeddings: {len(embedding_dict)}")
        print(embedding_dict[0].keys())

    # Perform K-Means Clustering on the embeddings
    if not os.path.exists("embeddings/active_learning_embeddings.pkl"):

        B = 50 # Number of new samples to query (active learning batch size)
        K = B
        NUM_ITERATIONS = 1000 # Also the number of total samples at the end.
        MAX_CLUSTERS = 500

        active_learning_embeddings = {}

        for i in range(NUM_ITERATIONS):
            print(f"Iteration: {i+1}/{NUM_ITERATIONS}")

            # Concatenate the embeddings
            all_embeddings = np.array([embedding_dict[i]["embedding"] for i in range(len(embedding_dict))])
            print(all_embeddings.shape)
            
            L_i_1 = len(active_learning_embeddings) # Number of embeddings already labelled
            K = min(L_i_1 + B, MAX_CLUSTERS) # Update K (same as paper, upper bounded by MAX_CLUSTERS)

            kmeans = KMeans(n_clusters=K, random_state=42).fit(all_embeddings)
            cluster_labels = kmeans.fit_predict(all_embeddings)

            print(cluster_labels.shape)

            # Add the most typical image to the active learning set
            most_typical_idx = select_most_typical(embedding_dict, cluster_labels, K)
            print(most_typical_idx)

            active_learning_embeddings[most_typical_idx] = embedding_dict[most_typical_idx]
            embedding_dict.pop(most_typical_idx)

            # Remap the embeddings:
            new_embedding_dict = {i: embedding_dict[key] for i, key in enumerate(embedding_dict.keys())}
            embedding_dict = new_embedding_dict

        print(f"Number of embeddings in active learning set: {len(active_learning_embeddings)}")
        
        os.makedirs("embeddings", exist_ok=True)
        with open(f"embeddings/active_learning_embeddings.pkl", "wb") as f:
            pickle.dump(active_learning_embeddings, f)
    
    else:
        with open("embeddings/active_learning_embeddings.pkl", "rb") as f:
            active_learning_embeddings = pickle.load(f)

    print(f"No. of embeddings/images for new dataset: {len(active_learning_embeddings)}")
    print("Done!")