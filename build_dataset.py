import torch
import numpy as np
import torchvision
import pickle
import os
import random
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

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

def select_most_typical(embedding_dict, cluster_labels, num_clusters, B, k_neighbours=20):
    """
    Selects the most typical image from each cluster based on the typicality scores
    computed using the k-nearest neighbours algorithm.

    Args:
        embedding_dict (Dict[int, Dict[str, Any]]): A dictionary containing the embeddings of the images.
        cluster_labels (np.ndarray): The cluster labels assigned by the clustering algorithm.
        num_clusters (int): The number of clusters.
        B (int): The number of most typical images to select at each iteration.
        k_neighbours (int): The number of neighbours to consider when computing the typicality scores.
                            Set to 20 by default, same as in the paper.
    """

    # Create a dictionary of cluster IDs to image indices
    cluster_dict = {}
    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        if len(cluster_indices) < 5:  # Drop clusters with fewer than 5 images
            continue
        cluster_dict[cluster_id] = cluster_indices
    
    # Sort the clusters by the length of the cluster
    sorted_clusters = sorted(cluster_dict.items(), key=lambda x: len(x[1]), reverse=True)

    # Select the B largest clusters, selecting one image from each cluster
    most_typical_indices = []

    for i in range(B):
        if i >= len(sorted_clusters):
            break
        cluster_indices = sorted_clusters[i][1]

        # Select the embeddings of the images in the cluster
        cluster_embeddings = np.array([embedding_dict[i]["embedding"] for i in cluster_indices])

        # Compute typicality scores for each image in the cluster
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbours, len(cluster_embeddings)), algorithm="auto").fit(cluster_embeddings)
        distances, _ = nbrs.kneighbors(cluster_embeddings)
        typicality_scores = 1 / np.mean(distances, axis=1)

        most_typical_idx = cluster_indices[np.argmax(typicality_scores)]
        most_typical_indices.append(most_typical_idx)
    
    print(most_typical_indices)
    return most_typical_indices

def plot_by_cluster_assignment(active_learning_embeddings, embedding_dict):
    """
    Plots the embeddings based on the cluster assignment of the images with cluster centers.
    """
    all_embeddings = np.array([embedding_dict[i]["embedding"] for i in range(len(active_learning_embeddings))])
    print(all_embeddings.shape)

    # Number of clusters (e.g., CIFAR-10)
    K = 10
    kmeans = KMeans(n_clusters=K, random_state=42).fit(all_embeddings)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Combine embeddings and cluster centers
    combined = np.vstack([all_embeddings, cluster_centers])

    # Apply t-SNE on combined data
    n_samples = combined.shape[0]
    perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than the number of samples
    print(perplexity)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate=200, init="pca")
    reduced_combined = tsne.fit_transform(combined)

    # Split back into embeddings and cluster centers
    reduced_embeddings = reduced_combined[:-K]  # First part is embeddings
    reduced_centers = reduced_combined[-K:]  # Last part is cluster centers

    # Plot clusters by k-means assignment
    plt.figure(figsize=(10, 8))
    for i in range(K):
        cluster_points = reduced_embeddings[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', s=5)

    # Plot cluster centers
    plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='black', marker='x', s=100, label='Cluster Center')

    plt.title('t-SNE of Embeddings Colored by Cluster Assignment')
    plt.legend()
    plt.show()

def plot_by_true_labels(active_learning_embeddings, embedding_dict):
    """
    Plots the embeddings based on the true labels of the images with cluster centers.
    """
    all_embeddings = np.array([embedding_dict[i]["embedding"] for i in range(len(active_learning_embeddings))])
    true_labels = np.array([embedding_dict[i]["label"] for i in range(len(active_learning_embeddings))])
    print(all_embeddings.shape)

    # Apply t-SNE for dimensionality reduction
    n_samples = all_embeddings.shape[0]
    perplexity = min(30, n_samples - 1)
    print(perplexity)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate=200, init="pca")
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    # Compute cluster centers based on true labels
    unique_labels = np.unique(true_labels)
    cluster_centers = np.array([
                                reduced_embeddings[true_labels == label].mean(axis=0) for label in unique_labels
                                ])

    # Plot clusters based on true labels
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        cluster_points = reduced_embeddings[true_labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Label {label}', s=5)

    # Plot cluster centers
    plt.scatter(
                cluster_centers[:, 0],
                cluster_centers[:, 1],
                c='black', 
                marker='x', 
                s=100, 
                label='Cluster Center'
                )

    plt.title('t-SNE of Embeddings Colored by True Labels')
    plt.legend()
    plt.show()

def plot_by_log_density(active_learning_embeddings, embedding_dict):
    """
    Plots the embeddings based on the log density of the embeddings.
    """
    all_embeddings = np.array([embedding_dict[i]["embedding"] for i in range(len(active_learning_embeddings))])
    print(all_embeddings.shape)

    # Apply t-SNE for dimensionality reduction
    n_samples = all_embeddings.shape[0]
    perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than the number of samples
    print(perplexity)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate=200, init="pca")
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    # Compute kernel density estimate (KDE)
    kde = gaussian_kde(reduced_embeddings.T)
    density = kde(reduced_embeddings.T)

    # Apply log transformation to the density
    log_density = np.log(density)

    # Plot the embeddings colored by log density
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=log_density, cmap='viridis', s=5)
    plt.colorbar(label='Log Density')

    plt.title('t-SNE of Embeddings Colored by Log Density')
    plt.show()

if __name__ == "__main__":

    np.random.seed(2004)
    torch.manual_seed(2004)
    random.seed(2004)

    B = 50 # Number of new samples to query (active learning batch size)
    K = B
    NUM_ITERATIONS = 100
    MAX_CLUSTERS = 500
    # Total number of samples at the end = NUM_ITERATIONS * B

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
    if not os.path.exists(f"embeddings/{NUM_ITERATIONS}_iterations"):
        os.makedirs(f"embeddings/{NUM_ITERATIONS}_iterations", exist_ok=True)

        num_active_learning_embeddings = 0
        for i in range(NUM_ITERATIONS):
            print(f"Iteration: {i+1}/{NUM_ITERATIONS} | K: {K} | Number of embeddings: {num_active_learning_embeddings}")

            # Concatenate the embeddings
            all_embeddings = np.array([embedding_dict[i]["embedding"] for i in range(len(embedding_dict))])
            # print(all_embeddings.shape)
            
            L_i_1 = num_active_learning_embeddings # Number of embeddings already labelled
            K = min(L_i_1 + B, MAX_CLUSTERS) # Update K (same as paper, upper bounded by MAX_CLUSTERS)

            # In paper, states K <= 50, use KMeans, else use MiniBatchKMeans
            if K <= 50:
                kmeans = KMeans(n_clusters=K, random_state=42).fit(all_embeddings)
            else:
                batch_size = min(max(256, K * 5), len(all_embeddings))
                kmeans = MiniBatchKMeans(n_clusters=K, batch_size=batch_size, random_state=42).fit(all_embeddings)
            cluster_labels = kmeans.fit_predict(all_embeddings)

            # print(cluster_labels.shape)
            del all_embeddings

            # Add the most typical image to the active learning set
            most_typical_indices = select_most_typical(
                                                        embedding_dict=embedding_dict, 
                                                        cluster_labels=cluster_labels, 
                                                        num_clusters=K, 
                                                        B=B
                                                        )
            # print(most_typical_idx)

            for most_typical_idx in most_typical_indices:
                # Select the most typical image and add it to the active learning set
                most_typical_embedding = embedding_dict[most_typical_idx]
                # active_learning_embeddings[most_typical_idx] = embedding_dict[most_typical_idx]
                embedding_dict.pop(most_typical_idx)
                # print(most_typical_embedding.keys())

                with open(f"embeddings/{NUM_ITERATIONS}_iterations/embedding_{num_active_learning_embeddings}.pkl", "wb") as f: # B embeddings per iteration
                    pickle.dump(most_typical_embedding, f)
                    
                num_active_learning_embeddings += 1
            
            # Remap the embeddings:
            new_embedding_dict = {i: embedding_dict[key] for i, key in enumerate(embedding_dict.keys())}
            embedding_dict = new_embedding_dict
        
        print(f"Number of embeddings in active learning set: {num_active_learning_embeddings}")
    
    # Load the active learning embeddings
    num_active_learning_embeddings = 0
    active_learning_embeddings = {}
    for i in range(len(os.listdir(f"embeddings/{NUM_ITERATIONS}_iterations"))):
        with open(f"embeddings/{NUM_ITERATIONS}_iterations/embedding_{i}.pkl", "rb") as f:
            embedding = pickle.load(f)
            num_active_learning_embeddings += 1
            active_learning_embeddings[i] = embedding
    
    print(f"No. of embeddings/images for new dataset: {num_active_learning_embeddings}")
    print("Done!")


    plot_by_cluster_assignment(
                                active_learning_embeddings=active_learning_embeddings,
                                embedding_dict=embedding_dict
                                )
    plot_by_true_labels(
                        active_learning_embeddings=active_learning_embeddings,
                        embedding_dict=embedding_dict
                        )
    
    plot_by_log_density(
                        active_learning_embeddings=active_learning_embeddings,
                        embedding_dict=embedding_dict
                        )