import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

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
    Plots the embeddings based on the log density of the embeddings with cluster centers.
    """
    all_embeddings = np.array([embedding_dict[i]["embedding"] for i in range(len(active_learning_embeddings))])
    print(all_embeddings.shape)

    # Apply t-SNE for dimensionality reduction
    n_samples = all_embeddings.shape[0]
    perplexity = min(30, n_samples - 1)
    print(perplexity)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate=200, init="pca")
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    # Compute kernel density estimate (KDE)
    kde = gaussian_kde(reduced_embeddings.T)
    density = kde(reduced_embeddings.T)

    # Apply log transformation to the density
    log_density = np.log(density)

    # Apply KMeans clustering to find cluster centers
    K = 10  # Number of clusters (adjust based on dataset)
    kmeans = KMeans(n_clusters=K, random_state=42).fit(reduced_embeddings)
    cluster_centers = kmeans.cluster_centers_

    # Plot the embeddings colored by log density
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=log_density, cmap='viridis', s=5)
    plt.colorbar(label='Log Density')

    # Plot cluster centers
    plt.scatter(
                cluster_centers[:, 0], 
                cluster_centers[:, 1],
                c='black', 
                marker='x', 
                s=100, 
                label='Cluster Center'
                )
    plt.title('t-SNE of Embeddings Colored by Log Density')
    plt.legend()
    plt.show()