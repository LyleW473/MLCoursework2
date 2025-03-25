import numpy as np
import pickle
import os

from typing import Dict, List
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

def select_most_typical(
                        embedding_dict:Dict[int, Dict[str, np.ndarray]],
                        cluster_labels:np.ndarray,
                        num_clusters:int,
                        B:int,
                        k_neighbours:int=20
                        ) -> List[int]:
    """
    Selects the most typical image from each cluster based on the typicality scores
    computed using the k-nearest neighbours algorithm.

    Args:
        embedding_dict (Dict[int, Dict[str, np.ndarray]]): A dictionary containing the embeddings of the images.
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

def perform_typiclust(
                    embedding_dict:Dict[int, Dict[str, np.ndarray]], 
                    num_iterations:int,
                    B:int,
                    setting:str,
                    max_clusters:int=500
                    ) -> None:
    """
    Creates the Typiclust dataset.

    Args:
        embedding_dict (Dict[int, Dict[str, np.ndarray]]): A dictionary containing the embeddings of the images.
        num_iterations (int): Number of iterations to run the algorithm for.
        B (int): The budget, i.e., the number of images to add to the dataset pool at each iteration.
        setting (str): The setting i.e., (top / bottom) in the paper.
        max_clusters (int): The maximum number of clusters at one point, 

    """

    # Perform K-Means Clustering on the embeddings
    if not os.path.exists(f"embeddings/typiclust/{setting}/{num_iterations}_iterations_B{B}"):
        os.makedirs(f"embeddings/typiclust/{setting}/{num_iterations}_iterations_B{B}", exist_ok=True)

        num_active_learning_embeddings = 0

        for i in range(num_iterations):

            # Concatenate the embeddings
            all_embeddings = np.array([embedding_dict[i]["embedding"] for i in range(len(embedding_dict))])
            # print(all_embeddings.shape)
            
            L_i_1 = num_active_learning_embeddings # Number of embeddings already labelled
            K = min(L_i_1 + B, max_clusters) # Update K (same as paper, upper bounded by MAX_CLUSTERS)

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

                with open(f"embeddings/typiclust/{setting}/{num_iterations}_iterations_B{B}/embedding_{num_active_learning_embeddings}.pkl", "wb") as f: # B embeddings per iteration
                    pickle.dump(most_typical_embedding, f)
                    
                num_active_learning_embeddings += 1
            
            # Remap the embeddings:
            new_embedding_dict = {i: embedding_dict[key] for i, key in enumerate(embedding_dict.keys())}
            embedding_dict = new_embedding_dict

            print(f"Iteration: {i+1}/{num_iterations} | K: {K} | Number of embeddings: {num_active_learning_embeddings}")

        print(f"Number of embeddings in active learning set: {num_active_learning_embeddings}")