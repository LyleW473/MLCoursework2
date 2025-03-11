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


def select_most_typical(embedding_dict, cluster_labels, num_clusters, k_neighbours=20):
    selected_indices = []

    for cluster_id in range(num_clusters):
        
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        if len(cluster_indices) == 0:
            continue
        
        cluster_embeddings = np.array([embedding_dict[i]["embedding"] for i in cluster_indices])

        # Compute typicality scores for each image in the cluster
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbours, len(cluster_embeddings)), algorithm="auto").fit(cluster_embeddings)
        distances, _ = nbrs.kneighbors(cluster_embeddings)
        typicality_scores = 1 / np.mean(distances, axis=1)

        # Select the most typical image from each cluster
        most_typical_idx = cluster_indices[np.argmax(typicality_scores)]
        selected_indices.append(most_typical_idx)

    return selected_indices

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
        B = 50 # Number of new samples to query (active learning batch size)
        K = B
        NUM_ITERATIONS = 100

        labeled_indices = []
    
        for i in range(NUM_ITERATIONS):
            print(f"Iteration: {i+1}/{NUM_ITERATIONS}")

            # Concatenate the embeddings
            all_embeddings = np.array([embedding_dict[i]["embedding"] for i in range(len(embedding_dict))])
            print(all_embeddings.shape)
                  
            kmeans = KMeans(n_clusters=K, random_state=42).fit(all_embeddings)
            cluster_labels = kmeans.fit_predict(all_embeddings)

            print(cluster_labels.shape)
            
            most_typical_indices = select_most_typical(embedding_dict, cluster_labels, K)
            print(most_typical_indices)
            print(len(most_typical_indices))

            for idx in most_typical_indices:
                labeled_indices.append(idx)
                embedding_dict.pop(idx)

            # Remap the embeddings:
            new_embedding_dict = {i: embedding_dict[key] for i, key in enumerate(embedding_dict.keys())}
            embedding_dict = new_embedding_dict

        labeled_indices = list(set(labeled_indices)) # Remove duplicates
        print(labeled_indices)
        print(f"Number of labeled indices: {len(labeled_indices)}")