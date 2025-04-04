import numpy as np
import pickle
import os

from PIL import Image
from typing import List, Tuple

def load_active_learning_embeddings(embeddings_dir:str) -> Tuple[List[Image.Image], List[np.ndarray], List[np.ndarray]]:
    """
    Loads the embeddings of the images from the active learning dataset.
    - The embeddings are stored in a dictionary with the image index as the key.
    - The dictionary contains the image, label and generated embeddings.

    Args:
        embeddings_dir (str): The directory containing the embeddings of the images.
    """
    active_learning_embeddings = {}
    num_embeddings = len(os.listdir(embeddings_dir))
    for i in range(num_embeddings):
        with open(f"{embeddings_dir}/embedding_{i}.pkl", "rb") as f:
            embeddings = pickle.load(f)
            active_learning_embeddings[i] = embeddings

    # print("Embeddings shape:", len(active_learning_embeddings))

    all_images = []
    all_labels = []
    all_embeddings = []
    for key, value_dict in active_learning_embeddings.items():
        image = value_dict["image"]
        # print(image.flatten().min(), image.flatten().max())
        image = Image.fromarray(image.astype("uint8")) # Convert to PIL image
        label = np.array(value_dict["label"]) # Convert scalar to 1D array
        embedding = np.array(value_dict["embedding"])
        # print(np.array(image).shape, label.shape)
        all_images.append(image)
        all_labels.append(label)
        all_embeddings.append(embedding)

    return all_images, all_labels, all_embeddings