import torch
import numpy as np
import torchvision
import pickle
import os

from PIL import Image
from typing import Dict

from SCAN.utils.config import create_config
from SCAN.utils.common_config import get_model, get_val_transformations

def create_simclr_embeddings() -> Dict[int, Dict[str, np.ndarray]]:
    """
    Creates the embeddings of the CIFAR-10 dataset using the pre-trained SimCLR model.
    """
    
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

    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=standard_transform) # Generate embeddings for the test set as well (for linear evaluation)
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=True)
    
    train_embedding_dict = generate_embeddings_dict(model, val_dl, val_transforms)
    test_embedding_dict = generate_embeddings_dict(model, test_dl, val_transforms)

    return train_embedding_dict, test_embedding_dict

def generate_embeddings_dict(
                            model:torch.nn.Module, 
                            dataloader:torch.utils.data.DataLoader, 
                            transforms:torchvision.transforms.Compose
                            ) -> Dict[int, Dict[str, np.ndarray]]:
    embedding_dict = {}
    for i, (images, labels) in enumerate(dataloader):

        # Use transformed images for creating the embeddings
        transformed_images = torch.stack([transforms(Image.fromarray(np.array(image))) for image in images])
        
        embeddings = model(transformed_images.cuda()).cpu().detach().numpy()
        
        # Normalise the embeddings (L2 Normalisation)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        for j in range(embeddings.shape[0]):
            embedding_dict[i * dataloader.batch_size + j] = {
                                                        "embedding": embeddings[j],
                                                        "image": np.array(images[j]), # Store original image
                                                        "label": np.array(labels[j])
                                                        }
            
        print(f"Batch: {i+1}/{len(dataloader)}")

    return embedding_dict

def get_simclr_embeddings() -> Dict[int, Dict[str, np.ndarray]]:
    """
    Loads or creates the embeddings of the CIFAR-10 dataset using the pre-trained SimCLR model,
    depending on whether the embeddings are already saved.

    Returns only the training embeddings.
    """
    if not (os.path.exists("embeddings/simclr_cifar10_embeddings.pkl") and os.path.exists("embeddings/simclr_cifar10_test_embeddings.pkl")):
        print("Hello")
        train_embedding_dict, test_embedding_dict = create_simclr_embeddings()

        # Save the embeddings
        os.makedirs("embeddings", exist_ok=True)
        with open("embeddings/simclr_cifar10_embeddings.pkl", "wb") as f:
            pickle.dump(train_embedding_dict, f)
            
        with open("embeddings/simclr_cifar10_test_embeddings.pkl", "wb") as f:
            pickle.dump(test_embedding_dict, f)
    else:
        with open("embeddings/simclr_cifar10_embeddings.pkl", "rb") as f:
            train_embedding_dict = pickle.load(f)

        print(f"Number of embeddings: {len(train_embedding_dict)}")
        print(train_embedding_dict[0].keys())
    return train_embedding_dict