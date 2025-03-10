import torch
import numpy as np
import torchvision
import pickle
import os

from PIL import Image

from SCAN.utils.config import create_config
from SCAN.utils.common_config import get_model, get_val_transformations

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