import torch
import numpy as np
import torchvision



from SCAN.utils.config import create_config
from SCAN.utils.common_config import get_model, get_val_transformations

if __name__ == "__main__":
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
    
    val_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=val_transforms) # Using the train set for creating the embeddings
    val_dl = torch.utils.data.DataLoader(val_set, batch_size=2, shuffle=True)
    
    for i, (images, labels) in enumerate(val_dl):
        print(images.shape)
        print(labels)
        
        embeddings = model(images.cuda()).cpu().detach().numpy()
        print(embeddings.shape)

        break