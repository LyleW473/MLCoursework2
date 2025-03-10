import torch

from SCAN.utils.config import create_config
from SCAN.utils.common_config import get_model

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

    embeddings = model(torch.randn(2, 3, 32, 32).cuda())
    print(embeddings.shape)
