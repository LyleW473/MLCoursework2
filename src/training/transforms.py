import torchvision.transforms as transforms

def get_transform():
    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=0), # Padding not mentioned in paper
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return TRANSFORM