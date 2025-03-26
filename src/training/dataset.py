import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label)
        return image, label

class LinearEvalDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, transform=None):
        super().__init__()
        self.embeddings = embeddings
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]

        if self.transform:
            embedding = self.transform(embedding)
        else:
            embedding = torch.tensor(embedding)
        label = torch.tensor(label)
        return embedding, label