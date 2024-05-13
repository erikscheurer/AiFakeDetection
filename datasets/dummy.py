# This is supposed to be a dummy dataset for quick prototyping

import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, data_path, n_samples=1000, shape=(3, 32, 32), n_classes=2, split="train"):
        self.n_samples = n_samples
        self.shape = shape
        self.n_classes = n_classes
        self.data = torch.randn(n_samples, *shape)
        self.targets = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __repr__(self):
        return f"DummyDataset(n_samples={self.n_samples}, shape={self.shape}, n_classes={self.n_classes})"