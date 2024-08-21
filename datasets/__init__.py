import torch

from .dummy import DummyDataset
from .GenImage import GenImageDataset


def create_dataloader(
    data_path,
    dataset="GenImage",
    split="train",
    batch_size=32,
    num_workers=4,
    shuffle=None,
    transform=GenImageDataset.TransformFlag.NONE,
    **kwargs,
):
    if dataset.lower() == "dummy" or dataset.lower() == "random":
        dataset = DummyDataset(data_path, split=split, **kwargs)
    elif dataset.lower() == "genimage":
        dataset = GenImageDataset(data_path, split=split, transform=transform, **kwargs)
    else:
        raise ValueError(f"dataset {dataset} not found")

    shuffle = (split == "train") if shuffle is None else shuffle
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def available_generators(dataset_path) -> list:
    import os

    generators = os.listdir(dataset_path)
    generators = [gen for gen in generators if os.path.isdir(f"{dataset_path}/{gen}")]
    return generators
