import torch

from .dummy import DummyDataset
from .GenImage import GenImageDataset

def create_dataloader(data_path, dataset='timyGenImage', split='train', batch_size=32, num_workers=4, **kwargs):
    if dataset.lower() == 'dummy' or dataset.lower() == 'random':
        dataset = DummyDataset(data_path, **kwargs)
    elif dataset.lower() == 'genimage':
        dataset = GenImageDataset(data_path, **kwargs)
    else:
        raise ValueError(f"dataset {dataset} not found")
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers)