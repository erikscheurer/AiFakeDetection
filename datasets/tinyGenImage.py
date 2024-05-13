import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, RandomResizedCrop, ToImage, ToDtype
from PIL import Image
import os
import glob

class TinyGenImageDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None, target_size=(128,128), generators_allowed=None):
        # structure of data: <data_path>/<generator>/<train/val>/<ai/nature>/<image.ending>
        generators = os.listdir(data_path)
        self.aidata = []
        self.realdata = []

        for generator in generators:
            if generators_allowed is not None and generator.split('_')[-1] not in generators_allowed:
                continue

            for file in glob.iglob(f"{data_path}/{generator}/{split}/ai/*.*"):
                self.aidata.append(file)
            for file in glob.iglob(f"{data_path}/{generator}/{split}/nature/*.*"):
                self.realdata.append(file)

        self.transform = Compose([
            RandomResizedCrop(target_size),
            ToImage(),
            ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.aidata)+len(self.realdata)
    
    def __getitem__(self, idx):
        if idx < len(self.aidata):
            img = Image.open(self.aidata[idx]).convert("RGB")
            return self.transform(img), 0 # 0 for ai
        else:
            img = Image.open(self.realdata[idx-len(self.aidata)]).convert("RGB")
            return self.transform(img), 1
        
    def __repr__(self):
        return f"TinyGenImageDataset({len(self.aidata)} ai images, {len(self.realdata)} real images)"
    
if __name__ == "__main__":
    dataset = TinyGenImageDataset("data/TinyGenImage", generators_allowed='biggan')
    print(dataset)
    print(dataset[0][0].shape)
    import matplotlib.pyplot as plt
    plt.imshow(dataset[0][0].permute(1,2,0))
    plt.show()