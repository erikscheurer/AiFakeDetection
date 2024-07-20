import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, RandomResizedCrop, ToImage, ToDtype, Resize
from PIL import Image
import os
import glob


generator_dict = {
    'ImageNet': -1,
    'stable_diffusion_v_1_4': 0,
    'glide': 1,
    'stable_diffusion_v_1_5': 2,
    'Midjourney': 3,
    'wukong': 4,
    'ADM': 5,
    'VQDM': 6,
    'BigGAN': 7,
    'Camera': 8,
    'MidjourneyV6': 9,
    'StableDiffusion3': 10,
    'StableDiffusion3OwnPrompts': 11,
}


class GenImageDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None, target_size=(128,128), generators_allowed: list = None, real=True, fake=True):
        """
        data_path: path to the data folder
        split: train or val
        transform: optional transform to be applied to the images. Not implemented yet
        target_size: target size for the images
        generators_allowed: list of generators to use, if None all generators are used. Choices: ['BigGAN', 'ADM', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong']
        """
        # structure of data: <data_path>/<generator>/<train/val>/<ai/nature>/<image.ending>
        generators = os.listdir(data_path)
        self.aidata = []
        self.realdata = []
        self.generatortype = []

        for generator in generators:
            if not os.path.isdir(f"{data_path}/{generator}"):
                continue
            
            newaidata = []
            newrealdata = []
            newgeneratortype = [] 
            if generators_allowed is not None and generator not in generators_allowed:
                print(f"Skipping generator {generator}")
                continue

            for file in glob.iglob(f"{data_path}/{generator}/{split}/ai/*.*"):
                newaidata.append(file.replace("//", "/"))
                newgeneratortype.append(generator_dict[generator])
            for file in glob.iglob(f"{data_path}/{generator}/{split}/nature/*.*"):
                newrealdata.append(file.replace("//", "/"))

            if len(newaidata) == 0: # temporary fix for extracted data structure
                for file in glob.iglob(f"{data_path}/{generator}/{generator}_extracted/{split}/ai/*.*"):
                    newaidata.append(file.replace("//", "/"))
                    newgeneratortype.append(generator_dict[generator])
                for file in glob.iglob(f"{data_path}/{generator}/{generator}_extracted/{split}/nature/*.*"):
                    newrealdata.append(file.replace("//", "/"))

            self.aidata.extend(newaidata)
            self.realdata.extend(newrealdata)
            self.generatortype.extend(newgeneratortype)

            print(f"Found {len(newaidata)} ai images and {len(newrealdata)} real images")

        if len(self.aidata) == 0:
            print(f"No ai images found for generator {generator}")
        if len(self.realdata) == 0:
            print(f"No real images found for generator {generator}")
        if len(self.aidata) != len(self.realdata):
            print(f"WARNING: Number of ai images ({len(self.aidata)}) and real images ({len(self.realdata)}) do not match")
            
        if transform is not None:
            raise NotImplementedError("Custom transforms are not supported yet")

        self.target_size = target_size
        self.transform = Compose([
            # RandomResizedCrop(target_size),
            Resize(self.target_size),
            ToImage(),
            ToDtype(torch.float32, scale=True)
        ])

        if not real:
            self.realdata = []
            print("WARNING: Disregarding real data")
        if not fake:
            self.aidata = []
            self.generatortype = []
            print("WARNING: Disregarding fake data")

    def __len__(self):
        return len(self.aidata)+len(self.realdata)
    
    def try_to_load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except:
            print(f"Could not load image {path}")
            return Image.new("RGB", self.target_size)

    def __getitem__(self, idx):
        if idx < len(self.aidata):
            img = self.try_to_load_image(self.aidata[idx])
            return self.transform(img), 0, self.generatortype[idx]  # 0 for ai
        else:
            img = self.try_to_load_image(self.realdata[idx-len(self.aidata)])
            return self.transform(img), 1, generator_dict["ImageNet"]
        
    def __repr__(self):
        return f"GenImageDataset({len(self.aidata)} ai images, {len(self.realdata)} real images)"



if __name__ == "__main__":
    dataset = GenImageDataset("data",split='val')#, generators_allowed=['glide'])',
    print(dataset)
    print(dataset[0][0].shape)
    import matplotlib.pyplot as plt
    plt.imshow(dataset[0][0].permute(1,2,0))
    plt.show()