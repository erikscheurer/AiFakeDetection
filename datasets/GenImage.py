import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    Compose,
    RandomResizedCrop,
    ToImage,
    ToDtype,
    Resize,
    RandomHorizontalFlip,
    GaussianBlur,
    JPEG,
    RandomGrayscale,
)
from PIL import Image
import os
import glob
from enum import Flag


class GenImageDataset(Dataset):
    class TransformFlag(Flag):
        NONE = 0
        RANDOM_HORIZONTAL_FLIP = 1
        GAUSSIAN_BLUR = 2
        RANDOM_JPEG = 4
        RANDOM_GRAYSCALE = 8
        ALL = 15

    def __init__(
        self,
        data_path,
        split="train",
        transform=None,
        target_size=(128, 128),
        generators_allowed: list = None,
    ):
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

        exclude_files = [
            "./data/GenImage/stable_diffusion_v_1_4/stable_diffusion_v_1_4_extracted/train/ai/033_sdv4_00137.png",
            "./data/GenImage/stable_diffusion_v_1_4/stable_diffusion_v_1_4_extracted/train/ai/033_sdv4_00134.png",
            "./data/GenImage/stable_diffusion_v_1_4/stable_diffusion_v_1_4_extracted/train/ai/033_sdv4_00152.png",
            "./data/GenImage/Midjourney/Midjourney_extracted/train/ai/208_midjourney_76.png",
            "./data/GenImage/Midjourney/Midjourney_extracted/train/ai/208_midjourney_92.png",
            "./data/GenImage/Midjourney/Midjourney_extracted/train/ai/208_midjourney_91.png",
            "./data/GenImage/BigGAN/BigGAN_extracted/train/ai/116_biggan_00107.png",
            "./data/GenImage/BigGAN/BigGAN_extracted/train/ai/116_biggan_00094.png",
            "./data/GenImage/BigGAN/BigGAN_extracted/train/ai/116_biggan_00081.png",
            "./data/GenImage/BigGAN/BigGAN_extracted/train/ai/116_biggan_00098.png",
        ]

        for generator in generators:
            if not os.path.isdir(f"{data_path}/{generator}"):
                continue

            newaidata = []
            newrealdata = []
            if generators_allowed is not None and generator not in generators_allowed:
                print(f"Skipping generator {generator}")
                continue

            for file in glob.iglob(f"{data_path}/{generator}/{split}/ai/*.*"):
                newaidata.append(file.replace("//", "/"))
            for file in glob.iglob(f"{data_path}/{generator}/{split}/nature/*.*"):
                newrealdata.append(file.replace("//", "/"))

            if len(newaidata) == 0:  # temporary fix for extracted data structure
                for file in glob.iglob(
                    f"{data_path}/{generator}/{generator}_extracted/{split}/ai/*.*"
                ):
                    newaidata.append(file.replace("//", "/"))
                for file in glob.iglob(
                    f"{data_path}/{generator}/{generator}_extracted/{split}/nature/*.*"
                ):
                    newrealdata.append(file.replace("//", "/"))

            self.aidata.extend(newaidata)
            self.realdata.extend(newrealdata)

            # remove files that are not images
            self.aidata = [file for file in self.aidata if file not in exclude_files]

            print(
                f"Found {len(newaidata)} ai images and {len(newrealdata)} real images"
            )

        if len(self.aidata) == 0:
            raise ValueError(f"No ai images found for generator {generator}")
        if len(self.realdata) == 0:
            raise ValueError(f"No real images found for generator {generator}")
        if len(self.aidata) != len(self.realdata):
            print(
                f"WARNING: Number of ai images ({len(self.aidata)}) and real images ({len(self.realdata)}) do not match"
            )

        self.target_size = target_size
        transform_list = [Resize(self.target_size), ToImage()]
        if transform is not None:
            if transform & self.TransformFlag.RANDOM_HORIZONTAL_FLIP:
                transform_list.append(RandomHorizontalFlip())
            if transform & self.TransformFlag.GAUSSIAN_BLUR:
                transform_list.append(
                    GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2.0))
                )
            if transform & self.TransformFlag.RANDOM_JPEG:
                transform_list.append(JPEG(quality=(50, 100)))
            if transform & self.TransformFlag.RANDOM_GRAYSCALE:
                transform_list.append(RandomGrayscale(p=0.1))
        transform_list.append(ToDtype(torch.float32, scale=True))

        self.transform = Compose(transform_list)

    def __len__(self):
        return len(self.aidata) + len(self.realdata)

    def try_to_load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except:
            print(f"Could not load image {path}")
            return Image.new("RGB", self.target_size)

    def __getitem__(self, idx):
        if idx < len(self.aidata):
            img = self.try_to_load_image(self.aidata[idx])
            return self.transform(img), 0  # 0 for ai
        else:
            img = self.try_to_load_image(self.realdata[idx - len(self.aidata)])
            return self.transform(img), 1

    def __repr__(self):
        return f"GenImageDataset({len(self.aidata)} ai images, {len(self.realdata)} real images)"


if __name__ == "__main__":
    dataset = GenImageDataset(
        "data/GenImage", split="val"
    )  # , generators_allowed=['glide'])',
    print(dataset)
    print(dataset[0][0].shape)
    import matplotlib.pyplot as plt

    plt.imshow(dataset[0][0].permute(1, 2, 0))
    plt.show()
