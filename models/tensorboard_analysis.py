# %% plot selected runs from tensorboard in matplotlib
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import os
import time
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    SCALARS,
    IMAGES,
    TENSORS,
)

os.chdir("/home/st/st_us-053030/st_st161532/AiFakeDetection")


# %%
@lru_cache(maxsize=None)
def get_event_acc(logdir, run, size_guidance=None):
    event_acc = EventAccumulator(
        path=os.path.join(logdir, run),
        size_guidance={
            SCALARS: size_guidance if size_guidance else 0,
        },
    )
    event_acc.Reload()
    return event_acc


# %%
def plot_smoothed_runs(
    logdir,
    runs,
    tags,
    title=None,
    window=10,
    save_path=None,
    colors="bgrcmyk",
    labels=None,
    maxiter=100000,
):
    cmap = get_cmap("tab10")
    fig, axs = plt.subplots(len(tags), 1)  # , figsize=(10, 5*len(tags)))
    if len(tags) == 1:
        axs = [axs]
    if title:
        fig.suptitle(title)
    for i, tag in enumerate(tags):
        print(f"Plotting tag {tag}")
        for j, run in enumerate(iterable=runs):
            print(f"Plotting run {run}")

            start = time.time()
            event_acc = get_event_acc(logdir, run, size_guidance=maxiter)
            print("Loading Time:", time.time() - start)

            # get data
            try:
                start = time.time()
                data = pd.DataFrame(event_acc.Scalars(tag))
                data = data[data["step"] < maxiter]
                print("Data Time:", time.time() - start)
            except KeyError:
                print(f"Could not find tag {tag} in run {run}")
                continue
            print(event_acc.Tags())
            start = time.time()
            to_plot_data = data["value"].rolling(window=window).mean()
            print("Rolling Time:", time.time() - start)
            # plot
            start = time.time()
            try:
                label = labels[j]
            except:
                label = run
            try:
                color = colors[j]
            except IndexError:
                color = cmap(j)
            axs[i].plot(data["step"], to_plot_data, label=label, color=color)
            # axs[i].set_xscale('log')
            print("Plotting Time:", time.time() - start)
            # axs[i].plot(data['step'], data['value'], alpha=0.1, color=color, label='_nolegend_', linewidth=.1)
            # print metrics
            print(run)
            print("Max", label, ":", data["value"].max())
            print("Smoothed max", label, ":", to_plot_data.max())

        # axs[i].set_title(tag)
        # axs[i].legend()
        axs[i].set_ylabel(tag)
        axs[i].set_xlabel("Step")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
logdir = "output/ResNet_DANN2/train"
# runs = ['ResNet_DANN22024-07-01_21-38-17[1, 2, 3, 4, 5, 6, 7]_[0]']#
runs = os.listdir(logdir)
for run in runs:
    assert os.path.exists(os.path.join(logdir, run)), f"Run {run} does not exist"
tags = [
    "transfer_fake_acc"
]  # ,'accuracy_fake','accuracy_real','transfer_all_acc','transfer_fake_acc','transfer_real_acc']
generator_dict = {
    "ImageNet": -1,
    "stable_diffusion_v_1_4": 0,
    "glide": 1,
    "stable_diffusion_v_1_5": 2,
    "Midjourney": 3,
    "wukong": 4,
    "ADM": 5,
    "VQDM": 6,
    "BigGAN": 7,
    "Camera": 8,
    "MidjourneyV6": 9,
    "StableDiffusion3": 10,
    "StableDiffusion3OwnPrompts": 11,
}
inverse_dict = {v: k for k, v in generator_dict.items()}
labels = [inverse_dict[int(run[-2])] for run in runs]
title = "Transfer Accuracy on Fake Images"
save_path = "F3Net_training.png"
plot_smoothed_runs(
    logdir=logdir,
    runs=runs,
    tags=tags,
    title=title,
    window=2000,
    save_path=save_path,
    labels=labels,
)


# %% plot 2 images, one fake and one real
from matplotlib import pyplot as plt
from datasets import GenImageDataset

dataset = GenImageDataset(
    "data/TinyGenImage", split="val"
)  # , generators_allowed='stable_diffusion_v_1_5')
# %%
index = 3
real_index = index + len(dataset.aidata)
print(real_index, index)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(dataset[index][0].permute(1, 2, 0))
plt.title("Fake Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(dataset[real_index][0].permute(1, 2, 0))
plt.title("Real Image")
plt.axis("off")

# %%
