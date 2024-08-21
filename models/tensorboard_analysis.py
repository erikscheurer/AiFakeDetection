# %% plot selected runs from tensorboard in matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

os.chdir("/home/st/st_us-053030/st_st161532/AiFakeDetection")


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
):
    fig, axs = plt.subplots(len(tags), 1, figsize=(10, 5 * len(tags)))
    if len(tags) == 1:
        axs = [axs]
    if title:
        fig.suptitle(title)
    for i, tag in enumerate(tags):
        for j, run in enumerate(runs):
            # get data
            event_acc = EventAccumulator(os.path.join(logdir, run)).Reload()
            data = pd.DataFrame(event_acc.Scalars(tag))
            to_plot_data = data["value"].rolling(window=window).mean()
            # plot
            label = labels[j] if labels else run
            axs[i].plot(data["step"], to_plot_data, label=label, color=colors[j])
            axs[i].plot(
                data["step"],
                data["value"],
                alpha=0.1,
                color=colors[j],
                label="_nolegend_",
                linewidth=0.1,
            )
            # print metrics
            print(run)
            print("Max", label, ":", data["value"].max())
            print("Smoothed max", label, ":", to_plot_data.max())

        axs[i].set_title(tag)
        axs[i].legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# %%
logdir = "output"
runs = [
    "F3Net/train/F3Net2024-06-02_00-29-20",
    "CNNDetection/train/CNNDetection2024-06-04_08-46-28",
]
for run in runs:
    assert os.path.exists(os.path.join(logdir, run)), f"Run {run} does not exist"
tags = [
    "accuracy",
]
title = ""
save_path = "F3Net_training.png"
plot_smoothed_runs(
    logdir=logdir,
    runs=runs,
    tags=tags,
    title=title,
    window=100,
    save_path=save_path,
    labels=["F3Net", "ResNet50"],
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
