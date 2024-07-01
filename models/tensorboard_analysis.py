#%% plot selected runs from tensorboard in matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS, IMAGES, TENSORS
os.chdir('/home/st/st_us-053030/st_st161532/AiFakeDetection')

#%%
def plot_smoothed_runs(logdir, runs, tags, title=None, window=10, save_path=None, colors = 'bgrcmyk', labels=None):
    cmap = get_cmap('tab10')
    fig, axs = plt.subplots(len(tags), 1, figsize=(10, 5*len(tags)))
    if len(tags) == 1:
        axs = [axs]
    if title:
        fig.suptitle(title)
    for j,run in enumerate(iterable=runs):
        print(f'Plotting run {run}')
        event_acc = EventAccumulator(path=os.path.join(logdir, run), size_guidance={
            SCALARS: 0,
        }).Reload()
        for i, tag in enumerate(tags):
            print(f'Plotting tag {tag}')
            # get data
            print(event_acc.Tags())
            try:
                data = pd.DataFrame(event_acc.Scalars(tag))
            except KeyError:
                print(f'Could not find tag {tag} in run {run}')
                continue
            to_plot_data = data['value'].rolling(window=window).mean()
            # plot
            try:
                label = labels[j] 
            except:
                label = run
            try:
                color = colors[j]
            except IndexError:
                color = cmap(j)
            axs[i].plot(data['step'], to_plot_data, label=label, color=color)
            axs[i].plot(data['step'], data['value'], alpha=0.1, color=color, label='_nolegend_', linewidth=.1)
            # print metrics
            print(run)
            print('Max',label,':',data['value'].max())
            print('Smoothed max',label,':',to_plot_data.max())

        axs[i].set_title(tag)
        axs[i].legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

#%%
logdir = 'output/ResNet_DANN/train'
runs = os.listdir(logdir)
for run in runs:
    assert os.path.exists(os.path.join(logdir, run)), f'Run {run} does not exist'
tags = ['accuracy','accuracy_fake','accuracy_real','transfer_all_acc','transfer_fake_acc','transfer_real_acc']
title = ''
save_path = 'F3Net_training.png'
plot_smoothed_runs(logdir=logdir, runs=runs, tags=tags, title=title, window=100, save_path=save_path,labels=['F3Net','ResNet50'])


#%% plot 2 images, one fake and one real
from matplotlib import pyplot as plt
from datasets import GenImageDataset

dataset = GenImageDataset("data/TinyGenImage",split='val')#, generators_allowed='stable_diffusion_v_1_5')
#%%
index = 3
real_index = index+len(dataset.aidata)
print(real_index,index)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(dataset[index][0].permute(1,2,0))
plt.title('Fake Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(dataset[real_index][0].permute(1,2,0))
plt.title('Real Image')
plt.axis('off')

# %%
