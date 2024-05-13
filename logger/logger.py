from tensorboardX import SummaryWriter
import time
import matplotlib.pyplot as plt
import os

class Logger:
    def __init__(self, opt):
        self.output_dir = opt.output_dir
        name = os.path.join(opt.output_dir, "train", opt.name+time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.train_writer = SummaryWriter(name)
        self.val_writer = SummaryWriter(name.replace("train", "val"))
        self.n_img_to_log = opt.train.n_img_to_log

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, writer='train'):
        if writer == 'train':
            self.train_writer.add_scalar(tag, scalar_value, global_step, walltime)
        elif writer == 'val':
            self.val_writer.add_scalar(tag, scalar_value, global_step, walltime)
        else:
            raise ValueError("writer should be 'train' or 'val'")

    def close(self):
        self.train_writer.close()
        self.val_writer.close()

    def log_images(self, tag, images, labels, predictions, global_step=None, walltime=None, writer='train'):
        images = images.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        n_img = min(self.n_img_to_log, images.shape[0])
        fig,axes = plt.subplots(2, n_img, figsize=(20,20))
        for i in range(n_img):
            axes[0,i].imshow(images[i].transpose(1,2,0))
            axes[0,i].set_title(f"Label: {"AI" if labels[i] == 0 else "Nature"}")
            axes[0,i].axis('off')
            # plot predictions probability of 0,1
            axes[1,i].bar([0,1], predictions[i])
            axes[1,i].set_xticks([0,1])
            axes[1,i].set_ylim(0,1)
            axes[1,i].set_title("Predictions")
        plt.tight_layout()

        if writer == 'train':
            self.train_writer.add_figure(tag, fig, global_step, walltime)
        elif writer == 'val':
            self.val_writer.add_figure(tag, fig, global_step, walltime)
        else:
            raise ValueError("writer should be 'train' or 'val'")