from tensorboardX import SummaryWriter
import time
import matplotlib.pyplot as plt
import os
import torch

class Logger:
    def __init__(self, opt, unique=''):
        self.output_dir = opt.output_dir
        self.start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        name = os.path.join(opt.output_dir, "train", opt.name+self.start_time+unique)
        self.train_writer = SummaryWriter(name)
        #! temporarily use train_writer for val_writer
        self.val_writer = self.train_writer # SummaryWriter(name.replace("train", "val"))
        self.n_img_to_log = opt.train.n_img_to_log
        self.save_img_freq = opt.train.save_img_freq

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, writer='train'):
        if writer == 'train':
            self.train_writer.add_scalar(tag, scalar_value, global_step, walltime)
        elif writer == 'val':
            print("Logging to val writer:", tag, scalar_value, global_step)
            self.val_writer.add_scalar(tag, scalar_value, global_step, walltime)
        else:
            raise ValueError("writer should be 'train' or 'val'")
        
    def calc_accuracy(self, output, labels, from_logits=False):
        if from_logits:
            predictions = torch.round(torch.sigmoid(output.squeeze()))
        else:
            assert all(output >= 0) and all(output <= 1), "output should be in [0,1], if not from logits. Did you forget to apply sigmoid or not specify from_logits=True?"
            predictions = torch.round(output.squeeze())
        assert predictions.shape == labels.shape
        correct = predictions == labels
        return correct.float().mean()
    
    def log_accuracy(self, output, labels, global_step=None, walltime=None, writer='train', from_logits=False):
        accuracy = self.calc_accuracy(output, labels, from_logits)
        fake_accuracy = self.calc_accuracy(output[labels==0], labels[labels==0], from_logits)
        real_accuracy = self.calc_accuracy(output[labels==1], labels[labels==1], from_logits)
        self.add_scalar('accuracy', accuracy, global_step, walltime, writer)
        self.add_scalar('accuracy_fake', fake_accuracy, global_step, walltime, writer)
        self.add_scalar('accuracy_real', real_accuracy, global_step, walltime, writer)

    def close(self):
        self.train_writer.close()
        self.val_writer.close()

    def log_images(self, tag, images, labels, predictions, global_step=None, walltime=None, writer='train'):
        if global_step % self.save_img_freq != 0:
            return
        images = images.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        n_img = min(self.n_img_to_log, images.shape[0])
        fig,axes = plt.subplots(2, n_img)
        for i in range(n_img):
            axes[0,i].imshow(images[i].transpose(1,2,0))
            axes[0,i].set_title(f"Label: {'AI' if labels[i] == 0 else 'Nature'}")
            axes[0,i].axis('off')
            # plot predictions probability of 0,1
            axes[1,i].bar([0,1], [1-predictions[i].item(), predictions[i].item()])
            axes[1,i].set_xticks([0,1])
            axes[1,i].set_xticklabels(['AI', 'Nature'])
            axes[1,i].set_ylim(0,1)
            axes[1,i].set_title("Predictions")
        plt.tight_layout()

        if writer == 'train':
            self.train_writer.add_figure(tag, fig, global_step, walltime)
        elif writer == 'val':
            self.val_writer.add_figure(tag, fig, global_step, walltime)
        else:
            raise ValueError("writer should be 'train' or 'val'")
        plt.close(fig)