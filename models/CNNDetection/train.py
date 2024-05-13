import os
import time
import torch.nn
from tensorboardX import SummaryWriter

from validate import validate
from datasets import create_dataloader
from earlystop import EarlyStopping
from models.CNNDetection.networks.trainer import Trainer
from config import load_config


if __name__ == '__main__':
    opt = load_config('models/CNNDetection/train.yaml')

    data_loader = create_dataloader(
        data_path=opt.train.dataset.path,
        dataset=opt.train.dataset.name,
        split='train',
        batch_size=opt.train.opt.batch_size,
    )

    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')

            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # # Validation
        # model.eval()
        # acc, ap = validate(model.model, val_opt)[:2]
        # val_writer.add_scalar('accuracy', acc, model.total_steps)
        # val_writer.add_scalar('ap', ap, model.total_steps)
        # print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        # early_stopping(acc, model)
        # if early_stopping.early_stop:
        #     cont_train = model.adjust_learning_rate()
        #     if cont_train:
        #         print("Learning rate dropped by 10, continue training...")
        #         early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
        #     else:
        #         print("Early stopping.")
        #         break
        # model.train()

