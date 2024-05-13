import os
import time
import torch.nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datasets import create_dataloader
# from earlystop import EarlyStopping
from models.CNNDetection.networks.trainer import Trainer
from config import load_config
from logger import Logger


if __name__ == '__main__':
    config = load_config('models/CNNDetection/train.yaml')

    data_loader = create_dataloader(
        data_path=config.train.dataset.path,
        dataset=config.train.dataset.name,
        split='train',
        batch_size=config.train.batch_size,
    )

    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    logger = Logger(config)

    model = Trainer(config)
    # early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(config.train.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            model.total_steps += 1
            epoch_iter += config.train.batch_size

            model.set_input(data)
            model_out = model.optimize_parameters()
            output, loss = model_out['output'], model_out['loss']

            if model.total_steps % config.train.loss_freq == 0:
                logger.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % config.train.save_img_freq == 0:
                logger.log_images('train', model.input, model.label, model.output, model.total_steps)

            if model.total_steps % config.train.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (config.name, epoch, model.total_steps))
                model.save_networks('latest')

                

        if epoch % config.train.save_epoch_freq == 0:
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

