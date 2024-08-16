import os
import time
import torch.nn
from tqdm import tqdm

from datasets import create_dataloader, available_generators
# from earlystop import EarlyStopping
from models.CNNDetection.networks.trainer import Trainer
from models.F3Net.utils import evaluate_GenImage
from config import load_config
from logger import Logger


if __name__ == '__main__':
    config = load_config('models/CNNDetection/train.yaml')
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generators = available_generators(config.train.dataset.path)
    print(f"Available generators: {generators}")
    source_generators = [generators[i] for i in config.train.DANN_config.source]
    target_generators = [generators[i] for i in config.train.DANN_config.target]
    print(f"Source generators: {source_generators}")
    print(f"Target generators: {target_generators}")

    data_loader = create_dataloader(
        data_path=config.train.dataset.path,
        dataset=config.train.dataset.name,
        split='train',
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        generators_allowed=source_generators+target_generators
    )

    dataset_size = len(data_loader)
    print(f'#training images = {dataset_size*config.train.batch_size}')

    logger = Logger(config, unique=f"{config.train.DANN_config.source}_{config.train.DANN_config.target}")

    model = Trainer(config)
    # early_stopping = EarlyStopping(patience=config.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(config.train.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()

        ##########################################################
        #                      Training                          #
        ##########################################################

        # tqdm only every 10 steps
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader),miniters=50):
            model.total_steps += 1
            if i >= config.train.epoch_size:
                break

            model.set_input(data)
            model_out = model.optimize_parameters()

            if model.total_steps % config.train.loss_freq == 0:
                for key in model_out:
                    if key == 'output':
                        output = model_out[key]
                        logger.log_accuracy(output.squeeze(), model.label, model.total_steps, from_logits=True)
                    else:
                        logger.add_scalar(key, model_out[key], model.total_steps)
                logger.add_scalar('lr', model.optimizer.param_groups[0]['lr'], model.total_steps)

            if model.total_steps % config.train.save_img_freq == 0:
                logger.log_images('train', model.input, model.label, model.output, model.total_steps)

            if model.total_steps % config.train.save_latest_freq == 0:
                print(f'saving the latest model (epoch {epoch}, total_steps {model.total_steps}), {logger.start_time} {target_generators}')
                model.save_networks('latest', logger.start_time+str(target_generators))


        if epoch % config.train.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}, iters {model.total_steps}, {logger.start_time} {target_generators}')
            model.save_networks('latest', logger.start_time+str(target_generators))
            model.save_networks(epoch, logger.start_time+str(target_generators))


        ##########################################################
        #                      Validation                        #
        ##########################################################

        model.eval()

        # same generators
        auc, r_acc, f_acc, overall_acc = evaluate_GenImage(model, config.train.dataset.path, generators=source_generators,val_size=config.train.val_size, batch_size=config.train.batch_size)
        logger.add_scalar('val_AUC', auc, epoch, writer='val')
        logger.add_scalar('val_real_acc', r_acc, epoch, writer='val')
        logger.add_scalar('val_fake_acc', f_acc, epoch, writer='val')
        logger.add_scalar('val_all_acc', overall_acc, epoch, writer='val')

        # different generator
        auc, r_acc, f_acc, overall_acc = evaluate_GenImage(model, config.train.dataset.path, generators=target_generators,val_size=config.train.val_size, batch_size=config.train.batch_size)
        logger.add_scalar('transfer_AUC', auc, epoch, writer='val')
        logger.add_scalar('transfer_real_acc', r_acc, epoch, writer='val')
        logger.add_scalar('transfer_fake_acc', f_acc, epoch, writer='val')
        logger.add_scalar('transfer_all_acc', overall_acc, epoch, writer='val')

        model.train()



        

