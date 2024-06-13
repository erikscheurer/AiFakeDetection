import os
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
else:
    osenvs = 0
import torch
import torch.nn

from utils import evaluate_GenImage, setup_logger
from trainer import Trainer
from logger import Logger

from datasets import create_dataloader
from omegaconf import OmegaConf
from tqdm import tqdm

opt = OmegaConf.create({
        'output_dir': './output/F3Net',
        'name': 'F3Net',
        'train': {
            'n_img_to_log': 10,
            'save_img_freq': 100,
        },
        'dataset_path': './data/GenImage/',
        'pretrained_path': './models/F3Net-main/pretrained/xception-b5690688.pth',
        'batch_size': 12,
        'gpu_ids': [*range(osenvs)] if osenvs > 0 else None,
        'mode': 'FAD',# ['Original', 'FAD', 'LFS', 'Both', 'Mix']
        'max_epoch': 5,
        'generators': None,
        'model_save_path': './output/F3Net/checkpoints',
        'loss_freq': 40,
        'model_save_freq': 100,
    })


if __name__ == '__main__':

    dataloader = create_dataloader(opt.dataset_path, 'GenImage', 'train', opt.batch_size, num_workers=4, target_size=(299,299))

    # init checkpoint and logger
    logger = setup_logger(opt.output_dir, 'result.log', 'logger')
    tensorboard_logger = Logger(opt)
    best_val = 0.
    ckpt_model_name = 'best.pkl'

    # train
    model = Trainer(opt.gpu_ids, opt.mode, opt.pretrained_path)
    model.total_steps = 0
    epoch = 0

    while epoch < opt.max_epoch:        
        logger.debug(f'Epoch {epoch}')

        for i,(inputs,labels) in enumerate(tqdm(dataloader)):
            model.total_steps += 1

            model.set_input(inputs,labels)
            loss, model_output = model.optimize_weight()

            tensorboard_logger.add_scalar('loss', loss, model.total_steps)
            tensorboard_logger.log_accuracy(model_output, model.label, model.total_steps, from_logits=True)
            tensorboard_logger.log_images('train', model.input, model.label, model_output, model.total_steps)

            if model.total_steps % opt.model_save_freq == 0:
                os.makedirs(os.path.dirname(opt.model_save_path), exist_ok=True)
                model.save(opt.model_save_path+f'_{epoch}_{model.total_steps}.pth')
                logger.info(f'saving the model at epoch {epoch}, iters {model.total_steps} at {opt.model_save_path}')

        epoch = epoch + 1
        model.model.eval()
        
        auc, r_acc, f_acc = evaluate_GenImage(model, opt.dataset_path, generators= opt.generators)
        logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
        model.model.train()

    model.model.eval()


    auc, r_acc, f_acc = evaluate_GenImage(model, opt.dataset_path, generators= opt.generators)
    logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
