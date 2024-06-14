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

from datasets import create_dataloader, available_generators
from config import load_config
from tqdm import tqdm


if __name__ == '__main__':
    opt = load_config('models/F3Net-main/train.yaml')
    generators = available_generators(opt.dataset_path)
    leave_out = generators.pop(opt.train.dataset.leave_out)

    dataloader = create_dataloader(opt.dataset_path, 'GenImage', 'train', opt.batch_size, num_workers=4, target_size=(299,299), generators_allowed=generators)

    # init checkpoint and logger
    logger = setup_logger(opt.output_dir, 'result.log', 'logger')
    tensorboard_logger = Logger(opt, unique=leave_out)
    best_val = 0.
    ckpt_model_name = 'best.pkl'

    # train
    model = Trainer([*range(osenvs)] if osenvs > 0 else None,
                    opt.mode,
                    opt.pretrained_path)
    model.total_steps = 0
    epoch = 0

    while epoch < opt.max_epoch:        
        logger.debug(f'Epoch {epoch}')

        for i,(inputs,labels) in enumerate(tqdm(dataloader)):
            if i>=opt.epoch_size:
                break
            model.total_steps += 1

            model.set_input(inputs,labels)
            loss, model_output = model.optimize_weight()

            tensorboard_logger.add_scalar('loss', loss, model.total_steps)
            tensorboard_logger.log_accuracy(model_output, model.label, model.total_steps, from_logits=True)
            tensorboard_logger.log_images('train', model.input, model.label, model_output, model.total_steps)

            if model.total_steps % opt.model_save_freq == 0:
                os.makedirs(os.path.dirname(opt.model_save_path), exist_ok=True)
                model.save(os.path.join(opt.model_save_path,f'{tensorboard_logger.start_time}_{epoch}_{model.total_steps}.pth'))
                logger.info(f'saving the model at epoch {epoch}, iters {model.total_steps} at {opt.model_save_path}')

        epoch = epoch + 1
        model.model.eval()
        
        # same generators
        auc, r_acc, f_acc, overall_acc = evaluate_GenImage(model, opt.dataset_path, generators=generators,val_size=opt.val_size, batch_size=opt.batch_size)
        logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
        tensorboard_logger.add_scalar('val_AUC', auc, epoch, writer='val')
        tensorboard_logger.add_scalar('val_real_acc', r_acc, epoch, writer='val')
        tensorboard_logger.add_scalar('val_fake_acc', f_acc, epoch, writer='val')
        tensorboard_logger.add_scalar('val_all_acc', overall_acc, epoch, writer='val')

        # different generator
        auc, r_acc, f_acc, overall_acc = evaluate_GenImage(model, opt.dataset_path, generators=[leave_out],val_size=opt.val_size, batch_size=opt.batch_size)
        logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
        tensorboard_logger.add_scalar('transfer_AUC', auc, epoch, writer='val')
        tensorboard_logger.add_scalar('transfer_real_acc', r_acc, epoch, writer='val')
        tensorboard_logger.add_scalar('transfer_fake_acc', f_acc, epoch, writer='val')
        tensorboard_logger.add_scalar('transfer_all_acc', overall_acc, epoch, writer='val')

        model.model.train()