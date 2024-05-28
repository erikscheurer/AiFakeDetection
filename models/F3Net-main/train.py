import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import torch
import torch.nn

from utils import setup_logger, evaluate_tinyGenImage
from trainer import Trainer

from tinyGenImage import TinyGenImageDataset

# config
dataset_path = './data/TinyGenImage/'
pretrained_path = './models/F3Net-main/pretrained/xception-b5690688.pth'
batch_size = 12
gpu_ids = [*range(osenvs)]
mode = 'FAD' # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = './output/F3Net/checkpoints/F3Net'
ckpt_name = 'FAD4_bz128'

max_epoch = 5
generators = 'biggan'
model_save_path = './output/F3Net/trained'

# how often to log the current loss
loss_freq = 40

if __name__ == '__main__':

    dataset = TinyGenImageDataset("./data/TinyGenImage", generators_allowed=generators, target_size=(299, 299))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8)
    
    len_dataloader = dataloader.__len__()


    # init checkpoint and logger
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    best_val = 0.
    ckpt_model_name = 'best.pkl'
    
    # train
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0
    epoch = 0
    
    while epoch < max_epoch:

        iterator = iter(dataloader)
        
        logger.debug(f'No {epoch}')
        i = 0

        while i < len_dataloader:
            
            i += 1
            model.total_steps += 1
            try:
                data = next(iterator)
            except StopIteration:
                break
            # -------------------------------------------------
        
            input,label = data

            model.set_input(input,label)
            loss = model.optimize_weight()

            if model.total_steps % loss_freq == 0:
                logger.debug(f'loss: {loss} at step: {model.total_steps}')

        epoch = epoch + 1
    
        model.model.eval()
        
        auc, r_acc, f_acc = evaluate_tinyGenImage(model, dataset_path, generators= generators)
        logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
        model.model.train()

    model.model.eval()

    # check if directory exists before saving
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    model.save(model_save_path)

    auc, r_acc, f_acc = evaluate_tinyGenImage(model, dataset_path, generators= generators)
    logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
