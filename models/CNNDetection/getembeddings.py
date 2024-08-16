import os
import time
import numpy as np
import torch.nn
from tqdm import tqdm

from datasets import create_dataloader, available_generators
# from earlystop import EarlyStopping
from models.CNNDetection.networks.trainer import Trainer
from config import load_config


if __name__ == '__main__':
    config = load_config('models/CNNDetection/train.yaml')
    # seed = config.seed
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    generators = available_generators(config.train.dataset.path)
    print(f"Available generators: {generators}")

    for generator in generators:
        data_loader = create_dataloader(
            data_path=config.train.dataset.path,
            dataset=config.train.dataset.name,
            split='train',
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            generators_allowed=[generator],
            real=True,fake=True,
            shuffle=False
        )

        dataset_size = len(data_loader)
        print(f'#training images = {dataset_size*config.train.batch_size}')

        # logger = Logger(config, unique=f"{config.train.DANN_config.source}_{config.train.DANN_config.target}")

        model = Trainer(config)
        # print(f"Loading model {config.checkpoint_names[generator]} for generator {generator}")
        model.load_networks(filename="output/ResNet_noPretraining/checkpoints/ResNet_noPretraining/2024-07-22_16-10-02['stable_diffusion_v_1_4']_model_epoch_40.pth")
        # model.load_networks(filename="output/ResNet_noPretrain_fourier/checkpoints/ResNet_noPretrain_fourier/2024-07-20_14-19-17_model_epoch_latest.pth")#config.checkpoint_names['stable_diffusion_v_1_4_dann'])
        model.eval()


        # tqdm only every 10 steps
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader),miniters=50):
            model.total_steps += 1
            if i >= config.train.epoch_size:
                break

            inp,label, gen_label = model.set_input(data)
            features, pred = model.model(inp)

            # save features
            features = features.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            batch_size = features.shape[0]
            for j,(f, l, p) in enumerate(zip(features, label, pred)):
                os.makedirs(f"{config.train.dataset.path.replace('GenImage','features_no_pretrain')}/{generator}/{'ai' if l==0 else 'real'}", exist_ok=True)
                np.save(f"{config.train.dataset.path.replace('GenImage','features_no_pretrain')}/{generator}/{'ai' if l==0 else 'real'}/{i*batch_size+j}_{p[0]}.npy", f)

