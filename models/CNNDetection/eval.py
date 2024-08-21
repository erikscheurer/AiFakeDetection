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


if __name__ == "__main__":
    config = load_config("models/CNNDetection/train.yaml")
    config.train.lr = 0.0
    config.train.save_img_freq = 1

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generators = available_generators("data/CurrentAI")
    print(f"Available generators: {generators}")

    data_loader = create_dataloader(
        data_path="data/CurrentAI",
        dataset=config.train.dataset.name,
        split="train",
        batch_size=config.train.n_img_to_log,
        num_workers=config.train.num_workers,
        generators_allowed=generators,
    )

    dataset_size = len(data_loader)
    print(f"#training images = {dataset_size*config.train.batch_size}")

    logger = Logger(
        config,
        unique=f"{config.train.DANN_config.source}_{config.train.DANN_config.target}",
    )

    model = Trainer(config)
    model.load_networks(filename=config.checkpoint_names["BigGAN"])

    # tqdm only every 10 steps
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), miniters=50):
        model.total_steps += 1

        inp, label, gen_label = model.set_input(data)
        model_out = model.optimize_parameters()

        # check if NaN or inf in input
        if torch.isnan(model.input).any() or torch.isinf(model.input).any():
            print("NaN or inf in input")
            break

        if model.total_steps % config.train.loss_freq == 0:
            for key in model_out:
                if key == "output":
                    output = model_out[key]
                    batch_size = output.shape[0]
                    if batch_size == 1:
                        logger.log_accuracy(
                            output.squeeze(),
                            model.label.item(),
                            model.total_steps,
                            from_logits=True,
                        )
                    else:
                        logger.log_accuracy(
                            output.squeeze(),
                            model.label.squeeze(),
                            model.total_steps,
                            from_logits=True,
                        )
                else:
                    logger.add_scalar(key, model_out[key], model.total_steps)
            logger.add_scalar(
                "lr", model.optimizer.param_groups[0]["lr"], model.total_steps
            )

        if True:
            logger.log_images(
                "train", model.input, model.label, model.output, model.total_steps
            )
