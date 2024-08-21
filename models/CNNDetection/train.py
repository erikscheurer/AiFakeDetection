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

    generators = available_generators(config.train.dataset.path)
    print(f"Available generators: {generators}")
    leave_out = generators.pop(config.train.dataset.leave_out)
    print(f"Leaving out generator: {leave_out}")

    data_loader = create_dataloader(
        data_path=config.train.dataset.path,
        dataset=config.train.dataset.name,
        split="train",
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        generators_allowed=generators,
    )

    dataset_size = len(data_loader)
    print(f"#training images = {dataset_size*config.train.batch_size}")

    logger = Logger(config, unique=leave_out)

    model = Trainer(config)
    # early_stopping = EarlyStopping(patience=config.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(config.train.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()

        ##########################################################
        #                      Training                          #
        ##########################################################

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            model.total_steps += 1
            if i >= config.train.epoch_size:
                break

            if config.fourier:
                # perform fourier transform on the input
                data[0] = torch.fft.rfft2(data[0]).log()
                # separate the real and imaginary parts
                data[0] = torch.cat([data[0].real, data[0].imag], dim=1)

            model.set_input(data)
            model_out = model.optimize_parameters()
            output, loss = model_out["output"], model_out["loss"]

            if model.total_steps % config.train.loss_freq == 0:
                logger.add_scalar("loss", model.loss, model.total_steps)
                logger.log_accuracy(
                    output, model.label, model.total_steps, from_logits=True
                )
                logger.add_scalar(
                    "lr", model.optimizer.param_groups[0]["lr"], model.total_steps
                )

            if model.total_steps % config.train.save_img_freq == 0:
                logger.log_images(
                    "train", model.input, model.label, model.output, model.total_steps
                )

            if model.total_steps % config.train.save_latest_freq == 0:
                print(
                    "saving the latest model %s (epoch %d, model.total_steps %d)"
                    % (config.name, epoch, model.total_steps)
                )
                model.save_networks("latest", logger.start_time)

        if epoch % config.train.save_epoch_freq == 0:
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, model.total_steps)
            )
            model.save_networks("latest", logger.start_time)
            model.save_networks(epoch, logger.start_time)

        ##########################################################
        #                      Validation                        #
        ##########################################################

        model.eval()

        # same generators
        auc, r_acc, f_acc, overall_acc = evaluate_GenImage(
            model,
            config.train.dataset.path,
            generators=generators,
            val_size=config.train.val_size,
            batch_size=config.train.batch_size,
        )
        logger.add_scalar("val_AUC", auc, epoch, writer="val")
        logger.add_scalar("val_real_acc", r_acc, epoch, writer="val")
        logger.add_scalar("val_fake_acc", f_acc, epoch, writer="val")
        logger.add_scalar("val_all_acc", overall_acc, epoch, writer="val")

        # different generator
        auc, r_acc, f_acc, overall_acc = evaluate_GenImage(
            model,
            config.train.dataset.path,
            generators=[leave_out],
            val_size=config.train.val_size,
            batch_size=config.train.batch_size,
        )
        logger.add_scalar("transfer_AUC", auc, epoch, writer="val")
        logger.add_scalar("transfer_real_acc", r_acc, epoch, writer="val")
        logger.add_scalar("transfer_fake_acc", f_acc, epoch, writer="val")
        logger.add_scalar("transfer_all_acc", overall_acc, epoch, writer="val")

        model.train()
