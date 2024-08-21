from models.CNNDetection.networks.trainer import Trainer
from config import load_config
from datasets import create_dataloader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

eps = 0.001  # max deviation from the original image
boxwidth = 1000  # width of the adversarial box


def get_random_mask(orig_image, boxwidth):
    if boxwidth > orig_image.size(2) or boxwidth > orig_image.size(3):
        return torch.ones_like(orig_image), 0, 0
    h = torch.randint(0, orig_image.size(2) - boxwidth, (1,)).item()
    w = torch.randint(0, orig_image.size(3) - boxwidth, (1,)).item()
    mask = torch.zeros_like(orig_image)
    mask[:, :, h : h + boxwidth, w : w + boxwidth] = 1
    return mask, h, w


config = load_config("models/CNNDetection/train.yaml")
data_loader = create_dataloader(
    data_path=config.train.dataset.path,
    dataset=config.train.dataset.name,
    split="train",
    batch_size=config.train.batch_size,
    num_workers=config.train.num_workers,
)

model = Trainer(config)
model.load_networks("latest")
loss_fn = nn.BCEWithLogitsLoss()

inputs = next(iter(data_loader))
inputs, labels = inputs[0], inputs[1]

# start adversarial attack
for i in range(10):  # attack the first 10 images
    orig_image = inputs[i : i + 1].clone()
    weights = torch.randn_like(orig_image)
    weights = nn.Parameter(weights, requires_grad=True)

    mask, h, w = get_random_mask(orig_image, boxwidth)

    # ensure max deviation:
    shifted_weights = torch.roll(weights, shifts=(h, w), dims=(2, 3))
    perturbed_img = (
        orig_image + mask * torch.tanh(shifted_weights) * eps
    )  # origial bild + tanh(noise) * eps
    # img1 = torch.clamp(img1, 0, 1)

    label1 = labels[i : i + 1].float()
    print(f"Original Label: {'AI' if label1 == 0 else 'Nature'}")

    opt = torch.optim.Adam([weights], lr=0.1)

    # get the model prediction
    model.eval()
    orig_output = model.model(orig_image)
    print(
        f"Original Prediction: {'AI' if torch.sigmoid(orig_output) < 0.5 else 'Nature'}"
    )
    print(f"Original Prediction Probability: {torch.sigmoid(orig_output)}")
    # optimize the input image
    try:
        for i in tqdm(range(500)):
            opt.zero_grad()
            mask, h, w = get_random_mask(orig_image, boxwidth)
            shifted_weights = torch.roll(weights, shifts=(h, w), dims=(2, 3))
            perturbed_img = orig_image + mask * torch.tanh(shifted_weights) * eps
            perturbed_img.clamp_(0, 1)
            output = model.model(perturbed_img)
            loss = loss_fn(output.squeeze(1), 1 - label1)
            loss.backward()
            opt.step()
            print(
                torch.sigmoid(output).item(),
                f"{'AI' if torch.sigmoid(output) < 0.5 else 'Nature'}",
            )
            if loss.item() < 1e-5:
                break
    except KeyboardInterrupt:
        pass

    # get the model prediction
    model.eval()
    output = model.model(perturbed_img)
    pred = torch.sigmoid(output) > 0.5
    print(f"Final Prediction: {'AI' if pred == 0 else 'Nature'}")
    print(f"Final Prediction Probability: {torch.sigmoid(output).item():.2f}")

    # plt.show()

    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(orig_image[0].permute(1, 2, 0))
    axes[0].set_title(
        f"Original Pred: {'AI' if label1 == 0 else 'Nature'}: {torch.sigmoid(orig_output).item():.2f}"
    )
    axes[0].axis("off")

    axes[1].imshow(perturbed_img[0].detach().permute(1, 2, 0))
    axes[1].set_title(
        f"Adversarial Pred: {'AI' if pred == 0 else 'Nature'}: {torch.sigmoid(output).item():.2f}"
    )
    axes[1].axis("off")

    dev = (perturbed_img - orig_image).detach()
    axes[2].imshow(dev[0].permute(1, 2, 0) + 0.5)
    axes[2].set_title(f"Deviation: {torch.mean(torch.abs(dev)).item():.2f}")
    axes[2].axis("off")

    axes[3].imshow(
        ((dev - dev.min()) / (dev.max() - dev.min())).squeeze().permute(1, 2, 0)
    )
    axes[3].set_title(f"Normalized Deviation")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
