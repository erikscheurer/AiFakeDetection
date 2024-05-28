from models.CNNDetection.networks.trainer import Trainer
from config import load_config
from datasets import create_dataloader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

eps = .5 # max deviation from the original image
boxwidth = 30 # width of the adversarial box
device='cpu'
learn = 'ai'#'inverse' 'ai' or 'nature'

def get_random_mask(orig_image, boxwidth):
    if boxwidth > orig_image.size(2) or boxwidth > orig_image.size(3):
        return torch.ones_like(orig_image),0,0
    h = torch.randint(0, orig_image.size(2)-boxwidth, (1,)).item()
    w = torch.randint(0, orig_image.size(3)-boxwidth, (1,)).item()
    mask = torch.zeros_like(orig_image)
    mask[:, :, h:h+boxwidth, w:w+boxwidth] = 1
    return mask,h,w

config = load_config('models/CNNDetection/train.yaml')
data_loader = create_dataloader(
    data_path=config.train.dataset.path,
    dataset=config.train.dataset.name,
    split='train',
    batch_size=10,#config.train.batch_size,
    num_workers=config.train.num_workers
)

model = Trainer(config)
model.load_networks('latest')
model.to(device)
loss_fn = nn.BCEWithLogitsLoss()

def get_next(n=100):
    inputs = next(iter(data_loader))
    inputs, labels = inputs[0][:n], inputs[1][:n].float()
    return inputs.to(device), labels.to(device)

# start adversarial attack
orig_image, labels = get_next()
weights = torch.randn_like(orig_image[:1])
weights = nn.Parameter(weights, requires_grad=True)

mask,h,w = get_random_mask(orig_image, boxwidth)

# ensure max deviation:
imgs = orig_image + mask*.5*torch.tanh(torch.roll(weights,(h,w),dims=(2, 3))) * eps
# img1 = torch.clamp(img1, 0, 1)


print(f"Original Label: {'AI' if labels[0] == 0 else 'Nature'}")

opt = torch.optim.Adam([weights], lr=0.1)


# get the model prediction
model.eval()
orig_output = model.model(imgs)
print(f"Original Prediction: {'AI' if torch.sigmoid(orig_output)[0] < 0.5 else 'Nature'}")
print(f"Original Prediction Probability: {torch.sigmoid(orig_output)[0].item():.3f}")
# optimize the input image
losses = []
for i in tqdm(range(200)):
    opt.zero_grad()
    mask,h,w = get_random_mask(orig_image, boxwidth)
    image,labels = get_next()
    imgs = image + mask*.5*torch.tanh(torch.roll(weights,(h,w),dims=(2, 3))) * eps
    imgs.clamp_(0, 1)
    output = model.model(imgs)
    if learn == 'inverse':
        fake_labels = 1-labels
    elif learn == 'ai':
        fake_labels = torch.zeros_like(labels)
    elif learn == 'nature':
        fake_labels = torch.ones_like(labels)
    loss = loss_fn(output.squeeze(1), fake_labels)
    losses.append(loss.item())
    loss.backward()
    opt.step()
    print(f'{loss.item():.2f}')
    if loss.item() < 1e-5:
        break
plt.plot(losses)
plt.show(block=False)
plt.pause(0.1)



images,labels = get_next()
print(images.shape)
image = orig_image + mask*.5*torch.tanh(torch.roll(weights,(h,w),dims=(2, 3))) * eps
image.clamp_(0, 1)
print(image.shape)

# get the model prediction
model.eval()
output = model.model(image)
fig,axes = plt.subplots(images.shape[0], ncols=4, figsize=(20, 20))
for i in range(images.shape[0]):
    print(f"Final Prediction: {'AI' if torch.sigmoid(output)[i] < 0.5 else 'Nature'}")
    print(f"Final Prediction Probability: {torch.sigmoid(output)[i].item():.2f}")

    # plt.show()

    axes[i,0].imshow(orig_image[i].permute(1, 2, 0).cpu())
    axes[i,0].set_title(f"Label: {'AI' if labels[0] == 0 else 'Nature'}: Pred: {'AI' if torch.sigmoid(orig_output)[i] < 0.5 else 'Nature'}: {torch.sigmoid(orig_output)[i].item():.2f}")
    axes[i,0].axis('off')

    axes[i,1].imshow(image[i].detach().cpu().permute(1, 2, 0))
    axes[i,1].set_title(f"Adversarial Pred: {'AI' if torch.sigmoid(output)[i] < 0.5 else 'Nature'}: {torch.sigmoid(output)[i].item():.2f}")
    axes[i,1].axis('off')

    dev = (image[i] - orig_image[i]).detach().cpu()
    axes[i,2].imshow(dev.permute(1, 2, 0)+0.5)
    axes[i,2].set_title(f"Deviation: {torch.mean(torch.abs(dev)).item():.2f}")
    axes[i,2].axis('off')

    axes[i,3].imshow(((dev-dev.min())/(dev.max()-dev.min())).squeeze().permute(1, 2, 0))
    axes[i,3].set_title(f"Normalized Deviation")
    axes[i,3].axis('off')

plt.tight_layout()
plt.show()