from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50
# import torchvision
# import torch

# model = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
# target_layers = [model.layer4[-1]]
# input_tensor = torch.rand(1, 3, 224, 224)
# # Note: input_tensor can be a batch tensor with several images!

# # Construct the CAM object once, and then re-use it on many images:
# cam = GradCAM(model=model, target_layers=target_layers)

# # You can also use it within a with statement, to make sure it is freed,
# # In case you need to re-create it inside an outer loop:
# # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
# #   ...

# # We have to specify the target we want to generate
# # the Class Activation Maps for.
# # If targets is None, the highest scoring category
# # will be used for every image in the batch.
# # Here we use ClassifierOutputTarget, but you can define your own custom targets
# # That are, for example, combinations of categories, or specific outputs in a non standard model.

# targets = [ClassifierOutputTarget(281)]

# # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# # In this example grayscale_cam has only one image in the batch:
# grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(input_tensor[0].permute(1, 2, 0).numpy(), grayscale_cam)
# print(visualization)
# import matplotlib.pyplot as plt
# plt.imshow(visualization)
# plt.show()


## now for our model

from models.CNNDetection.networks.trainer import Trainer
from config import load_config
from datasets import create_dataloader
import matplotlib.pyplot as plt

config = load_config('models/CNNDetection/train.yaml')
data_loader = create_dataloader(
    data_path=config.train.dataset.path,
    dataset=config.train.dataset.name,
    split='train',
    batch_size=config.train.batch_size,
    num_workers=config.train.num_workers
)

model = Trainer(config)
model.load_networks('latest')

# Construct the CAM object
target_layers = [model.model.layer4[-1]]
cam = GradCAM(model=model.model, target_layers=target_layers)

inputs = next(iter(data_loader))
inputs, labels = inputs[0], inputs[1]
targets = [BinaryClassifierOutputTarget(t) for t in labels]

grayscale_cam = cam(input_tensor=inputs, targets=targets)
visualizations = []
for i in range(len(grayscale_cam)):
    curr = grayscale_cam[i, :]
    visualization = show_cam_on_image(inputs[i].permute(1, 2, 0).numpy(), curr)
    visualizations.append(visualization)

plt.figure(figsize=(23, 20))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(visualizations[i])
    plt.title(f"Label: {'AI' if labels[i] == 0 else 'Nature'}")
    plt.axis('off')
plt.tight_layout()
plt.show()
