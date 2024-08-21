# Ai Fake Detection

## Notes for Tutors:
Our code is distributed along three branches, main, DANN and DataAugmentations. The main branch contains code for the 7/8 training (pre and not pretrained is set [in model code](models/CNNDetection/networks/resnet.py#189)) and Fourier training. The rest can be set [in the config](models/CNNDetection/train.yaml).
We also have the F3Net model implemented in this repo even though it is not mentioned in the report, for time reasons and because we wanted to stay consistent with a single model.
The training for the models is in the [folder of the specific model](models/CNNDetection/train.py).

## Choices

- 2 Datasets: GenImage, Cifake
    - Genimage is way larger and multiple generators
    - Cifake is only Stable diffusion 1.4 and is almost solved (ie >95% accuracy)
    - -> Use Genimage
    - There is Tiny GenImage [on Kaggle](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) with only 5k images -> use this for debugging to start and then switch to full dataset on cluster
- [GenImage](https://github.com/GenImage-Dataset/GenImage/tree/main?tab=readme-ov-file#genimage-a-million-scale-benchmark-for-detecting-ai-generated-image-homepage) Repo has multiple networks implemented
    - Thinking about what networks we want to train:
        - Until now, detectors have only been trained on one generator at a time (They use Stable diffusion 1.4 as this is the one that best generalizes to other generators)
        - We could train a detector on all generators at once
        - -> Use the preexisting code but train on all generators at once


## Todo
- [x] Data
    - [x] Dataloader for Tiny GenImage / GenImage
    - [x] Download full GenImage dataset
- [x] Netzwerke
    - [ ] implementierungen aus GenImage Repo bzw die aus dem Paper
        - Implementierungen angucken und bewerten wie sinnvoll die für uns sind (aka wie einfach kann man sie für uns anpassen)
        - Auf einen einfachen dataloader mal anwenden
    - [ ] Trainingsskripte
- [ ] Evaluation
    - [ ] Generalisierungs-performance
    - [ ] Vergleiche verschiedene Modelle

## Setup

- man braucht auf bwuni cluster einen workspace, dann verlinken mit `ln -s /path/to/your/workspace/ ~/aacv_ws`
- in workspace `git clone <this repo>`
- verlinken zu home `ln -s $(pwd)/AiFakeDetection/ ~/AiFakeDetection`

### python venv

```bash
module devel/python/3.11.7_intel_2021.4.0
python -m venv aacv_venv
ln -s $(pwd)/aacv_venv/ ~/aacv_venv
source ~/aacv_venv/bin/activate
pip install -r ~/AiFakeDetection/requirements.txt
```
