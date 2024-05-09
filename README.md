# Ai Fake Detection

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
- [ ] Data
    - [ ] Dataloader for Tiny GenImage / GenImage (Erik)
    - [ ] Download full GenImage dataset (Lukas)
- [ ] Netzwerke
    - [ ] implementierungen aus GenImage Repo bzw die aus dem Paper (Yannik)
        - Implementierungen angucken und bewerten wie sinnvoll die für uns sind (aka wie einfach kann man sie für uns anpassen)
        - Auf einen einfachen dataloader mal anwenden
    - [ ] Trainingsskripte
- [ ] Evaluation
    - [ ] Generalisierungs-performance
    - [ ] Vergleiche verschiedene Modelle

## Fragen an die Tutoren
- Wie ist das mit parallel processing auf dem bw uni cluster, wie bekommt man grafikkarten zugewiesen?