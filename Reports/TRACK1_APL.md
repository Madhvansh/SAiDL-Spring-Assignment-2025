# Track 1 - active-passive losses under label noise

## Scope

This track explores loss functions from the normalized-loss and active-passive-loss literature using a CIFAR-10 ResNet-18 modified for 32-by-32 images. Symmetric training-label noise is injected at fixed rates of 0.2, 0.4, 0.6, and 0.8.

The primary evidence is the stored output in [`CoreML/Google Colab/SAiDL_(2).ipynb`](../CoreML/Google%20Colab/SAiDL_(2).ipynb). The table below is a transcription of that notebook's final comparison, not a fresh rerun.

## Recorded notebook output

Best observed test accuracy over 100 epochs:

| Loss | eta=0.2 | eta=0.4 | eta=0.6 | eta=0.8 |
|---|---:|---:|---:|---:|
| CE | 85.56% | 79.90% | 73.18% | 39.92% |
| NCE | 83.46% | 75.76% | 61.03% | 34.66% |
| FL | 84.26% | 79.05% | 71.49% | 41.89% |
| NFL | 79.98% | 71.32% | 56.98% | 35.08% |
| NCE+MAE | 90.14% | 85.15% | 78.05% | 49.55% |
| NCE+RCE | 83.83% | 77.87% | 67.59% | 37.01% |
| NFL+MAE | 89.83% | 84.22% | 75.62% | 49.58% |
| NFL+RCE | 84.23% | 77.92% | 68.47% | 37.71% |

Within this stored run, the MAE-combined losses have the highest recorded values at each listed noise rate. The protocol limitations below prevent turning that observation into a general performance claim.

## Recorded protocol

- Dataset/model: CIFAR-10 and a torchvision ResNet-18 adapted for small images.
- Noise: class-symmetric replacement labels with seed 42 and fixed `eta`.
- Optimizer: SGD, learning rate 0.1, momentum 0.9, weight decay `5e-4`.
- Schedule: cosine annealing over 100 epochs.
- Selection: maximum test accuracy observed during training.
- Compared objectives: CE, focal loss, normalized CE, normalized focal loss, and four active-passive combinations.

## Limitations that affect interpretation

1. One transform containing random crop and horizontal flip is passed to both the training and test datasets. Test accuracy is therefore measured on randomly augmented test images rather than a deterministic clean test transform.
2. The test set is evaluated every epoch and its maximum is reported. It functions as a model-selection set, so these are not unbiased final-test estimates.
3. The table represents one stored run. The repository has no repeated seeds, uncertainty intervals, checkpoints, metric CSV, environment manifest tied to the run, or dataset checksum.
4. The notebook output includes multiprocessing cleanup warnings. They are not represented as accuracy failures, but they are another reason to rerun in a controlled environment.
5. The exact implemented RCE expression and active/passive weights should be reviewed against the cited paper before making replication claims.

## Replication status

**Historical notebook output only.** `scripts/check_repository.py` checks that the eight rows above remain present in the notebook. It does not retrain the model or validate the accuracy values independently.

## Recommended corrective rerun

- create separate stochastic training and deterministic test transforms;
- split the training set into train/validation subsets;
- select epochs and hyperparameters using validation only;
- evaluate the test set once after selection;
- run multiple seeds and report mean, dispersion, and every run;
- export configuration, environment, metrics, and checkpoint checksums; and
- add small unit tests for each loss against hand-computed tensors.

## Reference

Xingjun Ma et al., [Normalized Loss Functions for Deep Learning with Noisy Labels](https://arxiv.org/abs/2006.13554), ICML 2020. This repository is an exploratory implementation and does not claim to reproduce all results from that paper.
