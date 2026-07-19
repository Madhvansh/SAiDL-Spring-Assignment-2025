# Track 2 - S4-inspired image sequence mixer

## Scope

This track treats CIFAR-10 images as sequences of learned patches and mixes them with a custom state-space kernel. The code borrows ideas associated with structured state-space sequence models, but it is an independent educational implementation rather than the official S4 or S4D code.

The primary evidence is the stored output in [`SSM/Google Colab/SSM.ipynb`](../SSM/Google%20Colab/SSM.ipynb), whose code corresponds closely to [`SSM/SSM.py`](../SSM/SSM.py).

## Recorded notebook output

- Best test accuracy observed across 200 epochs: **86.83%**.
- Epoch 200 test accuracy: **86.63%** (training loss `0.9280`).

These values are historical notebook output, not independently rerun measurements.

## Recorded protocol

- Dataset: CIFAR-10.
- Training augmentation: random crop, horizontal flip, CIFAR-10 AutoAugment, normalization, and random erasing.
- Test preprocessing: tensor conversion and normalization only.
- Model: convolutional patch embedding, 12 custom S4-inspired residual blocks at dimension 512, mean pooling, and a linear classifier.
- Optimizer: AdamW with OneCycleLR, gradient clipping, and label-smoothed cross entropy.
- Training length: 200 epochs.
- Selection: checkpoint with the highest observed test accuracy.

## Limitations that affect interpretation

1. Although test preprocessing is deterministic, the test set is evaluated every epoch and used to choose `s4_2d_best.pth`. The best value is therefore selected on the test set, not on a validation split.
2. Only one stored run is available. There are no repeated seeds, confidence intervals, committed checkpoint, machine-readable metric export, or checkpoint checksum.
3. The script does not explicitly establish all sources of determinism or record hardware and dependency versions for this run.
4. The state-space parameterization is custom. Class and variable names containing `S4D` describe inspiration, not conformance with an official implementation or paper reproduction.
5. The PDFs in `SSM/SSM Plots/` are derived from a training-log string embedded in `vizualization.py`; they are visualizations of the stored log, not independent evidence.

## Replication status

**Historical notebook output only.** `scripts/check_repository.py` verifies the two recorded terminal strings and the structural validity of the source/notebooks. It does not download CIFAR-10 or execute training.

## Recommended corrective rerun

- create a validation split from training data and select checkpoints on it;
- reserve the test set for one final evaluation;
- add an explicit seed and deterministic-mode record;
- repeat across multiple seeds and report dispersion;
- export per-epoch metrics to CSV or JSON;
- store configuration, commit, environment, hardware, and checkpoint hashes; and
- compare with parameter- and compute-matched baselines.
