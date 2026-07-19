# SAiDL Spring Assignment 2025 - reproducibility archive

[![Repository checks](https://github.com/Madhvansh/SAiDL-Spring-Assignment-2025/actions/workflows/ci.yml/badge.svg)](https://github.com/Madhvansh/SAiDL-Spring-Assignment-2025/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains my work completed for the **2025 SAiDL Spring Assignment at BITS Goa**. It is a personal submission and portfolio record. The repository does **not** by itself claim formal SAiDL membership, selection, endorsement, or an official BITS Goa publication.

The work explores two CIFAR-10 directions:

1. active-passive loss combinations under symmetric label noise; and
2. a custom S4-inspired state-space sequence mixer over image patches.

The notebooks retain historical outputs from the original experiments. The reports below document what those outputs do and do not establish.

## At a glance

| Track | Implementation | Recorded notebook output | Evidence status |
|---|---|---|---|
| CoreML / noisy labels | CIFAR-10 ResNet-18 with CE, focal, normalized, and active-passive losses | four noise rates and eight loss configurations | Historical single-run output; test augmentation and test-set model selection prevent treating the table as a clean held-out benchmark |
| SSM | custom S4-inspired kernel and patch sequence classifier | best test accuracy `86.83%`; epoch 200 test accuracy `86.63%` | Historical single-run output; the test set was used to select the best checkpoint and no checkpoint is committed |

These values are transcribed from stored notebook output and checked by `scripts/check_repository.py`; they are not claims of independent replication.

## Repository map

```text
CoreML/
  APL_Models/APL.py                  exploratory active-passive-loss script
  Google Colab/SAiDL_(2).ipynb      primary historical notebook and outputs
  Noisy_Data_(vizualization)/       label-noise visualization experiment
  Normalized/                       earlier normalized-loss experiments
SSM/
  SSM.py                            custom S4-inspired CIFAR-10 model
  Google Colab/SSM.ipynb            primary historical notebook and outputs
  SSM Plots/                        plots derived from the stored training log
Reports/
  TRACK1_APL.md                     protocol, output table, and limitations
  TRACK2_SSM.md                     protocol, output summary, and limitations
scripts/check_repository.py         dependency-free source/evidence check
```

`CoreML/Normalized/temp.py` is retained as a legacy exploratory ResNet-18 variant. It is not a temporary build artifact and is not the primary experiment entry point.

## Reproduce the environment

Python 3.11 is the documented target. A minimal pip environment is:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate saidl-spring-2025
```

`environment.txt` is retained only as a historical Windows-specific environment snapshot. It is not the recommended cross-platform install file.

Before a long training run, execute the repository checks:

```bash
python scripts/check_repository.py
```

Then open the relevant notebook, or run `python SSM/SSM.py`. The experiments download CIFAR-10 and the full configurations are compute-intensive. The CoreML notebook executes 32 training configurations (eight losses across four noise rates, 100 epochs each); it should not be presented as a quickstart.

## Important evaluation limitations

- The CoreML notebook applies `RandomHorizontalFlip` and `RandomCrop` to both training and test datasets. Its reported test values therefore include random test-time augmentation.
- The CoreML table reports the maximum test accuracy observed across 100 epochs, so the test set is also used for model selection.
- The SSM test transform is deterministic, but the best checkpoint is likewise selected using test accuracy.
- The stored results are single runs. There are no repeated-seed statistics, confidence intervals, committed checkpoints, or machine-readable metric exports.
- The SSM code is a custom, S4-inspired implementation. It is not an official S4/S4D implementation or a reproduction of an official CIFAR-10 benchmark.
- CIFAR-10 is downloaded at runtime and is not redistributed by this repository.

A future benchmark-quality rerun should use separate train/validation/test transforms and splits, select checkpoints on validation data, evaluate the untouched test set once, record seeds and dependency versions, and export metrics/checksums as machine-readable artifacts.

## Reports and provenance

- [Track 1: active-passive losses](Reports/TRACK1_APL.md)
- [Track 2: S4-inspired sequence mixer](Reports/TRACK2_SSM.md)
- [Report corrections and provenance](Reports/README.md)
- [Research resources](Resources.md)

## Contributing

Reproductions, test-split fixes, small runnable configs, and corrections are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), the [Code of Conduct](CODE_OF_CONDUCT.md), and the [Security Policy](SECURITY.md) before opening a contribution.

## License

Code and original documentation in this repository are available under the [MIT License](LICENSE). CIFAR-10, cited papers, PyTorch, and torchvision have their own terms and are not relicensed here.
