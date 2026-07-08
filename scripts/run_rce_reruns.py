"""Re-run the two RCE-based APL configurations with the restored paper RCE
(ROADMAP L5): NCE+RCE and NFL+RCE at eta in {0.2, 0.4, 0.6, 0.8} — 8 runs of
100 epochs each (~2.5-3 A100-hours total on Colab).

Writes one per-epoch CSV per run to --outdir (default: results/), flushed
every epoch, and skips runs whose CSV is already complete, so the job can be
resumed after a Colab disconnect by simply re-running the same command.

Protocol notes — identical to the original Table 1 runs except for the
restored RCE formula:
  - train_model defaults: 100 epochs, SGD momentum 0.9, cosine annealing,
    batch 512, symmetric label noise seeded with 42.
  - alpha=0.5 (NCE+RCE) and alpha=0.3 (NFL+RCE), beta=1: the values the
    original runs used — a deliberate, documented deviation from the paper's
    alpha=beta=1 convention, kept for comparability with the existing table.
  - torch.manual_seed(42) before each run. (The original 32 runs seeded once
    and trained sequentially, so their exact per-run RNG streams are not
    reconstructable; per-run seeding is the closest well-defined protocol.)

Usage (from the repo root):  python scripts/run_rce_reruns.py --outdir results
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from CoreML.APL_Models.APL import (
    APL, NormalizedCrossEntropy, NormalizedFocalLoss, RCE, train_model,
)

CONFIGS = {
    'nce_rce': lambda: APL(NormalizedCrossEntropy(), RCE(), alpha=0.5, beta=1),
    'nfl_rce': lambda: APL(NormalizedFocalLoss(), RCE(), alpha=0.3, beta=1),
}


def csv_complete(path, epochs):
    if not os.path.exists(path):
        return False
    with open(path, newline='') as f:
        rows = [r for r in csv.reader(f) if r]
    return len(rows) >= epochs + 1  # header + one row per epoch


def main():
    ap = argparse.ArgumentParser(
        description='8 RCE re-runs with the restored paper formula (ROADMAP L5)')
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--etas', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument('--configs', nargs='+', choices=sorted(CONFIGS),
                    default=sorted(CONFIGS))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary = []
    for name in args.configs:
        for eta in args.etas:
            out = os.path.join(args.outdir, f'{name}_eta{eta:g}.csv')
            if csv_complete(out, args.epochs):
                print(f'[skip] {out} already complete')
                continue
            print(f'\n=== {name} @ eta={eta:g} -> {out} ===')
            torch.manual_seed(42)
            metrics = train_model(eta, CONFIGS[name](),
                                  num_epochs=args.epochs, log_csv=out)
            summary.append((name, eta, metrics['best_acc']))
            print(f'[done] {name} eta={eta:g}: best test acc {metrics["best_acc"]:.2f}%')

    if summary:
        print('\nSummary (best test accuracy over epochs; selection rule: max over epochs):')
        for name, eta, acc in summary:
            print(f'  {name:8s} eta={eta:g}: {acc:.2f}%')


if __name__ == '__main__':
    main()
