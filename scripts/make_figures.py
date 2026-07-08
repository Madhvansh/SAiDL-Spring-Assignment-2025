"""Regenerate Track-2 figures from the real training configuration
(ROADMAP L2). Currently regenerates: the learning-rate schedule.

The previous lr_schedule.pdf was a hand-drawn triangle ("mock_lr_schedule" in
SSM/SSM Plots/vizualization.py, since removed). OneCycleLR is deterministic
given its configuration, so the honest figure is obtained by instantiating
the exact scheduler from SSM/SSM.py train_advanced() and recording its state
at every optimizer step. The figure is labeled "as configured" because the
training run itself did not log learning rates.

Run (from the repo root):  python scripts/make_figures.py
"""
import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# Exactly the configuration in SSM/SSM.py train_advanced():
#   optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1, betas=(0.9, 0.98))
#   OneCycleLR(optimizer, max_lr=5e-4, total_steps=200 * len(train_loader),
#              pct_start=0.05, anneal_strategy='cos')
# with len(train_loader) = ceil(50000 / 128) = 391 (CIFAR-10 train set,
# batch_size=128, drop_last=False).
EPOCHS = 200
STEPS_PER_EPOCH = math.ceil(50000 / 128)
MAX_LR = 5e-4
PCT_START = 0.05


def real_onecycle_lrs():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=MAX_LR,
                                  weight_decay=0.1, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR,
        total_steps=EPOCHS * STEPS_PER_EPOCH,
        pct_start=PCT_START,
        anneal_strategy='cos'
    )
    lrs = []
    for _ in range(EPOCHS * STEPS_PER_EPOCH):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return lrs


def make_lr_schedule(out_path):
    lrs = real_onecycle_lrs()
    epochs_axis = [step / STEPS_PER_EPOCH for step in range(len(lrs))]

    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(epochs_axis, lrs, lw=2, color='tab:green')
    plt.title('OneCycleLR Schedule — as configured in SSM/SSM.py\n'
              f'(max_lr={MAX_LR:g}, pct_start={PCT_START:g}, cosine anneal, '
              f'{EPOCHS} epochs x {STEPS_PER_EPOCH} steps)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    peak_step = max(range(len(lrs)), key=lrs.__getitem__)
    print(f'wrote {out_path}')
    print(f'  steps={len(lrs)}, initial lr={lrs[0]:.3e}, '
          f'peak lr={lrs[peak_step]:.3e} at epoch {peak_step / STEPS_PER_EPOCH:.1f}, '
          f'final lr={lrs[-1]:.3e}')


if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out = os.path.join(repo_root, 'SSM', 'SSM Plots', 'lr_schedule.pdf')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=default_out)
    args = ap.parse_args()
    make_lr_schedule(args.out)
