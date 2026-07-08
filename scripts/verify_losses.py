"""Unit check (ROADMAP L5 acceptance): the loss implementations in
CoreML/APL_Models/APL.py match their closed forms on a toy batch.

Closed forms (Ma et al., "Normalized Loss Functions for Deep Learning with
Noisy Labels", ICML 2020), with p = softmax(logits), y the target class,
K the number of classes:

  NCE  = mean_b [ -log p_y / (-sum_k log p_k) ]
  NFL  = mean_b [ (1-p_y)^g (-log p_y) / (sum_k (1-p_k)^g (-log p_k)) ]   (g=2)
  RCE  = mean_b [ -A (1 - p_y) ],  A = -4  (paper text; the official repo's
         1e-4 prob clamp would give effective A = ln(1e-4) ~= -9.21)
  MAE  = mean_b [ 2 (1 - p_y) ]  -- paper form. This repo's MAELoss reduces
         with mean over the K classes instead of sum, so the implementation
         computes 2(1-p_y)/K: a constant factor K vs the paper, absorbed by
         the beta weight in APL. Asserted exactly as implemented, and the
         relation to the paper form is asserted too.

Run:  python scripts/verify_losses.py     (exit 0 = all pass)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from CoreML.APL_Models.APL import (
    NormalizedCrossEntropy, NormalizedFocalLoss, MAELoss, RCE,
)

torch.manual_seed(0)
B, K = 16, 10
logits = torch.randn(B, K) * 2.0
targets = torch.randint(0, K, (B,))

probs = F.softmax(logits, dim=1)
p_y = probs[torch.arange(B), targets]

failures = []


def check(name, actual, expected):
    actual = float(actual)
    expected = float(expected)
    ok = abs(actual - expected) <= 1e-6 + 1e-4 * abs(expected)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: implementation={actual:.8f} closed_form={expected:.8f}")
    if not ok:
        failures.append(name)


# NCE
expected_nce = ((-torch.log(p_y)) / (-torch.log(probs).sum(dim=1))).mean()
check("NCE  = -log p_y / -sum_k log p_k", NormalizedCrossEntropy()(logits, targets), expected_nce)

# NFL (gamma = 2, the repo/paper default)
g = 2.0
nfl_num = (1 - p_y) ** g * (-torch.log(p_y))
nfl_den = ((1 - probs) ** g * (-torch.log(probs))).sum(dim=1)
check("NFL  = focal(y) / sum_k focal(k)", NormalizedFocalLoss(gamma=g)(logits, targets),
      (nfl_num / nfl_den).mean())

# RCE — the L5 restoration: paper closed form -A(1-p_y) with A = -4
rce = RCE()
assert rce.A == -4.0, f"RCE default A must be -4 (paper text), got {rce.A}"
A = rce.A
check("RCE  = -A(1 - p_y), A=-4", rce(logits, targets), (-A * (1 - p_y)).mean())

# MAE — as implemented (mean over classes): 2(1-p_y)/K
mae_impl = MAELoss()(logits, targets)
check("MAE  = 2(1 - p_y)/K  (impl reduction)", mae_impl, (2 * (1 - p_y) / K).mean())
# ... and its fixed relation to the paper form 2(1-p_y)
check("MAE  * K = 2(1 - p_y)  (paper form)", mae_impl * K, (2 * (1 - p_y)).mean())

if failures:
    print(f"\n{len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
print("\nAll loss implementations match their closed forms.")
