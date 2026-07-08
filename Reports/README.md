# Reports

The three original PDF reports (`CoreML.pdf`, `COREML_OUTPUT.pdf`, `SSM.pdf`)
have been **withdrawn** from this repository and are being replaced by
Markdown reports.

## Why

An internal audit found that parts of the PDFs were not supported by the code
and logs in this repository — including a figure generated from random
placeholder data that shipped with an invented caption, and several claims
and numbers with no committed artifact behind them. No source files exist for
the PDFs, so they cannot be patched; they are being rewritten from scratch as
Markdown, where every number links to the committed evidence (notebook cell
or `results/` CSV) that produced it.

## Where the old PDFs live

They remain retrievable from git history at commit `a5c4f17`:

```
git show a5c4f17:Reports/CoreML.pdf        > CoreML.pdf
git show a5c4f17:Reports/COREML_OUTPUT.pdf > COREML_OUTPUT.pdf
git show a5c4f17:Reports/SSM.pdf           > SSM.pdf
```

This is a deliberate, documented withdrawal — the history is untouched.

## What replaces them

- `TRACK1_APL.md` — Active-Passive Losses (Ma et al., ICML 2020) replication
  under symmetric label noise on CIFAR-10 *(rewrite in progress)*
- `TRACK2_SSM.md` — an S4-style SSM as a token mixer on CIFAR-10 patch
  sequences *(rewrite in progress)*

Each rewritten report will carry a **Corrections & Provenance** section that
owns every fix made during the audit, with one-line explanations and links to
the evidence.
