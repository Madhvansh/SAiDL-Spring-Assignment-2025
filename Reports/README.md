# Reports, corrections, and provenance

The repository's earlier PDF reports were removed from the current tree after a review found claims and visual material that could not be traced cleanly to committed experiment evidence. The Git history is unchanged; the files remain available at commit `a5c4f17` for audit purposes.

They are replaced by:

- [TRACK1_APL.md](TRACK1_APL.md), covering active-passive losses under symmetric label noise; and
- [TRACK2_SSM.md](TRACK2_SSM.md), covering the custom S4-inspired image classifier.

The replacement reports follow three rules:

1. A number is labeled as a **recorded notebook output**, not as a rerun, unless a fresh machine-readable artifact exists.
2. Known protocol limitations are stated next to the result they affect.
3. Model names describe the committed code. “S4-inspired” does not mean an official S4/S4D reproduction.

The dependency-free checker verifies that the notebook evidence referenced by both reports still exists:

```bash
python scripts/check_repository.py
```

To inspect the withdrawn PDFs without changing history:

```bash
git show a5c4f17:Reports/CoreML.pdf > CoreML.pdf
git show a5c4f17:Reports/COREML_OUTPUT.pdf > COREML_OUTPUT.pdf
git show a5c4f17:Reports/SSM.pdf > SSM.pdf
```

Those historical files should not be treated as the repository's current account of the experiments.
