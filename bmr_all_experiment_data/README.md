# BMR All Experiment Data

This directory is the broad experiment-data archive for the BMR/SubMerge work.
It is intentionally larger than `bmr_paper_results/`, which is only a curated
paper-facing best-result summary.

## Included

- `results/`: all lightweight result JSON files, including main BMR runs,
  smoke runs, historical KDR-DTK-BMR / KDR-DTK-CCR / KDR-TDK-BMR outputs,
  PETS E77 reruns, and Cars/PETS lr-aligned reruns.
- `analysis/`: all Git-friendly analysis outputs, including mechanism plots,
  endpoint summaries, geometry summaries, clean trigger dormancy audits,
  patch optimization summaries, CCR diagnostics, and smoke diagnostics.
- `logs/`: run logs for main experiments, ablations, diagnostics, smoke tests,
  and lr-aligned rescue runs.
- `docs/`: local method/result notes used during the project.
- `scripts/`: shell scripts used to run or reproduce the experiments.
- `figures/`: small generated visualization files outside the analysis tree.
- `bmr_paper_results/`: the curated paper-facing summary copied here for
  convenience.
- `MANIFEST.tsv`: file path, byte size, and SHA256 for every archived file.

## Excluded

Large or binary tensor/model artifacts are intentionally excluded:

- model checkpoints: `*.pt`, `*.pth`
- trigger tensors: `*.npy`
- raw tensor dumps and cached margins: `*.npz`
- background-cache tensors under `analysis/**/background_cache/`

Those files are reproducibility artifacts, not Git-friendly experiment tables.
The corresponding configs, summaries, logs, and figures are included.

## Current Paper Summary

Use `bmr_paper_results/README.md` for the current best single-seed table. This
full archive also keeps non-best settings such as old Cars `lr=5e-7`, PETS E5,
smoke tests, and method-discovery ablations.
