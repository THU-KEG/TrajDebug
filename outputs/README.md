# Output examples

This directory contains ten sampled TRAJDEBUG examples for each of the seven
paper datasets. Both correct and incorrect predictions are included. The
selection is reproducible and guarantees at least two exact critical-step
matches per dataset. The artifacts can be opened directly with the
[local viewer](../viewer/README.md).

## Directory convention

Outputs are organized by dataset and pipeline stage:

```text
outputs/
├── <dataset>_stage_a/   # Multi-granularity trajectory compression
├── <dataset>_stage_b/   # Per-step error triggers and evidence
├── <dataset>_phase1/    # Error-instance clustering
├── <dataset>_phase2/    # Error-state classification
├── <dataset>_final/     # Critical-step prediction and causal attribution
└── <dataset>_report/    # Feedback generated from the diagnosis
```

Files sharing the same task ID belong to one trajectory. For a quick review,
start with `<dataset>_final/<task_id>_final.json`. The `_stage_a`, `_stage_b`,
`_phase1`, and `_phase2` files expose the intermediate reasoning artifacts;
`_report` contains the optional generated feedback.

## Sampling

The examples were selected with fixed seed `20260805`. For each dataset, two
exactly correct predictions were sampled first, followed by eight random
predictions from the remaining available outputs. The latter may include
additional correct predictions.

[`examples_manifest.json`](examples_manifest.json) records the complete
selection procedure, task IDs, ground-truth and predicted steps, correctness,
and per-stage file counts. Each dataset has ten final outputs and between two
and seven exact matches. The historical WhoAndWhen run did not retain Stage A
for the selected tasks; all other selected examples include Stage A/B,
Phase 1/2, final, and report artifacts.

These examples are for inspection only and should not be used to reproduce the
full-dataset scores reported in the paper.

To inspect them:

```bash
python -m viewer.server --dataset alfworld --output-dir outputs
```
