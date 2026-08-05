# Data processing

> [中文说明](README_zh.md)

This package converts heterogeneous agent traces into the single schema consumed by the detector.

## Unified schema

Each `data/unified/<dataset>/<task_id>.json` contains:

```json
{
  "messages": [
    {"step": 0, "role": "user", "name": "human", "content": "..."}
  ],
  "metadata": {
    "dataset": "example",
    "task_id": "task-1",
    "task_description": "...",
    "reward": 0,
    "annotation": {
      "critical_error_step": 7,
      "critical_error_type": "act.WrongTool"
    },
    "extra": {}
  }
}
```

Required invariants:

- `messages` is non-empty and `messages[i].step == i`.
- `role` is one of `user`, `assistant`, `tool`, or `system`.
- `content` is a string.
- `reward` is `0` or `1`.
- A non-null `critical_error_step` is a valid unified-message index and does not point to a `user` message.
- Successful trajectories have null critical-error annotations.

`schema.py` defines and validates this contract.

## Build included datasets

From the repository root:

```bash
python -m data_processing.build_unified_dataset --all
python -m data_processing.build_unified_dataset --dataset tau2bench
python -m data_processing.build_unified_dataset --dataset swebenchpro
```

Registered keys are `alfworld`, `gaia`, `webshop`, `whoandwhen`, `whoandwhen_algorithm`, `tau2bench`, and `swebenchpro`. Outputs go to `data/unified/<dataset>` by default.

## Add a dataset

1. Add `data_processing/<name>.py` with `convert_directory(src, out[, labels])`.
2. Normalize roles and assign contiguous zero-based `step` values.
3. Construct the required metadata and map labels to `<module>.<subtype>` when available.
4. Validate every output with `validate_unified`.
5. Register the converter in `DATASETS` in `build_unified_dataset.py`.
6. Add dataset-specific judging/compression settings to `dataset_config.json` when needed.

For message-style sources similar to ALFWorld, GAIA, or WebShop, reuse `_standard_messages.convert_standard_record`. For custom structures, construct the schema directly; `whoandwhen.py`, `tau2bench.py`, and `swebenchpro.py` are examples.

Build and run:

```bash
python -m data_processing.build_unified_dataset \
  --dataset my_dataset \
  --src data/MyDataset \
  --out data/unified/my_dataset

DATASETS="my_dataset" DETECTOR_MODEL=your-model bash run_pipeline.sh
```

Detector outputs default to `outputs`; data conversion never writes detector outputs.
