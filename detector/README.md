# Detector

> [中文说明](README_zh.md)

The detector localizes a critical error in a failed trajectory through an OpenAI-compatible LLM endpoint.

## Pipeline

```text
data/unified/<dataset>/*.json
  → Stage A: multi-granularity compression
  → Stage B: evidence-grounded error triggers
  → Stage C1: object-anchored instance clustering
  → Stage C2: resolution and terminal-impact classification
  → Stage C3: candidate-guided causal attribution
  → outputs/<dataset>_final
```

Entry points:

- `stage_a_diagnosis.py`: creates `*_stage_a.json`.
- `stage_b_per_step.py`: creates `*_stage_b.json` with verbatim conflict evidence.
- `stage_c_phase1_cluster.py`: groups repeated triggers into error instances.
- `stage_c_phase2_state.py`: classifies repair status, state, terminal connection, and chain membership.
- `stage_c_phase3_assemble.py`: selects the critical step and writes `*_final.json`.
- `score_steps.py`: evaluates predictions against `metadata.annotation`.

## Core contract

All step indices are zero-based positions in the unified `messages` array. A trigger must quote both the wrong commitment and its violated reference. Instances are grouped by the concrete violated object, not merely by taxonomy label. Phase C retains terminal-relevant instances and attributes the final failure among those candidates.

The five agent error modules are `plan`, `reason`, `act`, `obs`, and `verify`; definitions live in `utils/error_definitions.py`.

## Run

From the repository root:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export DETECTOR_MODEL="your-model-name"

DATASETS="alfworld" bash run_pipeline.sh
```

The runner is resumable and writes to `outputs` by default:

```text
outputs/<dataset>_stage_a/
outputs/<dataset>_stage_b/
outputs/<dataset>_phase1/
outputs/<dataset>_phase2/
outputs/<dataset>_final/
outputs/score_<dataset>.json
```

Override locations with `UNIFIED_ROOT` and `OUTPUT_ROOT`; tune requests with `FILE_CONCURRENCY`, `LLM_CONCURRENCY`, `MAX_TOKENS`, and `TEMPERATURE`.

## Evaluate

```bash
python detector/score_steps.py \
  --unified-dir data/unified/alfworld \
  --pred-dir outputs/alfworld_final \
  --out outputs/score_alfworld.json

python detector/score_steps_breakdown.py \
  --unified-dir data/unified/alfworld \
  --pred-dir outputs/alfworld_final \
  --stage-b-dir outputs/alfworld_stage_b \
  --phase1-dir outputs/alfworld_phase1 \
  --phase2-dir outputs/alfworld_phase2 \
  --out outputs/breakdown_alfworld.json
```

The scorer reports exact, loose-1, and loose-2 step accuracy plus taxonomy metrics where labels are available. Successful trajectories and system failures are excluded.

## Optional Section 6 feedback

`applications/generate_feedback.py` implements the paper's Section 6 feedback application on top of completed Stage C outputs. It is not a core detector stage and is not called by `run_pipeline.sh`.

```bash
python applications/generate_feedback.py \
  --final_dir outputs/alfworld_final \
  --trajectory_dir data/unified/alfworld \
  --stage_a_dir outputs/alfworld_stage_a \
  --output_dir outputs/alfworld_report \
  --base_url "$OPENAI_BASE_URL" \
  --model "$DETECTOR_MODEL" \
  --api_key "$OPENAI_API_KEY" \
  --resume
```

Reports are stored as `outputs/<dataset>_report/<task_id>_report.json`. The final report's `fix_suggestion.hint_sentence` field contains actionable guidance and the path matches the viewer's convention.

## Development notes

- Run CLIs from the repository root so detector utility imports resolve consistently.
- Preserve `messages[i].step == i` and never target a `user` message as a critical step.
- Keep Stage B evidence verbatim and keep the taxonomy stable; label changes invalidate existing annotations.
- Keep compression state local to each trajectory when changing concurrency-sensitive code.
- New datasets require only unified-schema files; see [`../data_processing/README.md`](../data_processing/README.md).
