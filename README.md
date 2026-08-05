# TRAJDEBUG

> [中文说明](README_zh.md)

**TRAJDEBUG: Tracing Error Lifecycle to Identify Critical Failures in Long-Horizon Agent Trajectories** is an evidence-grounded framework for locating the earliest decisive error responsible for a failed LLM-agent trajectory. This repository provides the detector, unified data adapters, evaluation tools, and a local viewer.

## Method

TRAJDEBUG first builds multi-granularity trajectory views, then performs three
auditable stages:

1. **Error trigger detection:** identifies wrong commitments and requires verbatim evidence for both the commitment and violated reference.
2. **Error state classification:** clusters triggers by violated object and classifies resolution and terminal impact.
3. **Causal attribution:** selects the failure-responsible origin from terminal-relevant candidates.

![Overview of the TRAJDEBUG pipeline](assets/%20pipeline.png)

See [`detector/README.md`](detector/README.md) for implementation details and [`13209_TRAJDEBUG_Tracing_Error_.pdf`](13209_TRAJDEBUG_Tracing_Error_.pdf) for the paper.

## Results

![Critical error detection results](assets/results.png)

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Include the local viewer:
pip install -e ".[viewer]"
```

Copy `.env.example` to `.env`, fill in your endpoint settings, and export them in your shell. The pipeline uses only OpenAI-compatible chat-completions APIs.

## Data

Detector inputs are one JSON file per trajectory under `data/unified/<dataset>/`. The included evaluation sets are:

| Dataset key | Trajectories |
|---|---:|
| `alfworld` | 100 |
| `gaia` | 50 |
| `webshop` | 50 |
| `whoandwhen` | 58 |
| `whoandwhen_algorithm` | 126 |
| `tau2bench` | 400 |
| `swebenchpro` | 86 |

TRAJERRBENCH comprises the 400 τ²-Bench and 86 SWE-Bench Pro failed trajectories. Build registered datasets with:

```bash
python -m data_processing.build_unified_dataset --all
```

Schema and custom-adapter instructions are in [`data_processing/README.md`](data_processing/README.md). Data licensing and attribution notes are in [`data/README.md`](data/README.md).

## OpenAI-compatible API

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export DETECTOR_MODEL="your-model-name"
```

`DETECTOR_API_KEY` and `DETECTOR_BASE_URL` may be used instead of the corresponding `OPENAI_*` variables.

## Self-hosting with SGLang

Install SGLang separately, then launch one OpenAI-compatible server:

```bash
MODEL_PATH=/models/your-model \
SERVED_MODEL_NAME=your-model \
TP_SIZE=8 \
bash deploy_qwen_router.sh
```

The default endpoint is `http://127.0.0.1:30000/v1`.

Run the detector against it with:

```bash
DETECTOR_BASE_URL=http://127.0.0.1:30000/v1 \
DETECTOR_API_KEY=EMPTY \
DETECTOR_MODEL=your-model \
bash run_pipeline.sh
```

## Run

```bash
DETECTOR_MODEL=your-model bash run_pipeline.sh

# Select datasets or tune concurrency:
DATASETS="alfworld gaia webshop" \
FILE_CONCURRENCY=2 \
LLM_CONCURRENCY=8 \
DETECTOR_MODEL=your-model \
bash run_pipeline.sh
```

Inputs default to `data/unified`; outputs default to `outputs`. Each run produces `<dataset>_stage_a`, `<dataset>_stage_b`, `<dataset>_phase1`, `<dataset>_phase2`, `<dataset>_final`, and `score_<dataset>.json`. See [`outputs/README.md`](outputs/README.md) for the directory convention, recommended reading order, and ten sampled examples per paper dataset.

## Evaluation

`run_pipeline.sh` runs exact-step evaluation automatically. To score existing predictions:

```bash
python detector/score_steps.py \
  --unified-dir data/unified/alfworld \
  --pred-dir outputs/alfworld_final \
  --out outputs/score_alfworld.json
```

## Feedback generation

The identified critical error step can be converted into actionable feedback to improve agent self-repair and failure-memory transfer (see Section 6 of the paper):

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

The generated feedback is available in `outputs/<dataset>_report/<task_id>_report.json` under `fix_suggestion.hint_sentence`.

## Viewer

```bash
python -m viewer.server --dataset alfworld --output-dir outputs
```

Open <http://localhost:8000>. See [`viewer/README.md`](viewer/README.md).

## Citation

The paper is currently anonymous. Use [`CITATION.cff`](CITATION.cff), and update its authors and public repository URL after deanonymization.

## License

Code is released under the [MIT License](LICENSE). Dataset components remain subject to their original licenses and terms.