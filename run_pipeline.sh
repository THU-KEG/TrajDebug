#!/usr/bin/env bash
# Run the TRAJDEBUG paper pipeline with any OpenAI-compatible endpoint.
#
# Required:
#   DETECTOR_MODEL=<model name>
# Optional:
#   OPENAI_API_KEY / DETECTOR_API_KEY
#   OPENAI_BASE_URL / DETECTOR_BASE_URL (default: https://api.openai.com/v1)
#   DATASETS="alfworld gaia webshop ..."
#   UNIFIED_ROOT=data/unified OUTPUT_ROOT=outputs

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_URL="${DETECTOR_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
API_KEY="${DETECTOR_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
MODEL="${DETECTOR_MODEL:-}"
UNIFIED_ROOT="${UNIFIED_ROOT:-data/unified}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
CACHE_ROOT="${CACHE_ROOT:-.cache/trajdebug}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
FILE_CONCURRENCY="${FILE_CONCURRENCY:-4}"
LLM_CONCURRENCY="${LLM_CONCURRENCY:-8}"
TOP_K="${TOP_K:-5}"
read -r -a DATASET_LIST <<< "${DATASETS:-alfworld gaia webshop whoandwhen whoandwhen_algorithm tau2bench swebenchpro}"

if [[ -z "$MODEL" ]]; then
  echo "DETECTOR_MODEL is required." >&2
  echo "Example: DETECTOR_MODEL=gpt-4.1-mini OPENAI_API_KEY=... bash run_pipeline.sh" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT"

echo "TRAJDEBUG pipeline"
echo "  endpoint: $BASE_URL"
echo "  model:    $MODEL"
echo "  datasets: ${DATASET_LIST[*]}"
echo "  outputs:  $OUTPUT_ROOT"

for dataset in "${DATASET_LIST[@]}"; do
  input_dir="$UNIFIED_ROOT/$dataset"
  if [[ ! -d "$input_dir" ]]; then
    echo "[$dataset] skipped: missing $input_dir" >&2
    continue
  fi

  stage_a="$OUTPUT_ROOT/${dataset}_stage_a"
  stage_b="$OUTPUT_ROOT/${dataset}_stage_b"
  phase1="$OUTPUT_ROOT/${dataset}_phase1"
  phase2="$OUTPUT_ROOT/${dataset}_phase2"
  final="$OUTPUT_ROOT/${dataset}_final"

  echo "[$dataset] Stage A: multi-granularity compression"
  "$PYTHON_BIN" detector/stage_a_diagnosis.py \
    --input "$input_dir" --output_dir "$stage_a" \
    --base_url "$BASE_URL" --model "$MODEL" --api_key "$API_KEY" \
    --cache "$CACHE_ROOT/${dataset}_stage_a.pkl" \
    --temperature "$TEMPERATURE" --max_tokens "$MAX_TOKENS" \
    --concurrency "$FILE_CONCURRENCY" --llm_concurrency "$LLM_CONCURRENCY" --resume

  echo "[$dataset] Stage 1: error trigger detection"
  "$PYTHON_BIN" detector/stage_b_per_step.py \
    --stage_a_dir "$stage_a" --trajectory_dir "$input_dir" --output_dir "$stage_b" \
    --base_url "$BASE_URL" --model "$MODEL" --api_key "$API_KEY" \
    --cache "$CACHE_ROOT/${dataset}_stage_b.pkl" \
    --temperature "$TEMPERATURE" --max_tokens "$MAX_TOKENS" \
    --concurrency "$FILE_CONCURRENCY" --llm_concurrency "$LLM_CONCURRENCY" --resume

  echo "[$dataset] Stage 2a: error instance clustering"
  "$PYTHON_BIN" detector/stage_c_phase1_cluster.py \
    --stage_b_dir "$stage_b" --output_dir "$phase1" \
    --base_url "$BASE_URL" --model "$MODEL" --api_key "$API_KEY" \
    --cache "$CACHE_ROOT/${dataset}_phase1.pkl" \
    --temperature "$TEMPERATURE" --max_tokens "$MAX_TOKENS" \
    --concurrency "$FILE_CONCURRENCY" --resume

  echo "[$dataset] Stage 2b: error state classification"
  "$PYTHON_BIN" detector/stage_c_phase2_state.py \
    --phase1_dir "$phase1" --trajectory_dir "$input_dir" --output_dir "$phase2" \
    --base_url "$BASE_URL" --model "$MODEL" --api_key "$API_KEY" \
    --cache "$CACHE_ROOT/${dataset}_phase2.pkl" \
    --temperature "$TEMPERATURE" --max_tokens "$MAX_TOKENS" \
    --concurrency "$FILE_CONCURRENCY" --llm_concurrency "$LLM_CONCURRENCY" --resume

  echo "[$dataset] Stage 3: candidate-guided causal attribution"
  "$PYTHON_BIN" detector/stage_c_phase3_assemble.py \
    --phase2_dir "$phase2" --trajectory_dir "$input_dir" --output_dir "$final" \
    --base_url "$BASE_URL" --model "$MODEL" --api_key "$API_KEY" \
    --cache "$CACHE_ROOT/${dataset}_phase3.pkl" \
    --max_tokens "$MAX_TOKENS" --top_k "$TOP_K" \
    --concurrency "$FILE_CONCURRENCY" --resume

  echo "[$dataset] Evaluation"
  "$PYTHON_BIN" detector/score_steps.py \
    --unified-dir "$input_dir" --pred-dir "$final" \
    --out "$OUTPUT_ROOT/score_${dataset}.json"
done

echo "Done. Results are under $OUTPUT_ROOT/."
