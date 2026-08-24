#!/usr/bin/env python3
"""Direct-LLM critical-step baseline (no detector pipeline).

This is a minimal comparison baseline for the structured detector
pipeline (`detector/fine_grained_analysis.py` -> `recovery_annotation.py`
-> `critical_error_detection.py`). Given a unified trajectory file (or
a directory of them), it asks the LLM in ONE call:

    "Here is TASK, here is the full trajectory, here is the taxonomy
    of 20 error types. Output the single critical_step, its taxonomy
    label (module.subtype), and a one-sentence rationale."

No Phase 1 / Phase 1.5 / Phase 2 annotations are provided. This
baseline is intentionally simple: one trajectory in, one JSON
prediction out. The output file schema matches what
`detector/score_steps.py` expects (`<stem>_critical_error.json` with a
`critical_error` object containing `critical_step`, `critical_module`,
`error_type`), so both the baseline and the pipeline can be scored
with the same script.

Usage (single file):

    python experiments/direct_critical_step_baseline.py \
        --input data/unified/alfworldsub/3.json \
        --output_dir output/direct_baseline \
        --base_url <endpoint> \
        --model <name>

Usage (batch over a directory):

    python experiments/direct_critical_step_baseline.py \
        --input data/unified/alfworldsub \
        --output_dir output/direct_baseline \
        --base_url <endpoint> \
        --model <name>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Let the baseline import from the detector package without being
# installed. The detector utilities (APIModel, ErrorDefinitionsLoader,
# trajectory flatten, clip_text_middle) are already tuned for this
# project's prompts / caching.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DETECTOR_DIR = _REPO_ROOT / "detector"
if str(_DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_DIR))

from utils.error_definitions import ErrorDefinitionsLoader  # noqa: E402
from utils.llm_compression import clip_text_middle, strip_think_tags  # noqa: E402
from utils.model import APIModel  # noqa: E402
from utils.trajectory_utils import (  # noqa: E402
    collect_trajectory_sources,
    flatten_unified_trajectory,
    load_trajectory_source,
    output_stem_for_source,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("direct_baseline")


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

DEFAULT_PER_STEP_CLIP_CHARS = 800
DEFAULT_MAX_TOKENS = 2048


def _render_taxonomy_block(loader: ErrorDefinitionsLoader) -> str:
    """One-line-per-tag rendering of the 20-tag taxonomy."""
    lines: List[str] = []
    for module_key in loader.get_all_modules():
        errors = loader.get_module_definitions(module_key)
        for subtype, details in errors.items():
            tag = f"{module_key}.{subtype}"
            definition = str(details.get("definition", "")).strip()
            # Keep the per-tag line reasonably short so the full block
            # fits comfortably in the prompt even for smaller models.
            if len(definition) > 320:
                definition = definition[:317] + "..."
            lines.append(f"- {tag}: {definition}")
    return "\n".join(lines)


def _render_trajectory_block(
    flat_steps: List[Dict[str, Any]],
    per_step_chars: int,
) -> str:
    """Render the trajectory as a plain step list for the direct prompt."""
    rendered: List[str] = []
    for item in flat_steps:
        step = item.get("step")
        role = item.get("message_role") or item.get("role") or "assistant"
        content = item.get("content") or ""
        clipped = clip_text_middle(content, per_step_chars)
        rendered.append(f"Step {step} [role: {role}]:\n{clipped}")
    return "\n\n".join(rendered)


def build_direct_prompt(
    task_description: str,
    flat_steps: List[Dict[str, Any]],
    taxonomy_block: str,
    per_step_chars: int,
) -> str:
    """Build the single-call direct baseline prompt."""
    trajectory_block = _render_trajectory_block(flat_steps, per_step_chars)
    last_step = flat_steps[-1].get("step") if flat_steps else 0

    return f"""You are an expert at identifying the SINGLE critical step that caused a failed agent trajectory.

An error is critical if:
- It represents the ROOT CAUSE that made task success impossible
- It caused a cascade of subsequent errors
- The trajectory could have succeeded if THIS specific error had
not occurred
- IMPORTANT: Correcting this specific error would fundamentally
change the trajectory toward success

Decide using only the TASK, the full trajectory, and the
taxonomy below.

=== TASK ===
{task_description}

TRAJECTORY OUTCOME: FAILED (reward=0, reported by the task environment).
LAST STEP INDEX: {last_step}

=== FULL TRAJECTORY (one entry per message) ===
{trajectory_block}

=== TAXONOMY (pick ONE label of the form <module>.<subtype>) ===
{taxonomy_block}

=== WHAT TO OUTPUT ===
Output ONE JSON object with these keys and nothing else:
- "critical_step": <int>  — the 0-indexed step number.
- "critical_module": one of ["plan", "reason", "act", "obs", "verify"].
- "error_type": one of the subtypes for that module (must be present in
  the TAXONOMY block above; e.g. "WrongTool", "GroundingFail",
  "PrematureTermination"). Do not invent new labels.
- "rationale": <string, 1-3 sentences> explaining why this step is
  critical.

=== REQUIRED OUTPUT FORMAT (single JSON object, no prose around it) ===
{{
  "critical_step": <int>,
  "critical_module": "<module>",
  "error_type": "<subtype>",
  "rationale": "<1-3 sentences>"
}}

Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------

def _balanced_regions(text: str) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        regions.append((start, i + 1))
                        i += 1
                        break
                i += 1
            else:
                break
        else:
            i += 1
    return regions


def _try_parse_dict(blob: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r",(\s*[}\]])", r"\1", blob)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        import ast
        py_str = cleaned.replace("true", "True").replace("false", "False").replace("null", "None")
        obj = ast.literal_eval(py_str)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_direct_response(
    response: str,
    loader: ErrorDefinitionsLoader,
) -> Dict[str, Any]:
    """Extract ``critical_step`` / ``critical_module`` / ``error_type`` /
    ``rationale`` from the LLM's direct response.

    On parse failure, returns a degenerate record with placeholders so
    the downstream scorer counts the trajectory as a wrong prediction
    rather than a missing prediction.
    """
    text = strip_think_tags(response or "")

    error_data: Optional[Dict[str, Any]] = None
    for start, end in reversed(_balanced_regions(text)):
        candidate = _try_parse_dict(text[start:end])
        if candidate is not None and "critical_step" in candidate:
            error_data = candidate
            break
    if error_data is None:
        for start, end in reversed(_balanced_regions(text)):
            candidate = _try_parse_dict(text[start:end])
            if candidate is not None:
                error_data = candidate
                break

    if not error_data:
        return {
            "critical_step": 1,
            "critical_module": "unknown",
            "error_type": "parse_error",
            "rationale": "Failed to parse direct baseline response.",
            "raw_response": text.strip()[:4000],
        }

    step_raw = error_data.get("critical_step", 1)
    try:
        step_val = int(step_raw)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+", str(step_raw))
        step_val = int(m.group(0)) if m else 1

    module_val = str(error_data.get("critical_module", "") or "").strip().lower()
    subtype_val = str(error_data.get("error_type", "") or "").strip()

    valid_modules = set(loader.get_all_modules())
    if module_val not in valid_modules:
        module_val = "unknown"

    if module_val in valid_modules:
        valid_subtypes = set(loader.get_valid_error_types(module_val))
        if subtype_val not in valid_subtypes or subtype_val == "no_error":
            subtype_val = ""

    rationale = str(error_data.get("rationale", "") or "").strip()

    return {
        "critical_step": step_val,
        "critical_module": module_val or "unknown",
        "error_type": subtype_val or "unknown",
        "rationale": rationale or "No rationale provided.",
        "raw_response": text.strip()[:4000],
    }


# ---------------------------------------------------------------------
# Per-trajectory processing
# ---------------------------------------------------------------------

def _extract_task_description(data: Dict[str, Any], flat_steps: List[Dict[str, Any]]) -> str:
    """Best-effort task description recovery: metadata first, then first
    user message.
    """
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if isinstance(metadata, dict):
        for key in ("task_description", "task", "prompt", "instruction"):
            val = metadata.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    for step in flat_steps:
        if str(step.get("role", "")).lower() == "user":
            content = step.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _extract_reward(data: Dict[str, Any]) -> Optional[int]:
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("reward")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _write_success_output(stem: str, output_dir: Path, source: str) -> Path:
    out_path = output_dir / f"{stem}_critical_error.json"
    payload = {
        "task_id": stem,
        "trajectory_source": source,
        "critical_error": None,
        "error_summary": {"message": "Task succeeded - direct baseline skipped"},
        "baseline": "direct_llm",
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def process_one(
    source: str,
    output_dir: Path,
    model: APIModel,
    loader: ErrorDefinitionsLoader,
    taxonomy_block: str,
    per_step_chars: int,
    max_tokens: int,
    temperature: float,
    force_json: bool,
    overwrite: bool,
) -> Tuple[str, Optional[Path], Optional[str]]:
    """Run the direct baseline on one unified trajectory file.

    Returns ``(status, out_path_or_none, reason_or_none)``:
      - status ∈ {"ok", "skipped", "failed", "success_task"}.
    """
    stem = output_stem_for_source(source)
    out_path = output_dir / f"{stem}_critical_error.json"
    if out_path.exists() and not overwrite:
        return ("skipped", out_path, "output file already exists")

    try:
        data = load_trajectory_source(source)
    except Exception as exc:  # noqa: BLE001
        return ("failed", None, f"load_trajectory_source failed: {exc}")

    flat_steps = flatten_unified_trajectory(data)
    if not flat_steps:
        return ("failed", None, "empty trajectory after flatten")

    reward = _extract_reward(data)
    if reward == 1:
        written = _write_success_output(stem, output_dir, source)
        return ("success_task", written, "reward=1")

    task_description = _extract_task_description(data, flat_steps)
    if not task_description:
        task_description = "(task description unavailable)"

    prompt = build_direct_prompt(
        task_description=task_description,
        flat_steps=flat_steps,
        taxonomy_block=taxonomy_block,
        per_step_chars=per_step_chars,
    )

    system_content = (
        "You are an expert at identifying critical failure points in agent trajectories. "
        "Respond with ONLY one valid JSON object that matches the requested schema. "
        "Do not invent taxonomy labels. Do not output placeholder strings like 'unknown' or 'parse_error'."
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    response_format = {"type": "json_object"} if force_json else None

    try:
        response = model.generate_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
    except Exception as exc:  # noqa: BLE001
        return ("failed", None, f"generate_chat failed: {exc}")

    parsed = parse_direct_response(response or "", loader)

    payload = {
        "task_id": (data.get("metadata") or {}).get("task_id", stem) if isinstance(data.get("metadata"), dict) else stem,
        "trajectory_source": source,
        "critical_error": {
            "critical_step": parsed["critical_step"],
            "critical_module": parsed["critical_module"],
            "error_type": parsed["error_type"],
            "root_cause": parsed["rationale"],
            "evidence": "(not provided by direct baseline)",
            "correction_guidance": "(not provided by direct baseline)",
            "instruction_guidance": "(not provided by direct baseline)",
            "cascading_effects": [],
            "confidence": 0.5,
            "selected_trigger": {},
            "override_phase1": False,
            "override_phase15": False,
            "overridden_steps": [],
            "selection_reasoning": parsed["rationale"],
            "persistence_evidence": "",
            "mechanism": "",
        },
        "error_summary": {
            "total_steps": len(flat_steps),
            "critical_at": (
                f"Step {parsed['critical_step']} - "
                f"{parsed['critical_module']}:{parsed['error_type']}"
            ),
            "confidence": 0.5,
        },
        "baseline": "direct_llm",
        "baseline_raw_response": parsed["raw_response"],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return ("ok", out_path, None)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Direct-LLM critical-step baseline (no Phase 1 / 1.5 / 2 "
            "annotations) — for comparison against the structured "
            "detector pipeline."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a unified trajectory JSON file or a directory of them.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where <stem>_critical_error.json files are written.",
    )
    parser.add_argument(
        "--base_url",
        required=True,
        help="OpenAI-compatible chat-completions endpoint (or /v1 base).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name / deployment identifier.",
    )
    parser.add_argument(
        "--api_key",
        default=os.getenv("API_KEY", "EMPTY"),
        help="API key (falls back to env API_KEY, then EMPTY).",
    )
    parser.add_argument(
        "--cache",
        default=".cache/direct_baseline.pkl",
        help="Prompt cache file path (reuses APIModel's cache).",
    )
    parser.add_argument(
        "--per_step_chars",
        type=int,
        default=DEFAULT_PER_STEP_CLIP_CHARS,
        help="Per-step middle-clip length when rendering the trajectory.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="max_tokens for the single LLM call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--extra_params",
        default=os.getenv("DIRECT_EXTRA_PARAMS", "{}"),
        help="JSON string of extra parameters for the model (from model profile).",
    )
    parser.add_argument(
        "--force_json",
        action="store_true",
        help="Pass response_format={'type': 'json_object'} to the backend.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even when the output file already exists.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of parallel worker threads for LLM calls (default: 64).",
    )
    args = parser.parse_args()

    sources = collect_trajectory_sources(args.input)
    if not sources:
        logger.warning("No trajectory files found at %s", args.input)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ErrorDefinitionsLoader()
    taxonomy_block = _render_taxonomy_block(loader)

    # Parse extra_params from environment variable if available
    extra_params = {}
    if hasattr(args, 'extra_params') and args.extra_params:
        import json
        try:
            extra_params = json.loads(args.extra_params)
        except (json.JSONDecodeError, TypeError):
            extra_params = {}

    model = APIModel(
        cache_url=args.cache,
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.api_key,
        extra_params=extra_params,
    )

    ok = skipped = failed = success_task = 0
    total = len(sources)
    num_workers = max(1, int(args.num_workers))
    logger.info("Running direct baseline on %d trajectories with num_workers=%d", total, num_workers)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Preserve submission order: submit in order, then iterate futures
        # in that same order so [idx/total] matches the input ordering.
        futures = [
            executor.submit(
                process_one,
                source,
                output_dir,
                model,
                loader,
                taxonomy_block,
                args.per_step_chars,
                args.max_tokens,
                args.temperature,
                args.force_json,
                args.overwrite,
            )
            for source in sources
        ]
        for idx, (source, future) in enumerate(zip(sources, futures), start=1):
            try:
                status, out_path, reason = future.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning("[%d/%d] fail %s (worker exception: %s)", idx, total, source, exc)
                continue
            if status == "ok":
                ok += 1
                logger.info("[%d/%d] ok   %s -> %s", idx, total, source, out_path)
            elif status == "skipped":
                skipped += 1
                logger.info("[%d/%d] skip %s (%s)", idx, total, source, reason)
            elif status == "success_task":
                success_task += 1
                logger.info("[%d/%d] pass %s (%s)", idx, total, source, reason)
            else:
                failed += 1
                logger.warning("[%d/%d] fail %s (%s)", idx, total, source, reason)

    try:
        model.save_cache()
    except Exception:  # noqa: BLE001
        pass

    logger.info(
        "direct baseline done: ok=%d skipped=%d success_task=%d failed=%d total=%d",
        ok, skipped, success_task, failed, len(sources),
    )


if __name__ == "__main__":
    main()
