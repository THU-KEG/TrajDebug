#!/usr/bin/env python3
"""Ablation 2: directly pick the critical step in ONE LLM call given
the Stage B per-step trigger output — WITHOUT any taxonomy.

This ablation reads the FULL pipeline's Stage B output (the v5
25-tag detector) and asks the LLM, in a single call, to pick the
single critical step. The prompt:

    * does NOT show the 25-tag taxonomy block,
    * does NOT ask the model to name a module / error_type,
    * does NOT show the Stage B taxonomy_tag / category /
      attribution / confidence fields,
    * ONLY shows:
        - the TASK,
        - the FULL trajectory,
        - a flat list of (step, wrong_content_quote) pairs from
          Stage B as candidate "this step is suspicious" hints.

The LLM returns one JSON object:

    {
      "critical_step": <int>,
      "rationale": "<1-3 sentences>"
    }

Output ``<output_dir>/<stem>_final.json`` is structured so that
``detector/score_steps.py`` can read it. Only the predicted
critical *step* matters for the metric; module / error_type are
filled with ``"unknown"`` placeholders since this ablation does
not produce a taxonomy label.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DETECTOR_DIR = _REPO_ROOT / "detector"
if str(_DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_DIR))

from _stage_common import (  # noqa: E402
    extract_last_json_object,
    resolve_extra_params,
)
from utils.llm_compression import (  # noqa: E402
    clip_text_middle,
    strip_think_tags,
)
from utils.model import APIModel  # noqa: E402
from utils.trajectory_utils import (  # noqa: E402
    collect_trajectory_sources,
    flatten_unified_trajectory,
    load_trajectory_source,
    output_stem_for_source,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ablation_direct_pick")


DEFAULT_PER_STEP_CLIP_CHARS = 800
DEFAULT_MAX_TOKENS = 4096


# =====================================================================
# Prompt construction — NO taxonomy at all
# =====================================================================


def _render_trajectory_block(
    flat_steps: List[Dict[str, Any]],
    per_step_chars: int,
) -> str:
    rendered: List[str] = []
    for item in flat_steps:
        step = item.get("step")
        role = item.get("message_role") or item.get("role") or "assistant"
        content = item.get("content") or ""
        clipped = clip_text_middle(content, per_step_chars)
        rendered.append(f"Step {step} [role: {role}]:\n{clipped}")
    return "\n\n".join(rendered)


def _render_stage_b_hints(triggers: List[Dict[str, Any]]) -> str:
    """Render Stage B triggers as taxonomy-free hints.

    Only ``step`` and ``wrong_content_quote`` are shown — taxonomy_tag /
    category / attribution / confidence are intentionally hidden so the
    LLM cannot use them.
    """
    if not triggers:
        return "(no suspicious-step hints available; pick the critical step from the trajectory yourself.)"
    # Aggregate quotes by step so the LLM sees one bullet per
    # suspicious step (instead of one bullet per trigger).
    by_step: Dict[int, List[str]] = {}
    order: List[int] = []
    for t in triggers:
        if not isinstance(t, dict):
            continue
        try:
            step = int(t.get("step"))
        except (TypeError, ValueError):
            continue
        wcq = str(t.get("wrong_content_quote", "") or "").strip()
        if step not in by_step:
            by_step[step] = []
            order.append(step)
        if wcq and wcq not in by_step[step]:
            by_step[step].append(wcq)

    lines: List[str] = []
    for step in sorted(order):
        quotes = by_step[step]
        if not quotes:
            lines.append(f"- step {step}: (flagged, no quote)")
            continue
        for q in quotes:
            qs = q if len(q) <= 240 else q[:237] + "..."
            lines.append(f"- step {step}: {qs}")
    return "\n".join(lines)


def build_direct_pick_prompt(
    task_description: str,
    flat_steps: List[Dict[str, Any]],
    stage_b_triggers: List[Dict[str, Any]],
    per_step_chars: int,
) -> str:
    trajectory_block = _render_trajectory_block(flat_steps, per_step_chars)
    hints_block = _render_stage_b_hints(stage_b_triggers)
    last_step = flat_steps[-1].get("step") if flat_steps else 0

    return f"""You are an expert at identifying the SINGLE critical step that caused a failed agent trajectory.

An error is critical if:
- It represents the ROOT CAUSE that made task success impossible
- It caused a cascade of subsequent errors

You are given:
- the TASK,
- the FULL TRAJECTORY,
- a list of SUSPICIOUS STEPS that an upstream annotator flagged as containing some erroneous content (they are hints; you may pick any other step from the trajectory if you disagree).

There is no fixed catalogue of error types. Use your own judgement.

=== TASK ===
{task_description}

TRAJECTORY OUTCOME: FAILED (reward=0, reported by the task environment).
LAST STEP INDEX: {last_step}

=== FULL TRAJECTORY (one entry per message) ===
{trajectory_block}

=== SUSPICIOUS STEPS (upstream hints; step number + quoted offending content) ===
{hints_block}

=== WHAT TO OUTPUT ===
Output ONE JSON object with these keys and nothing else:
- "critical_step": <int>  — the step number of the critical error.
- "rationale": <string, 1-3 sentences> explaining why this step is critical.

=== REQUIRED OUTPUT FORMAT (single JSON object, no prose around it) ===
{{
  "critical_step": <int>,
  "rationale": "<1-3 sentences>"
}}

Return ONLY the JSON object.
"""


# =====================================================================
# Response parsing — only critical_step is required
# =====================================================================


def parse_direct_pick_response(response: str) -> Dict[str, Any]:
    text = strip_think_tags(response or "")
    if text.strip().startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))

    data = extract_last_json_object(text, must_have_key="critical_step")
    if data is None:
        data = extract_last_json_object(text)
    if not isinstance(data, dict):
        # Fallback: try to extract a bare integer as critical_step.
        m = re.search(r'"?critical_step"?\s*[:=]\s*(-?\d+)', text)
        if m:
            try:
                return {
                    "critical_step": int(m.group(1)),
                    "rationale": "Recovered critical_step from raw text; JSON parse failed.",
                    "raw_response": text.strip()[:4000],
                }
            except ValueError:
                pass
        return {
            "critical_step": None,
            "rationale": "Failed to parse direct-pick response.",
            "raw_response": text.strip()[:4000],
        }

    step_raw = data.get("critical_step")
    step_val: Optional[int]
    try:
        step_val = int(step_raw) if step_raw is not None else None
    except (TypeError, ValueError):
        m = re.search(r"-?\d+", str(step_raw))
        step_val = int(m.group(0)) if m else None

    rationale = str(data.get("rationale", "") or "").strip()

    return {
        "critical_step": step_val,
        "rationale": rationale or "No rationale provided.",
        "raw_response": text.strip()[:4000],
    }


# =====================================================================
# Per-trajectory processing
# =====================================================================


def _extract_task_description(
    data: Dict[str, Any], flat_steps: List[Dict[str, Any]]
) -> str:
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


def _write_success_output(
    stem: str, output_dir: Path, source: str, stage_b_file: Optional[str]
) -> Path:
    out_path = output_dir / f"{stem}_final.json"
    payload = {
        "task_id": stem,
        "trajectory_source": source,
        "stage_b_file": stage_b_file,
        "critical_error": None,
        "error_summary": {"message": "Task succeeded - direct-pick ablation skipped"},
        "ablation": "direct_pick_no_taxonomy",
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def _build_payload(
    stem: str,
    source: str,
    stage_b_file: str,
    parsed: Dict[str, Any],
    n_steps: int,
    n_stage_b_triggers: int,
) -> Dict[str, Any]:
    """Build a v5-compatible final.json payload.

    Only ``critical_step`` is produced by the LLM in this ablation.
    Module / error_type are placeholder ``"unknown"`` strings — they
    are not part of the metric here. ``score_steps.py`` reads
    ``unified_critical_step`` (or ``critical_step`` as a fallback) for
    the step-accuracy comparison.
    """
    step_val = parsed.get("critical_step")
    rationale = parsed.get("rationale", "") or ""

    critical_error: Dict[str, Any] = {
        # v5 fields (preferred by score_steps.py)
        "unified_critical_step": step_val,
        "llm_unified_critical_step": step_val,
        "agent_critical_step": step_val,
        "llm_agent_critical_step": step_val,
        "agent_affected_module": "unknown",
        "agent_category": "unknown",
        "agent_pick_rationale": rationale,
        "agent_pick_source": "ablation_direct_pick_no_taxonomy",
        "unified_source": "agent",
        # Legacy fields (for backwards-compat with score_steps.py & friends)
        "critical_step": step_val,
        "critical_module": "unknown",
        "error_type": "unknown",
        "root_cause": rationale,
        "selection_reasoning": rationale,
        "evidence": "",
        "selected_trigger": {},
        "confidence": 0.5,
    }

    return {
        "task_id": stem,
        "trajectory_source": source,
        "stage_b_file": stage_b_file,
        "total_steps": n_steps,
        "num_stage_b_triggers": n_stage_b_triggers,
        "critical_error": critical_error,
        "error_summary": {
            "total_steps": n_steps,
            "critical_at": f"Step {step_val}" if step_val is not None else "(none)",
            "confidence": 0.5,
        },
        "ablation": "direct_pick_no_taxonomy",
        "ablation_raw_response": parsed.get("raw_response", ""),
    }


def process_one(
    stage_b_file: str,
    trajectory_file: str,
    output_dir: Path,
    model: APIModel,
    per_step_chars: int,
    max_tokens: int,
    temperature: float,
    force_json: bool,
    overwrite: bool,
) -> Tuple[str, Optional[Path], Optional[str]]:
    stem = Path(stage_b_file).stem.replace("_stage_b", "")
    out_path = output_dir / f"{stem}_final.json"
    if out_path.exists() and not overwrite:
        return ("skipped", out_path, "output file already exists")

    try:
        with open(stage_b_file, "r", encoding="utf-8") as fh:
            stage_b_payload = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return ("failed", None, f"load stage_b failed: {exc}")
    if not isinstance(stage_b_payload, dict):
        return ("failed", None, "stage_b payload is not a dict")

    try:
        traj_data = load_trajectory_source(trajectory_file)
    except Exception as exc:  # noqa: BLE001
        return ("failed", None, f"load trajectory failed: {exc}")

    flat_steps = flatten_unified_trajectory(traj_data)
    if not flat_steps:
        return ("failed", None, "empty trajectory after flatten")

    reward = _extract_reward(traj_data)
    if reward == 1 or stage_b_payload.get("task_success"):
        written = _write_success_output(stem, output_dir, trajectory_file, stage_b_file)
        return ("success_task", written, "reward=1")

    task_description = _extract_task_description(traj_data, flat_steps) or "(task description unavailable)"
    stage_b_triggers = stage_b_payload.get("step_triggers") or []
    if not isinstance(stage_b_triggers, list):
        stage_b_triggers = []

    prompt = build_direct_pick_prompt(
        task_description=task_description,
        flat_steps=flat_steps,
        stage_b_triggers=stage_b_triggers,
        per_step_chars=per_step_chars,
    )

    system_content = (
        "You are an expert at identifying critical failure points in agent trajectories. "
        "Respond with ONLY one valid JSON object that matches the requested schema."
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

    parsed = parse_direct_pick_response(response or "")
    payload = _build_payload(
        stem=stem,
        source=trajectory_file,
        stage_b_file=stage_b_file,
        parsed=parsed,
        n_steps=len(flat_steps),
        n_stage_b_triggers=len(stage_b_triggers),
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return ("ok", out_path, None)


# =====================================================================
# Indexing helpers
# =====================================================================

_STAGE_B_SUFFIX = "_stage_b.json"


def _collect_stage_b_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return [str(x) for x in sorted(p.rglob(f"*{_STAGE_B_SUFFIX}"))]
    raise FileNotFoundError(f"Path not found: {path}")


def _index_trajectories_by_stem(trajectory_dir: str) -> Dict[str, str]:
    files = collect_trajectory_sources(trajectory_dir)
    return {output_stem_for_source(fp): fp for fp in files}


def _strip_stage_b_suffix(stem: str) -> str:
    if stem.endswith("_stage_b"):
        return stem[: -len("_stage_b")]
    return stem


def _guess_trajectory_for_stage_b(
    stage_b_file: str,
    trajectory_index: Dict[str, str],
) -> Optional[str]:
    try:
        with open(stage_b_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ts = payload.get("trajectory_source")
        if isinstance(ts, str) and ts.strip() and Path(ts).exists():
            return ts
    except Exception:
        pass
    stem = _strip_stage_b_suffix(Path(stage_b_file).stem)
    if stem in trajectory_index:
        return trajectory_index[stem]
    for key, val in trajectory_index.items():
        if key.startswith(stem) or stem.startswith(key):
            return val
    return None


# =====================================================================
# CLI
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation 2: directly pick the critical step in ONE LLM "
            "call given Stage B per-step triggers, WITHOUT showing "
            "the model any taxonomy. Skips Phase 1/2/3 entirely. "
            "Evaluation only compares predicted vs. ground-truth "
            "critical step (step accuracy)."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage_b_file", help="Single Stage B result json (*_stage_b.json)")
    group.add_argument("--stage_b_dir", help="Directory containing Stage B result json files")

    parser.add_argument("--trajectory_file", help="Single original unified trajectory file")
    parser.add_argument("--trajectory_dir", help="Directory containing unified trajectory files")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_key", default=os.getenv("API_KEY", "EMPTY"))
    parser.add_argument(
        "--cache",
        default=".cache/ablation/direct_pick.pkl",
    )
    parser.add_argument("--per_step_chars", type=int, default=DEFAULT_PER_STEP_CLIP_CHARS)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--extra_params",
        default=os.getenv("DIRECT_EXTRA_PARAMS", "{}"),
        help="JSON string of extra parameters for the model (from model profile).",
    )
    parser.add_argument("--force_json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--model_profile", default=None)

    args = parser.parse_args()

    if args.stage_b_file and not args.trajectory_file:
        raise ValueError("--stage_b_file requires --trajectory_file.")
    if args.stage_b_dir and not args.trajectory_dir:
        raise ValueError("--stage_b_dir requires --trajectory_dir.")

    if args.stage_b_file:
        stage_b_inputs = [args.stage_b_file]
    else:
        stage_b_inputs = _collect_stage_b_files(args.stage_b_dir)
    if not stage_b_inputs:
        logger.warning("No Stage B files found.")
        return

    trajectory_index: Dict[str, str] = {}
    if args.trajectory_dir:
        trajectory_index = _index_trajectories_by_stem(args.trajectory_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_params: Dict[str, Any] = {}
    if args.model_profile:
        try:
            extra_params = resolve_extra_params(args.model_profile)
        except Exception:
            extra_params = {}
    else:
        try:
            extra_params = json.loads(args.extra_params)
            if not isinstance(extra_params, dict):
                extra_params = {}
        except (json.JSONDecodeError, TypeError):
            extra_params = {}

    model = APIModel(
        cache_url=args.cache,
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.api_key,
        extra_params=extra_params,
    )

    # Resolve per-stage-B trajectory matches up front so the worker
    # never has to touch the index.
    jobs: List[Tuple[str, str]] = []
    skipped: List[Dict[str, str]] = []
    for sb in stage_b_inputs:
        if args.trajectory_file:
            traj = args.trajectory_file
        else:
            traj = _guess_trajectory_for_stage_b(sb, trajectory_index)
        if not traj:
            skipped.append({"stage_b_file": sb, "reason": "no matching trajectory"})
            continue
        jobs.append((sb, traj))

    ok = skipped_cnt = failed = success_task = 0
    total = len(jobs)
    num_workers = max(1, int(args.num_workers))
    logger.info(
        "Running direct-pick ablation on %d trajectories with num_workers=%d",
        total, num_workers,
    )

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_one,
                sb,
                traj,
                output_dir,
                model,
                args.per_step_chars,
                args.max_tokens,
                args.temperature,
                args.force_json,
                args.overwrite,
            )
            for (sb, traj) in jobs
        ]
        for idx, ((sb, traj), future) in enumerate(zip(jobs, futures), start=1):
            try:
                status, out_path, reason = future.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning(
                    "[%d/%d] fail %s (worker exception: %s)", idx, total, sb, exc,
                )
                continue
            if status == "ok":
                ok += 1
                logger.info("[%d/%d] ok   %s -> %s", idx, total, sb, out_path)
            elif status == "skipped":
                skipped_cnt += 1
                logger.info("[%d/%d] skip %s (%s)", idx, total, sb, reason)
            elif status == "success_task":
                success_task += 1
                logger.info("[%d/%d] pass %s (%s)", idx, total, sb, reason)
            else:
                failed += 1
                logger.warning("[%d/%d] fail %s (%s)", idx, total, sb, reason)

    try:
        model.save_cache()
    except Exception:  # noqa: BLE001
        pass

    logger.info(
        "direct-pick ablation done: ok=%d skipped=%d success_task=%d failed=%d skipped_match=%d total=%d",
        ok, skipped_cnt, success_task, failed, len(skipped), len(stage_b_inputs),
    )


if __name__ == "__main__":
    main()
