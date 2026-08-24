#!/usr/bin/env python3
"""Ablation 1: per-step error detection WITHOUT any taxonomy.

This script is a drop-in replacement for ``detector/stage_b_per_step.py``
in the ablation pipeline. It runs the same per-judgable-step LLM call,
consumes the same Stage A pool / unified trajectory, and writes the same
``<stem>_stage_b.json`` schema (a flat ``step_triggers`` array, plus
``step_compressions``) so the existing Stage C Phase 1 / 2 / 3 chain in
``detector/`` can be reused unchanged.

DIFFERENCE from Stage B
-----------------------
The system prompt drops the entire v5 25-tag taxonomy and CHECKLIST
(A)/(B)/(C)/(D) machinery. The model is only asked: "for the current
step, is there an error? if yes, quote the offending span." It returns
JSON of the form

    {
      "is_error": true | false,
      "wrong_content_quote": "<verbatim substring of CURRENT STEP>",
      "reasoning": "<1-2 sentences>"
    }

To remain plug-compatible with Stage C (which keys off
``taxonomy_tag`` / ``category`` / ``attribution``), each detected
``is_error=true`` step is wrapped into a placeholder trigger with a
fixed tag (``reason.WrongChoice`` / ``cat-2`` / ``agent``). The
ablation evaluation only compares the predicted critical *step* to the
ground-truth ``critical_error_step``, so taxonomy fields are not part
of the metric.

Usage:

    python baselines/ablation_stageb_no_taxonomy.py \
        --stage_a_dir <path>/<ds>_stage_a \
        --trajectory_dir data/unified/<ds> \
        --output_dir output-ablation/<ds>_stage_b \
        --base_url <endpoint> \
        --model <name> \
        --cache .cache/ablation/no_taxonomy_<ds>_stage_b.pkl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the detector package directly; this baseline lives outside it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DETECTOR_DIR = _REPO_ROOT / "detector"
if str(_DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_DIR))

from _stage_common import (  # noqa: E402
    dump_step_compressions,
    extract_last_json_object,
    load_step_compressions_from_payload,
    load_unified_for_stage,
    render_history_for_focus,
    resolve_extra_params,
)
from stage_b_per_step import (  # noqa: E402
    StageBTrigger,
    _build_user_prompt,
)
from utils.llm_compression import (  # noqa: E402
    DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS,
    DEFAULT_STEP_TH1_MAX_CHARS,
    DEFAULT_STEP_TH2_MAX_CHARS,
    DEFAULT_STEP_TH3_MAX_CHARS,
    strip_think_tags,
)
from utils.model import APIModel  # noqa: E402
from utils.trajectory_utils import (  # noqa: E402
    collect_trajectory_sources,
    output_stem_for_source,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ablation_stageb_no_taxonomy")


# =====================================================================
# Ablation system prompt — NO taxonomy at all
# =====================================================================
#
# Compared to detector/stage_b_per_step.STAGE_B_SYSTEM_PROMPT this
# prompt drops:
#   * the cat-1 / cat-2 / cat-3 / env category framework;
#   * the 25-tag closed vocabulary;
#   * CHECKLIST (A)/(B)/(C)/(D) and HARD RULES (C1–C5);
#   * the agent-vs-env / upstream back-reference scoping;
#   * confidence levels.
#
# What stays: TASK / HISTORY / CURRENT STEP context (rendered by the
# shared ``_build_user_prompt``) and a request for one verbatim quote
# as evidence so the answer is auditable.
ABLATION_SYSTEM_PROMPT = """You are an error annotator for one step of an LLM-agent trajectory. You are shown the TASK, HISTORY (earlier steps), and the CURRENT STEP. Decide whether the CURRENT STEP contains an error.

You may use any reasoning you find helpful. Use your own judgement to decide what counts as an error: there is no fixed checklist of error types and no taxonomy.

Output ONE JSON object and nothing else:

{
  "is_error": true | false,
  "wrong_content_quote": "<verbatim substring of CURRENT STEP that is wrong; empty string if is_error=false>",
  "reasoning": "<1-2 sentences explaining why this step is wrong, or empty if is_error=false>"
}

If the CURRENT STEP has no error, set "is_error": false and leave the other fields empty.
"""


# =====================================================================
# Placeholder trigger fields — kept fixed so Stage C Phase 1/2/3 can
# still consume the output without exposing the model to taxonomy.
# Step accuracy (the only metric for this ablation) does not depend on
# any of these.
# =====================================================================

_PLACEHOLDER_TAG = "reason.WrongChoice"
_PLACEHOLDER_CATEGORY = "cat-2"
_PLACEHOLDER_ATTRIBUTION = "agent"
_PLACEHOLDER_CONFIDENCE = "medium"


def _parse_ablation_response(
    response: str, step_num: int
) -> List[StageBTrigger]:
    """Parse the ablation prompt response into 0 or 1 placeholder triggers.

    Returns an empty list when ``is_error`` is false or the response
    cannot be parsed (treated as no error).
    """
    text = strip_think_tags(response or "")
    if text.strip().startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        )

    data = extract_last_json_object(text, must_have_key="is_error")
    if data is None:
        data = extract_last_json_object(text)
    if not isinstance(data, dict):
        return []

    raw = data.get("is_error")
    if isinstance(raw, bool):
        is_error = raw
    elif isinstance(raw, str):
        is_error = raw.strip().lower() in ("true", "yes", "1")
    elif isinstance(raw, (int, float)):
        is_error = bool(raw)
    else:
        is_error = False

    if not is_error:
        return []

    quote = str(data.get("wrong_content_quote", "") or "").strip()
    reasoning = str(data.get("reasoning", "") or "").strip()

    return [
        StageBTrigger(
            step=int(step_num),
            category=_PLACEHOLDER_CATEGORY,
            taxonomy_tag=_PLACEHOLDER_TAG,
            attribution=_PLACEHOLDER_ATTRIBUTION,
            wrong_content_quote=quote,
            reference_quote="",
            confidence=_PLACEHOLDER_CONFIDENCE,
            confidence_reasoning=reasoning,
        )
    ]


# =====================================================================
# Per-step driver (mirrors detector.StageBDetector but with the ablated
# prompt + parser).
# =====================================================================


class AblationStageBDetector:
    """Per-step error detector — taxonomy-free ablation."""

    def __init__(self, api_config: Dict[str, Any]):
        self.config = api_config
        cache_url = api_config.get("cache_url")
        if not cache_url:
            raise ValueError("Missing cache_url in api_config. Please provide --cache.")
        self.model = APIModel(
            cache_url,
            api_config["base_url"],
            api_config["model"],
            api_config.get("api_key", "EMPTY"),
            extra_params=resolve_extra_params(api_config.get("model_profile")),
        )
        self.history_overall_cap_chars = int(
            api_config.get("history_overall_cap_chars", DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS)
        )
        self.judgement_force_json = bool(api_config.get("judgement_force_json", True))
        self.semaphore = asyncio.Semaphore(
            max(1, int(api_config.get("llm_concurrency", 10)))
        )
        self.fallback_chars = {
            "th1": int(api_config.get("step_th1_max_chars", DEFAULT_STEP_TH1_MAX_CHARS)),
            "th2": int(api_config.get("step_th2_max_chars", DEFAULT_STEP_TH2_MAX_CHARS)),
            "th3": int(api_config.get("step_th3_max_chars", DEFAULT_STEP_TH3_MAX_CHARS)),
        }

    def close(self) -> None:
        self.model.close()

    async def _call_llm(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {"type": "json_object"} if self.judgement_force_json else None
        max_tokens = int(self.config.get("max_tokens", 8192))
        temperature = float(self.config.get("temperature", 0.0))
        async with self.semaphore:
            result = await asyncio.to_thread(
                self.model.generate_chat,
                messages,
                max_tokens,
                temperature,
                response_format,
            )
        return result or ""

    async def analyze_step(
        self,
        step_data: Dict[str, Any],
        flat_steps: List[Dict[str, Any]],
        step_pool: Dict[int, Dict[str, str]],
        task_description: str,
        agent_framework_description: str,
    ) -> List[StageBTrigger]:
        step_num = int(step_data["step"])

        pool_entry = step_pool.get(step_num) or {}
        current_content = (
            pool_entry.get("th1")
            or pool_entry.get("th2")
            or pool_entry.get("th3")
            or str(step_data.get("content", "") or "")
        )

        warn_state: List[bool] = [False]
        rendered_history = render_history_for_focus(
            flat_steps=flat_steps,
            focus_step=step_num,
            step_pool=step_pool,
            warn_state=warn_state,
            fallback_chars=self.fallback_chars,
            th1_max_distance=2,
            th2_max_distance=5,
            include_focus=False,
            history_only_before=True,
        )

        user_prompt = _build_user_prompt(
            agent_framework_description=agent_framework_description,
            task_message=task_description,
            rendered_history=rendered_history,
            current_step=step_num,
            current_step_content=current_content,
        )

        response = await self._call_llm(ABLATION_SYSTEM_PROMPT, user_prompt)
        return _parse_ablation_response(response, step_num)

    async def analyze_trajectory(
        self,
        stage_a_payload: Dict[str, Any],
        trajectory_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_description = trajectory_data["task_description"]
        flat_steps = trajectory_data["flat_steps"]
        agent_framework_description = trajectory_data["agent_framework_description"]
        step_pool = load_step_compressions_from_payload(stage_a_payload)

        tasks = []
        for step in flat_steps:
            if not step.get("judgable", True):
                continue
            tasks.append(self.analyze_step(
                step_data=step,
                flat_steps=flat_steps,
                step_pool=step_pool,
                task_description=task_description,
                agent_framework_description=agent_framework_description,
            ))

        all_trigger_lists: List[List[StageBTrigger]] = await asyncio.gather(*tasks)
        all_triggers: List[Dict[str, Any]] = []
        for trig_list in all_trigger_lists:
            for t in trig_list:
                all_triggers.append(t.to_dict())

        return {
            "task_id": trajectory_data["task_id"],
            "task_description": task_description,
            "task_success": trajectory_data["task_success"],
            "task_outcome": trajectory_data["task_outcome"],
            "reward": trajectory_data["reward"],
            "environment": trajectory_data["environment"],
            "total_steps": trajectory_data["total_steps"],
            "trajectory_source": trajectory_data["trajectory_source"],
            "agent_framework_description": agent_framework_description,
            "step_triggers": all_triggers,
            "step_compressions": dump_step_compressions(step_pool),
            "trajectory_file_path": trajectory_data.get("file_path") or trajectory_data.get("source_file") or "",
            "metadata": {
                "dataset": trajectory_data.get("metadata", {}).get("dataset"),
                "annotation": trajectory_data.get("metadata", {}).get("annotation"),
                "extra": trajectory_data.get("metadata", {}).get("extra"),
            },
            "ablation": "no_taxonomy",
        }

    async def process_file(
        self,
        stage_a_file: str,
        trajectory_file: str,
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            with open(stage_a_file, "r", encoding="utf-8") as fh:
                stage_a_payload = json.load(fh)
        except Exception as exc:
            logger.error("Failed to load Stage A file %s: %s", stage_a_file, exc)
            return None

        try:
            trajectory_data = load_unified_for_stage(trajectory_file)
        except Exception as exc:
            logger.error("Failed to load trajectory %s: %s", trajectory_file, exc)
            return None

        if trajectory_data["task_success"]:
            payload: Dict[str, Any] = {
                "task_id": trajectory_data["task_id"],
                "task_description": trajectory_data["task_description"],
                "task_success": True,
                "task_outcome": "success",
                "reward": trajectory_data["reward"],
                "environment": trajectory_data["environment"],
                "total_steps": trajectory_data["total_steps"],
                "trajectory_source": trajectory_data["trajectory_source"],
                "agent_framework_description": trajectory_data["agent_framework_description"],
                "step_triggers": [],
                "step_compressions": stage_a_payload.get("step_compressions") or {},
                "trajectory_file_path": trajectory_data.get("file_path") or trajectory_data.get("source_file") or "",
                "metadata": {
                    "dataset": trajectory_data.get("metadata", {}).get("dataset"),
                    "annotation": trajectory_data.get("metadata", {}).get("annotation"),
                    "extra": trajectory_data.get("metadata", {}).get("extra"),
                },
                "ablation": "no_taxonomy",
            }
        else:
            payload = await self.analyze_trajectory(stage_a_payload, trajectory_data)

        out_stem = output_stem_for_source(trajectory_file)
        out_fp = Path(output_dir) / f"{out_stem}_stage_b.json"
        with out_fp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        n_triggers = len(payload.get("step_triggers") or [])
        logger.info(
            "Ablation Stage B [%s]: %d error steps across %d steps",
            out_stem, n_triggers, payload["total_steps"],
        )
        return payload


# =====================================================================
# Batch runner
# =====================================================================


def _is_valid_stage_b_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    return isinstance(data.get("step_triggers"), list)


def _ensure_chat_completions_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


def _index_trajectories_by_stem(trajectory_dir: str) -> Dict[str, str]:
    files = collect_trajectory_sources(trajectory_dir)
    return {output_stem_for_source(fp): fp for fp in files}


_STAGE_A_SUFFIX = "_stage_a.json"


def _strip_stage_a_suffix(stem: str) -> str:
    if stem.endswith("_stage_a"):
        return stem[: -len("_stage_a")]
    return stem


def _guess_trajectory_for_stage_a(
    stage_a_file: str,
    trajectory_index: Dict[str, str],
) -> Optional[str]:
    try:
        with open(stage_a_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ts = payload.get("trajectory_source")
        if isinstance(ts, str) and ts.strip():
            return ts
    except Exception:
        pass
    stem = _strip_stage_a_suffix(Path(stage_a_file).stem)
    if stem in trajectory_index:
        return trajectory_index[stem]
    for key, val in trajectory_index.items():
        if key.startswith(stem) or stem.startswith(key):
            return val
    return None


async def run_batch(
    stage_a_inputs: List[str],
    trajectory_dir: Optional[str],
    explicit_trajectory: Optional[str],
    output_dir: str,
    api_config: Dict[str, Any],
    concurrency: int = 4,
    resume: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    api_config = dict(api_config)
    api_config["base_url"] = _ensure_chat_completions_url(api_config["base_url"])

    detector = AblationStageBDetector(api_config)
    trajectory_index = _index_trajectories_by_stem(trajectory_dir) if trajectory_dir else {}

    jobs: List[Tuple[str, str]] = []
    skipped_list: List[Dict[str, Any]] = []
    for sa in stage_a_inputs:
        if explicit_trajectory:
            traj = explicit_trajectory
        else:
            traj = _guess_trajectory_for_stage_a(sa, trajectory_index)
        if not traj:
            skipped_list.append({"stage_a_file": sa, "reason": "No matching trajectory found"})
            continue
        jobs.append((sa, traj))

    file_sem = asyncio.Semaphore(max(1, int(concurrency)))
    total = len(jobs)

    async def _process_one(idx: int, sa: str, traj: str) -> Tuple[str, str, str, Optional[str]]:
        out_fp = Path(output_dir) / f"{output_stem_for_source(traj)}_stage_b.json"
        if overwrite:
            pass
        elif resume and _is_valid_stage_b_output(out_fp):
            return ("skipped", sa, traj, f"valid cached: {out_fp.name}")
        async with file_sem:
            logger.info(
                "Ablation Stage B (%d/%d): stage_a=%s  traj=%s",
                idx, total, sa, traj,
            )
            try:
                r = await detector.process_file(sa, traj, output_dir)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return ("failed", sa, traj, repr(exc))
        if r is None:
            return ("failed", sa, traj, "process_file returned None")
        return ("ok", sa, traj, None)

    ok = failed = 0
    failures: List[Dict[str, Any]] = []
    try:
        async_tasks = [_process_one(i, sa, traj) for i, (sa, traj) in enumerate(jobs, start=1)]
        results = await asyncio.gather(*async_tasks)
        for status, sa, traj, info in results:
            if status == "ok":
                ok += 1
            elif status == "failed":
                failed += 1
                failures.append({"stage_a_file": sa, "trajectory_file": traj, "error": info or ""})
            else:
                skipped_list.append({"stage_a_file": sa, "trajectory_file": traj, "reason": info or ""})
    finally:
        detector.close()

    summary = {
        "output_dir": output_dir,
        "num_stage_a_inputs": len(stage_a_inputs),
        "num_jobs": len(jobs),
        "num_ok": ok,
        "num_failed": failed,
        "num_skipped": len(skipped_list),
        "file_concurrency": int(concurrency),
        "llm_concurrency": int(api_config.get("llm_concurrency", 10)),
        "resume": resume,
        "overwrite": overwrite,
        "skipped": skipped_list[:20],
        "failures": failures[:20],
    }
    logger.info("Ablation Stage B batch done: %s", summary)
    return summary


def _collect_stage_a_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return [str(x) for x in sorted(p.rglob(f"*{_STAGE_A_SUFFIX}"))]
    raise FileNotFoundError(f"Path not found: {path}")


_BOOL_FALSE_STRINGS = {"0", "false", "no", "off", "n", "f"}


def _parse_bool_flag(value: Any) -> bool:
    return str(value).strip().lower() not in _BOOL_FALSE_STRINGS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation 1: per-step error detection without the v5 "
            "25-tag taxonomy. Output schema is identical to "
            "detector/stage_b_per_step.py so Stage C Phase 1/2/3 can "
            "run unchanged on top of it. Evaluation only compares "
            "predicted vs. ground-truth critical step (step accuracy)."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage_a_file", help="Single Stage A result json (*_stage_a.json)")
    group.add_argument("--stage_a_dir", help="Directory containing Stage A result json files")

    parser.add_argument("--trajectory_file", help="Single original unified trajectory file")
    parser.add_argument("--trajectory_dir", help="Directory containing unified trajectory files")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--api_key", default=os.getenv("API_KEY"))
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--cache", required=True)

    parser.add_argument("--history_overall_cap_chars", type=int,
                        default=DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS)
    parser.add_argument("--judgement_force_json", type=_parse_bool_flag, default=True)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--llm_concurrency", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model_profile", default=None)

    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Missing --api_key and env API_KEY is not set.")
    if args.stage_a_file and not args.trajectory_file:
        raise ValueError("--stage_a_file requires --trajectory_file.")
    if args.stage_a_dir and not args.trajectory_dir:
        raise ValueError("--stage_a_dir requires --trajectory_dir.")

    api_config = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "timeout": args.timeout,
        "cache_url": args.cache,
        "max_tokens": args.max_tokens,
        "history_overall_cap_chars": args.history_overall_cap_chars,
        "judgement_force_json": args.judgement_force_json,
        "llm_concurrency": args.llm_concurrency,
        "model_profile": args.model_profile,
    }

    if args.stage_a_file:
        stage_a_inputs = [args.stage_a_file]
    else:
        stage_a_inputs = _collect_stage_a_files(args.stage_a_dir)

    summary = asyncio.run(
        run_batch(
            stage_a_inputs=stage_a_inputs,
            trajectory_dir=args.trajectory_dir,
            explicit_trajectory=args.trajectory_file,
            output_dir=args.output_dir,
            api_config=api_config,
            concurrency=args.concurrency,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
