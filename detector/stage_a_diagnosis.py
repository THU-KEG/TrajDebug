#!/usr/bin/env python3
"""Stage A — per-step three-tier compression substrate.

Per v5 spec §5.2, Stage A's only job is to precompute the three-tier
``th1 / th2 / th3`` compressed views of every step in a unified
trajectory. Stage B / Stage C-Phase 1/2 then pick a tier per step
according to ``compress_role`` (baked by ``data_processing/bake_v5_fields``)
and the focal-distance rule.

Stage A no longer issues a trajectory-level failure-form diagnosis call;
the audit fields ``trajectory_summary`` / ``failure_form`` /
``failure_reason`` / ``error_phase`` are dropped entirely.

Output file: ``<output_dir>/<stem>_stage_a.json`` carrying:

  * the three-tier ``step_compressions`` pool;
  * minimal task metadata (task_id / description / environment / reward /
    agent_framework_description) so downstream stages can skip re-parsing
    the unified file when convenient.

The file name is preserved (``_stage_a.json``) for pipeline backward-
compatibility; only the schema shrinks.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _stage_common import (
    dump_step_compressions,
    load_unified_for_stage,
    resolve_extra_params,
)
from utils.llm_compression import (
    DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS,
    DEFAULT_STAGE_A_MAX_TOKENS,
    DEFAULT_STEP_TH1_MAX_CHARS,
    DEFAULT_STEP_TH2_MAX_CHARS,
    DEFAULT_STEP_TH3_MAX_CHARS,
    DEFAULT_TH1_CHUNK_SIZE,
    DEFAULT_TH2_CHUNK_SIZE,
    DEFAULT_TH3_CHUNK_SIZE,
    aprecompute_step_compressions,
    aprecompute_step_compressions_unified,
)
from utils.model import APIModel
from utils.trajectory_utils import collect_trajectory_sources, output_stem_for_source


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================
# Compressor (LLM-driven per-step three-tier precompute)
# =====================================================================


class StageACompressor:
    """Wraps the APIModel and the three-tier compression precompute.

    The class exposes only ``process_file`` / ``run_batch``; there is no
    diagnosis LLM call. ``compress_cache_url`` defaults to ``cache_url``
    so existing callers that supplied a single ``--cache`` keep working.
    """

    def __init__(self, api_config: Dict[str, Any]):
        self.config = api_config
        cache_url = api_config.get("cache_url")
        if not cache_url:
            raise ValueError("Missing cache_url in api_config. Please provide --cache.")
        compress_cache_url = api_config.get("compress_cache_url") or cache_url
        self.compress_model = APIModel(
            compress_cache_url,
            api_config["base_url"],
            api_config["model"],
            api_config.get("api_key", "EMPTY"),
            extra_params=resolve_extra_params(api_config.get("model_profile")),
        )

        self.step_th1_max_chars = int(
            api_config.get("step_th1_max_chars", DEFAULT_STEP_TH1_MAX_CHARS)
        )
        self.step_th2_max_chars = int(
            api_config.get("step_th2_max_chars", DEFAULT_STEP_TH2_MAX_CHARS)
        )
        self.step_th3_max_chars = int(
            api_config.get("step_th3_max_chars", DEFAULT_STEP_TH3_MAX_CHARS)
        )
        self.step_th1_chunk_size = int(
            api_config.get("step_th1_chunk_size", DEFAULT_TH1_CHUNK_SIZE)
        )
        self.step_th2_chunk_size = int(
            api_config.get("step_th2_chunk_size", DEFAULT_TH2_CHUNK_SIZE)
        )
        self.step_th3_chunk_size = int(
            api_config.get("step_th3_chunk_size", DEFAULT_TH3_CHUNK_SIZE)
        )
        self.history_overall_cap_chars = int(
            api_config.get(
                "history_overall_cap_chars",
                DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS,
            )
        )

        # ``force_json`` controls whether the compression LLM call is
        # made with ``response_format={"type": "json_object"}``. This is
        # the *only* LLM call Stage A issues, so we honour both the
        # legacy ``--compress_force_json`` flag and the unified
        # ``--judgement_force_json`` flag (the latter takes precedence
        # when explicitly supplied so a single shell driver can pin the
        # response_format for every stage).
        if "judgement_force_json" in api_config and api_config["judgement_force_json"] is not None:
            self.compress_force_json = bool(api_config["judgement_force_json"])
        else:
            self.compress_force_json = bool(api_config.get("compress_force_json", True))
        # Recorded for bookkeeping only.
        self.judgement_force_json = self.compress_force_json

        # ``--max_tokens`` is the real LLM-API ``max_tokens`` cap on the
        # compression call's *output* (i.e. how many tokens the model is
        # allowed to emit). ``--compress_max_tokens`` is a separate, prompt-
        # level signal hinting the model how short the compressed view
        # should be; it does NOT cap the API response and is currently
        # accepted for forward-compat / bookkeeping.
        _api_mt = api_config.get("max_tokens", None)
        self.llm_max_tokens: Optional[int] = int(_api_mt) if _api_mt is not None else None

        _ct = api_config.get("compress_max_tokens", None)
        self.compress_target_tokens: Optional[int] = int(_ct) if _ct is not None else None

        # Unified-mode per-step max_tokens. When present, each step's
        # compression LLM call uses this as the output budget; the
        # old per-tier path is bypassed entirely.
        _sa_mt = api_config.get("stage_a_max_tokens", None)
        self.stage_a_max_tokens: int = (
            int(_sa_mt) if _sa_mt is not None
            else (self.llm_max_tokens or DEFAULT_STAGE_A_MAX_TOKENS)
        )

        self.semaphore = asyncio.Semaphore(
            max(1, int(api_config.get("llm_concurrency", 10)))
        )

    def close(self) -> None:
        self.compress_model.close()

    async def _agenerate(self, prompt: str, **kwargs: Any) -> str:
        async with self.semaphore:
            result = await asyncio.to_thread(self.compress_model.generate, prompt, **kwargs)
        return result or ""

    async def compress_trajectory(
        self,
        trajectory_data: Dict[str, Any],
    ) -> Dict[int, Dict[str, str]]:
        """Run three-tier compression precompute for every step.

        Uses the unified single-call-per-step path: one LLM call per step
        produces th1/th2/th3 simultaneously, reducing total calls from
        N + ceil(N/3) + ceil(N/5) to N.

        Returns the ``step_compressions`` pool keyed by step id. The pool
        is also populated for successful trajectories so downstream
        analysers can replay them uniformly.
        """
        flat_steps = trajectory_data.get("flat_steps")
        if not isinstance(flat_steps, list):
            return {}

        step_compressions = await aprecompute_step_compressions_unified(
            agenerate_fn=self._agenerate,
            flat_steps=flat_steps,
            th1_max=self.step_th1_max_chars,
            th2_max=self.step_th2_max_chars,
            th3_max=self.step_th3_max_chars,
            stage_a_max_tokens=self.stage_a_max_tokens,
            force_json=self.compress_force_json,
        )
        return step_compressions

    async def process_file(
        self,
        file_path: str,
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """Compress one trajectory and write ``<stem>_stage_a.json``."""
        try:
            trajectory_data = load_unified_for_stage(file_path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path, exc)
            return None

        # Per-file token usage accumulator (concurrency-safe via closure)
        usage_acc = {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0}

        async def _tracked_agenerate(prompt: str, **kwargs: Any) -> str:
            async with self.semaphore:
                result, usage = await asyncio.to_thread(
                    self.compress_model.generate, prompt, return_usage=True, **kwargs
                )
            if usage:
                usage_acc["input_tokens"] += usage.get("input_tokens", 0)
                usage_acc["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
                usage_acc["output_tokens"] += usage.get("output_tokens", 0)
            return result or ""

        flat_steps = trajectory_data.get("flat_steps")
        if not isinstance(flat_steps, list):
            step_compressions: Dict[int, Dict[str, str]] = {}
        else:
            step_compressions = await aprecompute_step_compressions_unified(
                agenerate_fn=_tracked_agenerate,
                flat_steps=flat_steps,
                th1_max=self.step_th1_max_chars,
                th2_max=self.step_th2_max_chars,
                th3_max=self.step_th3_max_chars,
                stage_a_max_tokens=self.stage_a_max_tokens,
                force_json=self.compress_force_json,
            )

        payload: Dict[str, Any] = {
            "task_id": trajectory_data["task_id"],
            "task_description": trajectory_data["task_description"],
            "task_prompt_context": trajectory_data["task_prompt_context"],
            "task_success": trajectory_data["task_success"],
            "task_outcome": trajectory_data["task_outcome"],
            "reward": trajectory_data["reward"],
            "environment": trajectory_data["environment"],
            "total_steps": trajectory_data["total_steps"],
            "step_granularity": "unified_message",
            "trajectory_source": trajectory_data["trajectory_source"],
            "agent_framework_description": trajectory_data["agent_framework_description"],
            "step_compressions": dump_step_compressions(step_compressions),
            "token_usage": usage_acc,
        }

        out_stem = output_stem_for_source(file_path)
        out_fp = Path(output_dir) / f"{out_stem}_stage_a.json"
        with out_fp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info(
            "Stage A [%s]: compressed %d steps (tokens: in=%d, reason=%d, out=%d)",
            out_stem, trajectory_data["total_steps"],
            usage_acc["input_tokens"], usage_acc["reasoning_tokens"], usage_acc["output_tokens"],
        )
        return payload


# =====================================================================
# CLI batch runner
# =====================================================================


def _is_valid_stage_a_output(path: Path) -> bool:
    """Cached stage_a file is valid iff it has a non-empty
    ``step_compressions`` pool. The diagnosis block is no longer required.
    """
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    pool = data.get("step_compressions")
    if not isinstance(pool, dict) or not pool:
        return False
    if "task_id" not in data or "task_success" not in data:
        return False
    return True


def _ensure_chat_completions_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


async def run_batch(
    input_path: str,
    output_dir: str,
    api_config: Dict[str, Any],
    concurrency: int = 4,
    resume: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """File-level parallel runner.

    Stage A issues ~N compression-precompute LLM calls per trajectory
    (no diagnosis call). ``concurrency`` caps how many trajectories
    share the analyser; the analyser's own ``llm_concurrency`` caps
    total in-flight LLM calls.
    """
    os.makedirs(output_dir, exist_ok=True)
    api_config = dict(api_config)
    api_config["base_url"] = _ensure_chat_completions_url(api_config["base_url"])

    analyzer = StageACompressor(api_config)
    files = collect_trajectory_sources(input_path)
    if not files:
        raise ValueError(f"No .json trajectory files found under: {input_path}")

    file_sem = asyncio.Semaphore(max(1, int(concurrency)))
    total = len(files)

    async def _process_one(idx: int, fp: str) -> Tuple[str, str, Optional[str]]:
        try:
            out_fp = Path(output_dir) / f"{output_stem_for_source(fp)}_stage_a.json"
            if overwrite:
                pass
            elif resume and _is_valid_stage_a_output(out_fp):
                logger.info(
                    "Skipping (%d/%d): %s (valid cached: %s)",
                    idx, total, fp, out_fp.name,
                )
                return ("skipped", fp, None)
            elif resume and out_fp.exists():
                logger.warning(
                    "Re-running (%d/%d): %s (cached output invalid: %s)",
                    idx, total, fp, out_fp.name,
                )
            async with file_sem:
                logger.info("Processing (%d/%d): %s", idx, total, fp)
                r = await analyzer.process_file(fp, output_dir)
            if r is None:
                return ("failed", fp, "process_file returned None")
            return ("ok", fp, None)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return ("failed", fp, repr(exc))

    ok = failed = skipped = 0
    failures: List[Dict[str, Any]] = []
    try:
        tasks = [_process_one(i, fp) for i, fp in enumerate(files, start=1)]
        results = await asyncio.gather(*tasks)
        for status, fp, err in results:
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append({"file": fp, "error": err or ""})
    finally:
        analyzer.close()

    summary = {
        "input_path": input_path,
        "output_dir": output_dir,
        "num_files": total,
        "num_ok": ok,
        "num_failed": failed,
        "num_skipped": skipped,
        "file_concurrency": int(concurrency),
        "llm_concurrency": int(api_config.get("llm_concurrency", 10)),
        "resume": resume,
        "overwrite": overwrite,
        "failures": failures[:20],
    }
    logger.info("Stage A batch done: %s", summary)
    return summary


_BOOL_FALSE_STRINGS = {"0", "false", "no", "off", "n", "f"}


def _parse_bool_flag(value: Any) -> bool:
    return str(value).strip().lower() not in _BOOL_FALSE_STRINGS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage A: per-step three-tier compression substrate (v5).",
    )
    parser.add_argument("--input", required=True, help="Unified-schema trajectory file or directory")
    parser.add_argument("--output_dir", required=True, help="Directory to write *_stage_a.json outputs")

    parser.add_argument("--api_key", default=os.getenv("API_KEY"))
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--cache",
        required=True,
        help="Cache file for the per-step compression LLM calls.",
    )
    parser.add_argument(
        "--compress_cache",
        default=None,
        help=(
            "Alias for --cache (kept for back-compat with v4 invocations "
            "that used a separate diagnosis cache). Falls back to --cache."
        ),
    )

    parser.add_argument("--step_th1_max_chars", type=int, default=DEFAULT_STEP_TH1_MAX_CHARS)
    parser.add_argument("--step_th2_max_chars", type=int, default=DEFAULT_STEP_TH2_MAX_CHARS)
    parser.add_argument("--step_th3_max_chars", type=int, default=DEFAULT_STEP_TH3_MAX_CHARS)
    parser.add_argument("--step_th1_chunk_size", type=int, default=DEFAULT_TH1_CHUNK_SIZE)
    parser.add_argument("--step_th2_chunk_size", type=int, default=DEFAULT_TH2_CHUNK_SIZE)
    parser.add_argument("--step_th3_chunk_size", type=int, default=DEFAULT_TH3_CHUNK_SIZE)
    parser.add_argument(
        "--history_overall_cap_chars",
        type=int,
        default=DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS,
    )
    parser.add_argument("--compress_force_json", type=_parse_bool_flag, default=True)
    parser.add_argument(
        "--compress_max_tokens",
        type=int,
        default=None,
        help=(
            "Prompt-level hint telling the compression model how short "
            "the compressed view should be (NOT the LLM API max_tokens). "
            "Use --max_tokens to cap the API response length."
        ),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help=(
            "max_tokens passed to the LLM API call (caps the model's "
            "output length). Stage A only issues compression calls, so "
            "this directly caps the per-tier compression response."
        ),
    )
    parser.add_argument(
        "--judgement_force_json",
        type=_parse_bool_flag,
        default=None,
        help=(
            "If set, force response_format={'type':'json_object'} on the "
            "LLM API call. Stage A has no judgement call; in Stage A this "
            "controls the compression LLM call's response_format and "
            "overrides --compress_force_json when explicitly supplied."
        ),
    )
    parser.add_argument(
        "--stage_a_max_tokens",
        type=int,
        default=None,
        help=(
            "max_tokens for the unified single-call-per-step compression. "
            "Each step's LLM call uses this as the output budget. Set high "
            "for reasoning models whose thinking tokens far exceed the "
            "compressed output. Falls back to --max_tokens if not set."
        ),
    )

    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--llm_concurrency", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--model_profile",
        default=None,
        help=(
            "Optional model profile name from utils/model_profiles.py. "
            "When set, the profile's `params` (excluding temperature / "
            "max_tokens which stay owned by CLI) are forwarded to the "
            "OpenAI-compatible API as extra request parameters."
        ),
    )

    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Missing --api_key and env API_KEY is not set.")

    api_config = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "timeout": args.timeout,
        "cache_url": args.cache,
        "compress_cache_url": args.compress_cache or args.cache,
        "response_format": {"type": "json_object"},
        "step_th1_max_chars": args.step_th1_max_chars,
        "step_th2_max_chars": args.step_th2_max_chars,
        "step_th3_max_chars": args.step_th3_max_chars,
        "step_th1_chunk_size": args.step_th1_chunk_size,
        "step_th2_chunk_size": args.step_th2_chunk_size,
        "step_th3_chunk_size": args.step_th3_chunk_size,
        "history_overall_cap_chars": args.history_overall_cap_chars,
        "compress_force_json": args.compress_force_json,
        "compress_max_tokens": args.compress_max_tokens,
        "max_tokens": args.max_tokens,
        "stage_a_max_tokens": args.stage_a_max_tokens,
        "judgement_force_json": args.judgement_force_json,
        "llm_concurrency": args.llm_concurrency,
        "model_profile": args.model_profile,
    }

    summary = asyncio.run(
        run_batch(
            input_path=args.input,
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
