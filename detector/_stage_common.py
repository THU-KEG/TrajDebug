#!/usr/bin/env python3
"""Shared rendering / parsing helpers used by all three pipeline stages
(A = trajectory-level failure-form diagnosis, B = per-step wrong-content
detection, C = global critical-step selection). The helpers intentionally
live OUTSIDE ``utils/`` so the v2 (Phase 1 / 1.5 / 2) utility modules stay
untouched; new additions for the 3-stage pipeline live here.

Three concerns are handled:

1. Loading a unified trajectory into the shape every stage consumes
   (``load_unified_for_stage``): task statement, environment, flat step
   list, per-dataset agent-framework description, reward, terminal
   outcome. Stages A / B / C must all see the same list of steps and the
   same task text, so this is the single source of truth.

2. Rendering the trajectory block per stage:
     - ``render_uniform_th3_block``: every step at ``th3`` (Stage A).
     - ``render_stage_c_trajectory_block``: ``th1`` for Stage B wrong
       steps, ``th1`` for steps inside Stage A's ``error_phase`` ± radius,
       ``th2`` for their ±2 neighbours, ``th3`` elsewhere (Stage C).

3. Parsing the last balanced ``{...}`` dict from an LLM response
   (``extract_last_json_object``) so each stage can accept reasoning
   traces that emit narrative before the final JSON envelope.

The module does not import ``fine_grained_analysis`` /
``critical_error_detection`` to avoid cycles; every stage imports only
``_stage_common`` and ``utils.*``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.llm_compression import clip_text_middle
from utils.trajectory_utils import (
    flatten_unified_trajectory,
    load_trajectory_source,
    output_stem_for_source,
)


# =====================================================================
# Unified trajectory loading
# =====================================================================

def load_unified_for_stage(file_path: str) -> Dict[str, Any]:
    """Read a unified-schema trajectory file and return the fields every
    stage consumes.

    The returned dict is stage-agnostic: Stage A uses ``flat_steps`` to
    render the uniform-th3 block; Stage B iterates ``flat_steps`` for the
    per-step judgement; Stage C uses ``flat_steps`` to render the mixed-
    tier trajectory block. Task text, reward, and outcome are read once
    here so each stage can persist them into its output without
    reparsing the unified file downstream.
    """
    data = load_trajectory_source(file_path)
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Unified trajectory missing 'metadata': {file_path}. "
            "Run `python -m data_processing.build_unified_dataset` first."
        )

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Unified trajectory has no 'messages': {file_path}")

    environment = str(metadata.get("dataset", "")).strip() or "unified"
    task_id = str(metadata.get("task_id", "")).strip() or output_stem_for_source(file_path)
    task_description = str(metadata.get("task_description", "")).strip()
    if not task_description:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                task_description = str(msg.get("content", "")).strip()
                break

    reward = metadata.get("reward")
    try:
        reward_int = int(reward) if reward is not None else None
    except (TypeError, ValueError):
        reward_int = None
    task_success = bool(reward_int == 1)
    task_outcome = "success" if task_success else "failure"

    extra = metadata.get("extra") if isinstance(metadata.get("extra"), dict) else {}
    agent_framework_description = str(
        extra.get("agent_framework_description", "") or ""
    )

    flat_steps = flatten_unified_trajectory(data)
    trajectory_source = str(data.get("_trajectory_source", file_path))

    return {
        "task_id": task_id,
        "task_description": task_description,
        "task_prompt_context": task_description,
        "environment": environment,
        "task_success": task_success,
        "task_outcome": task_outcome,
        "reward": reward_int,
        "flat_steps": flat_steps,
        "total_steps": len(flat_steps),
        "agent_framework_description": agent_framework_description,
        "trajectory_source": trajectory_source,
        "metadata": metadata,
    }


# =====================================================================
# Compression-pool line rendering
# =====================================================================


def pool_line(
    sid: int,
    tier: str,
    raw_step: Dict[str, Any],
    step_pool: Dict[int, Dict[str, str]],
    warn_state: List[bool],
    fallback_chars: Dict[str, int],
) -> str:
    """Return the rendered "Step N: ..." line for one step at ``tier``.

    ``step_pool`` is the three-tier compression pool saved by Stage A /
    Phase 1; ``warn_state`` is a single-element mutable list ``[False]``
    used to throttle the "pool missing" warning to one log line per
    trajectory. ``fallback_chars`` maps ``"th1"/"th2"/"th3"`` to the
    per-tier fallback clip budget used only when the pool is missing for
    this step (legacy / broken files); in normal operation the pool is
    populated by Stage A and the fallback branch never runs.
    """
    entry = step_pool.get(sid)
    if entry:
        text = entry.get(tier) or entry.get("th2") or entry.get("th3") or entry.get("th1")
        if text:
            return text
    if not warn_state[0]:
        # Keep the warning a one-liner; the caller identifies the stage.
        import logging

        logging.getLogger(__name__).warning(
            "step_compressions pool missing tier=%s step=%s; falling back to raw clip.",
            tier,
            sid,
        )
        warn_state[0] = True
    cap = int(
        fallback_chars.get(tier, fallback_chars.get("th2", 512))
    )
    raw = str(raw_step.get("content", "") or "")
    role = str(raw_step.get("message_role", "") or raw_step.get("role", "") or "")
    clipped = clip_text_middle(raw, cap) if raw else ""
    header = f"Step {sid}"
    if role:
        header += f" [role: {role}]"
    if not clipped:
        return f"{header}:\n  (empty)"
    return f"{header}:\n  {clipped}"


# =====================================================================
# Stage A — uniform th3 trajectory block
# =====================================================================


def render_uniform_th3_block(
    trajectory_steps: List[Dict[str, Any]],
    step_pool: Dict[int, Dict[str, str]],
    warn_state: List[bool],
    fallback_chars: Dict[str, int],
    skip_middle: bool = False,
    keep_head: int = 4,
    keep_tail: int = 4,
) -> Tuple[str, str, int]:
    """Render the full trajectory at a uniform ``th3`` compression tier
    plus a raw TERMINAL STATE block (last 2 steps).

    Stage A's only trajectory rendering: one prompt sees every step at
    equal resolution so the holistic "what went wrong across the whole
    run" judgement is not biased toward any particular step. ``skip_middle``
    elides the middle when the trajectory is long; used by the 1-shot
    retry path to tame over-long prompts.

    Returns plain text (not JSON): each ``pool_line`` body already begins
    with its own ``Step N [role: X]:`` header, so wrapping the bodies in a
    JSON envelope would double-print the step / role and bloat the prompt
    with escape sequences for zero analytic gain. Steps are concatenated
    with a blank line between them; an elided-middle marker is inserted
    inline when ``skip_middle`` drops the centre of a long trajectory.
    """
    step_lookup: Dict[int, Dict[str, Any]] = {}
    for item in trajectory_steps:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item.get("step"))
        except Exception:
            continue
        step_lookup[sid] = item

    all_sids = sorted(step_lookup.keys())
    if not all_sids:
        return "(no trajectory steps)", "(no trajectory steps)", 0
    last_sid = all_sids[-1]

    render_sids: List[int] = list(all_sids)
    elided = 0
    if skip_middle and len(all_sids) > keep_head + keep_tail:
        render_sids = list(all_sids[:keep_head]) + list(all_sids[-keep_tail:])
        elided = len(all_sids) - len(render_sids)

    blocks: List[str] = []
    last_rendered: Optional[int] = None
    for sid in render_sids:
        if last_rendered is not None and sid - last_rendered > 1 and elided:
            blocks.append(
                f"Steps {last_rendered + 1}..{sid - 1}:\n  ... {elided} steps elided for brevity ..."
            )
        raw_step = step_lookup.get(sid, {})
        body = pool_line(sid, "th3", raw_step, step_pool, warn_state, fallback_chars)
        blocks.append(body)
        last_rendered = sid

    traj_text = "\n\n".join(blocks)
    terminal_text = _render_terminal_block(
        trajectory_steps=trajectory_steps,
        fallback_chars=fallback_chars,
    )
    return traj_text, terminal_text, last_sid


# =====================================================================
# Stage C — mixed-tier trajectory block with Stage B annotations merged
# =====================================================================


def render_stage_c_trajectory_block(
    trajectory_steps: List[Dict[str, Any]],
    stage_b_analyses: List[Dict[str, Any]],
    step_pool: Dict[int, Dict[str, str]],
    warn_state: List[bool],
    fallback_chars: Dict[str, int],
    error_phase: Optional[Tuple[int, int]] = None,
    focus_radius: int = 3,
) -> Tuple[str, str, int]:
    """Return ``(trajectory_json_block, terminal_block, last_step_id)``
    for Stage C.

    Tier routing (hidden from the LLM):
      - last K steps (K=4) → ``th1``.
      - any Stage B flagged step (``has_wrong_content=true``) → ``th1``.
      - any step inside ``error_phase`` ± ``focus_radius`` → ``th1``.
      - ±2 neighbours of a ``th1`` step → ``th2``.
      - everything else → ``th3``.

    Each entry carries the Stage B record inline (``stage_b`` field):
    ``{has_wrong_content, triggers[], on_error_chain, on_chain_rationale}``
    so the Stage C prompt has everything it needs in one JSON array.
    """
    step_lookup: Dict[int, Dict[str, Any]] = {}
    for item in trajectory_steps:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item.get("step"))
        except Exception:
            continue
        step_lookup[sid] = item

    stage_b_by_step: Dict[int, Dict[str, Any]] = {}
    for analysis in stage_b_analyses or []:
        if not isinstance(analysis, dict):
            continue
        try:
            sid = int(analysis.get("step"))
        except Exception:
            continue
        stage_b_by_step[sid] = analysis

    all_sids = sorted(step_lookup.keys())
    if not all_sids:
        return "[]", "(no trajectory steps)", 0
    last_sid = all_sids[-1]

    final_k = 4  # last step + 3 predecessors
    recent_sids: set = set(all_sids[-final_k:])

    flagged_sids: set = set()
    for sid, analysis in stage_b_by_step.items():
        wc = analysis.get("wrong_content") or {}
        triggers = wc.get("triggers") or []
        if isinstance(triggers, list) and any(isinstance(t, dict) for t in triggers):
            if wc.get("has_wrong_content"):
                flagged_sids.add(sid)

    focus_sids: set = set()
    if error_phase is not None:
        try:
            f_start = int(error_phase[0])
            f_end = int(error_phase[1])
        except Exception:
            f_start = f_end = None  # type: ignore[assignment]
        if f_start is not None and f_end is not None:
            if f_end < f_start:
                f_start, f_end = f_end, f_start
            radius = max(0, int(focus_radius))
            lo = f_start - radius
            hi = f_end + radius
            for sid in all_sids:
                if lo <= sid <= hi:
                    focus_sids.add(sid)

    th1_sids = recent_sids | flagged_sids | focus_sids

    th2_sids: set = set()
    for s in th1_sids:
        for d in (-2, -1, 1, 2):
            neighbour = s + d
            if neighbour in step_lookup and neighbour not in th1_sids:
                th2_sids.add(neighbour)

    def _tier_for(sid: int) -> str:
        if sid in th1_sids:
            return "th1"
        if sid in th2_sids:
            return "th2"
        return "th3"

    entries: List[Dict[str, Any]] = []
    for sid in all_sids:
        raw_step = step_lookup.get(sid, {})
        tier = _tier_for(sid)
        body = pool_line(sid, tier, raw_step, step_pool, warn_state, fallback_chars)
        role = str(raw_step.get("message_role") or raw_step.get("role") or "")

        entry: Dict[str, Any] = {
            "step": sid,
            "role": role,
            "content": body,
        }
        analysis = stage_b_by_step.get(sid)
        if analysis is not None:
            wc = analysis.get("wrong_content") or {}
            triggers_raw = wc.get("triggers") or []
            triggers_out: List[Dict[str, Any]] = []
            if isinstance(triggers_raw, list):
                for t in triggers_raw:
                    if not isinstance(t, dict):
                        continue
                    triggers_out.append({
                        "wrong_kind": t.get("wrong_kind", ""),
                        "taxonomy_tag": t.get("taxonomy_tag", ""),
                        "wrong_content_quote": t.get("wrong_content_quote", ""),
                        "reference_quote": t.get("reference_quote", ""),
                        "why_wrong": t.get("why_wrong", ""),
                    })
            entry["stage_b"] = {
                "has_wrong_content": bool(wc.get("has_wrong_content", False)),
                "triggers": triggers_out,
                "on_error_chain": bool(analysis.get("on_error_chain", False)),
                "on_chain_rationale": str(analysis.get("on_chain_rationale", "") or ""),
            }
        else:
            # User / non-judged step — represent explicitly so the LLM can
            # see it was not a Stage B candidate rather than guessing.
            entry["stage_b"] = None
        entries.append(entry)

    traj_json = json.dumps(entries, ensure_ascii=False, indent=2)
    terminal_text = _render_terminal_block(
        trajectory_steps=trajectory_steps,
        fallback_chars=fallback_chars,
    )
    return traj_json, terminal_text, last_sid


def _render_terminal_block(
    trajectory_steps: List[Dict[str, Any]],
    fallback_chars: Dict[str, int],
) -> str:
    """Return the last-2-steps TERMINAL STATE block (raw, clipped to
    ``fallback_chars["th1"]``).
    """
    step_lookup: Dict[int, Dict[str, Any]] = {}
    for item in trajectory_steps:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item.get("step"))
        except Exception:
            continue
        step_lookup[sid] = item
    all_sids = sorted(step_lookup.keys())
    if not all_sids:
        return "(no terminal step content)"
    cap = int(fallback_chars.get("th1", 3000))
    blocks: List[str] = []
    for sid in all_sids[-2:]:
        raw_step = step_lookup.get(sid, {})
        role = str(raw_step.get("message_role") or raw_step.get("role") or "")
        raw_text = str(raw_step.get("content", "") or "")
        capped = clip_text_middle(raw_text, cap) if raw_text else "(empty)"
        header = f"Step {sid}"
        if role:
            header += f" [role: {role}]"
        blocks.append(f"{header}:\n{capped}")
    return "\n\n".join(blocks) if blocks else "(no terminal step content)"


# =====================================================================
# Compression-pool IO
# =====================================================================


def load_step_compressions_from_payload(payload: Dict[str, Any]) -> Dict[int, Dict[str, str]]:
    """Coerce a serialized ``step_compressions`` pool (string keys) back
    to ``{int: {tier: str}}``. Tolerates legacy / broken payloads by
    returning an empty dict.
    """
    pool: Dict[int, Dict[str, str]] = {}
    raw = payload.get("step_compressions") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return pool
    for key, val in raw.items():
        try:
            sid = int(key)
        except Exception:
            continue
        if isinstance(val, dict):
            pool[sid] = {k: str(v) for k, v in val.items()}
    return pool


def dump_step_compressions(pool: Dict[int, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Serialize a pool back to string keys so JSON round-trips cleanly."""
    return {str(sid): dict(tiers) for sid, tiers in pool.items()}


# =====================================================================
# JSON extraction (shared parser helper)
# =====================================================================


_BALANCED_REGION_RE = re.compile(r"[{]")


def balanced_regions(text: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` indices (end exclusive) of every balanced
    ``{...}`` region in ``text``. Used by every stage's parser to tolerate
    reasoning-model outputs that emit stray ``{}`` fragments in their
    pre-JSON narrative.
    """
    regions: List[Tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != '{':
            i += 1
            continue
        depth = 0
        start = i
        while i < n:
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    regions.append((start, i + 1))
                    i += 1
                    break
            i += 1
        else:
            break
    return regions


def try_parse_dict(blob: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse ``blob`` as a JSON dict. Tolerates trailing
    commas, stray smart quotes, and a ``true/false/null`` → Python
    ``ast.literal_eval`` fallback.
    """
    cleaned = re.sub(r",(\s*[}\]])", r"\1", blob)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        fixed_str = cleaned
        fixed_str = re.sub(r'([a-zA-Z])"s\s', r"\1's ", fixed_str)
        fixed_str = re.sub(r'([a-zA-Z])"t\s', r"\1't ", fixed_str)
        fixed_str = re.sub(r'([a-zA-Z])"ll\s', r"\1'll ", fixed_str)
        fixed_str = re.sub(r'([a-zA-Z])"ve\s', r"\1've ", fixed_str)
        fixed_str = re.sub(r'([a-zA-Z])"re\s', r"\1're ", fixed_str)
        fixed_str = re.sub(r'([a-zA-Z])"d\s', r"\1'd ", fixed_str)
        obj = json.loads(fixed_str)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        import ast
        py_str = cleaned.replace('true', 'True').replace('false', 'False').replace('null', 'None')
        obj = ast.literal_eval(py_str)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_last_json_object(
    response: str,
    must_have_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Find the LATEST balanced ``{...}`` region in ``response`` that
    parses as a dict (and, if ``must_have_key`` is set, contains that
    key). Reasoning-model outputs conventionally emit narrative before
    the final JSON answer, so scanning from the back is more reliable
    than from the front.
    """
    if not response:
        return None
    regions = balanced_regions(response)
    for start, end in reversed(regions):
        cand = try_parse_dict(response[start:end])
        if cand is None:
            continue
        if must_have_key and must_have_key not in cand:
            continue
        return cand
    if must_have_key is None:
        return None
    # Final fallback: any parseable dict.
    for start, end in reversed(regions):
        cand = try_parse_dict(response[start:end])
        if cand is not None:
            return cand
    return None


# =====================================================================
# v5 — tier router (compress_role + focal-distance)
# =====================================================================


def select_tier_for_step(
    step_meta: Dict[str, Any],
    *,
    distance: int,
    th1_max_distance: int = 1,
    th2_max_distance: int = 4,
) -> str:
    """Pick a compression tier for one step under the v5 §5.2.3 rule.

    ``distance`` is the absolute step-index delta to the focal step
    (Stage B = current step; Stage C-Phase 2 = origin / last_trigger /
    last_trigger+N; Stage C-Phase 1 = no focus, callers should pass a
    large distance so everything falls back to th2/th3).

    Routing:
      * ``compress_role == "compress"`` → always ``th3``;
      * else if ``distance <= th1_max_distance``                 → ``th1``;
      * else if ``distance <= th2_max_distance``                 → ``th2``;
      * else                                                      → ``th3``.

    For ``compress_role == "preserve"`` the same distance ladder applies;
    the user-visible difference is whether the step is also subject to
    the global "compress everything in this role" override above.
    ``"default"`` falls through the same ladder.
    """
    role = (step_meta or {}).get("compress_role") if isinstance(step_meta, dict) else None
    if role == "compress":
        return "th3"
    if distance <= int(th1_max_distance):
        return "th1"
    if distance <= int(th2_max_distance):
        return "th2"
    return "th3"


def render_step_block(
    sid: int,
    tier: str,
    step_meta: Dict[str, Any],
    step_pool: Dict[int, Dict[str, str]],
    warn_state: List[bool],
    fallback_chars: Dict[str, int],
) -> str:
    """Render a single step's block with a header that carries only the
    basic step identity (``step`` / ``role`` / optional ``name``).

    Tier selection is handled by the caller via
    ``render_history_for_focus`` and ``pool_line``.
    """
    role = str((step_meta or {}).get("role") or "")
    name = (step_meta or {}).get("name")
    body = pool_line(sid, tier, step_meta or {}, step_pool, warn_state, fallback_chars)
    header_bits: List[str] = [f"step={sid}", f"role={role or 'assistant'}"]
    if isinstance(name, str) and name:
        header_bits.append(f"name={name}")
    header = "--- " + " ".join(header_bits) + " ---"
    return f"{header}\n{body}"


def render_history_for_focus(
    flat_steps: List[Dict[str, Any]],
    focus_step: int,
    step_pool: Dict[int, Dict[str, str]],
    *,
    warn_state: Optional[List[bool]] = None,
    fallback_chars: Optional[Dict[str, int]] = None,
    th1_max_distance: int = 1,
    th2_max_distance: int = 4,
    include_focus: bool = False,
    extra_th1_steps: Optional[List[int]] = None,
    sid_filter: Optional[List[int]] = None,
    history_only_before: bool = False,
) -> str:
    """Render every step in ``flat_steps`` (or the subset in ``sid_filter``)
    using the v5 tier router, focused on ``focus_step``.

    Stage B passes ``include_focus=False`` and ``history_only_before=True``
    so that only steps strictly before the current step are rendered as
    HISTORY (the current step is rendered separately at ``th1``). Stage C
    passes the focal step list via ``extra_th1_steps`` so origin /
    last_trigger / last_trigger+N are all pinned to ``th1``, and renders
    the full trajectory (no ``history_only_before``).
    """
    if warn_state is None:
        warn_state = [False]
    if fallback_chars is None:
        fallback_chars = {"th1": 4096, "th2": 1024, "th3": 256}

    extra_th1_set = {int(s) for s in (extra_th1_steps or []) if isinstance(s, int) or (isinstance(s, str) and s.lstrip("-").isdigit())}
    sid_filter_set: Optional[set] = None
    if sid_filter is not None:
        sid_filter_set = {int(s) for s in sid_filter}

    blocks: List[str] = []
    for step in flat_steps:
        try:
            sid = int(step.get("step"))
        except Exception:
            continue
        if sid_filter_set is not None and sid not in sid_filter_set:
            continue
        if (not include_focus) and sid == int(focus_step):
            continue
        if history_only_before and sid > int(focus_step):
            continue
        if sid in extra_th1_set:
            tier = "th1"
        elif step.get("compress_role") == "compress":
            tier = "th3"
        else:
            distance = abs(sid - int(focus_step))
            tier = select_tier_for_step(
                step,
                distance=distance,
                th1_max_distance=th1_max_distance,
                th2_max_distance=th2_max_distance,
            )
        blocks.append(render_step_block(sid, tier, step, step_pool, warn_state, fallback_chars))
    return "\n\n".join(blocks) if blocks else "(no history)"


# =====================================================================
# Model profile resolution
# =====================================================================

# Keys that are already controlled by per-stage CLI flags (--temperature,
# --max_tokens). They MUST be stripped from the profile's `params` before
# being passed as ``extra_params`` so the CLI value remains authoritative
# regardless of which profile is selected.
_PROFILE_PARAM_KEYS_RESERVED_BY_CLI = {"temperature", "max_tokens"}


def resolve_extra_params(model_profile: Optional[str]) -> Dict[str, Any]:
    """Resolve a ``--model_profile`` CLI value into ``extra_params`` for
    ``APIModel``.

    Behaviour:
      * ``None`` / empty string → returns ``{}`` (legacy callers see no
        change in behaviour).
      * Otherwise looks up the profile via
        ``utils.model_profiles.get_model_profile`` and returns its
        ``params`` dict with ``temperature`` / ``max_tokens`` stripped
        (those are owned by per-stage CLI flags).

    The profile's ``base_url`` / ``model`` / ``api_key`` are intentionally
    NOT returned — those are still expected to come through CLI flags so
    the shell driver remains the single source of truth for routing.
    """
    if not model_profile:
        return {}
    # Local import to avoid pulling utils.model_profiles when unused.
    from utils.model_profiles import get_model_profile

    profile = get_model_profile(model_profile)
    backend = profile.get("backend")
    if backend not in {"openai", "self_hosted"}:
        raise ValueError(
            f"Unsupported model profile backend: {backend!r}. "
            "Only OpenAI-compatible profiles are supported."
        )
    raw_params = profile.get("params") or {}
    if not isinstance(raw_params, dict):
        raise ValueError(
            f"Model profile {model_profile!r} has a non-object 'params' value."
        )
    extra: Dict[str, Any] = {}
    for k, v in raw_params.items():
        if k in _PROFILE_PARAM_KEYS_RESERVED_BY_CLI:
            continue
        extra[k] = v
    return extra


__all__ = [
    "load_unified_for_stage",
    "pool_line",
    "render_uniform_th3_block",
    "render_stage_c_trajectory_block",
    "load_step_compressions_from_payload",
    "dump_step_compressions",
    "balanced_regions",
    "try_parse_dict",
    "extract_last_json_object",
    # v5 additions
    "select_tier_for_step",
    "render_step_block",
    "render_history_for_focus",
    # model profile
    "resolve_extra_params",
]
