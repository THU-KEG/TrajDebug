#!/usr/bin/env python3
"""Stage C — Phase 3 (v5): Mechanical assembly of final output.

Per v5 spec §5.4.3 / §5.6.4, Phase 3 does NOT call any LLM. It reads
Phase 2 outputs and mechanically assembles:
  * agent_chain / env_chain
  * agent_critical_step / env_critical_step / unified_critical_step
  * top-K candidates
  * empty-chain rescue (optional LLM fallback)

Inputs:
  * ``<stem>_stage_c_phase2.json``

Output: ``<output_dir>/<stem>_final.json``
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _stage_common import (
    extract_last_json_object,
    load_step_compressions_from_payload,
    load_unified_for_stage,
    render_history_for_focus,
    resolve_extra_params,
)
from utils.llm_compression import strip_think_tags
from utils.model import APIModel
from utils.trajectory_utils import (
    collect_trajectory_sources,
    output_stem_for_source,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Category priority for tie-breaking: cat-1 > cat-2 > cat-3 > env
_CATEGORY_PRIORITY = {"cat-1": 0, "cat-2": 1, "cat-3": 2, "env": 3}
# State priority for tie-breaking: regular > dormant
_STATE_PRIORITY = {"regular": 0, "dormant": 1}
# Terminal-connection TIE-BREAK priority (v6.2):
#   irreversible > semantic > budget_debt > semantic_uncertain > none.
# v6.2 change: this is NO LONGER the primary sort key. The primary
# sort key is qualified_origin_step (earliness). TC priority is used
# only to break ties between instances with the same qos. Rationale:
# we want the EARLIEST root cause regardless of carrier strength;
# carrier strength only matters when two equally-early instances
# compete for the chain pick.
_TC_TIE_PRIORITY = {
    "irreversible": 0,
    "semantic": 1,
    "budget_debt": 2,
    "semantic_uncertain": 3,
    "none": 99,
}
# Backwards-compat alias (some external callers / tests reference
# _TC_PRIORITY by name).
_TC_PRIORITY = _TC_TIE_PRIORITY
# v6.2 budget_debt promotion gate. If a fixed-but-costly instance
# wasted more than this fraction of total_steps, its budget_debt
# carrier is treated as semantic-strength in the tie-break (i.e. it
# is no longer second-class to genuine semantic violations). This is
# a SOFT promotion: it only matters when qos is tied; it never
# overrides earliness.
_BUDGET_DEBT_RATIO_GATE = 0.5
# State-confidence priority: normal > low.
_CONF_PRIORITY = {"normal": 0, "low": 1}


def _effective_tc_priority(inst: Dict[str, Any]) -> int:
    """Return the tie-break TC priority WITH ratio promotion (v6.2).

    A budget_debt carrier whose `wasted_step_count / total_steps` ratio
    exceeds ``_BUDGET_DEBT_RATIO_GATE`` is promoted one tier so it ties
    with `semantic` carriers. This gives fixed-but-costly errors that
    consumed a meaningful fraction of the trajectory the same
    tie-break weight as visible-at-T semantic violations, while still
    leaving genuine `irreversible` carriers strictly stronger.
    """
    tc = inst.get("terminal_connection", "semantic_uncertain")
    base = _TC_TIE_PRIORITY.get(tc, _TC_TIE_PRIORITY["semantic_uncertain"])
    if tc != "budget_debt":
        return base
    re_obj = inst.get("resource_effect") or {}
    if not isinstance(re_obj, dict):
        return base
    try:
        wasted = int(re_obj.get("wasted_step_count") or 0)
    except (TypeError, ValueError):
        wasted = 0
    try:
        total = int(inst.get("_total_steps") or 0)
    except (TypeError, ValueError):
        total = 0
    if total <= 0 or wasted <= 0:
        return base
    ratio = wasted / total
    if ratio > _BUDGET_DEBT_RATIO_GATE:
        # Promote to the semantic tier (1) so it ties with semantic
        # carriers in tie-breaks. Never go above 1; irreversible (0)
        # remains strictly stronger.
        return _TC_TIE_PRIORITY["semantic"]
    return base


# =====================================================================
# Core assembly logic (no LLM)
# =====================================================================


def _qos(inst: Dict[str, Any]) -> Optional[int]:
    """Return the qualified_origin_step for chain_min_v2.

    Falls back to origin_step if Phase 2 omitted the field (e.g. an
    older Phase 2 output). If the instance was suppressed by
    pure_exploration in Phase 2, returns None so the caller filters
    it out.
    """
    if inst.get("exploration_suppressed") is True:
        return None
    if "qualified_origin_step" in inst:
        v = inst.get("qualified_origin_step")
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    # Backward compat: pre-v6 Phase 2 output had no qualified_origin_step.
    try:
        return int(inst.get("origin_step", 0))
    except (TypeError, ValueError):
        return None


def _is_in_chain(inst: Dict[str, Any]) -> bool:
    """Decide whether an instance is in the terminal-connected chain.

    v6 Phase 2 emits explicit ``chain_membership`` and
    ``terminal_connection`` fields; we trust those when available.
    Pre-v6 outputs fall back to fix_status=="active" / span_exceeds_cap
    (the legacy "active pool" definition).
    """
    if inst.get("exploration_suppressed") is True:
        return False
    if _qos(inst) is None:
        return False
    if "chain_membership" in inst:
        return bool(inst.get("chain_membership"))
    # Legacy fallback.
    return (
        inst.get("fix_status") == "active"
        or inst.get("span_exceeds_cap") is True
    )


def _chain_min_v2(
    chain_instances: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Methodology v6.2 §4.1 chain_min_v2 (cross-channel earliest-first)
    + v6.3 single-trigger guard.

    Pick the EARLIEST terminal-connected instance, with TC strength
    used only as a tie-break. v6.2 sort key (lower wins):

      1. qualified_origin_step (earliness wins).
      2. effective TC priority
         (irreversible > semantic > budget_debt > semantic_uncertain),
         with budget_debt promoted to semantic tier when
         wasted/total_steps > 0.15 (see _effective_tc_priority).
      3. category priority (cat-1 > cat-2 > cat-3 > env).
      4. state confidence (normal > low / parse_failed).
      5. state=regular before dormant (carrier strength).

    Rationale (v6.2): we want the EARLIEST root cause. Pure-TC-first
    sorting in v6.0/v6.1 had a known failure mode where a late
    high-TC instance beat the actual earliest cause. Earliness-first
    sorting matches the human-annotated “first-fault” intuition; TC is
    retained only to disambiguate when two instances genuinely share
    a qos.

    v6.3 SINGLE-TRIGGER GUARD: chain_min was prone to picking an
    early instance with only ONE trigger over a slightly-later
    instance with multiple triggers, even though the multi-trigger
    instance is the more persistent fault. We add a soft preference
    for multi-trigger instances when they exist within a small
    earliness window (5 steps) of the earliest single-trigger
    instance:

      - If the chain pool contains BOTH multi-trigger (≥2 triggers)
        and single-trigger (1 trigger) instances, AND
      - The earliest multi-trigger qos is at most 5 steps later
        than the earliest single-trigger qos,

    we restrict the sort to multi-trigger instances only. The 5-step
    window ensures we do not evict a genuinely earlier root cause:
    a single-trigger instance >5 steps before any multi-trigger is
    treated as the genuine first-fault and kept in the pool.
    """
    if not chain_instances:
        return None, None

    # v6.3 single-trigger guard
    multi = [m for m in chain_instances if len(m.get("trigger_indices") or []) >= 2]
    single = [m for m in chain_instances if len(m.get("trigger_indices") or []) < 2]
    pool_to_sort = chain_instances
    if multi and single:
        es = min(
            (_qos(m) for m in single if _qos(m) is not None), default=None
        )
        em = min(
            (_qos(m) for m in multi if _qos(m) is not None), default=None
        )
        if es is not None and em is not None and (em - es) <= 5:
            # Multi-trigger competitor is close enough; prefer it.
            pool_to_sort = multi

    def sort_key(inst: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        qos = _qos(inst)
        qos_key = qos if qos is not None else 10**9
        tc_p = _effective_tc_priority(inst)
        cat_p = _CATEGORY_PRIORITY.get(inst.get("category", "cat-2"), 1)
        conf = inst.get("state_confidence") or (
            "low" if inst.get("parse_failed") else "normal"
        )
        conf_p = _CONF_PRIORITY.get(conf, 0)
        state_p = _STATE_PRIORITY.get(_effective_state(inst), 0)
        # v6.2: qos first, TC second.
        return (qos_key, tc_p, cat_p, conf_p, state_p)

    pool_sorted = sorted(pool_to_sort, key=sort_key)
    chosen = pool_sorted[0]
    return _qos(chosen), chosen


def _per_channel_chain_min(
    chain_instances: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Per-channel earliest-qos picker, used for agent_/env_critical_step
    audit fields. Sort key (lower wins):
      1. qualified_origin_step (earliest).
      2. terminal_connection priority.
      3. category priority.
      4. state confidence.
      5. state=regular before dormant.
    """
    if not chain_instances:
        return None, None

    def sort_key(inst: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        qos = _qos(inst)
        qos_key = qos if qos is not None else 10**9
        tc_p = _effective_tc_priority(inst)
        cat_p = _CATEGORY_PRIORITY.get(inst.get("category", "cat-2"), 1)
        conf = inst.get("state_confidence") or (
            "low" if inst.get("parse_failed") else "normal"
        )
        conf_p = _CONF_PRIORITY.get(conf, 0)
        state_p = _STATE_PRIORITY.get(_effective_state(inst), 0)
        return (qos_key, tc_p, cat_p, conf_p, state_p)

    pool_sorted = sorted(chain_instances, key=sort_key)
    chosen = pool_sorted[0]
    return _qos(chosen), chosen


def _chain_min(active_instances: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """DEPRECATED legacy wrapper kept only for any external caller.

    Delegates to ``_chain_min_v2`` so the deterministic decision is
    always anchored on qualified_origin_step.
    """
    return _chain_min_v2(active_instances)


# =====================================================================
# Cascade audit — REMOVED in v6.
# Rationale: every observable cascade signal (same_violated_object,
# strategy_persistence, budget_debt_carryover, irreversible_carryover)
# is already represented in Phase 2's terminal_connection /
# qualified_origin / chain_membership fields. The cascade audit only
# re-encoded those signals via a non-recursive single-shot rewrite,
# which (a) introduced a subtle circular dependency for parse_failed
# instances (§4.4 self-reference) and (b) could be replaced wholesale
# by tightening Phase 2's terminal_connection labelling. The unified
# critical step is now anchored on a single function: chain_min_v2.
# =====================================================================


def _effective_state(inst: Dict[str, Any]) -> str:
    """Return the effective state for sorting.

    Span-cap instances are forced to 'regular' at the Phase 3 level;
    their `_effective_state` field reflects this override.
    """
    return inst.get("_effective_state") or inst.get("state", "regular")


def _build_top_k(
    active_instances: List[Dict[str, Any]],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Build top-K candidate list sorted by chain_min_v2 priority."""
    def sort_key(inst: Dict[str, Any]) -> Tuple[int, int, int, int]:
        qos = _qos(inst)
        qos_key = qos if qos is not None else 10**9
        tc_p = _effective_tc_priority(inst)
        cat_p = _CATEGORY_PRIORITY.get(inst.get("category", "cat-2"), 1)
        state_p = _STATE_PRIORITY.get(_effective_state(inst), 0)
        return (qos_key, tc_p, cat_p, state_p)

    sorted_list = sorted(active_instances, key=sort_key)
    top_k = []
    for inst in sorted_list[:k]:
        top_k.append({
            "instance_id": inst.get("instance_id"),
            "origin_step": inst.get("origin_step"),
            "qualified_origin_step": _qos(inst),
            "category": inst.get("category"),
            "attribution": inst.get("attribution"),
            "state": inst.get("state"),
            "terminal_connection": inst.get("terminal_connection"),
            "chain_membership": inst.get("chain_membership"),
            "parse_failed": inst.get("parse_failed", False),
            "error_content": inst.get("error_content", ""),
            "chain_explanation": inst.get("chain_explanation", ""),
        })
    return top_k


def assemble_final(phase2_payload: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """Mechanical assembly of the final output from Phase 2 results.

    Methodology v6 pipeline (cascade has been REMOVED in v6):
      1. chain_min_v2 sorts ALL terminal-connected instances ACROSS
         channels by (terminal_connection priority, qualified_origin,
         category, confidence, state) and picks the winner. The
         winner's qualified_origin_step is unified_critical_step.
      2. agent_critical_step / env_critical_step are reported per
         channel using the per-channel earliest-qos picker, purely
         for audit. They no longer drive unified.
      3. Empty-chain fallback (§4.5): if task failed and no
         terminal-connected instance exists, we relax to the active
         pool (fix_status=active OR span_exceeds_cap) and re-run
         chain_min_v2 there, marking unified_source=
         "empty_chain_fallback".
    """
    instances = phase2_payload.get("instances") or []
    instance_states = phase2_payload.get("instance_states") or []

    # Merge instance metadata with instance states
    inst_by_id = {int(inst.get("instance_id", -1)): inst for inst in instances}
    merged: List[Dict[str, Any]] = []
    for state_result in instance_states:
        iid = int(state_result.get("instance_id", -1))
        base = inst_by_id.get(iid, {})
        merged.append({**base, **state_result})

    # Effective-state annotation: span_exceeds_cap is now an auxiliary
    # signal only. We still surface it via `_effective_state=regular`
    # for downstream sorting / LLM picker rationales.
    # Also stamp `_total_steps` onto every instance so the budget_debt
    # ratio promotion in `_effective_tc_priority` (v6.2) can compute
    # wasted/total without threading the value through every helper.
    _total_steps_for_inst = phase2_payload.get("total_steps")
    for m in merged:
        if m.get("span_exceeds_cap") is True:
            m["_effective_state"] = "regular"
        m["_total_steps"] = _total_steps_for_inst

    # ----- channel pools ---------------------------------------------
    # chain_pool (per channel) = terminal-connected instances. This is
    # the input to chain_min_v2.
    agent_chain = [
        m for m in merged
        if m.get("attribution") == "agent" and _is_in_chain(m)
    ]
    env_chain = [
        m for m in merged
        if m.get("attribution") == "env" and _is_in_chain(m)
    ]

    # Legacy active pool (kept for LLM picker / top-K compatibility).
    # An instance enters the active pool if fix_status==\"active\" OR
    # span_exceeds_cap is True. Note this is a SUPERSET of chain_pool
    # because chain_pool additionally requires terminal_connection !=
    # \"none\".
    agent_active = [
        m for m in merged
        if m.get("attribution") == "agent"
        and (m.get("fix_status") == "active" or m.get("span_exceeds_cap") is True)
    ]
    env_active = [
        m for m in merged
        if m.get("attribution") == "env"
        and (m.get("fix_status") == "active" or m.get("span_exceeds_cap") is True)
    ]
    # Dedup active pools by instance_id (defensive).
    def _dedup(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for m in lst:
            iid = int(m.get("instance_id", -1))
            if iid in seen:
                continue
            seen.add(iid)
            out.append(m)
        return out
    agent_active = _dedup(agent_active)
    env_active = _dedup(env_active)
    all_active = agent_active + env_active

    # ----- §4.1 chain_min_v2 (CROSS-CHANNEL unified, v6) ------------
    # The unified pick is the cross-channel TC-first sort. The
    # per-channel picks are the legacy earliest-qos picks, retained
    # for audit. This guarantees that a strong-TC env instance can
    # win against an early but weak-TC agent instance.
    full_chain = agent_chain + env_chain

    # v6.4 ENV-FIRST GUARD (with multi-trigger override).
    # When an env instance is in the chain pool, prefer it over agent
    # instances UNLESS the agent_chain contains a multi-trigger
    # (>= 2 triggers) instance whose qualified_origin_step is strictly
    # earlier than the earliest env instance. The intuition:
    #   * env-channel triggers usually represent short, local
    #     environment anomalies; chain_min_v2 alone tends to let the
    #     earliest single-trigger agent instance win, even when the
    #     env instance is the actual root cause (e.g. WhoAndWhen 1
    #     where ad-overlay hijacks dominate).
    #   * BUT we must not let env override a genuine "persistent agent
    #     fault" (an agent instance that fired multiple triggers)
    #     occurring earlier than the env exposure.
    # If the multi-trigger agent override applies we fall through to
    # the original cross-channel chain_min_v2.
    if env_chain:
        env_pred_step, env_pred_inst = _chain_min_v2(env_chain)
        agent_multi_earlier = []
        if env_pred_step is not None:
            for m in agent_chain:
                if len(m.get("trigger_indices") or []) < 2:
                    continue
                m_qos = _qos(m)
                if m_qos is not None and m_qos < env_pred_step:
                    agent_multi_earlier.append(m)
        if not agent_multi_earlier and env_pred_inst is not None:
            unified_chosen_step, unified_chosen = env_pred_step, env_pred_inst
        else:
            unified_chosen_step, unified_chosen = _chain_min_v2(full_chain)
    else:
        unified_chosen_step, unified_chosen = _chain_min_v2(full_chain)

    agent_chosen_step, agent_chosen = _per_channel_chain_min(agent_chain)
    env_chosen_step, env_chosen = _per_channel_chain_min(env_chain)

    # Empty-chain fallback (§4.5): if no terminal-connected instance
    # was found but the task failed, relax to the active pool. We use
    # chain_min_v2 over the active pool too so the same TC-first sort
    # applies (most active instances have a non-none TC; those that
    # don't sort to the bottom).
    used_empty_chain_fallback = False
    task_success = bool(phase2_payload.get("task_success"))
    if unified_chosen is None and not task_success and all_active:
        unified_chosen_step, unified_chosen = _chain_min_v2(all_active)
        if unified_chosen is not None:
            used_empty_chain_fallback = True
        # Per-channel fallback as well, so audit fields are populated.
        if agent_chosen is None and agent_active:
            agent_chosen_step, agent_chosen = _per_channel_chain_min(agent_active)
        if env_chosen is None and env_active:
            env_chosen_step, env_chosen = _per_channel_chain_min(env_active)

    unified_critical_step = unified_chosen_step
    agent_critical_step = agent_chosen_step
    env_critical_step = env_chosen_step

    # Determine which channel contributed to the unified pick.
    if unified_chosen is None:
        unified_channel = None
        unified_source = None
    else:
        unified_channel = unified_chosen.get("attribution") or None
        unified_source = (
            "empty_chain_fallback" if used_empty_chain_fallback
            else "chain_min_v2"
        )

    agent_pick_source = "chain_min_v2" if agent_chosen is not None else None
    env_pick_source = "chain_min_v2" if env_chosen is not None else None

    # Build the affected_module and taxonomy_tag from the chosen instance
    agent_module = _extract_module(agent_chosen) if agent_chosen else None
    env_module = _extract_module(env_chosen) if env_chosen else None
    agent_taxonomy_tag = _extract_taxonomy_tag(agent_chosen) if agent_chosen else None
    env_taxonomy_tag = _extract_taxonomy_tag(env_chosen) if env_chosen else None
    # unified taxonomy_tag: prefer the channel that won the unified pick
    if unified_chosen is not None:
        unified_taxonomy_tag = _extract_taxonomy_tag(unified_chosen)
    else:
        unified_taxonomy_tag = None

    # Top-K candidates from all chain instances (fall back to active
    # pool if chain_pool is empty so the user still sees something).
    top_k_pool = (agent_chain + env_chain) or all_active
    top_k_candidates = _build_top_k(top_k_pool, k=top_k)

    # ----- audit / evaluation fields (§9) ----------------------------
    chain_count = len(agent_chain) + len(env_chain)
    parse_failed_in_chain = sum(
        1 for m in (agent_chain + env_chain) if m.get("parse_failed")
    )
    parse_failed_share = (
        parse_failed_in_chain / chain_count if chain_count else 0.0
    )

    # Compose output
    result: Dict[str, Any] = {
        "task_id": phase2_payload.get("task_id"),
        "task_description": phase2_payload.get("task_description"),
        "task_success": phase2_payload.get("task_success"),
        "task_outcome": phase2_payload.get("task_outcome"),
        "reward": phase2_payload.get("reward"),
        "environment": phase2_payload.get("environment"),
        "total_steps": phase2_payload.get("total_steps"),
        "trajectory_source": phase2_payload.get("trajectory_source"),
        "trajectory_file_path": phase2_payload.get("trajectory_file_path") or "",
        "metadata": phase2_payload.get("metadata") or {},
        "critical_error": {
            "unified_critical_step": unified_critical_step,
            "unified_source": unified_source,
            "unified_channel": unified_channel,
            "unified_taxonomy_tag": unified_taxonomy_tag,
            "agent_critical_step": agent_critical_step,
            "agent_category": agent_chosen.get("category") if agent_chosen else None,
            "agent_affected_module": agent_module,
            "agent_taxonomy_tag": agent_taxonomy_tag,
            "agent_chain_explanation": agent_chosen.get("chain_explanation", "") if agent_chosen else None,
            "agent_pick_source": agent_pick_source,
            "env_critical_step": env_critical_step,
            "env_category": env_chosen.get("category") if env_chosen else None,
            "env_affected_module": env_module,
            "env_taxonomy_tag": env_taxonomy_tag,
            "env_chain_explanation": env_chosen.get("chain_explanation", "") if env_chosen else None,
            "env_pick_source": env_pick_source,
            # Pre-fallback snapshot (the chain_min_v2 winner step
            # over the chain pool, before the empty-chain fallback).
            "chain_min_unified_critical_step": (
                unified_critical_step if not used_empty_chain_fallback else None
            ),
            "chain_min_agent_critical_step": agent_critical_step,
            "chain_min_env_critical_step": env_critical_step,
            # Empty-chain fallback indicator (§4.5).
            "empty_chain_fallback_used": used_empty_chain_fallback,
            # LLM-pick is auxiliary in v6: it can rewrite
            # agent_/env_critical_step but NEVER unified_critical_step
            # (which always comes from chain_min_v2 above).
            "llm_unified_critical_step": None,
            "llm_agent_critical_step": None,
            "llm_env_critical_step": None,
        },
        "top_k_candidates": top_k_candidates,
        "chain_summary": {
            "num_agent_chain": len(agent_chain),
            "num_env_chain": len(env_chain),
            "num_agent_active": len(agent_active),
            "num_env_active": len(env_active),
            "num_fixed": sum(
                1 for m in merged
                if isinstance(m.get("fix_status"), str)
                and m.get("fix_status", "").startswith("fixed")
            ),
            "num_dormant": sum(
                1 for m in merged
                if m.get("state") == "dormant" and not m.get("span_exceeds_cap")
            ),
            "num_span_cap": sum(1 for m in merged if m.get("span_exceeds_cap")),
            "num_exploration_suppressed": sum(
                1 for m in merged if m.get("exploration_suppressed")
            ),
            "num_parse_failed": sum(1 for m in merged if m.get("parse_failed")),
            "parse_failed_share": round(parse_failed_share, 4),
            "num_total_instances": len(merged),
        },
        "full_instance_states": merged,
        "step_triggers": phase2_payload.get("step_triggers") or [],
    }

    return result


def _extract_module(chosen: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract affected_module from an instance's triggers.

    Prefers ``qualified_origin_taxonomy_tag`` (set by phase2) which has
    the form ``"<module>.<subtype>"``; falls back to the instance category.
    """
    if not chosen:
        return None
    tag = chosen.get("qualified_origin_taxonomy_tag") or ""
    if tag and "." in tag:
        return tag.split(".", 1)[0]
    cat = chosen.get("category", "")
    return cat if cat else None


def _extract_taxonomy_tag(chosen: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the full ``qualified_origin_taxonomy_tag`` from a chosen instance."""
    if not chosen:
        return None
    tag = chosen.get("qualified_origin_taxonomy_tag") or ""
    return tag.strip() or None


# =====================================================================
# Empty-chain rescue (optional LLM call)
# =====================================================================

RESCUE_SYSTEM_PROMPT = r"""You are a critical-step rescuer. The per-step error detection pipeline found NO active error instances for this FAILED trajectory. This should be rare. Identify the single most likely critical error step.

Look at the trajectory and select the step where the agent most clearly deviated from a correct path. Output:

{
  "rescued_critical_step": <int>,
  "rescued_category": "cat-1" | "cat-2" | "cat-3" | "env",
  "rescued_explanation": "<2-3 sentences>"
}
"""


# =====================================================================
# Critical-instance picker (LLM ranking, used when active pool >= 2)
# =====================================================================

PICK_CRITICAL_SYSTEM_PROMPT = (
    "You are an expert at identifying critical failure points in agent trajectories. "
    "Respond with ONLY one valid JSON object that matches the requested schema."
)

PICK_CRITICAL_USER_PREAMBLE = f"""You are an expert at identifying the SINGLE critical step that caused a failed agent trajectory.

An error is critical if:
- It represents the ROOT CAUSE that made task success impossible
- It caused a cascade of subsequent errors
- The trajectory could have succeeded if THIS specific error had not occurred
- IMPORTANT: Correcting this specific error would fundamentally change the trajectory toward success

# Hints
1. Consider the ENTIRE trajectory from a global perspective - understand
   the task goal and how the agent's path diverged from success.
2. Early exploration steps (steps 1-3) are often normal and should **NOT**
   be marked as critical unless they clearly introduce the root cause.
3. Prefer the first step where the agent had enough reasonable information
   to proceed correctly but nevertheless introduced the error locally.
   This includes producing an incorrect output, misinterpreting or misusing
   prior information, or turning a correct intermediate state into an incorrect one.
4. Do not choose a later step merely because the failure becomes more visible there.
   If a later step only repeats, propagates, or amplifies an earlier mistake,
   select the earlier step where the mistake originated.
5. If a flawed strategy is repeated across multiple steps, attribute the failure
   to the step where that strategy was first introduced. If the strategy was
   reasonable but the execution result was wrong, attribute the failure to the
   first execution step that produced the incorrect result.
6. Use recoverability only as a tie-breaker: among otherwise similar candidates,
   prefer the earliest step whose error made recovery unlikely or blocked the
   trajectory from returning to a successful path.

You are given:
- the TASK,
- the FULL TRAJECTORY,
- a list of SUSPICIOUS STEPS that an upstream annotator flagged as containing some erroneous content (they are hints; you may pick any other step from the trajectory if you disagree).

"""


def _format_candidate_for_pick(inst: Dict[str, Any]) -> Dict[str, Any]:
    """Render an active instance into a compact JSON-friendly dict for
    the picker prompt. We deliberately strip large fields (e.g.
    trigger_indices, full quotes) to keep the prompt token-light.
    """
    re_obj = inst.get("resource_effect") or {}
    return {
        "instance_id": inst.get("instance_id"),
        "origin_step": inst.get("origin_step"),
        "last_trigger_step": inst.get("last_trigger_step"),
        "category": inst.get("category"),
        "attribution": inst.get("attribution"),
        "terminal_connection": inst.get("terminal_connection"),
        "fix_status": inst.get("fix_status"),
        "wasted_step_count": re_obj.get("wasted_step_count"),
        "span_exceeds_cap": inst.get("span_exceeds_cap"),
        "chain_membership": inst.get("chain_membership"),
        "error_content": inst.get("error_content", "") or inst.get("description", ""),
        "chain_explanation": inst.get("chain_explanation", ""),
    }


# =====================================================================
# Inline-trigger trajectory renderer (used by the LLM critical picker)
# =====================================================================


_QUOTE_TRUNC = 240  # max chars per quote in the inline render


def _truncate(s: Any, n: int = _QUOTE_TRUNC) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[:n] + " ...(truncated)"


def _is_active_or_span_cap(merged_inst: Dict[str, Any]) -> bool:
    return (
        merged_inst.get("fix_status") == "active"
        or merged_inst.get("span_exceeds_cap") is True
    )


def _is_chain_pool(inst: Dict[str, Any]) -> bool:
    """True if the instance is in the chain pool (legal pick candidate).

    Chain pool = _is_in_chain() passes (chain_membership=True and not
    exploration_suppressed) OR span_exceeds_cap=True.
    span_exceeds_cap instances are kept even when chain_membership is
    absent/False, because they represent persistent faults whose span
    overflows Phase 2's tracking cap.
    """
    if inst.get("exploration_suppressed") is True:
        return False
    if inst.get("span_exceeds_cap") is True:
        return True
    return bool(inst.get("chain_membership"))


def _build_chain_pool_legal_picks(
    merged_instances: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Build a unified (cross-channel) legal pick map for Design B.

    Returns a dict keyed by instance_id, each value containing:
      {
        "inst": <instance dict>,
        "trigger_steps": [
            {"step": int, "tags": [str, ...]},
            ...
        ]
      }

    Only chain-pool instances are included (chain_membership=True or
    span_exceeds_cap=True, and not exploration_suppressed).
    """
    result: Dict[int, Dict[str, Any]] = {}
    for inst in merged_instances:
        if not _is_chain_pool(inst):
            continue
        iid = int(inst.get("instance_id", -1))
        if iid < 0:
            continue
        # Collect (step -> {tags, wrong_quote}) for this instance's triggers
        step_tags: Dict[int, List[str]] = {}
        step_wrong: Dict[int, str] = {}
        for ti in inst.get("trigger_indices") or []:
            try:
                ti_int = int(ti)
            except (TypeError, ValueError):
                continue
            if not (0 <= ti_int < len(triggers)):
                continue
            t = triggers[ti_int]
            step = t.get("step")
            if not isinstance(step, int):
                continue
            tag = t.get("taxonomy_tag") or "?"
            step_tags.setdefault(step, []).append(tag)
            # keep first non-empty wrong_quote per step
            if step not in step_wrong:
                wq = (t.get("wrong_content_quote") or t.get("wrong_quote") or "").strip()
                if wq:
                    step_wrong[step] = wq
        if not step_tags:
            continue
        trigger_steps = [
            {"step": s, "tags": step_tags[s], "wrong_quote": step_wrong.get(s, "")}
            for s in sorted(step_tags)
        ]
        result[iid] = {"inst": inst, "trigger_steps": trigger_steps}
    return result


def _render_legal_picks_block(
    chain_pool_picks: Dict[int, Dict[str, Any]],
) -> str:
    """Render the LEGAL PICKS block for the LLM prompt.

    Format per instance:
      Inst #N  [AGENT / plan_error / terminal:semantic / fix:active / wasted:12]
        error_content: ...
        chain_explanation: ...
        trigger_steps:
          step 12 — tag: hallucinated_constraint
          step 18 — tag: hallucinated_constraint, plan_repeats_failed_query
    """
    lines: List[str] = ["You MUST output a (supporting_instance_id, critical_step) "
                        "pair from this list."]
    lines.append("A 'supporting instance' is a cluster of evidence that a "
                 "persistent error damaged the trajectory;")
    lines.append("its trigger_steps are the steps where that error commitment is visible.")
    lines.append("")

    # Sort: agent first, then by instance_id
    ordered = sorted(
        chain_pool_picks.items(),
        key=lambda kv: (
            0 if kv[1]["inst"].get("attribution") == "agent" else 1,
            kv[0],
        ),
    )
    for iid, entry in ordered:
        inst = entry["inst"]
        attribution = (inst.get("attribution") or "agent").upper()
        category = inst.get("category") or "?"
        terminal = inst.get("terminal_connection") or "?"
        fix_status = inst.get("fix_status") or "active"
        re_obj = inst.get("resource_effect") or {}
        wasted = re_obj.get("wasted_step_count")
        span_cap = inst.get("span_exceeds_cap") is True

        wasted_str = f"{wasted}" if wasted is not None else "?"
        cap_str = " [span_exceeds_cap]" if span_cap else ""
        header = (f"  Inst #{iid}  [{attribution} / {category} / "
                  f"terminal:{terminal} / fix:{fix_status} / wasted:{wasted_str}]{cap_str}")
        lines.append(header)

        error_content = (inst.get("error_content") or inst.get("description") or "").strip()
        if error_content:
            lines.append(f"    error_content: {error_content[:200]}")

        chain_exp = (inst.get("chain_explanation") or "").strip()
        if chain_exp:
            lines.append(f"    chain_explanation: {chain_exp[:300]}")

        lines.append("    trigger_steps:")
        for ts in entry["trigger_steps"]:
            tag_str = ", ".join(ts["tags"])
            lines.append(f"      step {ts['step']} — tag: {tag_str}")
        lines.append("")
    return "\n".join(lines)


def _render_trajectory_with_inline_triggers(
    flat_steps: List[Dict[str, Any]],
    step_pool: Dict[int, Dict[str, str]],
    merged_instances: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]],
) -> str:
    """Render the full trajectory with trigger annotations inlined.

    Design B rendering rules:
    - Only chain-pool instances (chain_membership=True or span_exceeds_cap)
      have their triggers rendered in full (with wrong/reference quotes).
    - Steps that ONLY have triggers from fixed/non-chain instances are
      summarised as a single folded line so the LLM is aware triggers
      fired without being tempted to pick them.
    """
    # Separate chain-pool vs non-chain instance ids
    chain_iids: set = set()
    for inst in merged_instances:
        if _is_chain_pool(inst):
            iid = int(inst.get("instance_id", -1))
            if iid >= 0:
                chain_iids.add(iid)

    # Build step -> [(trigger, parent_instance)] map for ALL instances
    step_to_chain: Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    step_to_fixed_count: Dict[int, int] = {}
    for inst in merged_instances:
        iid = int(inst.get("instance_id", -1))
        in_chain = iid in chain_iids
        for ti in inst.get("trigger_indices") or []:
            try:
                ti_int = int(ti)
            except (TypeError, ValueError):
                continue
            if not (0 <= ti_int < len(triggers)):
                continue
            t = triggers[ti_int]
            step = t.get("step")
            if not isinstance(step, int):
                continue
            if in_chain:
                step_to_chain.setdefault(step, []).append((t, inst))
            else:
                step_to_fixed_count[step] = step_to_fixed_count.get(step, 0) + 1

    parts: List[str] = []
    for fs in flat_steps:
        step = int(fs.get("step", -1))
        role = fs.get("role", "?")

        body = ""
        pool = step_pool.get(step) if step_pool else None
        if pool:
            body = pool.get("th1") or pool.get("th2") or pool.get("th3") or ""
        if not body:
            body = str(fs.get("content", "") or "")

        parts.append(f"--- step={step} role={role} ---")
        parts.append(body.rstrip())

        chain_trig_list = step_to_chain.get(step, [])
        fixed_cnt = step_to_fixed_count.get(step, 0)

        if chain_trig_list:
            parts.append(f"  TRIGGERS @ step {step}:")
            for t, inst in chain_trig_list:
                iid = int(inst.get("instance_id", -1))
                span_cap = inst.get("span_exceeds_cap") is True
                inst_label = f"Inst #{iid} ACTIVE" + (" (span_exceeds_cap)" if span_cap else "")
                attribution = (t.get("attribution") or inst.get("attribution") or "agent").upper()
                tag = t.get("taxonomy_tag", "?")
                category = t.get("category", inst.get("category", "?"))
                conf = t.get("confidence", "?")
                wrong = _truncate(t.get("wrong_content_quote") or t.get("wrong_quote"))
                ref = _truncate(t.get("reference_quote") or t.get("ref_quote"))
                parts.append(
                    f"    • [{inst_label}] [{attribution}] {tag} cat={category} conf={conf}"
                )
                if wrong:
                    parts.append(f"        wrong_quote: \"{wrong}\"")
                if ref:
                    parts.append(f"        reference_quote: \"{ref}\"")
            if fixed_cnt:
                parts.append(f"    ({fixed_cnt} fixed/non-chain trigger(s) omitted)")
        elif fixed_cnt:
            parts.append(f"  TRIGGERS @ step {step}: ({fixed_cnt} fixed/non-chain trigger(s) omitted)")

        parts.append("")  # blank line between steps

    return "\n".join(parts)


def _render_instance_summary(
    chain_pool_picks: Dict[int, Dict[str, Any]],
) -> str:
    """Compact summary of chain-pool instances for the LLM prompt.

    Placed after the trajectory. Only chain-pool instances are listed.
    """
    lines: List[str] = []
    ordered = sorted(
        chain_pool_picks.items(),
        key=lambda kv: (
            0 if kv[1]["inst"].get("attribution") == "agent" else 1,
            kv[0],
        ),
    )
    for iid, entry in ordered:
        inst = entry["inst"]
        attribution = (inst.get("attribution") or "agent").upper()
        category = inst.get("category", "?")
        span_cap = inst.get("span_exceeds_cap") is True
        status = "ACTIVE (span_exceeds_cap)" if span_cap else "ACTIVE"
        origin = inst.get("origin_step", "?")
        last = inst.get("last_trigger_step", "?")
        sub_goal = (inst.get("sub_goal") or "").strip()
        error_content = (inst.get("error_content") or "").strip()
        what_violated = (inst.get("what_is_being_violated") or "").strip()
        chain_explanation = (inst.get("chain_explanation") or "").strip()
        lines.append(
            f"[Instance #{iid}] {attribution} / {category} / {status}  "
            f"(origin_step={origin}, last_trigger_step={last})"
        )
        if sub_goal:
            lines.append(f"    sub_goal:        {sub_goal}")
        if error_content:
            lines.append(f"    error_content:   {_truncate(error_content, 360)}")
        if what_violated:
            lines.append(f"    violated:        {_truncate(what_violated, 360)}")
        if chain_explanation:
            lines.append(f"    chain_explain:   {_truncate(chain_explanation, 480)}")
        lines.append("")
    return "\n".join(lines)


def _render_candidate_instances(
    chain_pool_picks: Dict[int, Dict[str, Any]],
) -> str:
    """Render the CANDIDATE INSTANCES block for the LLM prompt.

    For each chain-pool instance shows:
      - instance ID and one-line error description
      - each trigger step with taxonomy_tag and wrong_content_quote (clipped)

    No wasted_step_count, terminal_connection, chain_explanation exposed.
    Sorted by instance_id.
    """
    _WQ_CLIP = 200

    lines: List[str] = []
    for iid in sorted(chain_pool_picks.keys()):
        entry = chain_pool_picks[iid]
        inst = entry["inst"]
        error_content = (
            inst.get("error_content") or inst.get("description") or ""
        ).strip()
        lines.append(f"Instance #{iid}: {error_content}")
        for ts in entry["trigger_steps"]:
            step = ts["step"]
            tags = ", ".join(ts["tags"]) if ts["tags"] else "?"
            wq = ts.get("wrong_quote", "")
            if wq and len(wq) > _WQ_CLIP:
                wq = wq[:_WQ_CLIP] + "..."
            if wq:
                lines.append(f"  - step {step} [{tags}]: {wq}")
            else:
                lines.append(f"  - step {step} [{tags}]")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_clean_trajectory(
    flat_steps: List[Dict[str, Any]],
    step_pool: Dict[int, Dict[str, str]],
    per_step_chars: int = 800,
) -> str:
    """Render trajectory WITHOUT any trigger annotations (ablation-style).

    Mirrors ablation_direct_pick_from_stageb._render_trajectory_block.
    """
    from utils.llm_compression import clip_text_middle  # local import to avoid circular
    parts: List[str] = []
    for fs in flat_steps:
        step = fs.get("step", "?")
        role = fs.get("role", "assistant")
        body = ""
        pool = step_pool.get(step) if step_pool and isinstance(step, int) else None
        if pool:
            body = pool.get("th1") or pool.get("th2") or pool.get("th3") or ""
        if not body:
            body = str(fs.get("content", "") or "")
        body = clip_text_middle(body, per_step_chars).rstrip()
        # Avoid duplicate header when body already starts with "Step N [role:"
        header = f"Step {step} [role: {role}]:"
        if body.startswith(f"Step {step} [role:"):
            parts.append(body)
        else:
            parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def _render_suspicious_steps(
    chain_pool_picks: Dict[int, Dict[str, Any]],
) -> str:
    """Render suspicious steps in ablation format: '- step N: <wrong_quote>'.

    Aggregates all trigger steps across all chain-pool instances.
    Deduplicates by step. Sorted by step number.
    Mirrors ablation_direct_pick_from_stageb._render_stage_b_hints.
    """
    _WQ_CLIP = 240

    by_step: Dict[int, List[str]] = {}
    for iid in sorted(chain_pool_picks.keys()):
        for ts in chain_pool_picks[iid]["trigger_steps"]:
            step = ts["step"]
            wq = ts.get("wrong_quote", "").strip()
            if step not in by_step:
                by_step[step] = []
            if wq and wq not in by_step[step]:
                by_step[step].append(wq)

    if not by_step:
        return "(No suspicious steps were identified by the upstream pipeline. " \
               "Please identify the critical error step based on the full trajectory alone.)"

    lines: List[str] = []
    for step in sorted(by_step):
        quotes = by_step[step]
        if not quotes:
            lines.append(f"- step {step}: (flagged, no quote)")
        else:
            for q in quotes:
                qs = q if len(q) <= _WQ_CLIP else q[:_WQ_CLIP] + "..."
                lines.append(f"- step {step}: {qs}")
    return "\n".join(lines)


def _match_taxonomy_for_step(
    step: int,
    iid: int,
    chain_pool_picks: Dict[int, Dict[str, Any]],
) -> str:
    """Return the first taxonomy_tag for the given step in the given instance."""
    entry = chain_pool_picks.get(iid)
    if not entry:
        return ""
    for ts in entry["trigger_steps"]:
        if ts["step"] == step and ts.get("tags"):
            return ts["tags"][0]
    return ""


class Phase3Assembler:
    """Phase 3 assembly + optional empty-chain rescue + LLM critical pick."""

    def __init__(self, api_config: Optional[Dict[str, Any]] = None):
        self.api_config = api_config
        self.model: Optional[APIModel] = None
        if api_config and api_config.get("cache_url"):
            self.model = APIModel(
                api_config["cache_url"],
                api_config["base_url"],
                api_config["model"],
                api_config.get("api_key", "EMPTY"),
                extra_params=resolve_extra_params(api_config.get("model_profile")),
            )
        self.enable_rescue = bool(api_config.get("enable_rescue", False)) if api_config else False
        # LLM critical pick is on by default whenever a model is available.
        # Set api_config["enable_critical_pick"] = False to disable it
        # explicitly (useful for ablation studies).
        if api_config is None:
            self.enable_critical_pick = False
        else:
            self.enable_critical_pick = bool(
                api_config.get("enable_critical_pick", True)
            )
        # max_tokens for both rescue and critical-pick LLM calls.
        # Reasoning models (e.g. qwen3-thinking) burn many tokens on
        # internal CoT before emitting the final JSON, so 2048 is too
        # tight for long trajectories; surface this as a hyperparameter
        # so callers can match it to the Phase 1 / Phase 2 budgets.
        if api_config and api_config.get("max_tokens") is not None:
            try:
                self.max_tokens = int(api_config["max_tokens"])
            except (TypeError, ValueError):
                self.max_tokens = 2048
        else:
            self.max_tokens = 2048
        # Per-attempt wall-clock timeout (seconds) for any LLM call
        # made by Phase 3 (rescue / critical-pick). Defaults to 180s
        # to match Phase 1 / Phase 2.
        if api_config and api_config.get("llm_timeout_seconds") is not None:
            try:
                self.llm_timeout_seconds = float(
                    api_config["llm_timeout_seconds"]
                )
            except (TypeError, ValueError):
                self.llm_timeout_seconds = 180.0
        else:
            self.llm_timeout_seconds = 180.0
        # unified_source_strategy controls which result becomes the
        # canonical unified_critical_step:
        #   "llm"       — LLM pick overrides chain_min_v2 (default, v6 behaviour)
        #   "chain_min" — chain_min_v2 is always used; LLM pick is stored
        #                 in llm_unified_critical_step for audit only and
        #                 never overwrites unified_critical_step.
        self.unified_source_strategy: str = (
            (api_config.get("unified_source_strategy") or "llm")
            if api_config else "llm"
        )

    def close(self) -> None:
        if self.model:
            self.model.close()

    async def _rescue_call(
        self, phase2_payload: Dict[str, Any], trajectory_data: Optional[Dict[str, Any]],
        usage_acc: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """LLM fallback when no active instances found but trajectory failed."""
        if not self.model or not trajectory_data:
            return {"rescued_critical_step": None}

        step_pool = load_step_compressions_from_payload(phase2_payload)
        flat_steps = trajectory_data["flat_steps"]
        rendered = render_history_for_focus(
            flat_steps=flat_steps,
            focus_step=len(flat_steps),
            step_pool=step_pool,
            include_focus=True,
        )
        agent_fw = str(phase2_payload.get("agent_framework_description", "") or "")
        task_desc = str(phase2_payload.get("task_description", "") or "")

        user_content = f"=== TASK ===\n{task_desc}\n\n=== TRAJECTORY ===\n{rendered}"
        if agent_fw.strip():
            user_content = f"=== AGENT FRAMEWORK ===\n{agent_fw}\n\n{user_content}"

        messages = [
            {"role": "system", "content": RESCUE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response_format = {"type": "json_object"}

        def _try_call(max_tokens: int, temperature: float) -> Optional[tuple]:
            """Returns (parsed_dict_or_None, usage_dict_or_None)."""
            try:
                raw, usage = self.model.generate_chat(
                    messages, max_tokens, temperature, response_format,
                    return_usage=True,
                )
            except Exception as exc:
                logger.warning("Rescue LLM call failed: %s", exc)
                return None, None
            raw = strip_think_tags(raw or "")
            d = extract_last_json_object(raw, must_have_key="rescued_critical_step")
            if not isinstance(d, dict):
                d = extract_last_json_object(raw)
            return (d if isinstance(d, dict) else None), usage

        # 3-attempt retry policy with wall-clock timeout per attempt:
        #   attempt 1: temp=0.0,         max_tokens
        #   attempt 2: temp=0.3,         max_tokens * 1.5
        #   attempt 3: temp=0.3,         max_tokens * 1.5
        base_temp = 0.0
        attempts = [
            (self.max_tokens, base_temp),
            (int(self.max_tokens * 1.5), min(1.0, base_temp + 0.3)),
            (int(self.max_tokens * 1.5), min(1.0, base_temp + 0.6)),
        ]
        per_attempt_timeout = float(self.llm_timeout_seconds)

        data: Optional[dict] = None
        for attempt_idx, (mt, temp) in enumerate(attempts, start=1):
            try:
                call_result = await asyncio.wait_for(
                    asyncio.to_thread(_try_call, mt, temp),
                    timeout=per_attempt_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Rescue attempt %d timed out after %.0fs "
                    "(max_tokens=%d, temp=%.2f).",
                    attempt_idx, per_attempt_timeout, mt, temp,
                )
                call_result = (None, None)
            if call_result is not None:
                data, usage = call_result
                if usage and usage_acc is not None:
                    usage_acc["input_tokens"] += usage.get("input_tokens", 0)
                    usage_acc["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
                    usage_acc["output_tokens"] += usage.get("output_tokens", 0)
            else:
                data = None
            if isinstance(data, dict) and data.get("rescued_critical_step") is not None:
                return data
            if attempt_idx < len(attempts):
                next_mt, next_temp = attempts[attempt_idx]
                logger.warning(
                    "Rescue attempt %d failed; retrying with "
                    "max_tokens=%d, temp=%.2f", attempt_idx, next_mt, next_temp,
                )

        return {"rescued_critical_step": None}

    async def _pick_critical_call(
        self,
        merged_instances: List[Dict[str, Any]],
        triggers: List[Dict[str, Any]],
        phase2_payload: Dict[str, Any],
        trajectory_data: Optional[Dict[str, Any]],
        usage_acc: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Design B: single unified LLM pick of (instance_id, step).

        The LLM sees the full trajectory with chain-pool triggers inlined,
        an instance summary, and a LEGAL PICKS block listing each chain-pool
        instance's trigger_steps with their taxonomy tags.

        Returns a dict with keys:
          supporting_instance_id, critical_step, taxonomy_tag,
          failure_mode, rationale, tag_reason, fix_suggestion,
          _legal_picks (serialisable form of the pick map)
        or None on LLM/parse failure (caller falls back to chain_min_v2).
        """
        if not self.model or not trajectory_data:
            return None

        # Build chain pool legal picks map (may be empty if no instances/triggers)
        chain_pool_picks = _build_chain_pool_legal_picks(
            merged_instances or [], triggers or []
        )

        step_pool = load_step_compressions_from_payload(phase2_payload)
        flat_steps = trajectory_data["flat_steps"]

        # Clean trajectory (no trigger annotations) — ablation style
        rendered_traj = _render_clean_trajectory(flat_steps, step_pool)
        # Suspicious steps block — fallback message if pool is empty
        rendered_suspicious = (
            _render_suspicious_steps(chain_pool_picks)
            if chain_pool_picks
            else "(No suspicious steps were identified by the upstream pipeline. "
                 "Please identify the critical error step based on the full trajectory alone.)"
        )

        agent_fw = str(phase2_payload.get("agent_framework_description", "") or "")
        task_desc = str(phase2_payload.get("task_description", "") or "")
        last_step = flat_steps[-1].get("step", 0) if flat_steps else 0

        user_parts: List[str] = []
        # preamble goes first — exactly as ablation build_direct_pick_prompt
        user_parts.append(PICK_CRITICAL_USER_PREAMBLE.rstrip())
        user_parts.append(f"=== TASK ===\n{task_desc}")
        user_parts.append(
            f"TRAJECTORY OUTCOME: FAILED (reward=0, reported by the task environment).\n"
            f"LAST STEP INDEX: {last_step}"
        )
        user_parts.append(f"=== FULL TRAJECTORY (one entry per message) ===\n{rendered_traj}")
        user_parts.append(
            f"=== SUSPICIOUS STEPS (upstream hints; step number + quoted offending content) ===\n"
            f"{rendered_suspicious}"
        )
        user_parts.append(
            "=== WHAT TO OUTPUT ===\n"
            "Output ONE JSON object with these keys and nothing else:\n"
            "- \"critical_step\": <int>  — the step number of the critical error.\n"
            "- \"rationale\": <string, 1-3 sentences> explaining why this step is critical.\n\n"
            "=== REQUIRED OUTPUT FORMAT (single JSON object, no prose around it) ===\n"
            "{\n"
            "  \"critical_step\": <int>,\n"
            "  \"rationale\": \"<1-3 sentences>\"\n"
            "}\n\n"
            "Return ONLY the JSON object."
        )
        user_content = "\n\n".join(user_parts)

        messages = [
            {"role": "system", "content": PICK_CRITICAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response_format = {"type": "json_object"}

        def _coerce_int(v: Any) -> Optional[int]:
            if v is None:
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        def _try_call(max_tokens: int, temperature: float) -> Optional[tuple]:
            """Single LLM call attempt; returns (parsed_dict_or_None, usage_dict_or_None)."""
            try:
                raw, usage = self.model.generate_chat(
                    messages, max_tokens, temperature, response_format,
                    return_usage=True,
                )
            except Exception as exc:
                logger.warning("Critical-pick LLM call failed: %s", exc)
                return None, None
            raw = strip_think_tags(raw or "")
            d = extract_last_json_object(raw, must_have_key="critical_step")
            if not isinstance(d, dict):
                d = extract_last_json_object(raw)
            return (d if isinstance(d, dict) else None), usage

        # 3-attempt retry policy with wall-clock timeout per attempt:
        #   attempt 1: base_temp,        max_tokens
        #   attempt 2: base_temp + 0.3,  max_tokens * 1.5
        #   attempt 3: base_temp + 0.3,  max_tokens * 1.5
        base_temp = 0.0
        attempts = [
            (self.max_tokens, base_temp),
            (int(self.max_tokens * 1.5), min(1.0, base_temp + 0.3)),
            (int(self.max_tokens * 1.5), min(1.0, base_temp + 0.3)),
        ]
        per_attempt_timeout = float(self.llm_timeout_seconds)

        data: Optional[dict] = None
        for attempt_idx, (mt, temp) in enumerate(attempts, start=1):
            try:
                call_result = await asyncio.wait_for(
                    asyncio.to_thread(_try_call, mt, temp),
                    timeout=per_attempt_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Critical-pick attempt %d timed out after %.0fs "
                    "(max_tokens=%d, temp=%.2f).",
                    attempt_idx, per_attempt_timeout, mt, temp,
                )
                call_result = (None, None)
            if call_result is not None:
                data, usage = call_result
                if usage and usage_acc is not None:
                    usage_acc["input_tokens"] += usage.get("input_tokens", 0)
                    usage_acc["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
                    usage_acc["output_tokens"] += usage.get("output_tokens", 0)
            else:
                data = None
            if isinstance(data, dict):
                break
            if attempt_idx < len(attempts):
                next_mt, next_temp = attempts[attempt_idx]
                logger.warning(
                    "Critical-pick attempt %d failed; retrying with "
                    "max_tokens=%d, temp=%.2f", attempt_idx, next_mt, next_temp,
                )

        if not isinstance(data, dict):
            logger.warning(
                "Critical-pick LLM response could not be parsed after "
                "3 retries; falling back."
            )
            return None

        step = _coerce_int(data.get("critical_step"))

        if step is None:
            logger.warning("Critical-pick LLM returned no critical_step after retry; falling back.")
            return None

        # Build all-legal-steps map: step -> instance_id (chain pool only)
        all_legal_steps: Dict[int, int] = {}
        for k, v in chain_pool_picks.items():
            for ts in v["trigger_steps"]:
                all_legal_steps[ts["step"]] = k

        # Post-processing: match step to instance and taxonomy_tag.
        # If LLM picked a step outside the pool, log a warning but still
        # return the pick (ablation allows picking any step from the trajectory).
        if step in all_legal_steps:
            iid = all_legal_steps[step]
            tag = _match_taxonomy_for_step(step, iid, chain_pool_picks)
        else:
            logger.warning(
                "Critical-pick LLM returned critical_step=%s outside chain-pool "
                "steps %s; keeping pick, iid/tag will be null.",
                step, sorted(all_legal_steps),
            )
            iid = None
            tag = ""

        # Build serialisable audit
        legal_picks_audit = {
            str(k): [ts["step"] for ts in v["trigger_steps"]]
            for k, v in chain_pool_picks.items()
        }

        return {
            "supporting_instance_id": iid,
            "critical_step": step,
            "taxonomy_tag": tag,
            "rationale": str(data.get("rationale") or ""),
            "why_wrong": str(data.get("why_wrong") or ""),
            "fix_suggestion": str(data.get("fix_suggestion") or ""),
            "failure_mode": str(data.get("failure_mode") or ""),
            "tag_reason": str(data.get("tag_reason") or ""),
            "_legal_picks": legal_picks_audit,
        }

    async def _apply_llm_critical_pick(
        self,
        result: Dict[str, Any],
        phase2_payload: Dict[str, Any],
        trajectory_data: Optional[Dict[str, Any]],
        usage_acc: Optional[Dict[str, int]] = None,
    ) -> None:
        """Design B: LLM unified pick, with strategy-controlled override.

        unified_source_strategy == "llm" (default):
          The LLM's (instance_id, step) becomes the canonical answer:
            critical_error.unified_critical_step
            critical_error.chain_min_unified_critical_step  (pre-override
              snapshot is kept under chain_min_unified_critical_step_pre_llm)
            critical_error.unified_critical_taxonomy_tag
            critical_error.unified_source = "llm_pick"

        unified_source_strategy == "chain_min":
          chain_min_v2 result is kept as the canonical answer.
          The LLM pick is stored for audit only:
            critical_error.llm_unified_critical_step
            critical_error.llm_pick_*
          unified_critical_step / unified_source are NOT changed.

        On LLM failure / invalid pick, chain_min_v2 decision is kept
        unchanged and llm_pick_status is set to "unavailable" or
        "null_or_invalid".
        """
        if not self.model or not trajectory_data:
            return
        critical_error = result.get("critical_error") or {}
        if not critical_error:
            return

        merged = result.get("full_instance_states") or []
        triggers = phase2_payload.get("step_triggers") or []

        pick = await self._pick_critical_call(
            merged_instances=merged,
            triggers=triggers,
            phase2_payload=phase2_payload,
            trajectory_data=trajectory_data,
            usage_acc=usage_acc,
        )

        if pick is None:
            critical_error.setdefault("llm_pick_status", "unavailable")
            result["critical_error"] = critical_error
            return

        iid = pick["supporting_instance_id"]
        step = pick["critical_step"]
        tag = pick.get("taxonomy_tag") or None

        # Derive channel from the chosen instance
        inst_dict = next(
            (m for m in merged if int(m.get("instance_id", -1)) == iid),
            None,
        )

        # Full pick audit fields (always written regardless of strategy)
        critical_error["llm_pick_supporting_instance_id"] = iid
        critical_error["llm_pick_step"] = step
        critical_error["llm_pick_taxonomy_tag"] = tag
        critical_error["llm_pick_why_wrong"] = pick.get("why_wrong", "")
        critical_error["llm_pick_fix_suggestion"] = pick.get("fix_suggestion", "")
        # legacy fields kept for backward compat with eval scripts
        critical_error["llm_pick_failure_mode"] = pick.get("failure_mode")
        critical_error["llm_pick_rationale"] = pick.get("rationale", "")
        critical_error["llm_pick_tag_reason"] = pick.get("tag_reason", "")
        critical_error["llm_pick_legal_picks"] = pick.get("_legal_picks", {})
        # Legacy fields for backward compat with eval scripts
        critical_error["llm_unified_critical_step"] = step
        critical_error["llm_agent_critical_step"] = (
            step if (inst_dict and inst_dict.get("attribution") == "agent") else None
        )
        critical_error["llm_env_critical_step"] = (
            step if (inst_dict and inst_dict.get("attribution") == "env") else None
        )

        if self.unified_source_strategy == "chain_min":
            # chain_min strategy: LLM result is audit-only; do NOT touch
            # unified_critical_step / unified_source / unified_channel.
            logger.info(
                "unified_source_strategy=chain_min: LLM pick (step=%s) stored "
                "for audit only; keeping chain_min_v2 result (step=%s).",
                step,
                critical_error.get("unified_critical_step"),
            )
        else:
            # llm strategy (default): LLM pick overrides unified_critical_step only.
            # chain_min_unified_critical_step is the deterministic result and must NOT be overwritten.
            critical_error["unified_critical_step"] = step
            critical_error["unified_critical_taxonomy_tag"] = tag
            critical_error["unified_source"] = "llm_pick"
            if inst_dict:
                critical_error["unified_channel"] = inst_dict.get("attribution")

        result["critical_error"] = critical_error

    async def assemble_file(
        self,
        phase2_file: str,
        output_dir: str,
        trajectory_file: Optional[str] = None,
        top_k: int = 5,
    ) -> Optional[Dict[str, Any]]:
        try:
            with open(phase2_file, "r", encoding="utf-8") as fh:
                phase2_payload = json.load(fh)
        except Exception as exc:
            logger.error("Failed to load Phase 2 file %s: %s", phase2_file, exc)
            return None

        if phase2_payload.get("task_success"):
            result: Dict[str, Any] = {
                "task_id": phase2_payload.get("task_id"),
                "task_description": phase2_payload.get("task_description"),
                "task_success": True,
                "task_outcome": "success",
                "reward": phase2_payload.get("reward"),
                "environment": phase2_payload.get("environment"),
                "total_steps": phase2_payload.get("total_steps"),
                "trajectory_source": phase2_payload.get("trajectory_source"),
                "trajectory_file_path": phase2_payload.get("trajectory_file_path") or "",
                "metadata": phase2_payload.get("metadata") or {},
                "critical_error": {
                    "unified_critical_step": None,
                    "unified_source": None,
                    "unified_channel": None,
                    "agent_critical_step": None,
                    "env_critical_step": None,
                    "chain_min_unified_critical_step": None,
                    "chain_min_agent_critical_step": None,
                    "chain_min_env_critical_step": None,
                    "empty_chain_fallback_used": False,
                    "llm_unified_critical_step": None,
                    "llm_agent_critical_step": None,
                    "llm_env_critical_step": None,
                },
                "top_k_candidates": [],
                "chain_summary": {
                    "num_agent_chain": 0,
                    "num_env_chain": 0,
                    "num_agent_active": 0,
                    "num_env_active": 0,
                    "num_fixed": 0,
                    "num_dormant": 0,
                    "num_span_cap": 0,
                    "num_exploration_suppressed": 0,
                    "num_parse_failed": 0,
                    "parse_failed_share": 0.0,
                    "num_total_instances": 0,
                },
                "token_usage": {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0},
            }
        else:
            result = assemble_final(phase2_payload, top_k=top_k)

            # Per-file token usage accumulator for Phase 3 LLM calls
            usage_acc: Dict[str, int] = {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0}

            # Pre-load trajectory once (used by both critical-pick and rescue).
            trajectory_data = None
            if trajectory_file:
                try:
                    trajectory_data = load_unified_for_stage(trajectory_file)
                except Exception:
                    pass

            # LLM critical pick: rank ACTIVE instances within each
            # channel (agent / env) by impact-on-outcome. Triggered
            # only when a channel has >= 2 active instances. Falls
            # back silently to the existing _chain_min decision.
            if self.enable_critical_pick and self.model is not None:
                await self._apply_llm_critical_pick(
                    result, phase2_payload, trajectory_data,
                    usage_acc=usage_acc,
                )

            # Empty-chain rescue
            if (
                self.enable_rescue
                and result["critical_error"]["unified_critical_step"] is None
            ):
                rescue = await self._rescue_call(phase2_payload, trajectory_data, usage_acc=usage_acc)
                if rescue.get("rescued_critical_step") is not None:
                    result["critical_error"]["unified_critical_step"] = int(rescue["rescued_critical_step"])
                    result["critical_error"]["unified_source"] = "rescue"
                    result["critical_error"]["rescue_explanation"] = rescue.get("rescued_explanation", "")
                    result["critical_error"]["rescue_category"] = rescue.get("rescued_category")

            result["token_usage"] = usage_acc

        stem = Path(phase2_file).stem.replace("_stage_c_phase2", "")
        out_fp = Path(output_dir) / f"{stem}_final.json"
        with out_fp.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        cs = result.get("critical_error", {}).get("unified_critical_step")
        token_usage = result.get("token_usage") or {}
        logger.info(
            "Phase 3 [%s]: unified_critical_step=%s (tokens: in=%d, reason=%d, out=%d)",
            stem, cs,
            token_usage.get("input_tokens", 0),
            token_usage.get("reasoning_tokens", 0),
            token_usage.get("output_tokens", 0),
        )
        return result


# =====================================================================
# Batch runner
# =====================================================================


def _is_valid_final_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    return isinstance(data, dict) and "critical_error" in data


def _ensure_chat_completions_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


_PHASE2_SUFFIX = "_stage_c_phase2.json"


def _collect_phase2_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return [str(x) for x in sorted(p.rglob(f"*{_PHASE2_SUFFIX}"))]
    raise FileNotFoundError(f"Path not found: {path}")


def _index_trajectories_by_stem(trajectory_dir: str) -> Dict[str, str]:
    files = collect_trajectory_sources(trajectory_dir)
    return {output_stem_for_source(fp): fp for fp in files}


def _guess_trajectory_for_phase2(
    phase2_file: str,
    trajectory_index: Dict[str, str],
) -> Optional[str]:
    try:
        with open(phase2_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ts = payload.get("trajectory_source")
        if isinstance(ts, str) and ts.strip():
            return ts
    except Exception:
        pass
    stem = Path(phase2_file).stem.replace("_stage_c_phase2", "")
    if stem in trajectory_index:
        return trajectory_index[stem]
    for key, val in trajectory_index.items():
        if key.startswith(stem) or stem.startswith(key):
            return val
    return None


async def run_batch(
    phase2_inputs: List[str],
    trajectory_dir: Optional[str],
    output_dir: str,
    api_config: Optional[Dict[str, Any]],
    top_k: int = 5,
    concurrency: int = 8,
    resume: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    if api_config:
        api_config = dict(api_config)
        if api_config.get("base_url"):
            api_config["base_url"] = _ensure_chat_completions_url(api_config["base_url"])

    assembler = Phase3Assembler(api_config)
    trajectory_index = _index_trajectories_by_stem(trajectory_dir) if trajectory_dir else {}

    file_sem = asyncio.Semaphore(max(1, int(concurrency)))
    total = len(phase2_inputs)

    async def _process_one(idx: int, p2: str) -> Tuple[str, str, Optional[str]]:
        stem = Path(p2).stem.replace("_stage_c_phase2", "")
        out_fp = Path(output_dir) / f"{stem}_final.json"
        if overwrite:
            pass
        elif resume and _is_valid_final_output(out_fp):
            return ("skipped", p2, f"valid cached: {out_fp.name}")
        traj = _guess_trajectory_for_phase2(p2, trajectory_index) if trajectory_index else None
        async with file_sem:
            logger.info("Phase 3 (%d/%d): %s", idx, total, p2)
            try:
                r = await assembler.assemble_file(p2, output_dir, trajectory_file=traj, top_k=top_k)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return ("failed", p2, repr(exc))
        if r is None:
            return ("failed", p2, "assemble_file returned None")
        return ("ok", p2, None)

    ok = failed = 0
    skipped_list: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    try:
        async_tasks = [_process_one(i, p2) for i, p2 in enumerate(phase2_inputs, start=1)]
        results = await asyncio.gather(*async_tasks)
        for status, p2, info in results:
            if status == "ok":
                ok += 1
            elif status == "failed":
                failed += 1
                failures.append({"phase2_file": p2, "error": info or ""})
            else:
                skipped_list.append({"phase2_file": p2, "reason": info or ""})
    finally:
        assembler.close()

    summary = {
        "output_dir": output_dir,
        "num_inputs": len(phase2_inputs),
        "num_ok": ok,
        "num_failed": failed,
        "num_skipped": len(skipped_list),
        "concurrency": int(concurrency),
        "resume": resume,
        "overwrite": overwrite,
        "skipped": skipped_list[:20],
        "failures": failures[:20],
    }
    logger.info("Phase 3 batch done: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage C Phase 3 (v5): Mechanical assembly + empty-chain rescue",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase2_file", help="Single Phase 2 result json")
    group.add_argument("--phase2_dir", help="Directory of Phase 2 results")

    parser.add_argument("--trajectory_dir", help="Directory of unified trajectories (for rescue)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    # Rescue-related (optional)
    parser.add_argument("--enable_rescue", action="store_true")
    parser.add_argument(
        "--disable_critical_pick",
        action="store_true",
        help=(
            "Disable the LLM critical-instance picker (Phase 3). When "
            "disabled, the unified/agent/env critical step is decided "
            "purely by the deterministic _chain_min rule (earliest "
            "origin_step). Defaults to ENABLED whenever LLM "
            "credentials (--api_key / --base_url / --model / --cache) "
            "are provided."
        ),
    )
    parser.add_argument("--api_key", default=os.getenv("API_KEY"))
    parser.add_argument("--base_url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--cache", default="")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help=(
            "max_tokens for the rescue / critical-pick LLM calls. "
            "Reasoning models need a much larger budget than the "
            "default 2048 because internal CoT consumes most of the "
            "tokens before any JSON is emitted; aligning this with "
            "PHASE1_MAX_TOKENS / PHASE2_MAX_TOKENS (32768) is a safe "
            "choice."
        ),
    )
    parser.add_argument(
        "--model_profile",
        default=None,
        help=(
            "Optional model profile name from utils/model_profiles.py. "
            "Consulted whenever an LLM call may be issued (rescue or "
            "critical-pick). Forwards the profile's `params` "
            "(excluding temperature / max_tokens) to the LLM call as "
            "extra params."
        ),
    )
    parser.add_argument(
        "--unified_source_strategy",
        default="llm",
        choices=["llm", "chain_min"],
        help=(
            "Controls which result becomes the canonical "
            "unified_critical_step. "
            "'llm' (default): the LLM critical-pick result overrides "
            "chain_min_v2 when available. "
            "'chain_min': chain_min_v2 is always the canonical answer; "
            "the LLM pick is stored in llm_unified_critical_step for "
            "audit only and never overwrites unified_critical_step."
        ),
    )
    parser.add_argument(
        "--llm_timeout_seconds",
        type=float,
        default=180.0,
        help=(
            "Per-attempt wall-clock timeout (seconds) for any Phase 3 "
            "LLM call (rescue / critical-pick). Each call is retried up "
            "to 3 times; if every retry exceeds this timeout the call "
            "falls back gracefully (chain_min_v2 result is kept)."
        ),
    )

    args = parser.parse_args()

    api_config: Optional[Dict[str, Any]] = None
    has_llm_credentials = bool(
        args.api_key and args.base_url and args.model and args.cache
    )
    enable_critical_pick = (
        has_llm_credentials and not args.disable_critical_pick
    )
    if args.enable_rescue or enable_critical_pick:
        if not has_llm_credentials:
            raise ValueError(
                "LLM features require --api_key, --base_url, --model, "
                "--cache."
            )
        api_config = {
            "api_key": args.api_key,
            "base_url": args.base_url,
            "model": args.model,
            "cache_url": args.cache,
            "enable_rescue": bool(args.enable_rescue),
            "enable_critical_pick": enable_critical_pick,
            "model_profile": args.model_profile,
            "max_tokens": int(args.max_tokens),
            "unified_source_strategy": args.unified_source_strategy,
            "llm_timeout_seconds": float(args.llm_timeout_seconds),
        }

    if args.phase2_file:
        inputs = [args.phase2_file]
    else:
        inputs = _collect_phase2_files(args.phase2_dir)

    summary = asyncio.run(
        run_batch(
            phase2_inputs=inputs,
            trajectory_dir=args.trajectory_dir,
            output_dir=args.output_dir,
            api_config=api_config,
            top_k=args.top_k,
            concurrency=args.concurrency,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()