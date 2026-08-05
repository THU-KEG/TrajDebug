#!/usr/bin/env python3
"""Diagnostic breakdown on top of ``score_steps.py`` for the 3-stage
pipeline.

``score_steps.py`` reports aggregate step / module / full-type accuracy
but does not explain *why* a trajectory was scored wrong. For a given
dataset we want to know whether the misses come from:

* ``stage_b_miss`` — Stage B never flagged the ground-truth step
  (``has_wrong_content=false`` AND ``on_error_chain=false`` at GT), so
  Stage C had no candidate to pick.
* ``stage_c_too_early`` — Stage B flagged the GT step (or Stage A's
  ``error_phase`` covered it) and Stage C picked a step earlier than GT
  (typical exploration over-flagging).
* ``stage_c_too_late`` — Stage B flagged the GT step and Stage C picked
  a later step (typical InfiniteRetry-tail / signal-acknowledgement
  selection — Case 1 in the closest-to-correction rule may move picks
  later by design; this bucket lets you tell whether that movement was
  too aggressive).
* ``stage_c_offgrid`` — Stage C picked a step that Stage B never flagged
  AND Stage A's ``error_phase`` did not cover, OR emitted
  ``unknown.unknown`` (Stage B was silent).
* ``correct`` — Stage C's step matches GT.

In addition, this script now performs **fine-grained error analysis**
across the three downstream stages to pinpoint where the pipeline
diverges from ground truth:

Each metric is gated on the upstream stage actually fingerprinting GT,
so the denominators reflect "trajectories where this stage even had a
chance to act on GT". Concretely:

1. **stage_b_gt_missed** — Trajectories whose Stage B ``step_triggers``
   contain zero entries with ``step == gt_step``.
   Denominator: trajectories whose Stage B file is available.

2. **phase1_gt_not_at_origin** — Trajectories where no Phase 1
   ``instance`` has ``origin_step == gt_step``.
   Denominator: Phase 1 file available AND Stage B fired ≥1 trigger on
   the GT step. (If Stage B never flagged GT, Phase 1 cannot cluster GT
   anywhere; counting it would inflate the ratio.)

3. **phase2_gt_fixed / phase2_gt_dormant** (strict) — Trajectories where
   the Phase 2 ``instance_state`` whose Phase 1 origin equals GT has
   ``fix_status`` matching ``fixed_at_step_<N>`` (fixed) or ``state ==
   "dormant"`` (dormant).
   Denominator: Phase 2 file available AND Phase 1 placed a GT-origin
   instance.

4. **phase2_gt_rescued_by_span_cap** — Among the strictly-fixed-or-dormant
   GT instances, how many also have ``span_exceeds_cap == True`` and are
   therefore re-pulled into Phase 3's active / regular candidate pool
   (the strict counts above thus over-state how many GTs are truly lost
   at Phase 2). Denominator: same as Metric 3.

5. **phase3_gt_in_selected** — Trajectories where Phase 3's selected
   instance (matched by ``unified_critical_step`` + ``unified_source``)
   has ``trigger_indices`` resolving to a step set that contains GT.
   Denominator: Phase 3 file available AND Phase 1 placed a GT-origin
   instance.

The legacy ``phase1_hit_but_all_overwritten`` bucket is gone (Phase 1.5
is no longer part of the pipeline). For backward compatibility the
``--phase15-dir`` CLI flag is still accepted; it is treated as a Stage B
directory.

The script also dumps a pred-minus-gt offset histogram and, for datasets
whose ground-truth ``critical_error_type`` is non-null, a lightweight
module/subtype confusion matrix.

Usage::

    python detector/score_steps_breakdown.py \\
        --unified-dir data/unified/webshopsub \\
        --pred-dir    output-XXX/webshopsub_final \\
        --stage-b-dir output-XXX/webshopsub_stage_b \\
        --phase1-dir  output-XXX/webshopsub_phase1 \\
        --phase2-dir  output-XXX/webshopsub_phase2 \\
        --out         output-XXX/breakdown_webshop.json

``--dump-mismatches`` adds per-trajectory records (GT vs pred, Stage B
flagged steps + on-error-chain steps, failure category) so we can re-read
specific cases.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from score_steps import (
    iter_json_files,
    iter_unified_labels,
    load_json,
    load_pred_index,
)


# ---------- Stage B per-step flagged loading ----------


# We accept Stage B outputs (the new naming) AND the legacy Phase 1 /
# 1.5 outputs so cross-version comparison runs cleanly through the same
# script.
_STAGE_B_SUFFIXES = (
    "_stage_b",
    "_error_detection_with_recovery",
    "_error_detection",
)


def _strip_stage_b_suffix(stem: str) -> str:
    for suffix in _STAGE_B_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _extract_flagged_steps(stage_b_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Return ``{step_index: {has_wrong_content, on_error_chain,
    triggers_count}}`` for every step in a Stage B / Phase 1 file.

    Supports two output formats:

    * **New format** (v5 pipeline): Stage B emits ``step_triggers``, a
      flat list of trigger dicts each with ``step``, ``category``,
      ``attribution``, etc. Every step that appears in this list is
      considered flagged (``has_wrong_content=True``,
      ``on_error_chain=True``).
    * **Legacy format** (v0430 pipeline): Stage B emits
      ``step_analyses`` per step with ``wrong_content.has_wrong_content``
      and ``on_error_chain`` fields.

    Legacy Phase 1 / Phase 1.5 outputs lack ``on_error_chain``; we
    treat ``has_wrong_content`` as the on-chain proxy in that case.
    """
    out: Dict[int, Dict[str, Any]] = {}
    if not isinstance(stage_b_data, dict):
        return out

    # --- New format: step_triggers (flat list) ---
    step_triggers = stage_b_data.get("step_triggers")
    if isinstance(step_triggers, list) and step_triggers:
        for trig in step_triggers:
            if not isinstance(trig, dict):
                continue
            step = trig.get("step")
            if not isinstance(step, int):
                continue
            if step not in out:
                out[step] = {
                    "has_wrong_content": True,
                    "on_error_chain": True,
                    "triggers_count": 0,
                }
            out[step]["triggers_count"] += 1
        if out:
            return out

    # --- Legacy format: step_analyses ---
    for entry in stage_b_data.get("step_analyses") or []:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        if not isinstance(step, int):
            continue

        wc = entry.get("wrong_content") or {}
        triggers = wc.get("triggers") or []
        has_wrong_content = bool(wc.get("has_wrong_content")) and bool(triggers)
        triggers_count = len(triggers) if isinstance(triggers, list) else 0
        on_error_chain = entry.get("on_error_chain")
        if isinstance(on_error_chain, bool):
            on_chain = on_error_chain
        else:
            # Legacy Phase 1 has no on_error_chain field; fall back to
            # has_wrong_content so the breakdown still makes sense.
            on_chain = has_wrong_content
        out[step] = {
            "has_wrong_content": has_wrong_content,
            "on_error_chain": on_chain,
            "triggers_count": triggers_count,
        }
    return out


def _extract_error_phase(stage_b_data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Pull Stage A's ``error_phase`` out of the Stage B output (it is
    echoed into ``stage_a``). Returns ``None`` for legacy files.
    """
    if not isinstance(stage_b_data, dict):
        return None
    sa = stage_b_data.get("stage_a")
    if not isinstance(sa, dict):
        return None
    ep = sa.get("error_phase")
    if not (isinstance(ep, list) and len(ep) == 2):
        return None
    try:
        return int(ep[0]), int(ep[1])
    except Exception:
        return None


def load_stage_b_index(
    stage_b_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Index Stage B (or legacy Phase 1 / 1.5) outputs by trajectory
    stem. Each value holds the per-step flagged map AND the Stage A
    ``error_phase`` (None for legacy files).
    """
    index: Dict[str, Dict[str, Any]] = {}
    for fp in iter_json_files(stage_b_dir):
        try:
            data = load_json(fp)
        except Exception:
            continue
        stem = _strip_stage_b_suffix(Path(fp).stem)
        index[stem] = {
            "flagged": _extract_flagged_steps(data),
            "error_phase": _extract_error_phase(data),
            # Also extract raw step_triggers for the new error analysis
            "step_triggers": data.get("step_triggers") or [],
        }
    return index


# ---------- Phase 1 (Stage C clustering) loading ----------

_PHASE1_SUFFIXES = ("_stage_c_phase1",)


def _strip_phase1_suffix(stem: str) -> str:
    for suffix in _PHASE1_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_phase1_index(phase1_dir: str) -> Dict[str, Dict[str, Any]]:
    """Index Phase 1 outputs by trajectory stem.

    Each value holds the ``instances`` list from the Phase 1 output.
    Each instance has ``origin_step``, ``trigger_indices``, etc.
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not phase1_dir:
        return index
    for fp in iter_json_files(phase1_dir):
        try:
            data = load_json(fp)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        stem = _strip_phase1_suffix(Path(fp).stem)
        index[stem] = {
            "instances": data.get("instances") or [],
            "step_triggers": data.get("step_triggers") or [],
        }
    return index


# ---------- Phase 2 (Stage C per-instance state) loading ----------

_PHASE2_SUFFIXES = ("_stage_c_phase2",)


def _strip_phase2_suffix(stem: str) -> str:
    for suffix in _PHASE2_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_phase2_index(phase2_dir: str) -> Dict[str, Dict[str, Any]]:
    """Index Phase 2 outputs by trajectory stem.

    Each value holds the ``instance_states`` list from the Phase 2 output.
    Each instance_state has ``origin_step``, ``fix_status``, ``state``, etc.
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not phase2_dir:
        return index
    for fp in iter_json_files(phase2_dir):
        try:
            data = load_json(fp)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        stem = _strip_phase2_suffix(Path(fp).stem)
        index[stem] = {
            "instance_states": data.get("instance_states") or [],
        }
    return index


# ---------- Phase 3 (Stage C final assembly) loading ----------

_PHASE3_SUFFIXES = (
    "_final",
    "_critical_error",
)


def _strip_phase3_suffix(stem: str) -> str:
    for suffix in _PHASE3_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_phase3_index(phase3_dir: str) -> Dict[str, Dict[str, Any]]:
    """Index Phase 3 final outputs (``*_final.json``) by trajectory stem.

    Each value captures ``critical_error`` (with ``unified_critical_step``
    and ``unified_source``), ``full_instance_states`` (which carry
    ``trigger_indices`` pointing into ``step_triggers``), and
    ``step_triggers`` (used to resolve each trigger index to its actual
    step number).
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not phase3_dir:
        return index
    for fp in iter_json_files(phase3_dir):
        try:
            data = load_json(fp)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        stem = _strip_phase3_suffix(Path(fp).stem)
        index[stem] = {
            "critical_error": data.get("critical_error") or {},
            "full_instance_states": data.get("full_instance_states") or [],
            "step_triggers": data.get("step_triggers") or [],
        }
    return index


# ---------- Failure classification ----------


def _pred_minus_gt_bucket(offset: int) -> str:
    """Bucket a signed pred-gt offset for histogram display."""
    if offset <= -10:
        return "<=-10"
    if offset <= -3:
        return "-9..-3"
    if offset <= 2:
        return "-2..2"
    if offset < 10:
        return "3..9"
    return ">=10"


def classify_trajectory(
    gt_step: int,
    pred_step: Optional[int],
    flagged: Dict[int, Dict[str, Any]],
    error_phase: Optional[Tuple[int, int]],
) -> str:
    """Classify *why* a trajectory's prediction diverges from GT.

    Returns one of: ``correct``, ``stage_b_miss``,
    ``stage_c_too_early``, ``stage_c_too_late``, ``stage_c_offgrid``.
    """
    if pred_step is None:
        return "stage_b_miss"

    if pred_step == gt_step:
        return "correct"

    # Was the GT step flagged by Stage B?
    gt_record = flagged.get(gt_step)
    gt_flagged = gt_record is not None and (
        gt_record.get("has_wrong_content") or gt_record.get("on_error_chain")
    )
    # Was the GT step covered by Stage A's error_phase?
    gt_covered_by_phase = False
    if error_phase is not None:
        gt_covered_by_phase = error_phase[0] <= gt_step <= error_phase[1]

    if gt_flagged or gt_covered_by_phase:
        # Stage B (or Stage A) flagged the GT step, but Stage C picked
        # a different step.
        if pred_step < gt_step:
            return "stage_c_too_early"
        return "stage_c_too_late"

    # GT was neither flagged by Stage B nor covered by Stage A.
    # Check whether the predicted step was even flagged.
    pred_record = flagged.get(pred_step)
    pred_flagged = pred_record is not None and (
        pred_record.get("has_wrong_content") or pred_record.get("on_error_chain")
    )
    if pred_flagged:
        return "stage_c_offgrid"

    # Stage B was silent on both GT and pred → miss
    return "stage_b_miss"





# ---------- Fine-grained error analysis ----------


def _gt_step_has_trigger_in_stage_b(
    step_triggers: List[Dict[str, Any]],
    gt_step: int,
) -> bool:
    """Return True iff Stage B's ``step_triggers`` contains at least one
    trigger whose ``step`` equals ``gt_step``.
    """
    for trig in step_triggers:
        if isinstance(trig, dict) and trig.get("step") == gt_step:
            return True
    return False





def _gt_at_origin_in_phase1(
    instances: List[Dict[str, Any]],
    gt_step: int,
) -> Tuple[bool, Optional[int]]:
    """Check whether Phase 1 produced an instance whose ``origin_step``
    equals ``gt_step``.

    Returns ``(is_at_origin, instance_id)`` where ``instance_id`` is the
    first matching instance's id (or None).
    """
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if inst.get("origin_step") == gt_step:
            return True, inst.get("instance_id")
    return False, None





def _gt_instance_state_in_phase2(
    instance_states: List[Dict[str, Any]],
    instances: List[Dict[str, Any]],
    gt_step: int,
) -> Optional[Dict[str, Any]]:
    """Find the instance_state whose corresponding Phase 1 instance has
    ``origin_step`` equal to ``gt_step`` and return its state info
    (``fix_status``, ``state``, etc.).

    Phase 2's ``instance_states`` have ``instance_id`` but no
    ``origin_step``, so we cross-reference with Phase 1's ``instances``
    list (which maps ``instance_id`` → ``origin_step``).

    Returns ``None`` if no such instance_state exists.
    """
    # Build a mapping from instance_id → origin_step from Phase 1
    id_to_origin: Dict[int, int] = {}
    for inst in instances:
        if isinstance(inst, dict):
            iid = inst.get("instance_id")
            origin = inst.get("origin_step")
            if isinstance(iid, int) and isinstance(origin, int):
                id_to_origin[iid] = origin

    for ist in instance_states:
        if not isinstance(ist, dict):
            continue
        iid = ist.get("instance_id")
        if isinstance(iid, int) and id_to_origin.get(iid) == gt_step:
            return ist
    return None


def _find_selected_phase3_instance(
    full_instance_states: List[Dict[str, Any]],
    critical_error: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Locate the single instance that Phase 3 picked as the final
    ``unified_critical_step``.

    Phase 3's rule: the chosen instance comes from the ``unified_source``
    chain (agent / env) and its ``origin_step`` equals
    ``unified_critical_step``. We match by (origin_step, attribution)
    when ``unified_source`` is available; otherwise by origin_step only.
    Returns ``None`` if no matching instance is found.
    """
    if not isinstance(critical_error, dict):
        return None
    selected_step = critical_error.get("unified_critical_step")
    if not isinstance(selected_step, int):
        return None
    unified_source = critical_error.get("unified_source")
    unified_source = (
        unified_source.strip().lower()
        if isinstance(unified_source, str) and unified_source.strip()
        else None
    )

    fallback: Optional[Dict[str, Any]] = None
    for inst in full_instance_states:
        if not isinstance(inst, dict):
            continue
        if inst.get("origin_step") != selected_step:
            continue
        attribution = inst.get("attribution")
        if unified_source is None:
            return inst
        if isinstance(attribution, str) and attribution.strip().lower() == unified_source:
            return inst
        if fallback is None:
            fallback = inst
    return fallback


def _instance_covers_step(
    selected_inst: Dict[str, Any],
    step_triggers: List[Dict[str, Any]],
    step: int,
) -> bool:
    """Return True iff the selected Phase 3 instance's ``trigger_indices``
    resolve (via ``step_triggers``) to a set of step numbers that
    contains ``step``.

    Note: ``trigger_indices`` are indices into ``step_triggers``, NOT
    step numbers themselves.
    """
    if not isinstance(selected_inst, dict):
        return False
    trigger_indices = selected_inst.get("trigger_indices") or []
    for idx in trigger_indices:
        if not isinstance(idx, int):
            continue
        if 0 <= idx < len(step_triggers):
            trig = step_triggers[idx]
            if isinstance(trig, dict) and trig.get("step") == step:
                return True
    return False


def _gt_in_selected_phase3_instance(
    phase3_entry: Dict[str, Any],
    gt_step: int,
) -> Tuple[bool, Optional[int], Optional[int]]:
    """Check whether ``gt_step`` is covered by the single instance that
    Phase 3 chose as its ``unified_critical_step``.

    Returns ``(gt_covered, selected_origin_step, selected_instance_id)``.
    """
    critical_error = phase3_entry.get("critical_error") or {}
    full_instance_states = phase3_entry.get("full_instance_states") or []
    step_triggers = phase3_entry.get("step_triggers") or []

    selected = _find_selected_phase3_instance(full_instance_states, critical_error)
    if selected is None:
        return False, critical_error.get("unified_critical_step"), None

    covered = _instance_covers_step(selected, step_triggers, gt_step)
    return covered, selected.get("origin_step"), selected.get("instance_id")


def analyze_pipeline_errors(
    labels: List[Tuple[str, Dict[str, Any]]],
    stage_b_index: Dict[str, Dict[str, Any]],
    phase1_index: Dict[str, Dict[str, Any]],
    phase2_index: Dict[str, Dict[str, Any]],
    phase3_index: Optional[Dict[str, Dict[str, Any]]] = None,
    dump_details: bool = False,
) -> Dict[str, Any]:
    """Fine-grained error analysis across Stage B → Phase 1 → Phase 2 → Phase 3.

    For every failure trajectory (reward=0, critical_step is not None),
    we trace the GT step through each pipeline stage and record where
    it was lost or misclassified.

    Returns a dict with per-stage counts and optional per-trajectory
    details.
    """
    if phase3_index is None:
        phase3_index = {}
    total = 0
    # --- Per-stage availability denominators ---
    # A trajectory only contributes to a stage's metric if that
    # stage's per-trajectory file exists. This avoids silently counting
    # missing files as "the GT was lost at that stage" (which used to
    # inflate e.g. phase1_gt_not_at_origin to 100% whenever
    # --phase1-dir was not supplied).
    stage_b_available = 0
    phase1_available = 0
    phase2_available = 0
    phase3_available = 0

    # Metric 1: Stage B missed GT (no trigger fired on GT step)
    stage_b_gt_missed = 0
    # Metric 2: Phase 1 did not place GT at origin_step of any instance
    phase1_gt_not_at_origin = 0
    # Metric 3a: Phase 2 marked the GT instance as fixed (strict)
    phase2_gt_fixed = 0
    # Metric 3b: Phase 2 marked the GT instance as dormant (strict)
    phase2_gt_dormant = 0
    # Metric 3c: GT instance was strictly fixed/dormant BUT also has
    # span_exceeds_cap=True, meaning Phase 3 re-pulls it into the
    # active/regular candidate pool. These are "rescued" — they should
    # not be counted as truly lost at Phase 2.
    phase2_gt_rescued_by_span_cap = 0
    # Metric 4: GT step is covered by the Phase 3 final-selected instance
    #           (i.e., gt_step appears in the steps resolved from that
    #           instance's trigger_indices via step_triggers).
    phase3_gt_in_selected = 0

    details: List[Dict[str, Any]] = []

    for stem, label in labels:
        if label["reward"] == 1 or label["critical_step"] is None:
            continue

        total += 1
        gt_step = label["critical_step"]

        # --- Metric 1: Stage B trigger coverage ---
        sb_entry = stage_b_index.get(stem)
        gt_has_sb_trigger: Optional[bool] = None
        if sb_entry is not None:
            stage_b_available += 1
            step_triggers_sb = sb_entry.get("step_triggers") or []
            gt_has_sb_trigger = _gt_step_has_trigger_in_stage_b(
                step_triggers_sb, gt_step
            )
            if not gt_has_sb_trigger:
                stage_b_gt_missed += 1

        # --- Metric 2: Phase 1 origin placement ---
        # Gate: only count when Phase 1 file is available AND Stage B
        # actually fired on the GT step (otherwise Phase 1 has no GT
        # trigger to cluster, so the metric is meaningless here).
        p1_entry = phase1_index.get(stem)
        instances: List[Dict[str, Any]] = []
        gt_at_origin: Optional[bool] = None
        gt_origin_inst_id: Optional[int] = None
        sb_fired_on_gt = bool(gt_has_sb_trigger)
        if p1_entry is not None and sb_fired_on_gt:
            phase1_available += 1
            instances = p1_entry.get("instances") or []
            gt_at_origin, gt_origin_inst_id = _gt_at_origin_in_phase1(
                instances, gt_step
            )
            if not gt_at_origin:
                phase1_gt_not_at_origin += 1
        elif p1_entry is not None:
            # Phase 1 file is loaded but Stage B never fired on GT —
            # we still need ``instances`` for downstream phase2 cross-
            # reference, but we do NOT count this trajectory toward the
            # phase1_not_at_origin metric.
            instances = p1_entry.get("instances") or []

        # --- Metric 3: Phase 2 fix_status / state (strict + rescued) ---
        # Gate: only count when Phase 2 file is available AND Phase 1
        # placed a GT-origin instance. Without a GT-origin instance,
        # Phase 2 has no fix/dormant verdict on GT to read.
        p2_entry = phase2_index.get(stem)
        gt_is_fixed = False
        gt_is_dormant = False
        gt_fix_status = None
        gt_state = None
        gt_span_exceeds_cap = False
        gt_rescued = False
        if p2_entry is not None and gt_at_origin:
            phase2_available += 1
            instance_states = p2_entry.get("instance_states") or []
            gt_ist = _gt_instance_state_in_phase2(
                instance_states, instances, gt_step
            )
            if gt_ist is not None:
                gt_fix_status = gt_ist.get("fix_status")
                gt_state = gt_ist.get("state")
                gt_span_exceeds_cap = gt_ist.get("span_exceeds_cap") is True
                # Strict fixed: fix_status matches the canonical
                # fixed_at_step_<N> shape produced by Phase 2.
                if (
                    isinstance(gt_fix_status, str)
                    and gt_fix_status.startswith("fixed_at_step_")
                ):
                    gt_is_fixed = True
                if gt_state == "dormant":
                    gt_is_dormant = True
                if gt_is_fixed:
                    phase2_gt_fixed += 1
                if gt_is_dormant:
                    phase2_gt_dormant += 1
                # Rescue: GT was strictly fixed/dormant but Phase 3
                # will re-pull it because span_exceeds_cap=True.
                if (gt_is_fixed or gt_is_dormant) and gt_span_exceeds_cap:
                    phase2_gt_rescued_by_span_cap += 1
                    gt_rescued = True

        # --- Metric 4: GT covered by the Phase 3 final-selected instance ---
        # Gate: only count when Phase 3 file is available AND Phase 1
        # placed a GT-origin instance (otherwise Phase 3 has no GT-
        # origin candidate to select; the trajectory is destined to
        # miss for an upstream reason already counted in Metric 2).
        p3_entry = phase3_index.get(stem)
        gt_in_selected: Optional[bool] = None
        selected_origin = None
        selected_iid = None
        if p3_entry is not None and gt_at_origin:
            phase3_available += 1
            gt_in_selected, selected_origin, selected_iid = (
                _gt_in_selected_phase3_instance(p3_entry, gt_step)
            )
            if gt_in_selected:
                phase3_gt_in_selected += 1

        if dump_details:
            details.append({
                "stem": stem,
                "task_id": label.get("task_id"),
                "gt_step": gt_step,
                "stage_b_gt_has_trigger": gt_has_sb_trigger,
                "phase1_gt_at_origin": gt_at_origin,
                "phase2_gt_fix_status": gt_fix_status,
                "phase2_gt_state": gt_state,
                "phase2_gt_span_exceeds_cap": gt_span_exceeds_cap,
                "phase2_gt_rescued_by_span_cap": gt_rescued,
                "phase3_selected_origin_step": selected_origin,
                "phase3_selected_instance_id": selected_iid,
                "phase3_gt_in_selected": gt_in_selected,
            })

    return {
        "trajectories_analyzed": total,
        "stage_b_available": stage_b_available,
        "phase1_available": phase1_available,
        "phase2_available": phase2_available,
        "phase3_available": phase3_available,
        "stage_b_gt_missed": stage_b_gt_missed,
        "stage_b_gt_missed_ratio": (
            stage_b_gt_missed / stage_b_available if stage_b_available else 0.0
        ),
        "phase1_gt_not_at_origin": phase1_gt_not_at_origin,
        "phase1_gt_not_at_origin_ratio": (
            phase1_gt_not_at_origin / phase1_available if phase1_available else 0.0
        ),
        "phase2_gt_fixed": phase2_gt_fixed,
        "phase2_gt_fixed_ratio": (
            phase2_gt_fixed / phase2_available if phase2_available else 0.0
        ),
        "phase2_gt_dormant": phase2_gt_dormant,
        "phase2_gt_dormant_ratio": (
            phase2_gt_dormant / phase2_available if phase2_available else 0.0
        ),
        "phase2_gt_rescued_by_span_cap": phase2_gt_rescued_by_span_cap,
        "phase2_gt_rescued_by_span_cap_ratio": (
            phase2_gt_rescued_by_span_cap / phase2_available if phase2_available else 0.0
        ),
        "phase3_gt_in_selected": phase3_gt_in_selected,
        "phase3_gt_in_selected_ratio": (
            phase3_gt_in_selected / phase3_available if phase3_available else 0.0
        ),
        "details": details,
    }


# ---------- Evaluate one directory ----------



def evaluate_breakdown(
    labels: List[Tuple[str, Dict[str, Any]]],
    pred_index: Dict[str, Dict[str, Any]],
    stage_b_index: Dict[str, Dict[str, Any]],
    dump_mismatches: bool = False,
    max_mismatches: int = 200,
) -> Dict[str, Any]:
    """Score predictions and classify every trajectory into a failure
    bucket.
    """
    cat_counts: Counter = Counter()
    offset_hist: Counter = Counter()
    pred_step_eq_1_or_3: Counter = Counter()
    type_total = 0
    module_correct = 0
    full_correct = 0
    type_confusion: List[Dict[str, Any]] = []
    failure_form_counts: Counter = Counter()
    self_correction_counts: Counter = Counter()
    scd_x_cat: Counter = Counter()
    mismatches: List[Dict[str, Any]] = []

    for stem, label in labels:
        if label["reward"] == 1 or label["critical_step"] is None:
            continue

        gt_step = label["critical_step"]
        pred = pred_index.get(stem)
        pred_step = pred.get("step") if isinstance(pred, dict) else None

        sb_entry = stage_b_index.get(stem) or {}
        flagged = sb_entry.get("flagged", {})
        error_phase = sb_entry.get("error_phase")

        category = classify_trajectory(gt_step, pred_step, flagged, error_phase)
        cat_counts[category] += 1

        # Pred-minus-GT offset histogram
        if isinstance(pred_step, int):
            offset = pred_step - gt_step
            offset_hist[_pred_minus_gt_bucket(offset)] += 1
            # Special tracking for very early picks (step 1 or 3)
            if pred_step in (1, 3):
                pred_step_eq_1_or_3[str(pred_step)] += 1

        # Module / subtype accuracy
        if label["module"] is not None and label["subtype"] is not None:
            type_total += 1
            if isinstance(pred, dict):
                pm = pred.get("module")
                ps = pred.get("subtype")
                if pm == label["module"]:
                    module_correct += 1
                if pm == label["module"] and ps == label["subtype"]:
                    full_correct += 1
                if pm != label["module"] or ps != label["subtype"]:
                    type_confusion.append({
                        "gt_type": f"{label['module']}.{label['subtype']}",
                        "pred_type": f"{pm}.{ps}" if pm and ps else None,
                        "stem": stem,
                    })

        # Stage C diagnostics
        if isinstance(pred, dict):
            ff = pred.get("failure_form")
            if ff:
                failure_form_counts[ff] += 1
            dist = pred.get("self_correction_distance")
            if dist:
                self_correction_counts[dist] += 1
                scd_x_cat[(dist, category)] += 1

        # Mismatch record
        if dump_mismatches and category != "correct" and len(mismatches) < max_mismatches:
            gt_record = flagged.get(gt_step, {})
            mismatches.append({
                "stem": stem,
                "task_id": label.get("task_id"),
                "gt_step": gt_step,
                "gt_type": label.get("critical_error_type"),
                "pred_step": pred_step,
                "pred_type": (
                    f"{pred.get('module')}.{pred.get('subtype')}"
                    if isinstance(pred, dict) and pred.get("module") and pred.get("subtype")
                    else None
                ),
                "pred_failure_form": pred.get("failure_form") if isinstance(pred, dict) else None,
                "pred_self_correction_distance": pred.get("self_correction_distance") if isinstance(pred, dict) else None,
                "stage_b_flagged_steps": sorted(flagged.keys()),
                "stage_b_gt_record": gt_record,
                "stage_a_error_phase": list(error_phase) if error_phase else None,
                "failure_category": category,
            })

    trajectories_scored = sum(cat_counts.values())
    return {
        "trajectories_scored": trajectories_scored,
        "categories": dict(cat_counts),
        "category_ratios": {k: v / trajectories_scored for k, v in cat_counts.items()}
        if trajectories_scored
        else {},
        "pred_minus_gt_histogram": dict(offset_hist),
        "pred_step_eq_1_or_3": dict(pred_step_eq_1_or_3),
        "type_total": type_total,
        "module_correct": module_correct,
        "full_correct": full_correct,
        "type_confusion": type_confusion[:50],
        "failure_form_distribution": dict(failure_form_counts),
        "self_correction_distance_distribution": dict(self_correction_counts),
        "self_correction_distance_x_category": [
            {"distance": dist, "category": cat, "count": cnt}
            for (dist, cat), cnt in sorted(scd_x_cat.items())
        ],
        "mismatches": mismatches,
    }





# ---------- Formatting ----------


def _format_breakdown(bd: Dict[str, Any]) -> str:
    """Format the breakdown classification for console output."""
    scored = bd.get("trajectories_scored", 0)
    cats = bd.get("categories", {})
    ratios = bd.get("category_ratios", {})
    lines = [f"  Failure Classification (N={scored}):"]
    for cat_name in ("correct", "stage_b_miss", "stage_c_too_early",
                      "stage_c_too_late", "stage_c_offgrid"):
        cnt = cats.get(cat_name, 0)
        ratio = ratios.get(cat_name, 0.0)
        lines.append(f"    {cat_name:<22s} {cnt:>4}  ({ratio*100:5.1f}%)")
    # Offset histogram
    hist = bd.get("pred_minus_gt_histogram", {})
    if hist:
        lines.append("  Pred-GT offset histogram:")
        for bucket in ("<=-10", "-9..-3", "-2..2", "3..9", ">=10"):
            lines.append(f"    {bucket:>8s}: {hist.get(bucket, 0)}")
    # Failure form
    ff = bd.get("failure_form_distribution", {})
    if ff:
        lines.append("  Failure form distribution:")
        for k, v in sorted(ff.items(), key=lambda x: -x[1]):
            lines.append(f"    {k}: {v}")
    # Self-correction distance
    scd = bd.get("self_correction_distance_distribution", {})
    if scd:
        lines.append("  Self-correction distance:")
        for k, v in sorted(scd.items(), key=lambda x: -x[1]):
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)





def _format_pipeline_analysis(analysis: Dict[str, Any]) -> str:
    """Format the fine-grained pipeline error analysis for console output."""
    total = analysis.get("trajectories_analyzed", 0)
    stage_b_n = analysis.get("stage_b_available", total)
    phase1_n = analysis.get("phase1_available", total)
    phase2_n = analysis.get("phase2_available", total)
    phase3_n = analysis.get("phase3_available", total)
    lines = [
        f"  Pipeline Error Analysis (N={total}):",
        f"    stage_b_gt_missed:           {analysis.get('stage_b_gt_missed', 0):>4}/{stage_b_n:<4} "
        f"({analysis.get('stage_b_gt_missed_ratio', 0)*100:.1f}%)",
        f"    phase1_gt_not_at_origin:     {analysis.get('phase1_gt_not_at_origin', 0):>4}/{phase1_n:<4} "
        f"({analysis.get('phase1_gt_not_at_origin_ratio', 0)*100:.1f}%)",
        f"    phase2_gt_fixed:             {analysis.get('phase2_gt_fixed', 0):>4}/{phase2_n:<4} "
        f"({analysis.get('phase2_gt_fixed_ratio', 0)*100:.1f}%)",
        f"    phase2_gt_dormant:           {analysis.get('phase2_gt_dormant', 0):>4}/{phase2_n:<4} "
        f"({analysis.get('phase2_gt_dormant_ratio', 0)*100:.1f}%)",
        f"    phase2_gt_rescued_by_span_cap: {analysis.get('phase2_gt_rescued_by_span_cap', 0):>4}/{phase2_n:<4} "
        f"({analysis.get('phase2_gt_rescued_by_span_cap_ratio', 0)*100:.1f}%)",
        f"    phase3_gt_in_selected:       {analysis.get('phase3_gt_in_selected', 0):>4}/{phase3_n:<4} "
        f"({analysis.get('phase3_gt_in_selected_ratio', 0)*100:.1f}%)",
    ]
    return "\n".join(lines)


# ---------- CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic breakdown of detector step predictions.",
    )
    parser.add_argument("--unified-dir", required=True)
    parser.add_argument(
        "--stage-b-dir",
        default="",
        help=(
            "Directory of Stage B per-step outputs (*_stage_b.json). "
            "Required for the new 3-stage pipeline. The legacy "
            "--phase15-dir flag is still accepted as an alias."
        ),
    )
    parser.add_argument(
        "--phase15-dir",
        default="",
        help=(
            "Legacy alias for --stage-b-dir. Accepts directories of "
            "*_error_detection_with_recovery.json or *_error_detection.json "
            "from the v0430 pipeline."
        ),
    )
    parser.add_argument(
        "--phase1-dir",
        default="",
        help=(
            "Directory of Stage C Phase 1 outputs "
            "(*_stage_c_phase1.json). Required for the fine-grained "
            "error analysis (phase1_gt_not_at_origin metric)."
        ),
    )
    parser.add_argument(
        "--phase2-dir",
        default="",
        help=(
            "Directory of Stage C Phase 2 outputs "
            "(*_stage_c_phase2.json). Required for the fine-grained "
            "error analysis (phase2_gt_fixed_or_dormant metric)."
        ),
    )
    parser.add_argument("--pred-dir", required=True,
                        help="Directory of detector final outputs (*_critical_error.json).")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--dump-mismatches", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    stage_b_dir = (args.stage_b_dir or args.phase15_dir or "").strip()
    if not stage_b_dir:
        parser.error("Provide --stage-b-dir (or legacy --phase15-dir) pointing at a "
                     "directory of Stage B / Phase 1.5 / Phase 1 per-step outputs.")

    phase1_dir = args.phase1_dir.strip()
    phase2_dir = args.phase2_dir.strip()
    pred_dir = args.pred_dir.strip()

    dataset_filter = {d.strip() for d in args.datasets.split(",") if d.strip()}
    labels = list(iter_unified_labels(args.unified_dir))
    if dataset_filter:
        labels = [
            (stem, label)
            for stem, label in labels
            if str(label.get("dataset", "")) in dataset_filter
        ]

    stage_b_index = load_stage_b_index(stage_b_dir)
    phase1_index = load_phase1_index(phase1_dir)
    phase2_index = load_phase2_index(phase2_dir)
    phase3_index = load_phase3_index(pred_dir)
    pred_index = load_pred_index(pred_dir)

    by_dataset: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for stem, label in labels:
        by_dataset[str(label.get("dataset") or "unknown")].append((stem, label))

    report: Dict[str, Any] = {
        "unified_dir": args.unified_dir,
        "pred_dir": pred_dir,
        "stage_b_dir": stage_b_dir,
        "num_labels": len(labels),
        "num_predictions": len(pred_index),
        "num_stage_b": len(stage_b_index),
        "datasets": {},
    }

    for ds, items in sorted(by_dataset.items()):
        # --- Breakdown classification ---
        breakdown = evaluate_breakdown(
            items,
            pred_index,
            stage_b_index,
            dump_mismatches=args.dump_mismatches,
        )
        # --- Pipeline error analysis (if phase1/phase2/phase3 available) ---
        pipeline_analysis = analyze_pipeline_errors(
            items,
            stage_b_index,
            phase1_index,
            phase2_index,
            phase3_index=phase3_index,
            dump_details=args.dump_mismatches,
        )
        breakdown["pipeline_analysis"] = pipeline_analysis
        report["datasets"][ds] = breakdown
        print(f"=== {ds} Breakdown (N={breakdown['trajectories_scored']}) ===")
        print(_format_breakdown(breakdown))
        if pipeline_analysis.get("trajectories_analyzed"):
            print(_format_pipeline_analysis(pipeline_analysis))

    overall_breakdown = evaluate_breakdown(
        labels,
        pred_index,
        stage_b_index,
        dump_mismatches=args.dump_mismatches,
    )
    overall_pipeline = analyze_pipeline_errors(
        labels,
        stage_b_index,
        phase1_index,
        phase2_index,
        phase3_index=phase3_index,
        dump_details=args.dump_mismatches,
    )
    overall_breakdown["pipeline_analysis"] = overall_pipeline
    report["overall"] = overall_breakdown
    print(f"=== OVERALL Breakdown (N={overall_breakdown['trajectories_scored']}) ===")
    print(_format_breakdown(overall_breakdown))
    if overall_pipeline.get("trajectories_analyzed"):
        print(_format_pipeline_analysis(overall_pipeline))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "load_stage_b_index",
    "load_phase1_index",
    "load_phase2_index",
    "load_phase3_index",
    "analyze_pipeline_errors",
    "classify_trajectory",
    "evaluate_breakdown",
    "_extract_flagged_steps",
    "_extract_error_phase",
    "_strip_stage_b_suffix",
]
