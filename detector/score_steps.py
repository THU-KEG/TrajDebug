#!/usr/bin/env python3
"""Evaluate detector predictions against unified-schema annotations.

The detector now consumes unified trajectories (``data/unified/<dataset>/*.json``)
whose ``metadata.annotation`` records the ground-truth ``critical_error_step``
and ``critical_error_type`` (``"<module>.<subtype>"`` or ``null``). For every
labeled trajectory we look up the matching detector output
(``<pred-dir>/<stem>_critical_error.json``) and report:

* **step accuracy** — always reported, computed over labels with a non-null
  ``critical_error_step`` AND a non-null reward of 0 (failure trajectories).
* **module accuracy** — reported only over the subset whose ground-truth
  ``critical_error_type`` is non-null and follows ``"<module>.<subtype>"``.
* **module+subtype accuracy** — same subset; both module and subtype must match.

Successful trajectories (``reward == 1`` or ``critical_error_step is null``)
are excluded from the step-accuracy denominator since there is no critical
step to predict.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from utils.error_definitions import split_full_error_type


# ---------- Generic JSON IO ----------


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_json_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_dir():
        return sorted(str(x) for x in p.rglob("*.json"))
    if p.is_file():
        return [str(p)]
    return []


# ---------- Prediction loading ----------


_PRED_SUFFIXES = (
    "_error_detection_with_recovery_critical_error",
    "_error_detection_critical_error",
    "_critical_error",
    "_final",
)


def _strip_pred_suffix(stem: str) -> str:
    for suffix in _PRED_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_pred_index(pred_dir: str) -> Dict[str, Dict[str, Any]]:
    """Index detector outputs by the unified-trajectory stem.

    For each ``*_critical_error.json`` we capture the predicted
    ``critical_step``, ``critical_module``, and ``error_type`` (if present).
    The 3-stage pipeline also emits ``failure_form`` and
    ``self_correction_distance``; we surface them here as
    ``failure_form`` / ``self_correction_distance`` so dataset-level
    distributions can be reported alongside step accuracy. Legacy v0430
    outputs that still carry ``mechanism`` are read transparently — both
    keys land in ``failure_form`` so downstream callers don't need to
    branch on schema version.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for fp in iter_json_files(pred_dir):
        try:
            data = load_json(fp)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        stem = _strip_pred_suffix(Path(fp).stem)
        ce = data.get("critical_error")
        entry: Dict[str, Any] = {
            "_pred_file": fp,
            "step": None,
            "chain_min_step": None,
            "chain_min_pre_llm_step": None,
            "llm_step": None,
            "module": None,
            "subtype": None,
            "taxonomy_tag": None,
            "failure_form": None,
            "self_correction_distance": None,
            "stage_a_agreement": None,
        }
        if isinstance(ce, dict):
            # v5 schema uses unified_critical_step; legacy uses critical_step
            # NOTE: use `is not None` to avoid treating step=0 as falsy
            _step_val = ce.get("unified_critical_step")
            step = _step_val if _step_val is not None else ce.get("critical_step")
            if isinstance(step, int):
                entry["step"] = step
            # chain_min_unified_critical_step
            chain_min_step = ce.get("chain_min_unified_critical_step")
            if isinstance(chain_min_step, int):
                entry["chain_min_step"] = chain_min_step
            # chain_min_unified_critical_step_pre_llm (pre-override snapshot)
            chain_min_pre_llm_step = ce.get("chain_min_unified_critical_step_pre_llm")
            if isinstance(chain_min_pre_llm_step, int):
                entry["chain_min_pre_llm_step"] = chain_min_pre_llm_step
            # llm_unified_critical_step
            llm_step = ce.get("llm_unified_critical_step")
            if isinstance(llm_step, int):
                entry["llm_step"] = llm_step
            # v5 schema uses agent_affected_module; legacy uses critical_module
            module = ce.get("agent_affected_module") or ce.get("critical_module")
            if isinstance(module, str) and module.strip():
                entry["module"] = module.strip().lower()
            subtype = ce.get("error_type")
            if isinstance(subtype, str) and subtype.strip():
                entry["subtype"] = subtype.strip()
            # taxonomy_tag: unified_taxonomy_tag (preferred) or agent_taxonomy_tag
            taxonomy_tag = ce.get("unified_taxonomy_tag") or ce.get("agent_taxonomy_tag")
            if isinstance(taxonomy_tag, str) and taxonomy_tag.strip():
                entry["taxonomy_tag"] = taxonomy_tag.strip()
                # derive module/subtype from taxonomy_tag if not already set
                if "." in taxonomy_tag:
                    tag_module, tag_subtype = taxonomy_tag.split(".", 1)
                    if not entry["module"]:
                        entry["module"] = tag_module.strip().lower()
                    if not entry["subtype"]:
                        entry["subtype"] = tag_subtype.strip()
            # v5: agent_category; legacy: failure_form / mechanism
            failure_form = ce.get("agent_category") or ce.get("failure_form") or ce.get("mechanism")
            if isinstance(failure_form, str) and failure_form.strip():
                entry["failure_form"] = failure_form.strip().lower()
            distance = ce.get("self_correction_distance")
            if isinstance(distance, str) and distance.strip():
                entry["self_correction_distance"] = distance.strip().lower()
            agreement = ce.get("stage_a_agreement")
            if isinstance(agreement, str) and agreement.strip():
                entry["stage_a_agreement"] = agreement.strip().lower()
            # v5 additions
            unified_source = ce.get("unified_source")
            if isinstance(unified_source, str) and unified_source.strip():
                entry["unified_source"] = unified_source.strip().lower()
        if stem not in index:
            index[stem] = entry
    return index


# ---------- Label loading from unified trajectories ----------


def iter_unified_labels(
    unified_dir: str,
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield ``(stem, label)`` pairs from a directory of unified trajectories.

    ``label`` is a dict with keys ``dataset``, ``task_id``, ``reward``,
    ``critical_step``, ``module``, ``subtype``, ``critical_error_type``,
    ``critical_failure_module``, ``trajectory_file``.

    ``critical_failure_module`` is pulled from
    ``metadata.extra.label_entry.critical_failure_module`` when present;
    it carries the dataset-native failure category (plan / action /
    memory / reflection / system / ...). It is used by ``evaluate`` to
    filter out ``system``-attributed trajectories (the failure was
    caused by environment / orchestration, not by the agent's own
    behaviour).
    """
    for fp in iter_json_files(unified_dir):
        try:
            data = load_json(fp)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        metadata = data.get("metadata") or {}
        annotation = metadata.get("annotation") or {} if isinstance(metadata, dict) else {}
        if not isinstance(annotation, dict):
            annotation = {}

        critical_step = annotation.get("critical_error_step")
        if not isinstance(critical_step, int):
            critical_step = None

        full_type = annotation.get("critical_error_type")
        module = subtype = None
        if isinstance(full_type, str) and full_type.strip():
            module, subtype = split_full_error_type(full_type)

        reward = metadata.get("reward") if isinstance(metadata, dict) else None
        try:
            reward_int = int(reward) if reward is not None else None
        except (TypeError, ValueError):
            reward_int = None

        # Pull the dataset-native failure module (plan / action /
        # memory / reflection / system / ...) from
        # metadata.extra.label_entry.critical_failure_module. It is
        # optional and may be missing on datasets without the
        # label_entry schema.
        extra = metadata.get("extra") if isinstance(metadata, dict) else None
        label_entry = extra.get("label_entry") if isinstance(extra, dict) else None
        critical_failure_module: Optional[str] = None
        if isinstance(label_entry, dict):
            cfm = label_entry.get("critical_failure_module")
            if isinstance(cfm, str) and cfm.strip():
                critical_failure_module = cfm.strip().lower()

        yield Path(fp).stem, {
            "dataset": str(metadata.get("dataset", "")) if isinstance(metadata, dict) else "",
            "task_id": metadata.get("task_id") if isinstance(metadata, dict) else None,
            "reward": reward_int,
            "critical_step": critical_step,
            "module": module,
            "subtype": subtype,
            "critical_error_type": full_type if isinstance(full_type, str) else None,
            "critical_failure_module": critical_failure_module,
            "trajectory_file": fp,
        }


# ---------- Assistant-step distance ----------


def _build_assistant_step_index(trajectory_file: str) -> Optional[List[int]]:
    """Build sorted list of step indices that are assistant (judgable) steps.

    Returns None if the file cannot be loaded.
    """
    try:
        data = load_json(trajectory_file)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    messages = data.get("messages") or []
    assistant_steps = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        judgable = msg.get("judgable", True)
        step = msg.get("step")
        if not isinstance(step, int):
            continue
        if role == "assistant" and judgable:
            assistant_steps.append(step)
    return sorted(assistant_steps)


def compute_assistant_distance(
    pred_step: int, label_step: int, assistant_steps: Optional[List[int]]
) -> Optional[int]:
    """Compute signed distance in assistant-step units.

    If pred=5, label=3, and assistant_steps=[1,3,5,7], then
    the distance is +1 (pred is 1 assistant step ahead of label).
    Returns None if either step is not in the assistant index.
    """
    if assistant_steps is None:
        return None
    try:
        pred_idx = assistant_steps.index(pred_step)
        label_idx = assistant_steps.index(label_step)
    except ValueError:
        # If exact step not in index, find nearest position
        import bisect
        pred_idx = bisect.bisect_left(assistant_steps, pred_step)
        label_idx = bisect.bisect_left(assistant_steps, label_step)
    return pred_idx - label_idx


# ---------- Scoring ----------


# Valid choices for score_source
_SCORE_SOURCE_CHOICES = ("unified", "llm", "chain_min", "chain_min_pre_llm")
# Map score_source name -> pred entry key
_SCORE_SOURCE_KEY = {
    "unified": "step",
    "llm": "llm_step",
    "chain_min": "chain_min_step",
    "chain_min_pre_llm": "chain_min_pre_llm_step",
}


def evaluate(
    labels: List[Tuple[str, Dict[str, Any]]],
    pred_index: Dict[str, Dict[str, Any]],
    show_mismatches: bool,
    max_mismatches: int,
    score_source: str = "unified",
) -> Dict[str, Any]:
    """Evaluate predictions.

    score_source controls which field is used as the primary step for
    step_accuracy / step_correct / step_missing / mismatches:
      "unified"          — unified_critical_step (default)
      "llm"              — llm_unified_critical_step
      "chain_min"        — chain_min_unified_critical_step
      "chain_min_pre_llm" — chain_min_unified_critical_step_pre_llm
    The other three sources are always reported as auxiliary stats
    regardless of score_source.
    """
    step_total = step_correct = step_missing = 0
    step_loose1 = step_loose2 = 0  # |distance| <= 1 and |distance| <= 2
    chain_min_total = chain_min_correct = chain_min_loose1 = chain_min_loose2 = 0
    llm_total = llm_correct = llm_loose1 = llm_loose2 = 0
    type_total = type_module_correct = type_full_correct = 0
    # taxonomy_tag accuracy (module prefix and full tag)
    tag_total = tag_module_correct = tag_full_correct = 0
    skipped_success = 0
    # Trajectories whose dataset-native
    # ``metadata.extra.label_entry.critical_failure_module`` equals
    # ``"system"`` are excluded from scoring: the failure was caused by
    # the environment / orchestration layer, not by the agent's own
    # behaviour, so asking the detector to locate a critical agent step
    # is ill-posed.
    skipped_system_module = 0
    mismatches: List[Dict[str, Any]] = []
    # Stage C diagnostic distributions, computed over the same
    # denominator as ``step_total`` so they line up with step accuracy.
    failure_form_counts: Dict[str, int] = {}
    distance_counts: Dict[str, int] = {}
    stage_a_agreement_counts: Dict[str, int] = {}
    # Assistant-step distance distribution
    assistant_distance_counts: Dict[int, int] = {}
    assistant_distance_sum = 0
    assistant_distance_abs_sum = 0
    assistant_distance_n = 0

    # Determine which pred key to use as the primary step
    primary_key = _SCORE_SOURCE_KEY.get(score_source, "step")

    for stem, label in labels:
        if label.get("critical_failure_module") == "system":
            skipped_system_module += 1
            # continue
        if label["reward"] == 1 or label["critical_step"] is None:
            skipped_success += 1
            continue

        pred = pred_index.get(stem)

        step_total += 1
        primary_step = pred.get(primary_key) if pred else None
        if not pred or not isinstance(primary_step, int):
            step_missing += 1
            if show_mismatches and len(mismatches) < max_mismatches:
                mismatches.append({
                    "id": label.get("task_id") or stem,
                    "stem": stem,
                    "label_step": label["critical_step"],
                    "pred_step": None,
                    "score_source": score_source,
                    "reason": "missing_pred",
                })
            continue

        step_match = primary_step == label["critical_step"]
        if step_match:
            step_correct += 1

        # Compute assistant-step distance (signed: positive = pred is later)
        adist = None
        traj_file = label.get("trajectory_file")
        if traj_file and isinstance(primary_step, int) and isinstance(label.get("critical_step"), int):
            ast_steps = _build_assistant_step_index(traj_file)
            adist = compute_assistant_distance(primary_step, label["critical_step"], ast_steps)
            if adist is not None:
                assistant_distance_counts[adist] = assistant_distance_counts.get(adist, 0) + 1
                assistant_distance_sum += adist
                assistant_distance_abs_sum += abs(adist)
                assistant_distance_n += 1
                if adist >= -1 and adist <= 0:
                    step_loose1 += 1
                if adist >= -2 and adist <= 0:
                    step_loose2 += 1

        # chain_min_unified_critical_step scoring
        if isinstance(pred.get("chain_min_step"), int):
            chain_min_total += 1
            chain_min_match = pred["chain_min_step"] == label["critical_step"]
            if chain_min_match:
                chain_min_correct += 1
            if traj_file:
                ast_steps = _build_assistant_step_index(traj_file)
                cm_adist = compute_assistant_distance(pred["chain_min_step"], label["critical_step"], ast_steps)
                if cm_adist is not None:
                    if cm_adist >= -1 and cm_adist <= 0:
                        chain_min_loose1 += 1
                    if cm_adist >= -2 and cm_adist <= 0:
                        chain_min_loose2 += 1

        # llm_unified_critical_step scoring
        if isinstance(pred.get("llm_step"), int):
            llm_total += 1
            llm_match = pred["llm_step"] == label["critical_step"]
            if llm_match:
                llm_correct += 1
            if traj_file:
                ast_steps = _build_assistant_step_index(traj_file)
                llm_adist = compute_assistant_distance(pred["llm_step"], label["critical_step"], ast_steps)
                if llm_adist is not None:
                    if llm_adist >= -1 and llm_adist <= 0:
                        llm_loose1 += 1
                    if llm_adist >= -2 and llm_adist <= 0:
                        llm_loose2 += 1

        # Tally Stage C diagnostics across every scored trajectory
        # (correct or not) so the distribution reflects the full
        # denominator of step accuracy.
        ff = pred.get("failure_form")
        if ff:
            failure_form_counts[ff] = failure_form_counts.get(ff, 0) + 1
        dist = pred.get("self_correction_distance")
        if dist:
            distance_counts[dist] = distance_counts.get(dist, 0) + 1
        agreement = pred.get("stage_a_agreement")
        if agreement:
            stage_a_agreement_counts[agreement] = (
                stage_a_agreement_counts.get(agreement, 0) + 1
            )

        if label["module"] is not None and label["subtype"] is not None:
            type_total += 1
            module_match = pred.get("module") == label["module"]
            subtype_match = pred.get("subtype") == label["subtype"]
            if module_match:
                type_module_correct += 1
            if module_match and subtype_match:
                type_full_correct += 1

        # taxonomy_tag accuracy: compare pred taxonomy_tag against label
        # critical_error_type (both have the form "<module>.<subtype>")
        label_full_type = label.get("critical_error_type")
        pred_tag = pred.get("taxonomy_tag") if pred else None
        if label_full_type and pred_tag:
            tag_total += 1
            label_tag_module = label_full_type.split(".", 1)[0].lower() if "." in label_full_type else label_full_type.lower()
            pred_tag_module = pred_tag.split(".", 1)[0].lower() if "." in pred_tag else pred_tag.lower()
            if pred_tag_module == label_tag_module:
                tag_module_correct += 1
            if pred_tag.lower() == label_full_type.lower():
                tag_full_correct += 1

        if not step_match and show_mismatches and len(mismatches) < max_mismatches:
            mismatches.append({
                "id": label.get("task_id") or stem,
                "stem": stem,
                "label_step": label["critical_step"],
                "pred_step": primary_step,
                "score_source": score_source,
                "assistant_distance": adist,
                "label_type": label.get("critical_error_type"),
                "pred_type": (
                    f"{pred.get('module')}.{pred.get('subtype')}"
                    if pred.get("module") and pred.get("subtype")
                    else None
                ),
                "reason": "mismatch",
            })

    step_acc = step_correct / step_total if step_total else 0.0
    module_acc = type_module_correct / type_total if type_total else 0.0
    full_acc = type_full_correct / type_total if type_total else 0.0
    chain_min_acc = chain_min_correct / chain_min_total if chain_min_total else 0.0
    chain_min_loose1_acc = chain_min_loose1 / chain_min_total if chain_min_total else 0.0
    chain_min_loose2_acc = chain_min_loose2 / chain_min_total if chain_min_total else 0.0
    llm_acc = llm_correct / llm_total if llm_total else 0.0
    llm_loose1_acc = llm_loose1 / llm_total if llm_total else 0.0
    llm_loose2_acc = llm_loose2 / llm_total if llm_total else 0.0
    tag_module_acc = tag_module_correct / tag_total if tag_total else 0.0
    tag_full_acc = tag_full_correct / tag_total if tag_total else 0.0
    return {
        "score_source": score_source,
        "step_total": step_total,
        "step_correct": step_correct,
        "step_accuracy": step_acc,
        "step_loose1": step_loose1,
        "step_loose1_accuracy": step_loose1 / assistant_distance_n if assistant_distance_n else 0.0,
        "step_loose2": step_loose2,
        "step_loose2_accuracy": step_loose2 / assistant_distance_n if assistant_distance_n else 0.0,
        "step_missing_pred": step_missing,
        # chain_min_unified_critical_step stats
        "chain_min_total": chain_min_total,
        "chain_min_correct": chain_min_correct,
        "chain_min_accuracy": chain_min_acc,
        "chain_min_loose1": chain_min_loose1,
        "chain_min_loose1_accuracy": chain_min_loose1_acc,
        "chain_min_loose2": chain_min_loose2,
        "chain_min_loose2_accuracy": chain_min_loose2_acc,
        # llm_unified_critical_step stats
        "llm_total": llm_total,
        "llm_correct": llm_correct,
        "llm_accuracy": llm_acc,
        "llm_loose1": llm_loose1,
        "llm_loose1_accuracy": llm_loose1_acc,
        "llm_loose2": llm_loose2,
        "llm_loose2_accuracy": llm_loose2_acc,
        "type_total": type_total,
        "module_correct": type_module_correct,
        "module_accuracy": module_acc,
        "full_type_correct": type_full_correct,
        "full_type_accuracy": full_acc,
        # taxonomy_tag accuracy
        "tag_total": tag_total,
        "tag_module_correct": tag_module_correct,
        "tag_module_accuracy": tag_module_acc,
        "tag_full_correct": tag_full_correct,
        "tag_full_accuracy": tag_full_acc,
        "skipped_success": skipped_success,
        "skipped_system_module": skipped_system_module,
        "mismatches": mismatches,
        # Stage C diagnostics (empty dicts on legacy outputs).
        "failure_form_distribution": failure_form_counts,
        "self_correction_distance_distribution": distance_counts,
        "stage_a_agreement_distribution": stage_a_agreement_counts,
        # Assistant-step distance stats
        "assistant_distance_distribution": dict(sorted(assistant_distance_counts.items())),
        "assistant_distance_mean": (assistant_distance_sum / assistant_distance_n) if assistant_distance_n else None,
        "assistant_distance_abs_mean": (assistant_distance_abs_sum / assistant_distance_n) if assistant_distance_n else None,
        "assistant_distance_n": assistant_distance_n,
    }


def format_summary(name: str, stats: Dict[str, Any]) -> str:
    lines = [
        f"{name} | step total={stats['step_total']} correct={stats['step_correct']} "
        f"acc={stats['step_accuracy']*100:.2f}% "
        f"loose1={stats.get('step_loose1', 0)} loose1_acc={stats.get('step_loose1_accuracy', 0.0)*100:.2f}% "
        f"loose2={stats.get('step_loose2', 0)} loose2_acc={stats.get('step_loose2_accuracy', 0.0)*100:.2f}% "
        f"missing_pred={stats['step_missing_pred']} "
        f"skipped_success={stats['skipped_success']} "
        f"skipped_system={stats.get('skipped_system_module', 0)}"
    ]
    if stats.get('chain_min_total', 0):
        lines.append(
            f"{name} | chain_min_step total={stats['chain_min_total']} "
            f"correct={stats['chain_min_correct']} acc={stats['chain_min_accuracy']*100:.2f}% "
            f"loose1={stats['chain_min_loose1']} loose1_acc={stats['chain_min_loose1_accuracy']*100:.2f}% "
            f"loose2={stats['chain_min_loose2']} loose2_acc={stats['chain_min_loose2_accuracy']*100:.2f}%"
        )
    if stats.get('llm_total', 0):
        lines.append(
            f"{name} | llm_step total={stats['llm_total']} "
            f"correct={stats['llm_correct']} acc={stats['llm_accuracy']*100:.2f}% "
            f"loose1={stats['llm_loose1']} loose1_acc={stats['llm_loose1_accuracy']*100:.2f}% "
            f"loose2={stats['llm_loose2']} loose2_acc={stats['llm_loose2_accuracy']*100:.2f}%"
        )
    if stats["type_total"]:
        lines.append(
            f"{name} | type total={stats['type_total']} module_acc="
            f"{stats['module_accuracy']*100:.2f}% full_acc={stats['full_type_accuracy']*100:.2f}%"
        )
    else:
        lines.append(f"{name} | type total=0 (no taxonomy-aligned labels)")
    if stats.get('tag_total', 0):
        lines.append(
            f"{name} | taxonomy_tag total={stats['tag_total']} "
            f"module_acc={stats['tag_module_accuracy']*100:.2f}% "
            f"full_acc={stats['tag_full_accuracy']*100:.2f}%"
        )
    ff_dist = stats.get("failure_form_distribution") or {}
    if ff_dist:
        ff_str = ", ".join(f"{k}={v}" for k, v in sorted(ff_dist.items()))
        lines.append(f"{name} | failure_form: {ff_str}")
    dist_dist = stats.get("self_correction_distance_distribution") or {}
    if dist_dist:
        d_str = ", ".join(f"{k}={v}" for k, v in sorted(dist_dist.items()))
        lines.append(f"{name} | self_correction_distance: {d_str}")
    # Assistant-step distance summary
    adist_n = stats.get("assistant_distance_n", 0)
    if adist_n:
        adist_mean = stats.get("assistant_distance_mean", 0)
        adist_abs_mean = stats.get("assistant_distance_abs_mean", 0)
        lines.append(
            f"{name} | assistant_distance: n={adist_n} "
            f"mean={adist_mean:+.2f} abs_mean={adist_abs_mean:.2f}"
        )
        adist_dist = stats.get("assistant_distance_distribution") or {}
        if adist_dist:
            # Show compact distribution for common values
            compact = {k: v for k, v in sorted(adist_dist.items()) if abs(k) <= 5}
            if compact:
                d_str = ", ".join(f"{k:+d}={v}" for k, v in sorted(compact.items()))
                lines.append(f"{name} | assistant_distance_dist(|d|<=5): {d_str}")
    return "\n".join(lines)


# ---------- Optional ID filter ----------


def load_id_filter(path: str) -> Optional[Set[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    raw = load_json(str(p))
    ids: Set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                ids.add(item)
            elif isinstance(item, dict):
                for key in ("id", "task_id", "stem"):
                    val = item.get(key)
                    if isinstance(val, str):
                        ids.add(val)
                        break
    return ids or None


# ---------- CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score detector predictions against unified annotations.",
    )
    parser.add_argument(
        "--unified-dir",
        required=True,
        help="Directory of unified trajectory JSON files (one dataset per dir, "
             "or pass the parent and use --datasets to filter by metadata.dataset).",
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory of detector phase-2 outputs (*_critical_error.json).",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset filter (matched against "
             "metadata.dataset). Empty means all.",
    )
    parser.add_argument("--show-mismatches", action="store_true")
    parser.add_argument("--max-mismatches", type=int, default=50)
    parser.add_argument(
        "--id-list-file",
        default="",
        help="Optional JSON file listing trajectory stems / task_ids to keep.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to write the full report as JSON.",
    )
    parser.add_argument(
        "--score-source",
        default="unified",
        choices=list(_SCORE_SOURCE_CHOICES),
        help=(
            "Which field to use as the primary step for step_accuracy. "
            "'unified' (default): unified_critical_step. "
            "'llm': llm_unified_critical_step. "
            "'chain_min': chain_min_unified_critical_step. "
            "'chain_min_pre_llm': chain_min_unified_critical_step_pre_llm "
            "(chain_min result before LLM override, only present when "
            "unified_source_strategy=llm was used). "
            "The other three sources are always reported as auxiliary stats."
        ),
    )
    args = parser.parse_args()

    dataset_filter = {d.strip() for d in args.datasets.split(",") if d.strip()}
    id_filter = load_id_filter(args.id_list_file)

    labels = list(iter_unified_labels(args.unified_dir))

    if dataset_filter:
        labels = [
            (stem, label)
            for stem, label in labels
            if str(label.get("dataset", "")) in dataset_filter
        ]
    if id_filter is not None:
        labels = [
            (stem, label)
            for stem, label in labels
            if stem in id_filter or str(label.get("task_id", "")) in id_filter
        ]

    pred_index = load_pred_index(args.pred_dir)

    report: Dict[str, Any] = {
        "unified_dir": os.path.abspath(args.unified_dir),
        "pred_dir": os.path.abspath(args.pred_dir),
        "num_labels": len(labels),
        "num_predictions": len(pred_index),
        "datasets": {},
        "overall": {},
    }

    by_dataset: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for stem, label in labels:
        ds = str(label.get("dataset") or "unknown")
        by_dataset.setdefault(ds, []).append((stem, label))

    for ds, items in sorted(by_dataset.items()):
        stats = evaluate(items, pred_index, args.show_mismatches, args.max_mismatches,
                         score_source=args.score_source)
        report["datasets"][ds] = stats
        print(format_summary(ds, stats))

    overall = evaluate(labels, pred_index, args.show_mismatches, args.max_mismatches,
                       score_source=args.score_source)
    report["overall"] = overall
    print(format_summary("OVERALL", overall))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
