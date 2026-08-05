"""SWE-Bench-Pro → unified schema converter.

Input
-----
A single JSONL file (``data/Swe-Bench-Pro/swe-bench-pro-res.jsonl``) whose
structure mirrors tau²-bench: one line = one labelling event, grouped by
``juhe``. The differences that matter for conversion are:

  * Each trajectory already contains its own ``messages_role == "system"``
    message at ``step_id == 0`` (the SWE-Agent system prompt). tau²-bench did
    not have one; we used airline/retail policy files instead.
  * ``claude`` / ``gemini`` / ``gpt`` model predictions are plain strings of
    the form ``"<MODULE>.<Subtype>\\n<rationale>"`` rather than JSON objects
    — no ``confidence`` field is available.
  * Every trajectory is a failure case (no reward labels shipped); we emit
    ``reward = 0``.
  * Seven human annotators rotate across trajectories; we still keep each
    independent labelling as per the tau²-bench design.

Everything else (juhe grouping, resultType handling, tag semantics, error
category mapping, critical-step shift) is identical to tau²-bench. We reuse
the tau²-bench helpers directly to stay DRY.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_processing._common import dump_json
from data_processing.schema import (
    ROLE_SYSTEM,
    ROLE_USER,
    UnifiedAnnotation,
    UnifiedMessage,
    UnifiedMetadata,
    UnifiedTrajectory,
    validate_unified,
)
from data_processing.tau2bench import (
    _RESULT_TYPE_FITTED,
    _RESULT_TYPE_PER_ANNOTATOR,
    _SYSTEM_STEP_OFFSET,
    _TAG_ERROR_CATEGORY,
    _TAG_IS_CRITICAL,
    _TAG_RATIONALE,
    _coerce_role,
    _collect_human_annotator_critical_steps,
    _group_by_juhe,
    _is_critical_row,
    _map_error_category,
    _row_step_id,
    _slugify_task_id,
    _tag_list,
    _tag_text,
)


DATASET_NAME = "swebenchpro"


AGENT_FRAMEWORK_DESCRIPTION = (
    "A single `assistant` works on an SWE-Agent-style software-engineering "
    "task: modifying a code repository to satisfy a GitHub-issue-shaped "
    "specification. The leading `system` message is the SWE-Agent system "
    "prompt that defines the agent's tools (e.g. `str_replace_editor`, "
    "`execute_bash`) and operating rules. The single `user` message that "
    "follows describes the issue to solve. The assistant then interleaves "
    "reasoning turns and tool calls; `tool` messages are the tool-call "
    "return values from shell / editor tools. Critical errors can only be "
    "attributed to an `assistant` step; `user` and `tool` steps are never "
    "the critical step."
)


def _parse_model_prediction(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse an SWE-Bench-Pro model-prediction string.

    The column contains either ``"-"`` (no prediction) or a string of the
    shape::

        PLAN.BadDecomposition
        <multi-line rationale>

    We return ``{"critical_error_type": "PLAN.BadDecomposition",
    "rationale_short": "<rest>"}``. When the first line does not look like
    ``MODULE.Subtype`` the whole blob is kept under ``"raw"`` so the
    evaluator can still inspect it.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s == "-":
        return None
    # Split off the first non-empty line.
    first_line, sep, rest = s.partition("\n")
    first_line = first_line.strip()
    rest = rest.strip() or None
    if "." in first_line and re.match(r"^[A-Za-z]+\.[A-Za-z0-9_]+", first_line):
        # Normalise trailing punctuation / extra whitespace.
        et_match = re.match(r"^([A-Za-z]+\.[A-Za-z0-9_]+)", first_line)
        critical_type = et_match.group(1) if et_match else first_line
        return {
            "critical_error_type": critical_type,
            "rationale_short": rest,
        }
    return {"raw": s}


def _collect_model_critical_steps(
    fitted_rows: List[Dict[str, Any]],
    original_step_to_unified: Dict[int, int],
    model_field: str,
) -> List[Dict[str, Any]]:
    """For each non-summary '拟合结果' row, if the model column parses to a
    critical-error prediction, emit one entry."""
    out: List[Dict[str, Any]] = []
    for row in fitted_rows:
        orig_sid = _row_step_id(row)
        if orig_sid is None or orig_sid < 0:
            continue
        c = row.get("datasetItemContent") or {}
        pred = _parse_model_prediction(c.get(model_field))
        if pred is None:
            continue
        entry: Dict[str, Any] = {
            "step": original_step_to_unified.get(orig_sid),
            "original_step_id": orig_sid,
        }
        entry.update(pred)
        out.append(entry)
    out.sort(key=lambda e: (e.get("original_step_id") or 0))
    return out


def _build_trajectory(
    juhe: str, rows: List[Dict[str, Any]]
) -> Optional[UnifiedTrajectory]:
    fitted_rows = [r for r in rows if r.get("resultType") == _RESULT_TYPE_FITTED]
    per_annotator_rows = [
        r for r in rows if r.get("resultType") == _RESULT_TYPE_PER_ANNOTATOR
    ]
    if not fitted_rows:
        return None

    # Identify task_id (uniform across rows of one juhe).
    task_id = ""
    for r in rows:
        tid = (r.get("datasetItemContent") or {}).get("task_id")
        if isinstance(tid, str) and tid:
            task_id = tid
            break
    if not task_id:
        return None

    # Bucket fitted rows by messages_role / step_id: summary + system + rest.
    summary_row: Optional[Dict[str, Any]] = None
    system_row: Optional[Dict[str, Any]] = None
    other_message_rows: List[Tuple[int, Dict[str, Any]]] = []
    for r in fitted_rows:
        c = r.get("datasetItemContent") or {}
        sid = _row_step_id(r)
        role = str(c.get("messages_role", "")).strip().lower()
        if sid is None:
            continue
        if sid == -1 or role == "summary":
            if summary_row is None:
                summary_row = r
            continue
        if role == "system":
            if system_row is None:
                system_row = r
            continue
        other_message_rows.append((sid, r))

    if not other_message_rows:
        return None

    def _order_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
        sid, r = item
        c = r.get("datasetItemContent") or {}
        try:
            rid = int(str(c.get("id")))
        except (TypeError, ValueError):
            rid = 0
        return (sid, rid)

    other_message_rows.sort(key=_order_key)

    # --- Build unified messages --------------------------------------------
    unified_messages: List[UnifiedMessage] = []

    # Step 0 is the original system prompt. If the data happens not to have
    # one, we still insert an empty system turn so the step offset stays at 1
    # (consistent with tau²-bench's convention).
    system_content = ""
    if system_row is not None:
        sc = system_row.get("datasetItemContent") or {}
        system_content = str(sc.get("message_content") or "")
    unified_messages.append(
        UnifiedMessage(
            step=0,
            role=ROLE_SYSTEM,
            content=system_content,
            name="system",
        )
    )

    # The non-summary non-system rows get unified indices 1, 2, 3, ...
    original_step_to_unified: Dict[int, int] = {}
    skipped: List[Tuple[int, str]] = []

    for sid, r in other_message_rows:
        c = r.get("datasetItemContent") or {}
        role = _coerce_role(str(c.get("messages_role", "")))
        content = c.get("message_content")
        if role is None:
            skipped.append((sid, f"unknown role {c.get('messages_role')!r}"))
            continue
        unified_messages.append(
            UnifiedMessage(
                step=len(unified_messages),
                role=role,
                content="" if content is None else str(content),
                name=str(c.get("messages_role") or "") or None,
            )
        )
        original_step_to_unified[sid] = len(unified_messages) - 1

    if len(unified_messages) <= 1:
        return None

    # We also want the model-prediction lookup to recognise the system-row's
    # step_id (it is 0 in the raw data, but we already placed the system
    # turn at unified step 0, so expose that mapping too — gated on the
    # system message not being `user`-roled (which it isn't)).
    if system_row is not None:
        sys_sid = _row_step_id(system_row)
        if isinstance(sys_sid, int) and sys_sid >= 0:
            original_step_to_unified[sys_sid] = 0

    # --- Annotation from first fitted row marked critical ------------------
    annotation = UnifiedAnnotation()
    critical_fitted_row: Optional[Dict[str, Any]] = None
    for r in fitted_rows:
        sid = _row_step_id(r)
        if sid is None or sid < 0:
            continue
        if _is_critical_row(r):
            critical_fitted_row = r
            break

    if critical_fitted_row is not None:
        orig_sid = _row_step_id(critical_fitted_row)
        unified_step = (
            original_step_to_unified.get(orig_sid) if orig_sid is not None else None
        )
        tags = (critical_fitted_row.get("detailLabel") or {}).get("tags") or []
        category = _tag_list(tags, _TAG_ERROR_CATEGORY)
        rationale = _tag_text(tags, _TAG_RATIONALE).strip() or None
        error_type = _map_error_category(category)

        if (
            isinstance(unified_step, int)
            and unified_messages[unified_step].role != ROLE_USER
        ):
            annotation.critical_error_step = unified_step
        annotation.critical_error_type = error_type
        annotation.human_rationale = rationale

    # Every SWE-Bench-Pro trajectory shipped so far is a failed attempt.
    reward = 0

    # --- Per-rater critical-step labels ------------------------------------
    critical_step_labels: Dict[str, Any] = {
        "claude": _collect_model_critical_steps(
            fitted_rows, original_step_to_unified, "claude"
        ),
        "gemini": _collect_model_critical_steps(
            fitted_rows, original_step_to_unified, "gemini"
        ),
        "gpt": _collect_model_critical_steps(
            fitted_rows, original_step_to_unified, "gpt"
        ),
        "human_annotators": _collect_human_annotator_critical_steps(
            per_annotator_rows, original_step_to_unified
        ),
    }

    # --- task_description & summary ----------------------------------------
    summary_text = ""
    if summary_row is not None:
        sc = summary_row.get("datasetItemContent") or {}
        summary_text = str(sc.get("message_content") or "")

    task_description = _extract_user_requirement(summary_text) or summary_text

    extra: Dict[str, Any] = {
        "agent_framework_description": AGENT_FRAMEWORK_DESCRIPTION,
        "juhe": juhe,
        "original_task_id": task_id,
        "summary": summary_text,
        "critical_step_labels": critical_step_labels,
    }
    if skipped:
        extra["skipped_rows"] = skipped

    metadata = UnifiedMetadata(
        dataset=DATASET_NAME,
        task_id=task_id,
        task_description=task_description,
        reward=reward,
        annotation=annotation,
        extra=extra,
    )

    return UnifiedTrajectory(messages=unified_messages, metadata=metadata)


def _extract_user_requirement(summary_text: str) -> Optional[str]:
    """Return just the ``【用户需求概括】`` block of the summary if present.

    The summary has four labelled sections:
    ``【用户需求概括】`` / ``【模型轨迹概括】`` / ``【测试错误概括】`` /
    ``【测试错误输出】``. Only the first one is task-describing; the others
    leak the ground-truth failure reason and should not end up in
    ``task_description``.
    """
    if not isinstance(summary_text, str):
        return None
    marker = "【用户需求概括】"
    idx = summary_text.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    # Stop at any of the other section markers.
    stop_idx = len(summary_text)
    for other in ("【模型轨迹概括】", "【测试错误概括】", "【测试错误输出】"):
        j = summary_text.find(other, start)
        if j != -1 and j < stop_idx:
            stop_idx = j
    return summary_text[start:stop_idx].strip() or None


def convert_jsonl(
    src_path: str | os.PathLike,
    out_dir: str | os.PathLike,
) -> List[str]:
    """Convert the SWE-Bench-Pro JSONL into unified per-trajectory JSONs."""
    src_path = Path(src_path)
    if not src_path.is_file():
        return []

    buckets = _group_by_juhe(src_path)
    written: List[str] = []

    out_dir_path = Path(out_dir)
    for juhe, rows in sorted(
        buckets.items(),
        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0],
    ):
        traj = _build_trajectory(juhe, rows)
        if traj is None:
            continue
        data = traj.to_dict()
        validate_unified(data)
        stem = _slugify_task_id(traj.metadata.task_id)
        out_path = out_dir_path / f"{stem}.json"
        dump_json(data, out_path)
        written.append(str(out_path))
    return written


def convert_directory(
    src_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
) -> List[str]:
    """Accept either the SWE-Bench-Pro directory or the JSONL file itself."""
    src = Path(src_dir)
    written: List[str] = []
    if src.is_file():
        return convert_jsonl(src, out_dir)
    if not src.is_dir():
        return written
    for jsonl_path in sorted(src.glob("*.jsonl")):
        written.extend(convert_jsonl(jsonl_path, out_dir))
    return written


__all__ = [
    "DATASET_NAME",
    "AGENT_FRAMEWORK_DESCRIPTION",
    "convert_jsonl",
    "convert_directory",
]
