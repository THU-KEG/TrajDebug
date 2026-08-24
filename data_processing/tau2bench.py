"""τ²-Bench (tau²-bench) → unified schema converter.

Input
-----
A single JSONL file (e.g. ``data/Tau^2-Bench/tau^2-bench-res.jsonl``) where each
line is **one labeling event** (not one trajectory, not one message):

    - ``datasetItemContent``: the actual message (``juhe`` = trajectory id,
      ``step_id`` = original position, ``messages_role``, ``message_content``,
      plus ``claude`` / ``gemini`` / ``gpt`` prediction strings).
    - ``resultType``: one of ``"标注结果"`` (per-annotator raw, three copies
      per message — one per human annotator), ``"拟合结果"`` (aggregated /
      fitted human label, authoritative), ``"最终结果"`` (final release copy).
    - ``detailLabel.tags``: per-row annotation tuple with a "人工-是否critical
      error" flag, an "错误类别" cascader value, and a "错误原因" rationale.
    - ``labeler``: populated on ``"标注结果"`` rows with the annotator's name.

One trajectory = one ``juhe``. Each ``juhe`` contains many (step_id, role,
text) message rows, each repeated 5× (3 × 标注结果 + 1 × 拟合结果 + 1 × 最终
结果).

Output
------
Per-trajectory unified JSON under ``data/unified/tau2bench/<task_id>.json``.
Rules:

  * Trajectory key: ``juhe``. ``task_id`` is uniquely determined by ``juhe``
    and we use it verbatim (slug-safe) as the output filename.
  * ``messages[0]`` is a synthesised ``system`` turn containing the domain
    policy document (``airline.txt`` or ``retail.txt``, selected by task_id /
    summary scene prefix). The remaining messages are the ``"拟合结果"`` rows
    with ``step_id >= 0``, sorted by ``int(step_id)``. Because of the inserted
    system turn, **every downstream unified ``step`` index is ``original
    step_id + 1``** — this shift is applied consistently to
    ``metadata.annotation.critical_error_step`` and to every ``step`` field
    inside ``metadata.extra.critical_step_labels``.
  * ``metadata.task_description`` keeps only the scenario + user-goal portion
    of the summary text. The "未满足的部分" block is stripped because it leaks
    the ground-truth failure reason.
  * ``metadata.annotation`` is derived from the FIRST fitted row whose
    "人工-是否critical error" equals "是":
      - ``critical_error_step``   = unified step index of that message
                                    (= original step_id + 1).
      - ``critical_error_type``   = mapped to detector taxonomy
        ``"<module>.<subtype>"`` (e.g. ``"reason.InvalidInference"``), or
        ``None`` if the mapping is not in the canonical white-list.
      - ``human_rationale``       = the aggregator's free-text "错误原因".
  * ``metadata.reward``:
      - ``1`` if ``"_success"`` appears in ``task_id``;
      - otherwise ``0`` (the current file only contains ``_failed_`` tasks).
  * ``metadata.extra`` holds tau²-bench-specific auxiliary information:
      - ``summary`` — the step_id=-1 ``message_content`` string (scene + goal
        + unsatisfied-assertion). The full original text is kept here in case
        evaluators want the ground-truth assertion.
      - ``scene`` — ``"airline"`` or ``"retail"``.
      - ``critical_step_labels`` — per-rater lists. See schema in code.
      - ``agent_framework_description`` — short prose for the Phase-1 prompt.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_processing._common import dump_json
from data_processing.schema import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    UnifiedAnnotation,
    UnifiedMessage,
    UnifiedMetadata,
    UnifiedTrajectory,
    validate_unified,
)

try:
    from detector.utils.error_definitions import is_valid_full_error_type
except Exception:  # pragma: no cover - keep the converter importable standalone
    def is_valid_full_error_type(full_type: str) -> bool:  # type: ignore[misc]
        return isinstance(full_type, str) and "." in full_type


DATASET_NAME = "tau2bench"


AGENT_FRAMEWORK_DESCRIPTION = (
    "A user interacts with a single `assistant` that must follow the scene "
    "policy shown in the leading `system` message while executing the user's "
    "request. `tool` messages are tool-call return values. Critical errors "
    "can only be attributed to an `assistant` step; `user` and `tool` steps "
    "are never the critical step. The assistant should try to fulfil the "
    "user's request as much as possible, but the user may make unreasonable "
    "or out-of-policy requests; whenever the user's request conflicts with "
    "the `system` policy, the `system` policy takes precedence and the "
    "assistant must refuse or redirect rather than comply.\n\n"
    "DATASET NOTES (tau²-bench). The leading `system` message lists "
    "binding procedural rules using \"必须\" / \"must\" / \"before X you "
    "must Y\" / \"only ... if\" / \"never\". These rules are hard "
    "constraints on every assistant step that touches the database "
    "(booking, modifying, cancelling, refunding, returning) — typical "
    "examples include verifying user identity (email lookup, name+zip) "
    "before any account-data operation. When the user refers to a "
    "target object indirectly (\"my recent order\", \"the items I want "
    "to cancel\") and the prior `tool` outputs visibly enumerate "
    "multiple candidate objects under that user (multiple `orders`, "
    "multiple `reservations`, multiple membership tiers), the candidate "
    "set is fully visible — picking one to act on is a substantive "
    "commitment, not a probe; the tool response for that one candidate "
    "yields no information about the others, so it is not "
    "self-correcting on the next turn. The system policy also "
    "constrains the option space the assistant may offer the user "
    "(e.g. cancellation may be whole-order-only; modification may be "
    "limited to same-product-type variants). When the assistant "
    "presents a procedural option to the user (\"you can return only "
    "this item\", \"you can wait until delivery and then return\", "
    "\"I can split the cancellation\"), that option must be authorised "
    "either by the system policy verbatim or by a prior tool result "
    "for the specific object — even if the assistant is only "
    "presenting choices and not yet executing, presenting an option "
    "the assistant has not verified to be permitted commits the "
    "assistant to a path that may be unworkable, and the user may "
    "rely on the option as if it were available."
)


# --- tau²-bench ``detailLabel.tags`` label text used for matching -------------
_TAG_IS_CRITICAL = "是否critical error"     # unique within "人工-是否critical error"
_TAG_ERROR_CATEGORY = "错误类别"
_TAG_RATIONALE = "错误原因"

_RESULT_TYPE_FITTED = "拟合结果"
_RESULT_TYPE_PER_ANNOTATOR = "标注结果"

_MODULE_MAP = {
    "plan": "plan",
    "planning": "plan",
    "reason": "reason",
    "reasoning": "reason",
    "act": "act",
    "action": "act",
    "obs": "obs",
    "observation": "obs",
    "verify": "verify",
    "verification": "verify",
    "reflection": "verify",
}

# Number of unified-index positions the inserted system turn pushes every
# original step_id forward by.
_SYSTEM_STEP_OFFSET = 1


# --- helpers -----------------------------------------------------------------

def _slugify_task_id(task_id: str) -> str:
    """Make a filesystem-safe stem. tau²-bench task_ids contain ``::``."""
    if not isinstance(task_id, str) or not task_id:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("_") or "unknown"


def _trim_scene_goal(summary_text: str) -> str:
    """Keep only the ``场景`` + ``用户目的`` portion of a summary, i.e. drop
    everything from ``未满足的部分`` onward (plus any trailing whitespace)."""
    if not isinstance(summary_text, str):
        return ""
    idx = summary_text.find("未满足的部分")
    if idx == -1:
        return summary_text.strip()
    return summary_text[:idx].rstrip()


def _detect_scene(task_id: str, summary_text: str) -> Optional[str]:
    """Return ``"airline"`` / ``"retail"`` / ``None``.

    Prefer ``task_id`` prefix (``airline_*`` / ``retail_*``); fall back to the
    summary's ``"场景：航空"`` / ``"场景：零售"`` marker.
    """
    tid = (task_id or "").lower()
    if tid.startswith("airline"):
        return "airline"
    if tid.startswith("retail"):
        return "retail"
    if isinstance(summary_text, str):
        if "场景：航空" in summary_text:
            return "airline"
        if "场景：零售" in summary_text:
            return "retail"
    return None


def _load_scene_policies(policy_dir: Path) -> Dict[str, str]:
    """Load ``airline.txt`` / ``retail.txt`` from the given directory.

    Missing files produce an empty string (the converter will still emit a
    system message, but with empty content).
    """
    out: Dict[str, str] = {}
    for scene, fname in (("airline", "airline.txt"), ("retail", "retail.txt")):
        fpath = policy_dir / fname
        if fpath.is_file():
            try:
                out[scene] = fpath.read_text(encoding="utf-8")
            except Exception:
                out[scene] = ""
        else:
            out[scene] = ""
    return out


def _find_tag(tags: List[Dict[str, Any]], label_contains: str) -> Optional[Dict[str, Any]]:
    for t in tags or []:
        if isinstance(t, dict) and label_contains in str(t.get("label", "")):
            return t
    return None


def _tag_text(tags: List[Dict[str, Any]], label_contains: str) -> str:
    tag = _find_tag(tags, label_contains)
    if tag is None:
        return ""
    v = tag.get("value")
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return "" if v is None else str(v)


def _tag_list(tags: List[Dict[str, Any]], label_contains: str) -> List[str]:
    tag = _find_tag(tags, label_contains)
    if tag is None:
        return []
    v = tag.get("value")
    if isinstance(v, list):
        return [str(x) for x in v]
    if v is None or v == "":
        return []
    return [str(v)]


def _parse_model_prediction(raw: Any) -> Optional[Dict[str, Any]]:
    """The ``claude`` / ``gemini`` / ``gpt`` columns are JSON-as-string or
    the literal ``"-"``. Returns a dict with ``critical_error_type`` /
    ``confidence`` / ``rationale_short`` when parseable, or ``None`` for
    empty / absent predictions."""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s == "-":
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {
                "critical_error_type": obj.get("critical_error_type"),
                "confidence": obj.get("confidence"),
                "rationale_short": obj.get("rationale_short"),
            }
    except Exception:
        pass
    return {"raw": s}


def _map_error_category(category: List[str]) -> Optional[str]:
    """Map tau²-bench ``错误类别`` cascader value to canonical
    ``"<module>.<subtype>"``. Returns ``None`` if not cleanly mappable or
    the result is not in the detector white-list."""
    if not category:
        return None
    head = str(category[0]).strip().lower()
    module = _MODULE_MAP.get(head)
    if module is None:
        return None
    if len(category) < 2:
        return None
    sub_raw = str(category[1]).strip()
    sub = sub_raw.split(":", 1)[0].strip()
    sub = re.sub(r"\s+", "", sub)
    if not sub:
        return None
    candidate = f"{module}.{sub}"
    return candidate if is_valid_full_error_type(candidate) else None


def _coerce_role(messages_role: str) -> Optional[str]:
    r = (messages_role or "").strip().lower()
    if r == "assistant":
        return ROLE_ASSISTANT
    if r == "user":
        return ROLE_USER
    if r == "tool":
        return ROLE_TOOL
    return None


def _stream_jsonl(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                continue


# --- core conversion ---------------------------------------------------------

def _group_by_juhe(src_path: str | os.PathLike) -> Dict[str, List[Dict[str, Any]]]:
    """Read the JSONL once and bucket rows by ``juhe``."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, row in _stream_jsonl(src_path):
        content = row.get("datasetItemContent") or {}
        juhe = content.get("juhe")
        if juhe is None or juhe == "":
            continue
        buckets[str(juhe)].append(row)
    return buckets


def _row_step_id(row: Dict[str, Any]) -> Optional[int]:
    c = row.get("datasetItemContent") or {}
    try:
        return int(str(c.get("step_id")))
    except (TypeError, ValueError):
        return None


def _is_critical_row(row: Dict[str, Any]) -> bool:
    tags = (row.get("detailLabel") or {}).get("tags") or []
    return "是" in _tag_list(tags, _TAG_IS_CRITICAL)


def _collect_model_critical_steps(
    fitted_rows: List[Dict[str, Any]],
    original_step_to_unified: Dict[int, int],
    model_field: str,
) -> List[Dict[str, Any]]:
    """For each non-summary '拟合结果' row, if ``row.datasetItemContent[model_field]``
    parses to a critical-error prediction, emit an entry."""
    out: List[Dict[str, Any]] = []
    for row in fitted_rows:
        orig_sid = _row_step_id(row)
        if orig_sid is None or orig_sid < 0:
            continue
        c = row.get("datasetItemContent") or {}
        pred = _parse_model_prediction(c.get(model_field))
        if pred is None:
            continue
        entry = {
            "step": original_step_to_unified.get(orig_sid),
            "original_step_id": orig_sid,
        }
        entry.update(pred)
        out.append(entry)
    out.sort(key=lambda e: (e.get("original_step_id") or 0))
    return out


def _collect_human_annotator_critical_steps(
    per_annotator_rows: List[Dict[str, Any]],
    original_step_to_unified: Dict[int, int],
) -> List[Dict[str, Any]]:
    """For each '标注结果' row flagged as critical, emit one entry per
    (labeler, step). Non-critical rows are skipped."""
    out: List[Dict[str, Any]] = []
    for row in per_annotator_rows:
        orig_sid = _row_step_id(row)
        if orig_sid is None or orig_sid < 0:
            continue
        if not _is_critical_row(row):
            continue
        tags = (row.get("detailLabel") or {}).get("tags") or []
        category = _tag_list(tags, _TAG_ERROR_CATEGORY)
        rationale = _tag_text(tags, _TAG_RATIONALE).strip() or None
        out.append({
            "labeler": row.get("labeler") or "",
            "step": original_step_to_unified.get(orig_sid),
            "original_step_id": orig_sid,
            "error_category": category,
            "mapped_error_type": _map_error_category(category),
            "rationale": rationale,
        })
    out.sort(key=lambda e: (e.get("labeler") or "", e.get("original_step_id") or 0))
    return out


def _build_trajectory(
    juhe: str,
    rows: List[Dict[str, Any]],
    scene_policies: Dict[str, str],
) -> Optional[UnifiedTrajectory]:
    fitted_rows: List[Dict[str, Any]] = [r for r in rows if r.get("resultType") == _RESULT_TYPE_FITTED]
    if not fitted_rows:
        return None

    # Identify the trajectory's task_id (uniform across all rows of one juhe).
    task_id = ""
    for r in rows:
        tid = (r.get("datasetItemContent") or {}).get("task_id")
        if isinstance(tid, str) and tid:
            task_id = tid
            break
    if not task_id:
        return None

    # Locate the summary row (step_id == -1) and the message rows from the
    # fitted (authoritative) subset.
    summary_row: Optional[Dict[str, Any]] = None
    message_rows: List[Tuple[int, Dict[str, Any]]] = []
    for r in fitted_rows:
        sid = _row_step_id(r)
        if sid is None:
            continue
        if sid == -1:
            if summary_row is None:
                summary_row = r
            continue
        message_rows.append((sid, r))

    if not message_rows:
        return None

    def _order_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
        sid, r = item
        c = r.get("datasetItemContent") or {}
        try:
            rid = int(str(c.get("id")))
        except (TypeError, ValueError):
            rid = 0
        return (sid, rid)

    message_rows.sort(key=_order_key)

    # --- Summary text & scene detection ------------------------------------
    summary_text = ""
    if summary_row is not None:
        sc = summary_row.get("datasetItemContent") or {}
        summary_text = str(sc.get("message_content") or "")
    scene = _detect_scene(task_id, summary_text)
    policy_text = scene_policies.get(scene or "", "") if scene else ""

    # --- Build unified messages --------------------------------------------
    unified_messages: List[UnifiedMessage] = []
    # Always insert the system policy as step 0 (even when policy_text is empty,
    # so that the unified-step offset (+1) is deterministic across the dataset).
    unified_messages.append(
        UnifiedMessage(
            step=0,
            role=ROLE_SYSTEM,
            content=policy_text,
            name="system",
        )
    )

    original_step_to_unified: Dict[int, int] = {}

    for sid, r in message_rows:
        c = r.get("datasetItemContent") or {}
        role = _coerce_role(str(c.get("messages_role", "")))
        content = c.get("message_content")
        if role is None:
            continue
        unified_messages.append(
            UnifiedMessage(
                step=len(unified_messages),
                role=role,
                content="" if content is None else str(content),
                name=str(c.get("messages_role") or "") or None,
            )
        )
        # With the system turn at index 0, every subsequent original step_id
        # maps to (step_id + _SYSTEM_STEP_OFFSET).
        original_step_to_unified[sid] = len(unified_messages) - 1

    # Sanity: only the synthesised system message if no other rows mapped.
    if len(unified_messages) <= 1:
        return None

    # --- Annotation (from the FIRST fitted row marked critical) -------------
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
        unified_step = original_step_to_unified.get(orig_sid) if orig_sid is not None else None

        tags = (critical_fitted_row.get("detailLabel") or {}).get("tags") or []
        category = _tag_list(tags, _TAG_ERROR_CATEGORY)
        error_type = _map_error_category(category)

        if isinstance(unified_step, int) and unified_messages[unified_step].role != ROLE_USER:
            annotation.critical_error_step = unified_step
        annotation.critical_error_type = error_type

    # Reward: failed by default; the current file only contains failed tasks.
    reward = 1 if "_success" in task_id.lower() else 0
    if reward == 1 and annotation.critical_error_step is not None:
        annotation = UnifiedAnnotation()  # schema: success -> no critical step

    # --- task_description ---------------------------------------------------
    task_description = _trim_scene_goal(summary_text)

    # Public releases retain only detector context. Internal grouping IDs,
    # generated summaries, per-rater labels, and skipped-row diagnostics are
    # intentionally excluded.
    extra: Dict[str, Any] = {
        "agent_framework_description": AGENT_FRAMEWORK_DESCRIPTION,
    }

    metadata = UnifiedMetadata(
        dataset=DATASET_NAME,
        task_id=task_id,
        task_description=task_description,
        reward=reward,
        annotation=annotation,
        extra=extra,
    )

    return UnifiedTrajectory(messages=unified_messages, metadata=metadata)


def convert_jsonl(
    src_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    *,
    policy_dir: Optional[str | os.PathLike] = None,
) -> List[str]:
    """Convert a tau²-bench JSONL file into unified per-trajectory JSONs.

    ``policy_dir`` must contain ``airline.txt`` and ``retail.txt``. It
    defaults to the parent directory of ``src_path`` (i.e. the
    ``data/Tau^2-Bench/`` directory that already ships the policy files).
    """
    src_path = Path(src_path)
    if not src_path.is_file():
        return []

    policy_path = Path(policy_dir) if policy_dir is not None else src_path.parent
    scene_policies = _load_scene_policies(policy_path)

    buckets = _group_by_juhe(src_path)
    written: List[str] = []

    out_dir_path = Path(out_dir)
    for juhe, rows in sorted(
        buckets.items(),
        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0],
    ):
        traj = _build_trajectory(juhe, rows, scene_policies)
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
    """Compatibility wrapper for ``build_unified_dataset.py``.

    ``src_dir`` may be either the directory containing the tau²-bench JSONL
    file or the JSONL file itself. In both cases we look for the scene
    policy files (``airline.txt`` / ``retail.txt``) next to the JSONL.
    """
    src = Path(src_dir)
    written: List[str] = []
    if src.is_file():
        return convert_jsonl(src, out_dir, policy_dir=src.parent)
    if not src.is_dir():
        return written
    for jsonl_path in sorted(src.glob("*.jsonl")):
        written.extend(convert_jsonl(jsonl_path, out_dir, policy_dir=src))
    return written


__all__ = [
    "DATASET_NAME",
    "AGENT_FRAMEWORK_DESCRIPTION",
    "convert_jsonl",
    "convert_directory",
]
