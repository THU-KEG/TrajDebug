"""Shared converter for the ALFWorld / GAIA / WebShop trajectory shape.

All three datasets share the same on-disk format:

  * ``messages``: a list of OpenAI-style ``{role, content}`` dicts that
    alternate ``user`` / ``assistant``. The first ``user`` message also
    embeds the natural-language task description.
  * ``metadata``: a small dict with ``environment``, ``won`` (the
    ground-truth success bool), and other bookkeeping.
  * Critical-error labels live in a separate file at
    ``data/<dataset>_labels.json``, keyed by ``trajectory_id`` (the
    file stem).

This module turns one such trajectory into the unified schema and
attaches the matching label entry (when one exists).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from data_processing._common import (
    extract_label_module_and_failure_type,
    map_label_to_detector_error_type,
)
from data_processing.schema import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    UnifiedAnnotation,
    UnifiedMessage,
    UnifiedMetadata,
    UnifiedTrajectory,
    assistant_turn_to_step,
    validate_unified,
)


_KNOWN_ROLES = {ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL, ROLE_SYSTEM}

# Datasets whose first `user` message embeds the full tool list /
# instructions / response-format boilerplate in addition to the task
# statement. For these datasets we keep only the task sentence (plus its
# trailing blank line) so downstream unified consumers never see the
# >3k-token boilerplate. All other datasets pass through unchanged.
_TASK_ONLY_DATASETS = {
    "alfworld", "alfworldsub",
    "gaia", "gaiasub",
    "webshop", "webshopsub",
}

def _normalize_role(raw_role: str) -> str:
    if not isinstance(raw_role, str):
        return ROLE_ASSISTANT
    role = raw_role.strip().lower()
    if role in _KNOWN_ROLES:
        return role
    if role in {"human", "task"}:
        return ROLE_USER
    return ROLE_ASSISTANT

def _strip_task_description(task_description: str, dataset: str) -> str:
    """Return only the task statement from the first-user-message blob.

    * gaia / gaiasub: keep the ``Task: ...`` paragraph, cut before the
      ``Available Tools:`` section, end with a trailing blank line so the
      shape matches ``"... 1:01.001.\\n\\n"``.
    * alfworld / alfworldsub / webshop / webshopsub: keep only the
      single ``Your task is: <goal>.`` line; drop the expert-agent
      preamble and the subsequent observation / admissible-action /
      ``<plan>`` boilerplate.
    * Every other dataset (or unexpected shapes) is returned unchanged.
    """
    ds = (dataset or "").strip().lower()
    if ds not in _TASK_ONLY_DATASETS or not task_description:
        return task_description

    text = task_description

    if ds in {"gaia", "gaiasub"}:
        task_idx = text.find("Task:")
        if task_idx < 0:
            return task_description
        tail = text[task_idx:]
        # Cut at the "Available Tools:" block that follows the task
        # paragraph. Prefer the "\n\nAvailable Tools:" form so we
        # preserve the trailing blank-line separator.
        for pat in ("\n\nAvailable Tools:", "\nAvailable Tools:", "Available Tools:"):
            cut = tail.find(pat)
            if cut >= 0:
                kept = tail[:cut].rstrip()
                return kept + "\n\n"
        return tail.strip() + "\n\n"

    # alfworld / alfworldsub / webshop / webshopsub.
    marker = "Your task is:"
    idx = text.find(marker)
    if idx < 0:
        return task_description
    tail = text[idx:]
    newline = tail.find("\n")
    if newline < 0:
        return tail.strip()
    return tail[:newline].strip()

def _extract_task_description(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            break
    return ""


def convert_standard_record(
    raw: Dict[str, Any],
    *,
    dataset: str,
    task_id: str,
    label_entry: Optional[Dict[str, Any]] = None,
    is_valid_full_error_type=None,
    agent_framework_description: str = "",
) -> Dict[str, Any]:
    """Convert one ALFWorld/GAIA/WebShop trajectory dict into the unified schema.

    ``agent_framework_description`` is a short prose paragraph describing
    the dataset's agent framework / turn-taking convention. When non-empty
    it is written into ``metadata.extra["agent_framework_description"]``
    and consumed by Phase 1's per-step prompt to disambiguate roles and
    step semantics (e.g. "subsequent user messages are env feedback, not
    new task instructions"). Leave empty for datasets that do not need
    this hint.
    """
    if is_valid_full_error_type is None:
        from detector.utils.error_definitions import is_valid_full_error_type as _check
        is_valid_full_error_type = _check

    raw_messages = raw.get("messages") or []
    messages: List[UnifiedMessage] = []
    for idx, msg in enumerate(raw_messages):
        if not isinstance(msg, dict):
            msg = {"role": "assistant", "content": str(msg)}
        role = _normalize_role(msg.get("role"))
        content = msg.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        messages.append(
            UnifiedMessage(
                step=idx,
                role=role,
                name=msg.get("name") if isinstance(msg.get("name"), str) else None,
                content=content,
            )
        )

    raw_metadata = raw.get("metadata") or {}
    won = bool(raw_metadata.get("won"))
    reward = 1 if won else 0

    critical_step: Optional[int] = None
    critical_error_type: Optional[str] = None
    if label_entry and reward == 0:
        # ALFWorld/GAIA/WebShop labels store the failure step as the
        # 1-indexed *assistant* turn it occurred on.
        raw_step = label_entry.get("critical_failure_step")
        unified_step = assistant_turn_to_step(
            [m.to_dict() for m in messages], raw_step, one_indexed=True
        )
        if unified_step is not None and messages[unified_step].role != ROLE_USER:
            critical_step = unified_step

        module_str, failure_type = extract_label_module_and_failure_type(label_entry)
        critical_error_type = map_label_to_detector_error_type(
            module_str,
            failure_type,
            is_valid_full_error_type=is_valid_full_error_type,
        )

    annotation = UnifiedAnnotation(
        critical_error_step=None if reward == 1 else critical_step,
        critical_error_type=None if reward == 1 else critical_error_type,
    )

    extra: Dict[str, Any] = {"original_metadata": raw_metadata}
    if label_entry:
        extra["label_entry"] = label_entry
    if agent_framework_description:
        extra["agent_framework_description"] = str(agent_framework_description)

    metadata = UnifiedMetadata(
        dataset=dataset,
        task_id=task_id,
        task_description=_strip_task_description(
            _extract_task_description(raw_messages), dataset
        ),
        reward=reward,
        annotation=annotation,
        extra=extra,
    )

    unified = UnifiedTrajectory(messages=messages, metadata=metadata).to_dict()
    validate_unified(unified)
    return unified


__all__ = ["convert_standard_record"]
