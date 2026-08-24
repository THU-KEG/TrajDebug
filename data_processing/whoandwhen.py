"""WhoAndWhen → unified schema converter.

The native WhoAndWhen format stores everything in a single ``history``
list of ``{role, content}`` dicts plus inline labels (``mistake_step``,
``mistake_agent``, ``mistake_reason``, ``is_corrected``). We map
``history`` 1:1 into ``messages`` so the unified ``step`` index simply
equals the original history index used by the inline label.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_processing._common import dump_json, load_json
from data_processing.schema import (
    ROLE_ASSISTANT,
    ROLE_USER,
    UnifiedAnnotation,
    UnifiedMessage,
    UnifiedMetadata,
    UnifiedTrajectory,
    validate_unified,
)


DATASET_NAME = "whoandwhen"

# Prose injected into Phase 1 prompts via
# ``metadata.extra["agent_framework_description"]``. WhoAndWhen uses a
# multi-agent system (Orchestrator + WebSurfer + FileSurfer + ...), so
# this description is materially different from the alfworld / gaia /
# webshop one.
AGENT_FRAMEWORK_DESCRIPTION = (
    "This trajectory is from a multi-agent system. Each assistant message's "
    "`role` field names which agent spoke. Canonical roles and what they do:\n"
    "  - Orchestrator / MagenticOneOrchestrator: plans and delegates sub-tasks.\n"
    "  - WebSurfer: browses the web via a headless browser.\n"
    "  - FileSurfer: reads / summarises local files.\n"
    "  - Coder: writes and executes code.\n"
    "  - ComputerTerminal: executes shell commands / returns tool output.\n"
    "  - user_proxy: forwards messages (often carries tool output).\n"
    "  - human: the task owner.\n\n"
    "DATASET NOTES (WhoAndWhen). The original TASK comes from `human` "
    "and is the binding constraint for the whole run. The Orchestrator "
    "subsequently dispatches sub-tasks to other agents; its dispatch "
    "messages often paraphrase the TASK. When the Orchestrator's "
    "paraphrase is STRICTLY NARROWER than the TASK, the narrower "
    "phrasing applies for that sub-task. When the Orchestrator's "
    "paraphrase is BROADER or LOOSER than the TASK (e.g. TASK says "
    "\"According to Google Finance\" but the Orchestrator writes "
    "\"using Google Finance or another credible financial resource\"), "
    "the original TASK constraint still governs the sub-agent's choice "
    "— a sub-agent that proceeds directly with a non-TASK source, "
    "without first attempting the TASK-named source, is acting against "
    "the TASK regardless of the Orchestrator's loosened phrasing. "
    "Sub-agents have specialised, bounded capabilities (WebSurfer: "
    "browser navigation and OCR-style page reads; FileSurfer: reads "
    "files; Coder: writes and runs code). Both the TASK and any "
    "leading role==system policy are binding when judging a sub-agent "
    "step; the Orchestrator's dispatch is contextual."
)


def _normalize_role(raw_role: str) -> str:
    """Map a WhoAndWhen ``history`` role to the unified ``role`` field.

    WhoAndWhen uses ``human`` for the user; everything else
    (``Orchestrator (thought)``, ``WebSurfer``, ``FileSurfer`` …) is an
    agent message and becomes ``assistant``. The original speaker label
    is preserved in ``message.name`` so the detector prompts can keep
    the role-aware framing.
    """
    if not isinstance(raw_role, str):
        return ROLE_ASSISTANT
    return ROLE_USER if raw_role.strip().lower() == "human" else ROLE_ASSISTANT


def convert_record(raw: Dict[str, Any], *, task_id: str) -> Dict[str, Any]:
    history = raw.get("history") or []
    messages: List[UnifiedMessage] = []
    for idx, entry in enumerate(history):
        if not isinstance(entry, dict):
            entry = {"role": "assistant", "content": str(entry)}
        role_raw = entry.get("role") or "assistant"
        content = entry.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        messages.append(
            UnifiedMessage(
                step=idx,
                role=_normalize_role(role_raw),
                name=role_raw if isinstance(role_raw, str) else None,
                content=content,
            )
        )

    is_corrected = bool(raw.get("is_corrected"))
    reward = 1 if is_corrected else 0

    mistake_step_raw = raw.get("mistake_step")
    critical_step: Optional[int] = None
    try:
        mistake_step_int = int(mistake_step_raw) if mistake_step_raw is not None else None
    except (TypeError, ValueError):
        mistake_step_int = None
    if (
        mistake_step_int is not None
        and 0 <= mistake_step_int < len(messages)
        and messages[mistake_step_int].role != ROLE_USER
    ):
        critical_step = mistake_step_int

    annotation = UnifiedAnnotation(
        critical_error_step=None if reward == 1 else critical_step,
        critical_error_type=None,
    )

    metadata = UnifiedMetadata(
        dataset=DATASET_NAME,
        task_id=task_id,
        task_description=str(raw.get("question") or ""),
        reward=reward,
        annotation=annotation,
        extra={
            "groundtruth": raw.get("groundtruth"),
            "level": raw.get("level"),
            "question_ID": raw.get("question_ID"),
            "mistake_agent": raw.get("mistake_agent"),
            "mistake_reason": raw.get("mistake_reason"),
            "mistake_step_native": raw.get("mistake_step"),
            "agent_framework_description": AGENT_FRAMEWORK_DESCRIPTION,
        },
    )

    unified = UnifiedTrajectory(messages=messages, metadata=metadata).to_dict()
    validate_unified(unified)
    return unified


def convert_directory(src_dir: str | os.PathLike, out_dir: str | os.PathLike) -> List[str]:
    src = Path(src_dir)
    written: List[str] = []
    if not src.is_dir():
        return written
    for path in sorted(src.glob("*.json")):
        raw = load_json(path)
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("question_ID") or path.stem)
        unified = convert_record(raw, task_id=task_id)
        out_path = Path(out_dir) / f"{path.stem}.json"
        dump_json(unified, out_path)
        written.append(str(out_path))
    return written


def _adapt_algorithm_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt the ``whoandwhen_algorithm`` schema to the canonical
    WhoAndWhen schema expected by :func:`convert_record`.

    Differences observed in ``data/whoandwhen_algorithm/*.json``:
      - ``is_correct``  -> ``is_corrected``
      - ``ground_truth`` -> ``groundtruth``
      - ``history[i]`` carries an explicit ``name`` (the agent label)
        alongside the generic ``role`` (``"assistant"``/``"human"``).
        The base ``convert_record`` reads the speaker from
        ``role`` and stores it as ``message.name``, so we promote the
        ``name`` field into ``role`` to preserve the agent identity.
      - extra ``system_prompt`` is preserved under ``extra``.
    The returned dict is a shallow copy; the original ``history`` list
    is rebuilt with patched entries (no in-place mutation of the input).
    """
    adapted: Dict[str, Any] = dict(raw)

    if "is_corrected" not in adapted and "is_correct" in raw:
        adapted["is_corrected"] = raw.get("is_correct")
    if "groundtruth" not in adapted and "ground_truth" in raw:
        adapted["groundtruth"] = raw.get("ground_truth")

    history = raw.get("history") or []
    new_history: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            new_history.append(entry)
            continue
        patched = dict(entry)
        role = patched.get("role")
        name = patched.get("name")
        # Only promote ``name`` when ``role`` is the generic
        # "assistant"/"user" tag and a more specific agent name exists.
        if (
            isinstance(name, str)
            and name.strip()
            and isinstance(role, str)
            and role.strip().lower() in {"assistant", "user"}
        ):
            patched["role"] = name
        new_history.append(patched)
    adapted["history"] = new_history
    return adapted


def convert_directory_algorithm(
    src_dir: str | os.PathLike, out_dir: str | os.PathLike
) -> List[str]:
    """Same as :func:`convert_directory` but for the
    ``whoandwhen_algorithm`` dataset variant which uses slightly
    different field names."""
    src = Path(src_dir)
    written: List[str] = []
    if not src.is_dir():
        return written
    for path in sorted(src.glob("*.json")):
        raw = load_json(path)
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("question_ID") or path.stem)
        adapted = _adapt_algorithm_record(raw)
        unified = convert_record(adapted, task_id=task_id)
        # Preserve algorithm-specific extras that the base record does
        # not know about.
        if isinstance(raw.get("system_prompt"), (dict, list, str)):
            unified.setdefault("metadata", {}).setdefault("extra", {})[
                "system_prompt"
            ] = raw.get("system_prompt")
        out_path = Path(out_dir) / f"{path.stem}.json"
        dump_json(unified, out_path)
        written.append(str(out_path))
    return written


__all__ = [
    "DATASET_NAME",
    "AGENT_FRAMEWORK_DESCRIPTION",
    "convert_record",
    "convert_directory",
    "convert_directory_algorithm",
]
