"""Unified trajectory schema for the EvalAgent critical-error detector.

A unified trajectory is a single JSON object with two top-level keys:

```
{
  "messages": [
    {"step": 0, "role": "user",      "name": "human",        "content": "..."},
    {"step": 1, "role": "assistant", "name": "Orchestrator", "content": "..."},
    {"step": 2, "role": "assistant", "name": "WebSurfer",    "content": "..."}
  ],
  "metadata": {
    "dataset": "whoandwhen",
    "task_id": "...",
    "task_description": "...",
    "reward": 0,
    "annotation": {
      "critical_error_step": 12,
      "critical_error_type": "act.WrongTool"
    },
    "extra": {}
  }
}
```

Hard rules:
  * ``step`` is the 0-indexed position of the message in the
    ``messages`` array. It MUST equal its array index.
  * ``role`` is one of ``user`` / ``assistant`` / ``tool`` / ``system``
    (coarse OpenAI-style). ``name`` (optional) preserves the original
    speaker label for the role-aware detector prompts.
  * ``role == "user"`` slots are kept but can never be the
    ``critical_error_step``. They represent human / task statements.
  * ``critical_error_step`` is the unified ``step`` of the message that
    introduced the critical error, or ``None`` for success trajectories.
  * ``critical_error_type`` is ``"<module>.<subtype>"`` from the canonical
    detector taxonomy in ``detector/utils/error_definitions.py``,
    or ``None`` when the source label cannot be cleanly mapped (in which
    case ``score_steps`` only reports step accuracy for that trajectory).
  * ``reward`` is ``0`` (failure) or ``1`` (success).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
ROLE_SYSTEM = "system"

VALID_ROLES = {ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL, ROLE_SYSTEM}


# ---------------------------------------------------------------------------
# Lightweight in-memory dataclasses (the on-disk format is always JSON dicts).
# ---------------------------------------------------------------------------
@dataclass
class UnifiedMessage:
    step: int
    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "step": int(self.step),
            "role": str(self.role),
            "content": str(self.content if self.content is not None else ""),
        }
        if self.name:
            out["name"] = str(self.name)
        return out


@dataclass
class UnifiedAnnotation:
    critical_error_step: Optional[int] = None
    critical_error_type: Optional[str] = None
    # Optional free-form rationale from the original human annotator.
    # Not consumed by the detector; surfaced for analysis / diagnostics.
    human_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "critical_error_step": (
                int(self.critical_error_step)
                if isinstance(self.critical_error_step, int)
                else None
            ),
            "critical_error_type": (
                str(self.critical_error_type)
                if isinstance(self.critical_error_type, str) and self.critical_error_type
                else None
            ),
        }
        if isinstance(self.human_rationale, str) and self.human_rationale.strip():
            out["human_rationale"] = self.human_rationale
        return out


@dataclass
class UnifiedMetadata:
    dataset: str
    task_id: str
    task_description: str
    reward: int
    annotation: UnifiedAnnotation = field(default_factory=UnifiedAnnotation)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": str(self.dataset),
            "task_id": str(self.task_id),
            "task_description": str(self.task_description or ""),
            "reward": int(self.reward),
            "annotation": self.annotation.to_dict(),
            "extra": dict(self.extra or {}),
        }


@dataclass
class UnifiedTrajectory:
    messages: List[UnifiedMessage]
    metadata: UnifiedMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata.to_dict(),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class UnifiedSchemaError(ValueError):
    """Raised when a dict does not conform to the unified schema."""


def validate_unified(data: Dict[str, Any]) -> None:
    """Raise :class:`UnifiedSchemaError` if ``data`` violates the schema.

    The function is intentionally strict: it is the contract the rest of
    the pipeline relies on, so silent acceptance of malformed records
    would propagate confusing failures downstream.
    """
    if not isinstance(data, dict):
        raise UnifiedSchemaError("trajectory must be a JSON object")

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise UnifiedSchemaError("messages must be a non-empty list")

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise UnifiedSchemaError(f"messages[{idx}] is not an object")
        step = msg.get("step")
        if not isinstance(step, int) or step != idx:
            raise UnifiedSchemaError(
                f"messages[{idx}].step must equal {idx}, got {step!r}"
            )
        role = msg.get("role")
        if role not in VALID_ROLES:
            raise UnifiedSchemaError(
                f"messages[{idx}].role must be one of {sorted(VALID_ROLES)}, got {role!r}"
            )
        if "content" not in msg or not isinstance(msg["content"], str):
            raise UnifiedSchemaError(
                f"messages[{idx}].content must be a string"
            )
        if "name" in msg and msg["name"] is not None and not isinstance(msg["name"], str):
            raise UnifiedSchemaError(
                f"messages[{idx}].name must be a string when present"
            )

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise UnifiedSchemaError("metadata must be a JSON object")

    for required in ("dataset", "task_id"):
        value = metadata.get(required)
        if not isinstance(value, str) or not value.strip():
            raise UnifiedSchemaError(f"metadata.{required} must be a non-empty string")

    reward = metadata.get("reward")
    if reward not in (0, 1):
        raise UnifiedSchemaError(f"metadata.reward must be 0 or 1, got {reward!r}")

    annotation = metadata.get("annotation")
    if not isinstance(annotation, dict):
        raise UnifiedSchemaError("metadata.annotation must be a JSON object")

    ces = annotation.get("critical_error_step")
    cet = annotation.get("critical_error_type")
    if ces is not None:
        if not isinstance(ces, int):
            raise UnifiedSchemaError(
                f"metadata.annotation.critical_error_step must be int or null, got {ces!r}"
            )
        if not (0 <= ces < len(messages)):
            raise UnifiedSchemaError(
                f"metadata.annotation.critical_error_step={ces} out of range [0,{len(messages)})"
            )
        if messages[ces].get("role") == ROLE_USER:
            raise UnifiedSchemaError(
                f"metadata.annotation.critical_error_step={ces} points at a 'user' message"
            )
    if cet is not None and not (isinstance(cet, str) and cet.strip()):
        raise UnifiedSchemaError(
            f"metadata.annotation.critical_error_type must be a non-empty string or null, got {cet!r}"
        )

    hr = annotation.get("human_rationale")
    if hr is not None and not isinstance(hr, str):
        raise UnifiedSchemaError(
            f"metadata.annotation.human_rationale must be a string when present, got {hr!r}"
        )

    if reward == 1 and ces is not None:
        raise UnifiedSchemaError(
            "metadata.annotation.critical_error_step must be null when reward == 1"
        )


# ---------------------------------------------------------------------------
# Step <-> assistant-turn helpers
# ---------------------------------------------------------------------------
def iter_assistant_indices(messages: Iterable[Dict[str, Any]]) -> List[int]:
    """Return the 0-indexed positions of every ``assistant`` message."""
    return [
        idx
        for idx, msg in enumerate(messages)
        if isinstance(msg, dict) and msg.get("role") == ROLE_ASSISTANT
    ]


def assistant_turn_to_step(
    messages: Iterable[Dict[str, Any]],
    assistant_turn: int,
    *,
    one_indexed: bool = True,
) -> Optional[int]:
    """Convert a 1- or 0-indexed assistant-turn count into a unified step.

    ALFWorld / GAIA / WebShop label files express ``critical_failure_step``
    as the N-th assistant message (1-indexed). This helper translates that
    to a position inside the unified ``messages`` array.

    Returns ``None`` when the requested assistant turn does not exist.
    """
    if not isinstance(assistant_turn, int):
        return None
    indices = iter_assistant_indices(messages)
    target = assistant_turn - 1 if one_indexed else assistant_turn
    if target < 0 or target >= len(indices):
        return None
    return indices[target]


def step_to_assistant_turn(
    messages: Iterable[Dict[str, Any]],
    step: int,
    *,
    one_indexed: bool = True,
) -> Optional[int]:
    """Inverse of :func:`assistant_turn_to_step`. Returns ``None`` when
    ``messages[step]`` is not an assistant message."""
    msgs = list(messages)
    if not (0 <= step < len(msgs)):
        return None
    if msgs[step].get("role") != ROLE_ASSISTANT:
        return None
    indices = iter_assistant_indices(msgs)
    try:
        position = indices.index(step)
    except ValueError:
        return None
    return position + 1 if one_indexed else position
