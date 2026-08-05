"""Shared helpers for the unified-format converters.

These helpers do *not* know about specific datasets. They handle JSON
IO, label-file extraction (the standard ALFWorld/GAIA/WebShop label
schema), and mapping label-file modules + failure types into the
canonical detector taxonomy.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# Map label-file module strings -> canonical detector module names.
# `system` (e.g. step_limit) has no detector counterpart and stays None
# so the converter records `critical_error_type = None`.
LABEL_MODULE_TO_DETECTOR_MODULE: Dict[str, Optional[str]] = {
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
    "system": None,
}


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def index_labels_by_trajectory_id(label_path: str | os.PathLike) -> Dict[str, Dict[str, Any]]:
    """Read a standard ALFWorld/GAIA/WebShop label file and index by ``trajectory_id``."""
    raw = load_json(label_path)
    if not isinstance(raw, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        tid = entry.get("trajectory_id")
        if isinstance(tid, str) and tid:
            out[tid] = entry
    return out


def _slugify(text: str) -> str:
    """Convert an arbitrary failure-type string into PascalCase suitable
    for matching the detector taxonomy (which uses PascalCase like
    ``WrongTool``)."""
    if not isinstance(text, str):
        return ""
    parts = re.split(r"[^A-Za-z0-9]+", text.strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def map_label_to_detector_error_type(
    label_module: Optional[str],
    label_failure_type: Optional[str],
    *,
    is_valid_full_error_type,
) -> Optional[str]:
    """Try to translate a ``(label_module, failure_type)`` pair into a
    canonical ``"<module>.<subtype>"`` string from the detector taxonomy.

    Returns ``None`` when either side cannot be mapped — that signals
    `score_steps` to fall back to step-only accuracy for this trajectory.

    ``is_valid_full_error_type`` is injected to keep this module free of
    hard imports from ``detector/``.
    """
    if not label_module:
        return None
    detector_module = LABEL_MODULE_TO_DETECTOR_MODULE.get(label_module.strip().lower())
    if not detector_module:
        return None

    slug = _slugify(label_failure_type or "")
    if slug:
        candidate = f"{detector_module}.{slug}"
        if is_valid_full_error_type(candidate):
            return candidate
    return None


def extract_label_module_and_failure_type(label_entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Pull ``critical_failure_module`` and the matching nested
    ``failure_type`` out of a label entry.

    The label files store the per-step annotation under a key whose name
    equals the module string, e.g.::

        {
            "step": 5,
            "planning": {"failure_type": "inefficient_plan", "reasoning": "..."}
        }

    This helper returns the most-relevant `(module, failure_type)` pair
    for the critical step.
    """
    module = label_entry.get("critical_failure_module")
    module_str = module.strip().lower() if isinstance(module, str) else None

    failure_type: Optional[str] = None
    step_annotations = label_entry.get("step_annotations")
    critical_step = label_entry.get("critical_failure_step")
    if isinstance(step_annotations, list):
        for ann in step_annotations:
            if not isinstance(ann, dict):
                continue
            if ann.get("step") != critical_step:
                continue
            for key, value in ann.items():
                if key == "step":
                    continue
                if not isinstance(value, dict):
                    continue
                if module_str and key.strip().lower() != module_str:
                    continue
                ft = value.get("failure_type")
                if isinstance(ft, str) and ft.strip():
                    failure_type = ft.strip()
                    break
            if failure_type:
                break

    return module_str, failure_type
