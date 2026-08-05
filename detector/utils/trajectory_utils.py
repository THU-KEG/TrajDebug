"""Trajectory IO and flattening for the unified-schema detector.

The detector now consumes only files emitted by ``data_processing/`` —
i.e. JSON objects with a ``messages`` array (each message has
``step``/``role``/``content`` and optional ``name``) and a ``metadata``
object whose ``annotation`` records ``critical_error_step`` and
``critical_error_type``.

This module exposes:

  * :func:`collect_trajectory_sources` — list every JSON trajectory file
    under a path (file or directory).
  * :func:`load_trajectory_source` — read one trajectory file.
  * :func:`output_stem_for_source` — stable per-trajectory stem used to
    name detector output files.
  * :func:`flatten_unified_trajectory` — turn a unified trajectory into
    the per-step dict list that ``fine_grained_analysis`` and
    ``critical_error_detection`` consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def collect_trajectory_sources(input_path: str) -> List[str]:
    """Return absolute paths for every trajectory file under ``input_path``.

    ``input_path`` may be a single ``.json`` file or a directory; in the
    directory case we recurse through ``*.json``.
    """
    path = Path(input_path)
    if path.is_file():
        if path.suffix != ".json":
            raise ValueError(f"Unsupported trajectory file type: {path}")
        return [str(path)]
    if path.is_dir():
        return [str(p) for p in sorted(path.rglob("*.json"))]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_trajectory_source(source: str) -> Dict[str, Any]:
    """Read one unified trajectory JSON file and stash the source path."""
    with open(source, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Trajectory file is not a JSON object: {source}")
    loaded = dict(data)
    loaded["_trajectory_source"] = source
    return loaded


def output_stem_for_source(source: str) -> str:
    """Stable per-trajectory output stem (the file stem of ``source``)."""
    return Path(source).stem


def flatten_unified_trajectory(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn a unified trajectory into the per-step list the detector uses.

    Output schema (one dict per message in the unified trajectory)::

        {
            "step": <int>,           # 0-indexed message position
            "message_role": <str>,    # `name` if present else `role`
            "role": <str>,            # canonical role (user / assistant / tool / system)
            "name": <str|None>,       # original `name` field (preserves agent identity)
            "content": <str>,
            "judgable": <bool>,       # v5: from data_processing/bake_v5_fields
            "compress_role": <str>,   # v5: "compress" | "preserve" | "default"
        }

    ALL messages are included — including ``user`` messages (task statement
    and any interleaved environment-style observations). The detector treats
    ``user`` messages the same as any other message for history / lookahead
    purposes; the only place they are filtered out is the per-step
    error-detection dispatch in :class:`ErrorTypeDetector.analyze_trajectory`,
    which never targets ``user`` / ``human`` messages as detection candidates.
    Keeping them in the flat list guarantees ``total_steps`` matches the
    original ``len(messages)`` of the unified trajectory.

    ``judgable`` defaults to ``True`` and ``compress_role`` defaults to
    ``"default"`` when the unified file pre-dates ``data_processing/
    bake_v5_fields.py``; rerun that script to populate them properly.
    """
    messages = data.get("messages") or []
    if not isinstance(messages, list):
        return []

    flat: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        try:
            step = int(msg.get("step"))
        except (TypeError, ValueError):
            step = idx
        content = msg.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        name = msg.get("name") if isinstance(msg.get("name"), str) and msg.get("name") else None

        judgable_raw = msg.get("judgable")
        if isinstance(judgable_raw, bool):
            judgable = judgable_raw
        else:
            # Legacy file: be conservative and only block roles that are
            # globally non-error (user / human). Bake the file properly to
            # honour dataset-level overrides like whoandwhensub's tool-bearing
            # WebSurfer.
            judgable = role not in {"user"} and (name not in {"human", "user_proxy"})

        compress_role_raw = msg.get("compress_role")
        if isinstance(compress_role_raw, str) and compress_role_raw in {"compress", "preserve", "default"}:
            compress_role = compress_role_raw
        else:
            compress_role = "default"

        flat.append({
            "step": step,
            "message_role": name or role or "assistant",
            "role": role or "assistant",
            "name": name,
            "content": content,
            "judgable": bool(judgable),
            "compress_role": compress_role,
        })
    return flat
