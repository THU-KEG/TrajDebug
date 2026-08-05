"""WebShop → unified schema converter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from data_processing._common import dump_json, index_labels_by_trajectory_id, load_json
from data_processing._standard_messages import convert_standard_record
from data_processing.alfworld import _SHARED_TURN_BASED_PREAMBLE


DATASET_NAME = "webshop"

# WebShop reuses the shared 2-role preamble unmodified — no dataset-
# specific hints yet. Add a "DATASET NOTES" paragraph here later if
# Stage B starts producing a clear pattern of WebShop misfires.
AGENT_FRAMEWORK_DESCRIPTION = _SHARED_TURN_BASED_PREAMBLE


def convert_directory(
    src_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    label_path: str | os.PathLike,
) -> List[str]:
    labels = index_labels_by_trajectory_id(label_path) if label_path else {}
    src = Path(src_dir)
    written: List[str] = []
    if not src.is_dir():
        return written
    for path in sorted(src.glob("*.json")):
        raw = load_json(path)
        if not isinstance(raw, dict):
            continue
        unified = convert_standard_record(
            raw,
            dataset=DATASET_NAME,
            task_id=path.stem,
            label_entry=labels.get(path.stem),
            agent_framework_description=AGENT_FRAMEWORK_DESCRIPTION,
        )
        out_path = Path(out_dir) / f"{path.stem}.json"
        dump_json(unified, out_path)
        written.append(str(out_path))
    return written


__all__ = ["DATASET_NAME", "AGENT_FRAMEWORK_DESCRIPTION", "convert_directory"]
