"""GAIA → unified schema converter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from data_processing._common import dump_json, index_labels_by_trajectory_id, load_json
from data_processing._standard_messages import convert_standard_record
from data_processing.alfworld import _SHARED_TURN_BASED_PREAMBLE


DATASET_NAME = "gaia"

# GAIA uses the shared 2-role preamble plus a dataset-specific "DATASET
# NOTES" paragraph for Stage B judgement. Unlike ALFWorld the environment
# is the open web / tool APIs, so the hints focus on final-answer
# epistemics (F1/F2) and how to treat partial search evidence.
AGENT_FRAMEWORK_DESCRIPTION = (
    _SHARED_TURN_BASED_PREAMBLE
    + "\n\nDATASET NOTES (GAIA). Each task expects a single, specific "
    "answer (a number, a name, a date, a short phrase). The agent's "
    "final assistant step typically returns that answer verbatim. "
    "GAIA tasks are knowledge / web-research oriented: the agent has "
    "access to web search and other research tools, and the load-"
    "bearing fact for the final answer is expected to come from those "
    "tool observations rather than from the agent's parametric "
    "knowledge. Web search snippets in HISTORY are not authoritative "
    "evidence on their own — they list candidate sources but do not "
    "always contain the queried entity's value; an answer must point "
    "to the specific snippet / page passage that names the queried "
    "entity. When the task names a specific source (e.g. \"According "
    "to Google Finance\", \"per Wikipedia\"), that source name is part "
    "of the task constraint, not just a hint."
)


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
