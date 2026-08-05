"""ALFWorld → unified schema converter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from data_processing._common import dump_json, index_labels_by_trajectory_id, load_json
from data_processing._standard_messages import convert_standard_record


DATASET_NAME = "alfworld"

# Prose injected into Phase 1 prompts via
# ``metadata.extra["agent_framework_description"]``. Shared verbatim with
# gaia.py and webshop.py because all three datasets use the same
# single-agent, 2-role, turn-based chat convention.
# Prose injected into Phase 1 prompts via
# ``metadata.extra["agent_framework_description"]``. Two constants live
# here:
#
#  * ``_SHARED_TURN_BASED_PREAMBLE`` — the generic 2-role, turn-based
#    description that is also consumed by gaia.py and webshop.py.
#  * ``AGENT_FRAMEWORK_DESCRIPTION`` — ALFWorld-specific; preamble plus a
#    trailing "DATASET NOTES" paragraph covering environment conventions
#    that matter for Stage B judgement (what the initial observation
#    contains, what an out-of-list action means, when a repeated action
#    is a retry vs. an informative probe).
#
# gaia.py / webshop.py still import ``_SHARED_TURN_BASED_PREAMBLE`` so
# their descriptions do NOT inherit ALFWorld-specific notes.
_SHARED_TURN_BASED_PREAMBLE = (
    "This trajectory comes from a turn-based environment with exactly two roles. "
    "The FIRST `user` message states the task. Every SUBSEQUENT `user` message is "
    "the environment / tool feedback (observation, admissible-action list, tool "
    "output, error string) — never a new task-level instruction. All `assistant` "
    "messages are produced by the SAME single agent: they contain the agent's "
    "planning, reasoning, and the action string the environment will execute "
    "next. When the agent self-narrates 'step N', it usually means the N-th "
    "assistant turn in a 1-indexed count; in this prompt we always refer to "
    "steps by their message index in the trajectory."
)

AGENT_FRAMEWORK_DESCRIPTION = (
    _SHARED_TURN_BASED_PREAMBLE
    + "\n\nDATASET NOTES (ALFWorld). Two observation tiers exist:\n"
    "  (a) the ROOM-level observation (the welcome screen, or after `look` "
    "in the open room) lists ONLY large furniture (desk, bed, drawer, "
    "sidetable, shelf, ...). Small task-relevant objects (desklamp, "
    "book, keychain, pencil) are NOT named at the room level — their "
    "absence at room level does not mean they don't exist; the agent has "
    "simply not approached any furniture yet.\n"
    "  (b) the FURNITURE-level observation (returned by `go to "
    "<furniture>`, and re-shown by `examine <furniture>`) is a complete "
    "surface inventory for that furniture. If a task-required object is "
    "not in that inventory, it is not on that furniture.\n"
    "Action repetition: calling `examine <furniture>` immediately after "
    "`go to <furniture>` typically returns the same surface inventory — "
    "the second occurrence is a redundant probe rather than a clear "
    "no-progress signal. A clear no-progress signal in this environment "
    "is a literal `Nothing happens.` reply, an empty payload, or a "
    "re-observation that is identical to a known earlier observation "
    "with no state change in between.\n"
    "Admissible-action list: each user observation includes an "
    "admissible-actions list. It is a hint about what the environment "
    "currently supports; the agent issuing an action outside the list "
    "is exploration unless HISTORY already shows that same action "
    "produced a bad outcome."
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
