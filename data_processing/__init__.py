"""Unified dataset conversion layer.

Each supported dataset (GAIA, ALFWorld, WebShop, WhoAndWhen) has a
converter module under this package that emits per-trajectory JSON files
in the unified `messages + metadata` schema defined in
:mod:`data_processing.schema`. The downstream detector
(``detector/``) consumes only that unified format.
"""

from data_processing.schema import (  # noqa: F401
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_SYSTEM,
    UnifiedMessage,
    UnifiedAnnotation,
    UnifiedMetadata,
    UnifiedTrajectory,
    validate_unified,
    iter_assistant_indices,
    assistant_turn_to_step,
    step_to_assistant_turn,
)
