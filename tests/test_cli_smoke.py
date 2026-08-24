import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    "detector/stage_a_diagnosis.py",
    "detector/stage_b_per_step.py",
    "detector/stage_c_phase1_cluster.py",
    "detector/stage_c_phase2_state.py",
    "detector/stage_c_phase3_assemble.py",
    "detector/score_steps.py",
    "applications/generate_feedback.py",
]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_entrypoint_help(entrypoint):
    result = subprocess.run(
        [sys.executable, entrypoint, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
