import json
from pathlib import Path

from data_processing.schema import validate_unified


REPO_ROOT = Path(__file__).resolve().parents[1]
SWE_BENCH_PRO = REPO_ROOT / "data" / "unified" / "swebenchpro"


def test_swebenchpro_release_has_86_valid_records():
    paths = sorted(SWE_BENCH_PRO.glob("*.json"))
    assert len(paths) == 86

    task_ids = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_unified(payload)
        messages = payload["messages"]
        metadata = payload["metadata"]
        annotation = metadata["annotation"]
        step = annotation["critical_error_step"]

        assert metadata["dataset"] == "swebenchpro"
        assert metadata["task_id"] not in task_ids
        assert isinstance(step, int) and 0 <= step < len(messages)
        assert messages[step]["role"] == "assistant"
        assert "critical_step_labels" not in metadata.get("extra", {})
        task_ids.add(metadata["task_id"])


def test_removed_swebenchpro_variant_is_absent():
    removed_variant = "swebenchpro" + "4model"
    assert not (SWE_BENCH_PRO.parent / removed_variant).exists()
