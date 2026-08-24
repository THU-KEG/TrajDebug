import json

from fastapi.testclient import TestClient

from viewer import server


def test_viewer_reads_public_output_layout(tmp_path, monkeypatch):
    unified_root = tmp_path / "data" / "unified"
    output_root = tmp_path / "outputs"
    dataset = "demo"
    task_id = "task-1"

    (unified_root / dataset).mkdir(parents=True)
    (output_root / f"{dataset}_report").mkdir(parents=True)
    (unified_root / dataset / f"{task_id}.json").write_text(
        json.dumps(
            {
                "messages": [{"step": 0, "role": "assistant", "content": "answer"}],
                "metadata": {
                    "dataset": dataset,
                    "task_id": task_id,
                    "annotation": {"critical_error_step": 0},
                },
            }
        ),
        encoding="utf-8",
    )
    (output_root / f"{dataset}_report" / f"{task_id}_report.json").write_text(
        json.dumps({"critical_error_analysis": {"critical_step": 0}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "BASE_DIR", tmp_path)
    datasets = server._list_datasets("outputs", "data/unified")
    assert datasets == [dataset]

    response = server.api_trajectory(
        dataset=dataset,
        task_id=task_id,
        output_dir="outputs",
        unified_root="data/unified",
    )
    assert response["report"]["critical_error_analysis"]["critical_step"] == 0
    assert response["files"]["report"].endswith(
        f"outputs/{dataset}_report/{task_id}_report.json"
    )


def test_viewer_http_smoke():
    response = TestClient(server.app).get("/api/config")
    assert response.status_code == 200
    assert "datasets" in response.json()
