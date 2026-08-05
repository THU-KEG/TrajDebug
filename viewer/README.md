# Trajectory diagnosis viewer

The viewer is a local web interface for inspecting unified trajectories together with Stage C diagnoses and staged reports. It reads local files only.

## Install and start

From the repository root:

```bash
pip install -e ".[viewer]"
python -m viewer.server --dataset alfworld --output-dir outputs
```

Open <http://localhost:8000>.

## File layout

For `<dataset>` and `<task_id>`, the viewer reads:

```text
data/unified/<dataset>/<task_id>.json
outputs/<dataset>_phase1/<task_id>_stage_c_phase1.json
outputs/<dataset>_phase2/<task_id>_stage_c_phase2.json
outputs/<dataset>_report/<task_id>_report.json
```

The unified trajectory is required. Phase 1, Phase 2, and report files are optional, so partially completed runs remain inspectable.

## Options

- `--dataset`: dataset selected on startup.
- `--output-dir`: detector output root; default `outputs`.
- `--unified-root`: unified data root; default `data/unified`.
- `--host`: bind address; default `127.0.0.1`.
- `--port`: web port; default `8000`.

Reports can be produced by the optional paper Section 6 application at `applications/generate_feedback.py`; it is not part of the default pipeline. When a report is available, the viewer displays its diagnosis and the actionable guidance from the final report's `fix_suggestion.hint_sentence` field.
