"""Local web app for inspecting agent trajectories alongside detector diagnoses.

Run:
    python -m viewer.server --dataset alfworld --output-dir outputs
    # then open http://localhost:8000

Given a dataset name and an output folder, the server locates three files per
trajectory and serves them to a single-page UI:

    unified trajectory : <unified_root>/<dataset>/<task_id>.json
    Phase 1            : <output_dir>/<dataset>_phase1/<task_id>_stage_c_phase1.json
    Phase 2            : <output_dir>/<dataset>_phase2/<task_id>_stage_c_phase2.json
    Feedback report    : <output_dir>/<dataset>_report/<task_id>_report.json

Everything is read from local disk only; nothing is uploaded or fetched.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "Missing dependency. Install the viewer requirements first:\n"
        "    pip install fastapi uvicorn\n"
        "(or: pip install -r viewer/requirements.txt)"
    ) from exc


HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

# Only allow simple identifiers for dataset / task_id so they cannot escape the
# configured root directories via slashes or "..".
SAFE_NAME = re.compile(r"^[A-Za-z0-9._=+-]+$")

# Extract metadata.annotation.critical_error_step from a Phase-1 file without a
# full JSON parse (those files can be hundreds of KB). The annotation block has
# no nested braces before the field, so a non-greedy scan is safe.
_GT_RE = re.compile(
    r'"annotation"\s*:\s*\{[^{}]*?"critical_error_step"\s*:\s*(\d+|null)',
    re.S,
)

# Runtime defaults, set from CLI args in main().
BASE_DIR = Path.cwd()
DEFAULT_DATASET = ""
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_UNIFIED_ROOT = "data/unified"

# mtime-keyed cache for the cheap ground-truth scan used by the list endpoint.
_GT_CACHE: dict[str, tuple[float, int | None]] = {}

app = FastAPI(title="Trajectory Diagnosis Viewer")


def _safe_name(name: str, what: str) -> str:
    if not name or not SAFE_NAME.match(name):
        raise HTTPException(status_code=400, detail=f"invalid {what}: {name!r}")
    return name


def _resolve_root(root: str) -> Path:
    """Resolve an output / unified root (CLI- or query-supplied)."""
    p = Path(root)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


def _read_gt_step(phase1_path: Path) -> int | None:
    """Cheaply read the ground-truth critical step from a Phase-1 file."""
    try:
        st = phase1_path.stat()
    except OSError:
        return None
    key = str(phase1_path)
    cached = _GT_CACHE.get(key)
    if cached is not None and cached[0] == st.st_mtime:
        return cached[1]
    gt: int | None = None
    try:
        text = phase1_path.read_text(encoding="utf-8")
        m = _GT_RE.search(text)
        if m and m.group(1) != "null":
            gt = int(m.group(1))
    except (OSError, ValueError):
        gt = None
    _GT_CACHE[key] = (st.st_mtime, gt)
    return gt


def _list_datasets(output_dir: str, unified_root: str) -> list[str]:
    names: set[str] = set()
    out = _resolve_root(output_dir)
    suffixes = ("_report", "_phase1", "_final", "_phase2", "_stage_b")
    if out.is_dir():
        for d in out.iterdir():
            if not d.is_dir():
                continue
            for suf in suffixes:
                if d.name.endswith(suf):
                    names.add(d.name[: -len(suf)])
                    break
    uni = _resolve_root(unified_root)
    if uni.is_dir():
        for d in uni.iterdir():
            if d.is_dir():
                names.add(d.name)
    return sorted(names)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def api_config(output_dir: Optional[str] = None, unified_root: Optional[str] = None):
    out = output_dir or DEFAULT_OUTPUT_DIR
    uni = unified_root or DEFAULT_UNIFIED_ROOT
    return {
        "default_dataset": DEFAULT_DATASET,
        "output_dir": out,
        "unified_root": uni,
        "datasets": _list_datasets(out, uni),
    }


@app.get("/api/trajectories")
def api_trajectories(
    dataset: str,
    output_dir: Optional[str] = None,
    unified_root: Optional[str] = None,
):
    dataset = _safe_name(dataset, "dataset")
    out = _resolve_root(output_dir or DEFAULT_OUTPUT_DIR)
    uni = _resolve_root(unified_root or DEFAULT_UNIFIED_ROOT)

    report_dir = out / f"{dataset}_report"
    phase1_dir = out / f"{dataset}_phase1"
    uni_dir = uni / dataset

    ids: dict[str, dict] = {}
    if uni_dir.is_dir():
        for f in uni_dir.glob("*.json"):
            ids.setdefault(f.stem, {})["has_unified"] = True
    if phase1_dir.is_dir():
        suffix = "_stage_c_phase1.json"
        for f in phase1_dir.glob(f"*{suffix}"):
            ids.setdefault(f.name[: -len(suffix)], {})["has_phase1"] = True
    if report_dir.is_dir():
        suffix = "_report.json"
        for f in report_dir.glob(f"*{suffix}"):
            ids.setdefault(f.name[: -len(suffix)], {})["has_report"] = True

    result = []
    for tid in sorted(ids):
        info = ids[tid]
        predicted = None
        outcome = None
        report_path = report_dir / f"{tid}_report.json"
        if report_path.is_file():
            try:
                rd = json.loads(report_path.read_text(encoding="utf-8"))
                predicted = rd.get("critical_error_analysis", {}).get("critical_step")
                outcome = rd.get("task_outcome")
            except (OSError, json.JSONDecodeError):
                pass
        phase1_path = phase1_dir / f"{tid}_stage_c_phase1.json"
        gt = _read_gt_step(phase1_path) if phase1_path.is_file() else None
        hit = predicted is not None and gt is not None and predicted == gt
        result.append(
            {
                "task_id": tid,
                "has_unified": info.get("has_unified", False),
                "has_phase1": info.get("has_phase1", False),
                "has_report": info.get("has_report", False),
                "predicted_step": predicted,
                "gt_step": gt,
                "hit": hit,
                "outcome": outcome,
            }
        )
    return result


@app.get("/api/trajectory")
def api_trajectory(
    dataset: str,
    task_id: str,
    output_dir: Optional[str] = None,
    unified_root: Optional[str] = None,
):
    dataset = _safe_name(dataset, "dataset")
    task_id = _safe_name(task_id, "task_id")
    out = _resolve_root(output_dir or DEFAULT_OUTPUT_DIR)
    uni = _resolve_root(unified_root or DEFAULT_UNIFIED_ROOT)

    unified_file = uni / dataset / f"{task_id}.json"
    phase1_file = out / f"{dataset}_phase1" / f"{task_id}_stage_c_phase1.json"
    phase2_file = out / f"{dataset}_phase2" / f"{task_id}_stage_c_phase2.json"
    report_file = out / f"{dataset}_report" / f"{task_id}_report.json"

    if not unified_file.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"unified trajectory not found: {unified_file}",
        )

    try:
        unified = json.loads(unified_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"bad unified JSON: {exc}") from exc

    def _maybe_load(path: Path):
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    metadata = unified.get("metadata", {})
    return {
        "task_id": task_id,
        "dataset": dataset,
        "messages": unified.get("messages", []),
        "annotation": metadata.get("annotation", {}),
        "metadata": metadata,
        "phase1": _maybe_load(phase1_file),
        "phase2": _maybe_load(phase2_file),
        "report": _maybe_load(report_file),
        "files": {
            "unified": str(unified_file),
            "phase1": str(phase1_file) if phase1_file.is_file() else None,
            "phase2": str(phase2_file) if phase2_file.is_file() else None,
            "report": str(report_file) if report_file.is_file() else None,
        },
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory Diagnosis Viewer")
    parser.add_argument("--dataset", default="", help="default dataset to open")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="folder containing <dataset>_phase1 / <dataset>_report",
    )
    parser.add_argument(
        "--unified-root",
        default="data/unified",
        help="root holding <dataset>/<task_id>.json unified trajectories",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global DEFAULT_DATASET, DEFAULT_OUTPUT_DIR, DEFAULT_UNIFIED_ROOT
    DEFAULT_DATASET = args.dataset
    DEFAULT_OUTPUT_DIR = args.output_dir
    DEFAULT_UNIFIED_ROOT = args.unified_root

    print(f"Trajectory Diagnosis Viewer → http://{args.host}:{args.port}")
    print(f"  output-dir   : {_resolve_root(args.output_dir)}")
    print(f"  unified-root : {_resolve_root(args.unified_root)}")
    if args.dataset:
        print(f"  dataset      : {args.dataset}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
