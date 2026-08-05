"""Bake v5 per-step hyperparameters into existing unified trajectory files.

For every message of every unified trajectory under ``data/unified/<dataset>/``
this script writes two additive fields:

* ``judgable``      : ``True`` iff the message can ever be marked as the
                      critical error step (``role`` / ``name`` not in
                      ``non_error_roles``).
* ``compress_role`` : ``"compress" | "preserve" | "default"`` — Stage A's
                      tier router consumes this to pick ``th3`` vs the
                      distance rule. Derived from ``compress_priority`` /
                      ``preserve_priority``; ``compress_priority`` wins on
                      conflict (matches v5 §5.0.1's ordering).

The script is idempotent: running twice over the same file gives the same
result. Existing fields are overwritten so the bake stays in sync with
``dataset_config.json`` after edits.

The ``agent_framework_description`` field is **not** touched here — it is
already written by ``data_processing/_standard_messages.py`` into
``metadata.extra`` and read directly by the detector
(``detector/_stage_common.py``).

CLI::

    python -m data_processing.bake_v5_fields                # all datasets
    python -m data_processing.bake_v5_fields --datasets alfworldsub gaiasub
    python -m data_processing.bake_v5_fields --dry-run      # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "dataset_config.json"
DEFAULT_UNIFIED_ROOT = REPO_ROOT / "data" / "unified"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Dict[str, List[str]]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Dict[str, List[str]]] = {}
    for dataset, entry in raw.items():
        if dataset.startswith("_"):
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"dataset_config[{dataset}] is not a dict")
        out[dataset] = {
            "non_error_roles":   list(entry.get("non_error_roles") or []),
            "compress_priority": list(entry.get("compress_priority") or []),
            "preserve_priority": list(entry.get("preserve_priority") or []),
        }
    return out


# ---------------------------------------------------------------------------
# Token matching
# ---------------------------------------------------------------------------
def _first_word(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str):
        return None
    parts = name.split()
    return parts[0] if parts else None


def _token_matches(token: str, role: Optional[str], name: Optional[str]) -> bool:
    if not isinstance(token, str) or not token:
        return False
    if isinstance(role, str) and token == role:
        return True
    if isinstance(name, str) and token == name:
        return True
    first = _first_word(name)
    if isinstance(first, str) and token == first:
        return True
    return False


def _matches_any(tokens: Iterable[str], role: Optional[str], name: Optional[str]) -> bool:
    return any(_token_matches(t, role, name) for t in tokens)


def derive_judgable(
    role: Optional[str],
    name: Optional[str],
    non_error_roles: List[str],
) -> bool:
    return not _matches_any(non_error_roles, role, name)


def derive_compress_role(
    role: Optional[str],
    name: Optional[str],
    compress_priority: List[str],
    preserve_priority: List[str],
) -> str:
    if _matches_any(compress_priority, role, name):
        return "compress"
    if _matches_any(preserve_priority, role, name):
        return "preserve"
    return "default"


# ---------------------------------------------------------------------------
# Per-file bake
# ---------------------------------------------------------------------------
def bake_trajectory(
    data: Dict[str, Any],
    *,
    non_error_roles: List[str],
    compress_priority: List[str],
    preserve_priority: List[str],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError("trajectory has no 'messages' list")

    stats = {"compress": 0, "preserve": 0, "default": 0, "judgable": 0, "non_judgable": 0}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        name = msg.get("name")

        judgable = derive_judgable(role, name, non_error_roles)
        compress_role = derive_compress_role(
            role, name, compress_priority, preserve_priority
        )

        msg["judgable"] = bool(judgable)
        msg["compress_role"] = compress_role

        stats[compress_role] += 1
        stats["judgable" if judgable else "non_judgable"] += 1

    return data, stats


def process_file(
    path: Path,
    cfg: Dict[str, List[str]],
    *,
    dry_run: bool,
) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data, stats = bake_trajectory(
        data,
        non_error_roles=cfg["non_error_roles"],
        compress_priority=cfg["compress_priority"],
        preserve_priority=cfg["preserve_priority"],
    )
    if not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _select_datasets(
    config: Dict[str, Dict[str, List[str]]],
    requested: Optional[List[str]],
    unified_root: Path,
) -> List[str]:
    available = sorted(p.name for p in unified_root.iterdir() if p.is_dir())
    if requested:
        missing_cfg = [d for d in requested if d not in config]
        missing_dir = [d for d in requested if d not in available]
        if missing_cfg:
            raise SystemExit(
                f"datasets {missing_cfg} not found in dataset_config.json"
            )
        if missing_dir:
            raise SystemExit(
                f"datasets {missing_dir} not found under {unified_root}"
            )
        return requested
    return [d for d in available if d in config]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"path to dataset_config.json (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--unified-root", type=Path, default=DEFAULT_UNIFIED_ROOT,
        help=f"root of unified trajectory dirs (default: {DEFAULT_UNIFIED_ROOT})",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="dataset folder names to process; default: all that have a config entry AND a folder",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="parse + compute fields but do not write files back",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = _select_datasets(config, args.datasets, args.unified_root)

    grand: Dict[str, int] = {}
    for dataset in datasets:
        cfg = config[dataset]
        ds_dir = args.unified_root / dataset
        files = sorted(ds_dir.glob("*.json"))
        ds_stats = {"compress": 0, "preserve": 0, "default": 0,
                    "judgable": 0, "non_judgable": 0}
        for path in files:
            try:
                stats = process_file(path, cfg, dry_run=args.dry_run)
            except Exception as exc:
                print(f"[WARN] {path}: {exc}", file=sys.stderr)
                continue
            for k, v in stats.items():
                ds_stats[k] = ds_stats.get(k, 0) + v
        n_files = len(files)
        print(
            f"[{dataset}] files={n_files} | "
            f"compress={ds_stats['compress']} preserve={ds_stats['preserve']} "
            f"default={ds_stats['default']} | "
            f"judgable={ds_stats['judgable']} non_judgable={ds_stats['non_judgable']}"
        )
        for k, v in ds_stats.items():
            grand[k] = grand.get(k, 0) + v

    if datasets:
        print(
            f"[total] datasets={len(datasets)} | "
            f"compress={grand.get('compress', 0)} preserve={grand.get('preserve', 0)} "
            f"default={grand.get('default', 0)} | "
            f"judgable={grand.get('judgable', 0)} non_judgable={grand.get('non_judgable', 0)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
