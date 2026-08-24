"""CLI: convert raw datasets into the unified schema.

Usage examples::

    python -m data_processing.build_unified_dataset --all
    python -m data_processing.build_unified_dataset --dataset whoandwhen
    python -m data_processing.build_unified_dataset \
        --dataset gaia --src data/GAIA --out data/unified/gaia \
        --labels data/gaia_labels.json

The CLI is idempotent: it re-emits one ``data/unified/<dataset>/<stem>.json``
per source trajectory and overwrites any existing file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from data_processing import (
    alfworld,
    gaia,
    swebenchpro,
    tau2bench,
    webshop,
    whoandwhen,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_UNIFIED_DIR = DEFAULT_DATA_DIR / "unified"


# Each entry: (src_subdir, label_filename_or_None, converter_fn)
DatasetSpec = Tuple[str, Optional[str], Callable[..., List[str]]]


DATASETS: Dict[str, DatasetSpec] = {
    "whoandwhen":   ("Who_and_When/Hand-Crafted",        None, whoandwhen.convert_directory),
    "whoandwhen_algorithm": (
        "Who_and_When/Algorithm-Generated", None, whoandwhen.convert_directory_algorithm,
    ),
    "alfworld":     ("AgentDebugBench/ALFWorld", "AgentDebugBench/alfworld_labels.json", alfworld.convert_directory),
    "gaia":         ("AgentDebugBench/GAIA",     "AgentDebugBench/gaia_labels.json",     gaia.convert_directory),
    "webshop":      ("AgentDebugBench/WebShop",  "AgentDebugBench/webshop_labels.json",  webshop.convert_directory),
    "tau2bench":    ("OurBench/Tau^2-Bench",     None,                                   tau2bench.convert_directory),
    "swebenchpro":  ("OurBench/Swe-Bench-Pro",   None,                                   swebenchpro.convert_directory),
}


def _convert_one(
    name: str,
    *,
    src: Optional[Path],
    out: Optional[Path],
    labels: Optional[Path],
) -> int:
    spec = DATASETS[name]
    src_subdir, default_label, fn = spec
    src_dir = src or (DEFAULT_DATA_DIR / src_subdir)
    out_dir = out or (DEFAULT_UNIFIED_DIR / name)
    label_path = labels
    if label_path is None and default_label:
        candidate = DEFAULT_DATA_DIR / default_label
        label_path = candidate if candidate.exists() else None

    print(f"[{name}] src={src_dir} out={out_dir} labels={label_path}")
    if not (src_dir.is_dir() or src_dir.is_file()):
        print(f"[{name}] WARNING: source path does not exist, skipping")
        return 0

    if default_label:
        written = fn(src_dir, out_dir, label_path)
    else:
        written = fn(src_dir, out_dir)
    print(f"[{name}] wrote {len(written)} unified trajectories")
    return len(written)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build unified-schema trajectories.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        help="Convert a single dataset (mutually exclusive with --all).",
    )
    parser.add_argument("--all", action="store_true", help="Convert every supported dataset.")
    parser.add_argument("--src", type=Path, default=None, help="Override source directory.")
    parser.add_argument("--out", type=Path, default=None, help="Override unified output directory.")
    parser.add_argument("--labels", type=Path, default=None, help="Override label JSON path.")
    args = parser.parse_args(argv)

    if not args.all and not args.dataset:
        parser.error("either --all or --dataset is required")
    if args.all and (args.src or args.out or args.labels):
        parser.error("--src/--out/--labels can only be used with --dataset")

    targets = sorted(DATASETS.keys()) if args.all else [args.dataset]
    total = 0
    for name in targets:
        total += _convert_one(name, src=args.src, out=args.out, labels=args.labels)
    print(f"Done. Wrote {total} unified trajectories total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
