#!/usr/bin/env python3
"""
Convert "pre_trial_format" JSON files into the per-trial directory layout.

This script copies each:
  /data/oasis/mc-multiplayer-eval-results_json/generated/pre_trial_format/<name>.json
to:
  <dst_root>/<name>/trial_1.json
and writes:
  <dst_root>/<name>/stats.json

Notes:
- Does NOT move or delete anything (copy only).
- By default, does NOT overwrite existing trial/stats files.
- The "directory format" is the one used by run_eval.py when saving per-trial outputs.

Example:
  python convert_pre_trial_format_to_trial_dirs.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import Any


DEFAULT_SRC_DIR = Path(
    "/data/oasis/mc-multiplayer-eval-results_json/generated/pre_trial_format"
)

DEFAULT_DST_ROOT = Path(
    "/data/oasis/mc-multiplayer-eval-results_json/generated"
)


def _read_episode_accuracy(trial_json_path: Path) -> float:
    with trial_json_path.open("r") as f:
        data = json.load(f)
    try:
        return float(data["episode_level_accuracy"]["episode_accuracy"])
    except Exception as e:
        raise KeyError(
            f"Missing episode_level_accuracy.episode_accuracy in {trial_json_path}"
        ) from e


def _write_stats_json(output_dir: Path, accuracies: list[float]) -> None:
    if not accuracies:
        stats: dict[str, Any] = {
            "metric": "episode_level_accuracy.episode_accuracy",
            "num_trials": 0,
            "trials": [],
            "mean": None,
            "median": None,
            "std": None,
        }
    else:
        stats = {
            "metric": "episode_level_accuracy.episode_accuracy",
            "num_trials": len(accuracies),
            "trials": [
                {"trial": i + 1, "episode_accuracy": acc}
                for i, acc in enumerate(accuracies)
            ],
            "mean": statistics.mean(accuracies),
            "median": statistics.median(accuracies),
            # Use population stddev so num_trials=1 yields 0.0
            "std": statistics.pstdev(accuracies),
        }

    stats_path = output_dir / "stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)


def convert_one(
    src_json: Path,
    dst_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[bool, str]:
    """Return (did_write_anything, message)."""
    name = src_json.stem
    out_dir = dst_root / name
    trial_path = out_dir / "trial_1.json"
    stats_path = out_dir / "stats.json"

    if not overwrite and (trial_path.exists() or stats_path.exists()):
        return (
            False,
            f"skip (exists): {name}",
        )

    if dry_run:
        return (
            True,
            f"would write: {out_dir}/{{trial_1.json,stats.json}}",
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy the JSON as trial_1.json
    shutil.copy2(src_json, trial_path)

    # Create stats.json for a single trial
    acc = _read_episode_accuracy(trial_path)
    _write_stats_json(out_dir, [acc])

    return (
        True,
        f"wrote: {out_dir.name}/trial_1.json (+ stats.json)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy pre_trial_format/*.json into per-trial directory layout "
            "(one trial per dataset/model)."
        )
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help=f"Source directory containing *.json (default: {DEFAULT_SRC_DIR})",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DEFAULT_DST_ROOT,
        help=f"Destination root directory (default: {DEFAULT_DST_ROOT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trial_1.json / stats.json if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files.",
    )
    args = parser.parse_args()

    src_dir: Path = args.src_dir
    dst_root: Path = args.dst_root

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    src_files = sorted(src_dir.glob("*.json"))
    if not src_files:
        print(f"No JSON files found in: {src_dir}")
        return 0

    if args.dry_run:
        print(f"[dry-run] src: {src_dir}")
        print(f"[dry-run] dst: {dst_root}")
    else:
        dst_root.mkdir(parents=True, exist_ok=True)
        print(f"src: {src_dir}")
        print(f"dst: {dst_root}")

    wrote = 0
    skipped = 0
    for src_json in src_files:
        did_write, msg = convert_one(
            src_json=src_json,
            dst_root=dst_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(msg)
        if did_write:
            wrote += 1
        else:
            skipped += 1

    print(f"\nDone. processed={len(src_files)} wrote={wrote} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

