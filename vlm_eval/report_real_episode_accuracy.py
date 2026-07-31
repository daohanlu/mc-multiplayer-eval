#!/usr/bin/env python3
"""
Report episode-level accuracy on ground-truth (real) episodes.

Reads stats.json and per-trial JSON files from results_json/real/{datasetEval}/
and prints a summary table to the console. Also writes a CSV file.

When both a base and "_long" version of a dataset exist, only the "_long"
version is reported.

Verifies that stats.json numbers match the per-trial JSON files.
Adds a combined "Turn To Look (Combined)" row with std recalculated
from the pooled episodes across both turn-to-look eval types.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent / "results_json" / "real"
OUTPUT_TSV = Path(__file__).parent / "real_episode_accuracy.tsv"

# Display names for the table (maps folder name → column header).
DISPLAY_NAMES: dict[str, str] = {
    "oneLooksAwayEval": "Grounding",
    "oneLooksAwayEval_long": "Grounding",
    "bothLookAwayEval": "Memory",
    "bothLookAwayEval_long": "Memory",
    "structureEval": "Building",
    "translationEval": "Translation",
    "rotationEval": "Rotation",
    "turnToLookEval": "Turn To Look",
    "turnToLookOppositeEval": "Turn To Look Opposite",
}

# Datasets whose episodes should be pooled into a combined row.
TURN_TO_LOOK_DATASETS = {"turnToLookEval", "turnToLookOppositeEval"}


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARNING: could not read {path} ({e})", file=sys.stderr)
        return None


def _read_trial_episode_counts(ds_dir: Path, num_trials: int) -> list[dict]:
    """Read per-trial JSONs and return list of {total, correct, accuracy}."""
    trials = []
    for i in range(1, num_trials + 1):
        data = _read_json(ds_dir / f"trial_{i}.json")
        if data is None:
            trials.append(None)
            continue
        ep = data.get("episode_level_accuracy", {})
        trials.append({
            "total": ep.get("total_episodes", 0),
            "correct": ep.get("fully_correct_episodes", 0),
            "accuracy": ep.get("episode_accuracy", 0.0),
        })
    return trials


def _verify_stats(dataset: str, stats: dict, trial_data: list[dict | None]) -> None:
    """Verify stats.json mean matches the per-trial episode accuracies."""
    stats_trials = stats.get("trials", [])
    for i, (st, td) in enumerate(zip(stats_trials, trial_data), start=1):
        if td is None:
            print(f"WARNING [{dataset}]: trial_{i}.json missing, cannot verify", file=sys.stderr)
            continue
        stats_acc = st.get("episode_accuracy")
        trial_acc = td["accuracy"]
        if abs(stats_acc - trial_acc) > 1e-6:
            print(
                f"MISMATCH [{dataset}] trial {i}: "
                f"stats.json says {stats_acc}, trial JSON says {trial_acc}",
                file=sys.stderr,
            )
        else:
            print(f"  OK [{dataset}] trial {i}: {trial_acc:.4f} "
                  f"({td['correct']}/{td['total']} episodes)")


def _discover_datasets(root: Path) -> list[str]:
    """Return dataset folder names, preferring _long variants over base ones."""
    if not root.is_dir():
        raise SystemExit(f"Results root not found: {root}")

    all_dirs = sorted(
        d.name for d in root.iterdir() if d.is_dir() and (d / "stats.json").exists()
    )

    # Build set of base names that have a _long variant
    long_variants = {d.removesuffix("_long") for d in all_dirs if d.endswith("_long")}

    # Keep _long version, drop the base when _long exists
    datasets = [d for d in all_dirs if not (d in long_variants and not d.endswith("_long"))]
    return datasets


def _build_combined_turn_to_look(rows: list[dict]) -> dict | None:
    """Pool episodes from both turn-to-look datasets across trials."""
    ttl_rows = [r for r in rows if r["dataset"] in TURN_TO_LOOK_DATASETS]
    if len(ttl_rows) != 2:
        return None

    # Both must have the same number of trials
    num_trials = ttl_rows[0]["num_trials"]
    if any(r["num_trials"] != num_trials for r in ttl_rows):
        print("WARNING: turn-to-look datasets have different trial counts", file=sys.stderr)
        return None

    # Pool episodes per trial: combine total/correct from both datasets
    combined_accuracies = []
    combined_trials_detail = []
    total_episodes_per_trial = None

    for trial_idx in range(num_trials):
        total = 0
        correct = 0
        for r in ttl_rows:
            td = r["trial_data"][trial_idx]
            if td is None:
                print(f"WARNING: missing trial data for combined row", file=sys.stderr)
                return None
            total += td["total"]
            correct += td["correct"]

        acc = (correct / total * 100) if total > 0 else 0.0
        combined_accuracies.append(acc)
        combined_trials_detail.append({"trial": trial_idx + 1, "episode_accuracy": acc})

        if total_episodes_per_trial is None:
            total_episodes_per_trial = total
        # Episodes per trial should be consistent
        if total != total_episodes_per_trial:
            print(f"WARNING: combined episode count varies across trials ({total} vs {total_episodes_per_trial})", file=sys.stderr)

    mean = statistics.mean(combined_accuracies)
    median = statistics.median(combined_accuracies)
    std = statistics.pstdev(combined_accuracies)

    return {
        "dataset": "turnToLook_combined",
        "display": "Turn To Look (Combined)",
        "mean": mean,
        "median": median,
        "std": std,
        "num_trials": num_trials,
        "trials": combined_trials_detail,
        "episodes": total_episodes_per_trial,
        "trial_data": None,
    }


def main() -> None:
    datasets = _discover_datasets(RESULTS_ROOT)

    if not datasets:
        raise SystemExit("No datasets with stats.json found.")

    # Collect data
    rows: list[dict] = []
    for ds in datasets:
        ds_dir = RESULTS_ROOT / ds
        stats = _read_json(ds_dir / "stats.json")
        if stats is None:
            continue

        display = DISPLAY_NAMES.get(ds, ds)
        num_trials = stats.get("num_trials", 0)

        # Read and verify per-trial data
        trial_data = _read_trial_episode_counts(ds_dir, num_trials)
        _verify_stats(ds, stats, trial_data)

        # Get episode count from first valid trial
        episodes = None
        for td in trial_data:
            if td is not None:
                episodes = td["total"]
                break

        rows.append({
            "dataset": ds,
            "display": display,
            "mean": stats.get("mean"),
            "median": stats.get("median"),
            "std": stats.get("std"),
            "num_trials": num_trials,
            "trials": stats.get("trials", []),
            "episodes": episodes,
            "trial_data": trial_data,
        })

    # Add combined turn-to-look row
    combined = _build_combined_turn_to_look(rows)
    if combined is not None:
        # Insert after the last turn-to-look row
        insert_idx = len(rows)
        for i, r in enumerate(rows):
            if r["dataset"] in TURN_TO_LOOK_DATASETS:
                insert_idx = i + 1
        rows.insert(insert_idx, combined)

    # --- Console table ---
    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    col_w = {
        "eval": max(len(r["display"]) for r in rows),
        "episodes": 10,
        "mean": 10,
        "median": 10,
        "std": 10,
        "trials": 8,
    }
    col_w["eval"] = max(col_w["eval"], len("Eval"))

    header = (
        f"{'Eval':<{col_w['eval']}}  "
        f"{'Episodes':>{col_w['episodes']}}  "
        f"{'Mean':>{col_w['mean']}}  "
        f"{'Median':>{col_w['median']}}  "
        f"{'Std':>{col_w['std']}}  "
        f"{'Trials':>{col_w['trials']}}"
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        ep_str = str(r["episodes"]) if r["episodes"] is not None else "N/A"
        print(
            f"{r['display']:<{col_w['eval']}}  "
            f"{ep_str:>{col_w['episodes']}}  "
            f"{_fmt(r['mean']):>{col_w['mean']}}  "
            f"{_fmt(r['median']):>{col_w['median']}}  "
            f"{_fmt(r['std']):>{col_w['std']}}  "
            f"{str(r['num_trials']):>{col_w['trials']}}"
        )

    print(sep)

    # --- CSV output ---
    csv_fields = ["Eval", "Episodes", "Mean", "Median", "Std", "Num Trials"]
    max_trials = max((r["num_trials"] for r in rows), default=0)
    for i in range(1, max_trials + 1):
        csv_fields.append(f"Trial {i}")

    with OUTPUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            tsv_row: dict[str, str] = {
                "Eval": r["display"],
                "Episodes": str(r["episodes"]) if r["episodes"] is not None else "",
                "Mean": str(r["mean"]) if r["mean"] is not None else "",
                "Median": str(r["median"]) if r["median"] is not None else "",
                "Std": str(r["std"]) if r["std"] is not None else "",
                "Num Trials": str(r["num_trials"]),
            }
            for i, trial in enumerate(r["trials"], start=1):
                tsv_row[f"Trial {i}"] = str(trial.get("episode_accuracy", ""))
            writer.writerow(tsv_row)

    print(f"\nTSV written to: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
