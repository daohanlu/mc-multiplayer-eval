#!/usr/bin/env python3
"""Run evaluations for all models and datasets."""

import argparse
import subprocess
import re
from pathlib import Path

# Base directories
GENERATIONS_DIR = Path("./generations")
DATASET_BASE = Path("./mc_multiplayer_v2_eval_max_speed")

# Mapping from generated video subdirectory key (e.g., "translation" from "*_eval_translation")
# to the dataset folder name (e.g., "translationEval")
#
# Model folders: generations/{model_name}/  (e.g., SF_CONCAT_FINAL_2)
# Eval subdirs:  step_{N}_multiplayer_v2_eval_{key}  (e.g., step_0002000_multiplayer_v2_eval_translation)
# Dataset folders: mc_multiplayer_v2_eval_max_speed/{datasetName}  (e.g., mc_multiplayer_v2_eval_max_speed/translationEval)
EVAL_TYPE_MAPPING = {
    "translation": "translationEval",
    "rotation": "rotationEval",
    "structure": "structureEval",
    "structure_no_place": "structureNoPlaceEval",
    "turn_to_look": "turnToLookEval",
    "turn_to_look_opposite": "turnToLookOppositeEval",
    "one_looks_away": "oneLooksAwayEval",
    "both_look_away": "bothLookAwayEval",
}

# Which eval types to actually run (comment out to skip)
ENABLED_EVAL_TYPES = [
    "translation",
    "rotation",
    "structure",
    # "structure_no_place"  # enable for diagnostics
    "turn_to_look",
    "turn_to_look_opposite",
    "one_looks_away",
    "both_look_away",
]


def extract_eval_type(folder_name: str) -> str | None:
    """Extract evaluation type from generated folder name.

    Examples:
        step_0080000_multiplayer_v2_eval_translation_ema_length_256 -> translation
        step_0002000_multiplayer_v2_eval_both_look_away -> both_look_away
        step_0001200_multiplayer_v2_eval_both_look_away_max_speed -> both_look_away
    """
    # Extract whatever comes after "eval_" and map it back to a canonical eval type key.
    #
    # We intentionally allow extra suffixes (e.g. "_ema_length_256", "_max_speed") and
    # match by prefix against known keys in EVAL_TYPE_MAPPING.
    _, _, suffix = folder_name.partition("eval_")
    if not suffix:
        return None

    for key in sorted(EVAL_TYPE_MAPPING.keys(), key=len, reverse=True):
        if suffix == key or suffix.startswith(f"{key}_"):
            return key

    return None


def main():
    parser = argparse.ArgumentParser(description="Run evaluations for all models and datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument(
        "--generations-dir",
        type=Path,
        default=GENERATIONS_DIR,
        help="Path to generations/ directory containing model folders",
    )
    parser.add_argument(
        "--dataset-base",
        type=Path,
        default=DATASET_BASE,
        help="Path to base dataset directory containing eval datasets (e.g., translationEval, rotationEval, ...)",
    )
    args = parser.parse_args()

    generations_dir: Path = args.generations_dir
    dataset_base: Path = args.dataset_base

    if not generations_dir.exists():
        print(f"Error: Generations directory not found: {generations_dir}")
        return 1

    # Get all model directories
    model_dirs = [d for d in generations_dir.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"No model directories found in {generations_dir}")
        return 1

    print(f"Found {len(model_dirs)} model(s) to evaluate")
    print(f"Enabled eval types: {ENABLED_EVAL_TYPES}")
    print(f"Generations dir: {generations_dir}")
    print(f"Dataset base: {dataset_base}")
    if args.dry_run:
        print("DRY RUN - commands will not be executed")
    print()

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        print(f"{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"{'=' * 80}")

        # Check which evaluation types this model has generated videos for
        eval_subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        available_eval_types = {}

        for eval_subdir in eval_subdirs:
            eval_type = extract_eval_type(eval_subdir.name)
            if eval_type:
                available_eval_types[eval_type] = eval_subdir

        print(f"Available eval types: {list(available_eval_types.keys())}")

        # Run evaluation for each enabled eval type
        for eval_type in ENABLED_EVAL_TYPES:
            if eval_type not in EVAL_TYPE_MAPPING:
                print(f"⚠ Unknown eval type in ENABLED_EVAL_TYPES: {eval_type}")
                continue

            dataset_name = EVAL_TYPE_MAPPING[eval_type]
            dataset_path = dataset_base / dataset_name

            if not dataset_path.exists():
                print(f"⊘ Skipping {eval_type} - dataset not found: {dataset_path}")
                continue

            if eval_type not in available_eval_types:
                print(f"⊘ Skipping {eval_type} - no generated videos for this model")
                continue

            # Construct the command
            cmd = [
                "python",
                "run_eval.py",
                str(dataset_path),
                "--generated",
                str(model_dir)
            ]

            print(f"\n{'[DRY RUN] Would run' if args.dry_run else 'Running'}: {' '.join(cmd)}")

            if args.dry_run:
                print(f"  → Dataset: {dataset_path}")
                print(f"  → Generated subdir: {available_eval_types[eval_type].name}")
                continue

            # Run the command
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✓ Success for {eval_type}")
                if result.stdout:
                    # Print just the summary section
                    lines = result.stdout.split('\n')
                    in_summary = False
                    for line in lines:
                        if 'EVALUATION SUMMARY' in line:
                            in_summary = True
                        if in_summary:
                            print(line)
            except subprocess.CalledProcessError as e:
                print(f"✗ Error for {eval_type}")
                print(f"Return code: {e.returncode}")
                if e.stdout:
                    print("STDOUT:", e.stdout[-1000:])  # Last 1000 chars
                if e.stderr:
                    print("STDERR:", e.stderr[-1000:])

        print()

    return 0


if __name__ == "__main__":
    exit(main())
