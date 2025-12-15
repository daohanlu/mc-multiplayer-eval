#!/usr/bin/env python3
"""Run evaluations for all models and datasets."""

import os
import subprocess
import re
from pathlib import Path

# Base directories
generations_dir = Path("./generations")

# Mapping from generation folder patterns to dataset names
# The generation folders have format: step_XXXXX_multiplayer_eval_TYPE_ema_length_256
# The dataset names have format: mc_multiplayer_eval_TYPE
eval_type_mapping = {
    "translation": "mc_multiplayer_eval_translation",
    "structure": "mc_multiplayer_eval_structure",
    # Uncomment to include additional eval types:
    # "rotation": "mc_multiplayer_eval_rotation",
    # "one_looks_away": "mc_multiplayer_eval_one_looks_away",
    # "both_look_away": "mc_multiplayer_eval_both_look_away",
}

def extract_eval_type(folder_name):
    """Extract evaluation type from folder name.

    Example: step_0080000_multiplayer_eval_translation_ema_length_256 -> translation
    """
    # Pattern: multiplayer_eval_<TYPE>_ema or multiplayer_eval_<TYPE> at end
    match = re.search(r'multiplayer_eval_(\w+?)(?:_ema|$)', folder_name)
    if match:
        eval_type = match.group(1)
        # Handle multi-word eval types
        return eval_type
    return None

def main():
    # Get all model directories
    model_dirs = [d for d in generations_dir.iterdir() if d.is_dir()]

    print(f"Found {len(model_dirs)} models to evaluate\n")

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        print(f"=" * 80)
        print(f"Model: {model_name}")
        print(f"=" * 80)

        # Check which evaluation types this model has
        eval_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        available_eval_types = set()

        for eval_dir in eval_dirs:
            eval_type = extract_eval_type(eval_dir.name)
            if eval_type:
                available_eval_types.add(eval_type)

        # Run evaluation for each dataset type we want to evaluate
        for eval_type, dataset_name in sorted(eval_type_mapping.items()):
            if eval_type in available_eval_types:
                # Construct the command - point to model directory, not specific eval folder
                cmd = [
                    "python",
                    "run_eval.py",
                    f"{dataset_name}/test",
                    "--generated",
                    str(model_dir)
                ]

                print(f"\nRunning: {' '.join(cmd)}")

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
                        print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"✗ Error for {eval_type}")
                    print(f"Return code: {e.returncode}")
                    if e.stdout:
                        print("STDOUT:", e.stdout)
                    if e.stderr:
                        print("STDERR:", e.stderr)
            else:
                print(f"⊘ Skipping {eval_type} - not available for this model")

        print()

if __name__ == "__main__":
    main()
