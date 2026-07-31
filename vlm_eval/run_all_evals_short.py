#!/usr/bin/env python3
"""Run evaluations for short (33-frame) ablation-generated videos.

This script evaluates generated videos from 33_frame_ablation_generations/ against
GT datasets for expected-answer computation. It only supports three eval types:
turnToLookEval, turnToLookOppositeEval, and translationEval.

Generated video frame extraction uses fixed indices:
  - Translation: always frame 0 (initial) and frame 32 (final)
  - Turn-to-look variants: frame min(computed_offset, 32)
"""

import argparse
import os
import subprocess
import concurrent.futures
import time
from pathlib import Path

# Base directories
GENERATIONS_DIR = Path("./33_frame_ablation_generations")
DATASET_BASE = Path("./mc_multiplayer_v2_eval_new_sneak_combined")
MAX_GEN_FRAME = 32

EVAL_TYPE_MAPPING = {
    "turn_to_look": "turnToLookEval",
    "turn_to_look_opposite": "turnToLookOppositeEval",
    "translation": "translationEval",
}

ENABLED_EVAL_TYPES = list(EVAL_TYPE_MAPPING.keys())


def find_generated_subdir(generations_dir: Path, eval_key: str) -> Path | None:
    """Find the generated-video subdirectory matching *eval_key* inside *generations_dir*.

    The 33-frame ablation directories use a naming scheme like:
        step_0120000_translation_33_frames_denoising_steps_50
        step_0120000_turn_to_look_33_frames_denoising_steps_50

    Matches ``_{eval_key}_`` in the folder name while excluding folders that
    match a longer key (e.g., ``turn_to_look`` won't match ``turn_to_look_opposite``).
    """
    longer_keys = [k for k in EVAL_TYPE_MAPPING if k != eval_key and k.startswith(eval_key)]

    candidates: list[Path] = []
    for subdir in generations_dir.iterdir():
        if not subdir.is_dir():
            continue
        name = subdir.name
        if f"_{eval_key}_" not in name and not name.endswith(f"_{eval_key}"):
            continue
        # Reject if a longer key also matches (e.g., turn_to_look vs turn_to_look_opposite)
        if any(f"_{lk}_" in name or name.endswith(f"_{lk}") for lk in longer_keys):
            continue
        candidates.append(subdir)

    if not candidates:
        return None

    # Prefer highest step number
    import re

    def _step_num(p: Path) -> int:
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=lambda p: (_step_num(p), p.name))


def _run_one_eval_type(
    eval_key: str,
    dataset_base: Path,
    generations_dir: Path,
    max_gen_frame: int,
    model_name: str,
    dry_run: bool,
    limit: int | None,
    num_trials: int,
    results_dir: Path | None,
) -> tuple[str, str, bool]:
    """Run a single eval type and return (eval_key, output_text, failed)."""
    out_lines: list[str] = []
    failed = False

    dataset_name = EVAL_TYPE_MAPPING[eval_key]
    dataset_path = dataset_base / dataset_name

    if not dataset_path.exists():
        out_lines.append(f"⊘ Skipping {eval_key} - dataset not found: {dataset_path}")
        return eval_key, "\n".join(out_lines), False

    gen_subdir = find_generated_subdir(generations_dir, eval_key)
    if gen_subdir is None:
        out_lines.append(f"⊘ Skipping {eval_key} - no generated subdir found in {generations_dir}")
        return eval_key, "\n".join(out_lines), False

    cmd = [
        "python",
        "run_eval.py",
        str(dataset_path),
        "--generated-subdir",
        str(gen_subdir),
        "--model-name",
        model_name,
        "--max-gen-frame",
        str(max_gen_frame),
    ]
    if results_dir is not None:
        cmd.extend(["--results-dir", str(results_dir)])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if num_trials != 1:
        cmd.extend(["--num-trials", str(num_trials)])

    out_lines.append(f"\n{'[DRY RUN] Would run' if dry_run else 'Running'}: {' '.join(cmd)}")

    if dry_run:
        out_lines.append(f"  → Dataset: {dataset_path}")
        out_lines.append(f"  → Generated subdir: {gen_subdir.name}")
        return eval_key, "\n".join(out_lines), False

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        out_lines.append(f"✓ Success for {eval_key}")
        if result.stdout:
            lines = result.stdout.split("\n")
            in_summary = False
            for line in lines:
                if "EVALUATION SUMMARY" in line:
                    in_summary = True
                if in_summary:
                    out_lines.append(line)
    except subprocess.CalledProcessError as e:
        failed = True
        out_lines.append(f"✗ Error for {eval_key}")
        out_lines.append(f"Return code: {e.returncode}")
        if e.stdout:
            out_lines.append("STDOUT: " + e.stdout[-1000:])
        if e.stderr:
            out_lines.append("STDERR: " + e.stderr[-1000:])

    return eval_key, "\n".join(out_lines), failed


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluations for short (33-frame) ablation-generated videos",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument(
        "--generations-dir",
        type=Path,
        default=GENERATIONS_DIR,
        help="Path to directory containing generated video subdirectories",
    )
    parser.add_argument(
        "--dataset-base",
        type=Path,
        default=DATASET_BASE,
        help="Path to base GT dataset directory (e.g., mc_multiplayer_v2_eval_new_sneak_combined)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Root directory for JSON outputs (forwarded to run_eval.py --results-dir).",
    )
    parser.add_argument("--limit", type=int, help="Limit number of episodes to process")
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of evaluation trials per eval type.",
    )
    parser.add_argument(
        "--max-gen-frame",
        type=int,
        default=MAX_GEN_FRAME,
        help="Max generated-video frame index (default: 32 for 33-frame videos).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override model name for output directories. Defaults to generations-dir name.",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=int,
        default=None,
        help="Max number of eval types to run in parallel.",
    )
    parser.add_argument(
        "--eval-types",
        nargs="+",
        help=(
            "Override which eval types to run. "
            f"Valid keys: {', '.join(sorted(EVAL_TYPE_MAPPING.keys()))}"
        ),
    )
    args = parser.parse_args()

    if args.num_trials < 1:
        raise SystemExit("--num-trials must be >= 1")

    generations_dir: Path = args.generations_dir
    dataset_base: Path = args.dataset_base
    model_name = args.model_name or generations_dir.name

    # Determine which eval types to run
    if args.eval_types:
        enabled = []
        for v in args.eval_types:
            for part in v.split(","):
                part = part.strip()
                if part:
                    enabled.append(part)
        unknown = [t for t in enabled if t not in EVAL_TYPE_MAPPING]
        if unknown:
            raise SystemExit(f"Unknown eval type(s): {unknown}. Valid: {sorted(EVAL_TYPE_MAPPING.keys())}")
    else:
        enabled = ENABLED_EVAL_TYPES

    print(f"Enabled eval types: {enabled}")
    print(f"Dataset base: {dataset_base}")
    print(f"Generations dir: {generations_dir}")
    print(f"Model name: {model_name}")
    print(f"Max gen frame: {args.max_gen_frame}")
    if args.dry_run:
        print("DRY RUN - commands will not be executed")
    print()

    if not generations_dir.exists():
        print(f"Error: Generations directory not found: {generations_dir}")
        return 1

    max_workers = args.max_workers or min(len(enabled), os.cpu_count() or 1)
    if max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")
    print(f"Parallelism: {max_workers} worker(s)")

    started = time.time()
    any_failed = False

    if max_workers == 1:
        for eval_key in enabled:
            name, text, failed = _run_one_eval_type(
                eval_key=eval_key,
                dataset_base=dataset_base,
                generations_dir=generations_dir,
                max_gen_frame=args.max_gen_frame,
                model_name=model_name,
                dry_run=args.dry_run,
                limit=args.limit,
                num_trials=args.num_trials,
                results_dir=args.results_dir,
            )
            print(text)
            any_failed = any_failed or failed
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _run_one_eval_type,
                    eval_key,
                    dataset_base,
                    generations_dir,
                    args.max_gen_frame,
                    model_name,
                    args.dry_run,
                    args.limit,
                    args.num_trials,
                    args.results_dir,
                ): eval_key
                for eval_key in enabled
            }
            for fut in concurrent.futures.as_completed(futures):
                name, text, failed = fut.result()
                print(text)
                any_failed = any_failed or failed

    elapsed = time.time() - started
    print(f"\n{'=' * 80}")
    print(f"Done in {elapsed:.1f}s. Failures: {'yes' if any_failed else 'no'}")
    print(f"{'=' * 80}")

    return 0


if __name__ == "__main__":
    exit(main())
