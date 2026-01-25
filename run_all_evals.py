#!/usr/bin/env python3
"""Run evaluations for all models and datasets."""

import argparse
import os
import subprocess
import concurrent.futures
import time
from dataclasses import dataclass
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


def _normalize_eval_types_arg(values: list[str] | None) -> list[str] | None:
    """Normalize --eval-types input.

    Supports:
    - space-separated values: --eval-types translation rotation
    - comma-separated values: --eval-types translation,rotation
    - mixed: --eval-types translation,rotation structure
    - special: --eval-types all
    """
    if not values:
        return None

    normalized: list[str] = []
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        normalized.extend(parts)

    if not normalized:
        return None

    if any(v.lower() == "all" for v in normalized):
        return sorted(EVAL_TYPE_MAPPING.keys())

    # De-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for v in normalized:
        if v not in seen:
            out.append(v)
            seen.add(v)

    return out


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


@dataclass(frozen=True)
class _EvalRun:
    eval_type: str
    cmd: list[str]
    dataset_path: Path
    generated_subdir_name: str | None


@dataclass(frozen=True)
class _EvalOutcome:
    eval_type: str
    ok: bool
    returncode: int
    summary: str


@dataclass(frozen=True)
class _ModelOutcome:
    model_name: str
    outcomes: list[_EvalOutcome]
    skipped: list[str]


def _build_model_plan(
    model_dir: Path,
    enabled_eval_types: list[str],
    dataset_base: Path,
    limit: int | None,
    num_trials: int,
) -> tuple[list[_EvalRun], list[str]]:
    """Build run plan (commands) for a model, plus skip reasons."""
    eval_subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
    available_eval_types: dict[str, Path] = {}
    for eval_subdir in eval_subdirs:
        eval_type = extract_eval_type(eval_subdir.name)
        if eval_type:
            available_eval_types[eval_type] = eval_subdir

    runs: list[_EvalRun] = []
    skipped: list[str] = []

    for eval_type in enabled_eval_types:
        dataset_name = EVAL_TYPE_MAPPING[eval_type]
        dataset_path = dataset_base / dataset_name

        if not dataset_path.exists():
            skipped.append(f"{eval_type} (dataset not found: {dataset_path})")
            continue

        if eval_type not in available_eval_types:
            skipped.append(f"{eval_type} (no generated videos)")
            continue

        cmd = [
            "python",
            "run_eval.py",
            str(dataset_path),
            "--generated",
            str(model_dir),
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])
        if num_trials != 1:
            cmd.extend(["--num-trials", str(num_trials)])

        runs.append(
            _EvalRun(
                eval_type=eval_type,
                cmd=cmd,
                dataset_path=dataset_path,
                generated_subdir_name=available_eval_types[eval_type].name,
            )
        )

    return runs, skipped


def _extract_summary(stdout: str) -> str:
    """Extract the 'EVALUATION SUMMARY' section from run_eval stdout."""
    if not stdout:
        return ""
    lines = stdout.splitlines()
    in_summary = False
    out_lines: list[str] = []
    for line in lines:
        if "EVALUATION SUMMARY" in line:
            in_summary = True
        if in_summary:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


def _run_one_model(
    model_dir: Path,
    enabled_eval_types: list[str],
    dataset_base: Path,
    dry_run: bool,
    limit: int | None,
    num_trials: int,
) -> _ModelOutcome:
    model_name = model_dir.name
    runs, skipped = _build_model_plan(
        model_dir=model_dir,
        enabled_eval_types=enabled_eval_types,
        dataset_base=dataset_base,
        limit=limit,
        num_trials=num_trials,
    )

    outcomes: list[_EvalOutcome] = []

    if dry_run:
        for r in runs:
            outcomes.append(
                _EvalOutcome(
                    eval_type=r.eval_type,
                    ok=True,
                    returncode=0,
                    summary=(
                        "[DRY RUN]\n"
                        f"cmd: {' '.join(r.cmd)}\n"
                        f"dataset: {r.dataset_path}\n"
                        f"generated subdir: {r.generated_subdir_name}"
                    ),
                )
            )
        return _ModelOutcome(model_name=model_name, outcomes=outcomes, skipped=skipped)

    for r in runs:
        try:
            result = subprocess.run(
                r.cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            ok = result.returncode == 0
            summary = _extract_summary(result.stdout)
            if not ok:
                tail_out = (result.stdout or "")[-1000:]
                tail_err = (result.stderr or "")[-1000:]
                summary = (
                    (summary + "\n\n" if summary else "")
                    + f"STDOUT (tail):\n{tail_out}\n\nSTDERR (tail):\n{tail_err}"
                ).strip()

            outcomes.append(
                _EvalOutcome(
                    eval_type=r.eval_type,
                    ok=ok,
                    returncode=result.returncode,
                    summary=summary,
                )
            )
        except Exception as e:
            outcomes.append(
                _EvalOutcome(
                    eval_type=r.eval_type,
                    ok=False,
                    returncode=1,
                    summary=f"Exception while running: {e!r}",
                )
            )

    return _ModelOutcome(model_name=model_name, outcomes=outcomes, skipped=skipped)


def main():
    parser = argparse.ArgumentParser(description="Run evaluations for all models and datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames from ground-truth videos (no --generated, for sanity checks)",
    )
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
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of episodes/queries to process (passed to run_eval.py)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of evaluation trials to run per (model, dataset) pair (passed to run_eval.py).",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Max number of models to evaluate in parallel. "
            "Defaults to min(num_models, CPU count). Set to 1 for sequential behavior."
        ),
    )
    parser.add_argument(
        "--eval-types",
        nargs="+",
        help=(
            "Override which eval types to run. "
            "Examples: --eval-types translation rotation | --eval-types translation,rotation | --eval-types all. "
            f"Valid keys: {', '.join(sorted(EVAL_TYPE_MAPPING.keys()))}"
        ),
    )
    args = parser.parse_args()
    if args.num_trials < 1:
        raise SystemExit("--num-trials must be >= 1")

    generations_dir: Path = args.generations_dir
    dataset_base: Path = args.dataset_base

    enabled_eval_types = _normalize_eval_types_arg(args.eval_types) or ENABLED_EVAL_TYPES
    unknown = [t for t in enabled_eval_types if t not in EVAL_TYPE_MAPPING]
    if unknown:
        valid = ", ".join(sorted(EVAL_TYPE_MAPPING.keys()))
        raise SystemExit(f"Unknown eval type(s): {unknown}. Valid keys: {valid}")

    print(f"Enabled eval types: {enabled_eval_types}")
    print(f"Dataset base: {dataset_base}")
    if args.dry_run:
        print("DRY RUN - commands will not be executed")
    if args.extract_frames:
        print("EXTRACT FRAMES MODE - extracting from ground-truth videos (no --generated)")
    print()

    # --extract-frames mode: run on ground-truth videos without --generated
    if args.extract_frames:
        print(f"{'=' * 80}")
        print("Extracting frames from ground-truth videos")
        print(f"{'=' * 80}")

        for eval_type in enabled_eval_types:
            dataset_name = EVAL_TYPE_MAPPING[eval_type]
            dataset_path = dataset_base / dataset_name

            if not dataset_path.exists():
                print(f"⊘ Skipping {eval_type} - dataset not found: {dataset_path}")
                continue

            # Construct the command (no --generated, add --extract-frames)
            cmd = [
                "python",
                "run_eval.py",
                str(dataset_path),
                "--extract-frames",
            ]
            if args.limit:
                cmd.extend(["--limit", str(args.limit)])
            if args.num_trials != 1:
                cmd.extend(["--num-trials", str(args.num_trials)])

            print(f"\n{'[DRY RUN] Would run' if args.dry_run else 'Running'}: {' '.join(cmd)}")

            if args.dry_run:
                print(f"  → Dataset: {dataset_path}")
                continue

            # Run the command
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=False,  # Show output in real-time
                    text=True
                )
                print(f"✓ Success for {eval_type}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Error for {eval_type}")
                print(f"Return code: {e.returncode}")

        return 0

    # Normal mode: evaluate generated videos
    if not generations_dir.exists():
        print(f"Error: Generations directory not found: {generations_dir}")
        return 1

    # Get all model directories
    model_dirs = [d for d in generations_dir.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"No model directories found in {generations_dir}")
        return 1

    print(f"Found {len(model_dirs)} model(s) to evaluate")
    print(f"Generations dir: {generations_dir}")
    if args.max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(len(model_dirs), cpu)
    else:
        max_workers = args.max_workers
    if max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")
    print(f"Parallelism: {max_workers} worker(s)")

    started = time.time()
    sorted_models = sorted(model_dirs)

    any_failed = False
    completed = 0

    if max_workers == 1:
        for model_dir in sorted_models:
            outcome = _run_one_model(
                model_dir=model_dir,
                enabled_eval_types=enabled_eval_types,
                dataset_base=dataset_base,
                dry_run=args.dry_run,
                limit=args.limit,
                num_trials=args.num_trials,
            )
            completed += 1
            print(f"{'=' * 80}")
            print(f"Model: {outcome.model_name}  ({completed}/{len(sorted_models)})")
            print(f"{'=' * 80}")
            if outcome.skipped:
                print("Skipped:")
                for s in outcome.skipped:
                    print(f"  - {s}")
            for o in outcome.outcomes:
                print(f"\n{'✓' if o.ok else '✗'} {o.eval_type} (rc={o.returncode})")
                if o.summary:
                    print(o.summary)
                if not o.ok:
                    any_failed = True
            print()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_model = {
                ex.submit(
                    _run_one_model,
                    model_dir,
                    enabled_eval_types,
                    dataset_base,
                    args.dry_run,
                    args.limit,
                    args.num_trials,
                ): model_dir
                for model_dir in sorted_models
            }

            for fut in concurrent.futures.as_completed(fut_to_model):
                outcome = fut.result()
                completed += 1
                print(f"{'=' * 80}")
                print(f"Model: {outcome.model_name}  ({completed}/{len(sorted_models)})")
                print(f"{'=' * 80}")
                if outcome.skipped:
                    print("Skipped:")
                    for s in outcome.skipped:
                        print(f"  - {s}")
                for o in outcome.outcomes:
                    print(f"\n{'✓' if o.ok else '✗'} {o.eval_type} (rc={o.returncode})")
                    if o.summary:
                        print(o.summary)
                    if not o.ok:
                        any_failed = True
                print()

    elapsed = time.time() - started
    print(f"{'=' * 80}")
    print(f"Done in {elapsed:.1f}s. Failures: {'yes' if any_failed else 'no'}")
    print(f"{'=' * 80}")

    # Preserve historical behavior: keep exit code 0 even if some evals fail.
    # (If you prefer non-zero on any failure, we can add a flag and flip this.)
    return 0


if __name__ == "__main__":
    exit(main())
