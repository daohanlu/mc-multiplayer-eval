#!/usr/bin/env python3
"""
Read-only sanity check for the late-episode query toggle.

For each eval dataset under ``mc_multiplayer_v2_eval_new_sneak_combined/``:

1. Auto-detect the generated horizon length GEN_LEN by probing the first
   ``video_*_side_by_side.mp4`` under the matching subdir of
   ``mc_multiplayer_v2_generations/flagship/``.
2. For each episode, run the unmodified handler (toggle OFF) to get the
   "last" query frame ``current_last_frame``.
3. Compute ``late_target_frame = frame1_idx + GEN_LEN - 20`` (clamped to the
   per-bot data length).
4. Count frames in ``(current_last_frame, late_target_frame]`` whose action
   is non-noop (using ``_is_noop`` from ``handlers/camera_utils.py``) and
   record which actions appear.

Prints a per-eval summary so we can see whether moving the query out to the
late horizon would silently include additional bot actions.

This script makes NO API calls and writes nothing to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make sure handlers / vlm_utils are importable
sys.path.insert(0, str(Path(__file__).parent))

from handlers import (  # noqa: E402
    MinecraftBothLookAwayHandler,
    MinecraftLooksAwayHandler,
    MinecraftRotationHandler,
    MinecraftStructureBuildingHandler,
    MinecraftStructureNoPlaceHandler,
    MinecraftTranslationHandler,
    MinecraftTurnToLookHandler,
    MinecraftTurnToLookOppositeHandler,
)
from handlers.camera_utils import LATE_EPISODE_END_MARGIN, _is_noop  # noqa: E402
from run_eval import find_mc_video_pairs, identify_handler  # noqa: E402
from vlm_utils import KeyframeQuery, VideoPair  # noqa: E402

# Default roots; can be overridden via CLI flags.
DEFAULT_DATASET_BASE = Path("mc_multiplayer_v2_eval_new_sneak_combined")
DEFAULT_GENERATIONS_DIR = Path("mc_multiplayer_v2_generations") / "flagship"

# Map dataset folder name -> generated subdir lookup key (matches the suffix
# stripping logic in vlm_utils.find_generated_video_subdir).
DATASET_TO_GEN_KEY: Dict[str, str] = {
    "translationEval": "translation",
    "rotationEval": "rotation",
    "structureEval": "structure",
    "structureNoPlaceEval": "structure_no_place",
    "turnToLookEval": "turn_to_look",
    "turnToLookOppositeEval": "turn_to_look_opposite",
    "oneLooksAwayEval": "one_looks_away",
    "oneLooksAwayEval_long": "one_looks_away_long",
    "bothLookAwayEval": "both_look_away",
    "bothLookAwayEval_long": "both_look_away_long",
}


def _find_generations_subdir(generations_dir: Path, dataset_name: str) -> Optional[Path]:
    """
    Find a ``step_*_multiplayer_v2_eval_<key>[...]`` subdir matching the dataset.

    Mirrors the lookup logic used by ``vlm_utils.find_generated_video_subdir``,
    but tolerates missing entries (returns ``None`` instead of raising).
    """
    key = DATASET_TO_GEN_KEY.get(dataset_name)
    if key is None or not generations_dir.exists():
        return None

    # Mirror vlm_utils' suffix-stripping behavior so we accept both
    # ``..._eval_translation`` and ``..._eval_translation_max_speed`` etc.
    strippable = ["_max_speed_long", "_max_speed"]
    replacements = ["_long", ""]

    candidates: List[Path] = []
    for subdir in generations_dir.iterdir():
        if not subdir.is_dir():
            continue
        _, _, suffix = subdir.name.partition("eval_")
        if not suffix:
            continue
        # Try direct match plus the known suffix swaps
        normalized_suffixes = [suffix]
        for strip, repl in zip(strippable, replacements):
            if suffix.endswith(strip):
                normalized_suffixes.append(suffix[: -len(strip)] + repl)
        if any(s == key for s in normalized_suffixes):
            candidates.append(subdir)

    if not candidates:
        return None
    # Prefer the lexicographically last one (usually the latest checkpoint).
    return sorted(candidates)[-1]


def _probe_gen_len(generations_subdir: Path) -> Optional[int]:
    """Probe the first ``video_*_side_by_side.mp4`` for its frame count."""
    try:
        import cv2
    except ImportError:
        print("⚠ opencv-python not installed; cannot probe generated video lengths.")
        return None

    candidates = sorted(generations_subdir.glob("video_*_side_by_side.mp4"))
    if not candidates:
        return None

    cap = cv2.VideoCapture(str(candidates[0]))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return frame_count if frame_count > 0 else None


def _load_pair_data(pair: VideoPair) -> Tuple[List[dict], List[dict]]:
    with open(pair.alpha_json) as f:
        alpha_data = json.load(f)
    with open(pair.bravo_json) as f:
        bravo_data = json.load(f)
    return alpha_data, bravo_data


def _bot_data_for_query(
    query: KeyframeQuery,
    alpha_data: List[dict],
    bravo_data: List[dict],
) -> List[Tuple[str, List[dict]]]:
    """
    Return the list of (bot_name, data) pairs whose actions matter for this
    query's late-horizon window.

    - Translation: ``moving_bot`` from metadata.
    - Rotation / looks-away: ``rotating_bot``.
    - Both look-away: the per-bot ``variant``.
    - Structure / structure-no-place: report both (plan calls out
      observer-side post-actions specifically; we include the builder for
      reference too).
    - Turn-to-look / opposite: both bots.
    """
    meta = query.metadata
    if meta.get("is_turn_to_look"):
        return [("alpha", alpha_data), ("bravo", bravo_data)]

    if "moving_bot" in meta:
        bot = meta["moving_bot"]
    elif "rotating_bot" in meta:
        bot = meta["rotating_bot"]
    elif "builder" in meta:
        # Structure: the variant in metadata is the observer.
        return [
            (f"observer({meta['variant']})", alpha_data if meta["variant"] == "alpha" else bravo_data),
            (f"builder({meta['builder']})", alpha_data if meta["builder"] == "alpha" else bravo_data),
        ]
    else:
        bot = meta.get("variant", "alpha")

    data = alpha_data if bot == "alpha" else bravo_data
    return [(bot, data)]


def _last_query_for_handler(handler, queries: List[KeyframeQuery]) -> List[KeyframeQuery]:
    """
    Filter ``queries`` down to just the "last" query per group that the
    late-episode toggle would replace.

    - Translation / rotation / structure / turn-to-look: 1 query per pair.
    - Looks-away / both-look-away: keep only ``player_position_turned_back``
      queries (the toggle leaves ``player_invisible_looked_away`` alone).
    """
    name = handler.__class__.__name__
    if "LooksAway" in name:
        return [q for q in queries if q.metadata.get("query_type") == "player_position_turned_back"]
    return list(queries)


def _current_last_frame(query: KeyframeQuery) -> int:
    """The frame index that the toggle replaces (== ``frame2``)."""
    if query.second_frame_index is not None:
        return query.second_frame_index
    return query.frame_index


def _count_nonnoop_actions(
    data: List[dict], lo_exclusive: int, hi_inclusive: int
) -> Tuple[int, Counter]:
    """
    Count non-noop frames in ``(lo_exclusive, hi_inclusive]`` and tally which
    boolean action keys (and 'camera') showed up.
    """
    if hi_inclusive <= lo_exclusive:
        return 0, Counter()

    count = 0
    actions: Counter = Counter()
    end = min(hi_inclusive + 1, len(data))
    for i in range(lo_exclusive + 1, end):
        action = data[i].get("action", {})
        if _is_noop(action):
            continue
        count += 1
        for key, value in action.items():
            if key == "camera":
                if any(abs(v) > 1e-4 for v in value):
                    actions["camera"] += 1
            elif isinstance(value, bool) and value:
                actions[key] += 1
    return count, actions


def _check_dataset(
    dataset_path: Path,
    generations_dir: Path,
    limit: Optional[int],
) -> Optional[dict]:
    dataset_name = dataset_path.name
    test_dir = dataset_path / "test"
    if not test_dir.exists():
        print(f"⊘ {dataset_name}: no test/ subdirectory at {test_dir}")
        return None

    try:
        # Toggle MUST be off so we get the "current" handler behavior.
        os.environ.pop("LATE_EPISODE_QUERY", None)
        os.environ.pop("LATE_EPISODE_GEN_LEN", None)
        handler = identify_handler(dataset_name)
    except ValueError as e:
        print(f"⊘ {dataset_name}: cannot identify handler ({e})")
        return None

    pairs = find_mc_video_pairs(test_dir)
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        print(f"⊘ {dataset_name}: no video pairs in {test_dir}")
        return None

    gen_subdir = _find_generations_subdir(generations_dir, dataset_name)
    gen_len: Optional[int] = None
    if gen_subdir is not None:
        gen_len = _probe_gen_len(gen_subdir)

    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name} ({len(pairs)} episodes)")
    if gen_subdir is None:
        print(f"  Generated subdir: <not found under {generations_dir}>")
    else:
        print(f"  Generated subdir: {gen_subdir.name}")
    print(f"  GEN_LEN (probed): {gen_len if gen_len is not None else '<unavailable>'}")
    print(f"{'=' * 80}")

    summary = {
        "dataset": dataset_name,
        "gen_len": gen_len,
        "num_episodes": 0,
        "episodes_with_post_action": 0,
        "total_post_action_frames": 0,
        "actions_seen": Counter(),
        "skipped_no_gen_len": gen_len is None,
        "skipped_episodes": 0,
    }

    for pair in pairs:
        try:
            queries = handler.extract_keyframes(pair)
        except (ValueError, KeyError) as e:
            print(f"  ⚠ ep{pair.episode_num} inst{pair.instance_num}: skipped ({e})")
            summary["skipped_episodes"] += 1
            continue

        last_queries = _last_query_for_handler(handler, queries)
        if not last_queries:
            continue

        alpha_data, bravo_data = _load_pair_data(pair)
        episode_post_action = 0

        for q in last_queries:
            frame1 = q.metadata["frame1"]
            current_last = _current_last_frame(q)
            for bot, data in _bot_data_for_query(q, alpha_data, bravo_data):
                if gen_len is None:
                    # GT-only fallback: mirror the handler helper's len(GT)-20.
                    late_target = max(0, len(data) - LATE_EPISODE_END_MARGIN)
                else:
                    late_target = min(
                        frame1 + gen_len - LATE_EPISODE_END_MARGIN,
                        len(data) - 1,
                    )
                if late_target <= current_last:
                    continue
                count, actions = _count_nonnoop_actions(data, current_last, late_target)
                if count > 0:
                    episode_post_action += count
                    summary["actions_seen"].update(actions)
                    print(
                        f"  ep{pair.episode_num} inst{pair.instance_num} {bot}: "
                        f"current_last={current_last} late={late_target} "
                        f"frame1={frame1} -> {count} non-noop frame(s) "
                        f"({dict(actions)})"
                    )

        summary["num_episodes"] += 1
        if episode_post_action > 0:
            summary["episodes_with_post_action"] += 1
            summary["total_post_action_frames"] += episode_post_action

    print(
        f"\n  Summary: {summary['episodes_with_post_action']}/{summary['num_episodes']} "
        f"episode(s) have post-action frame(s); "
        f"total post-action frames = {summary['total_post_action_frames']}; "
        f"action histogram = {dict(summary['actions_seen'])}"
    )
    if summary["skipped_episodes"]:
        print(f"  Skipped {summary['skipped_episodes']} episode(s) due to handler errors.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-base",
        type=Path,
        default=DEFAULT_DATASET_BASE,
        help="Root containing the eval dataset folders (default: %(default)s).",
    )
    parser.add_argument(
        "--generations-dir",
        type=Path,
        default=DEFAULT_GENERATIONS_DIR,
        help="Model generations dir to probe for GEN_LEN (default: %(default)s).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Restrict to these dataset folder names (default: all under --dataset-base).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Cap on number of episodes per dataset (for quick sanity).",
    )
    args = parser.parse_args()

    base: Path = args.dataset_base
    if not base.exists():
        print(f"Error: dataset base not found: {base}")
        return 1

    if args.datasets:
        dataset_paths = [base / d for d in args.datasets]
    else:
        dataset_paths = sorted(d for d in base.iterdir() if d.is_dir())

    print(f"Dataset base: {base}")
    print(f"Generations dir: {args.generations_dir}")
    print(f"Datasets to check: {[d.name for d in dataset_paths]}")

    summaries = []
    for dp in dataset_paths:
        summary = _check_dataset(dp, args.generations_dir, args.limit)
        if summary is not None:
            summaries.append(summary)

    print(f"\n{'#' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'#' * 80}")
    for s in summaries:
        flag = " ⚠" if s["episodes_with_post_action"] else ""
        print(
            f"  {s['dataset']:<28} gen_len={s['gen_len']!s:<6} "
            f"episodes_with_post_action={s['episodes_with_post_action']}/"
            f"{s['num_episodes']} "
            f"total_post_action_frames={s['total_post_action_frames']} "
            f"actions={dict(s['actions_seen'])}{flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
