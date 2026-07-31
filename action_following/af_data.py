#!/usr/bin/env python3
"""Shared data plumbing for the action-following study.

The generated clips ship as ``video_<id>_side_by_side.mp4``. Each one is a
2x2 grid:

    +----------------+----------------+
    | alpha  GT      | alpha  GEN     |   rows 0   .. H/2
    +----------------+----------------+
    | bravo  GT      | bravo  GEN     |   rows H/2 .. H
    +----------------+----------------+
      cols 0 .. W/2    cols W/2 .. W

Generated frame ``n`` is ground-truth frame ``frame1_idx + 1 + n``, where
``frame1_idx`` is the episode's reference frame that ``run_eval.py`` computes
from the handler. ``video_<id>`` counts the episodes that produced at least one
query, in sorted ``(episode, instance)`` order, which is what ``run_eval.py``
does with ``current_video_id``. This module reproduces that mapping so the
per-frame action stream can be lined up with the pixels.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vlm_eval"))

from run_eval import find_mc_video_pairs, identify_handler  # noqa: E402
from vlm_utils import find_generated_video_subdir  # noqa: E402

DATASET_BASE = ROOT / "mc_multiplayer_v2_eval_new_sneak_combined_simplified_naming"
GENERATIONS_BASE = ROOT / "mc_multiplayer_v2_generations"

# Minecraft's fov option is the *vertical* field of view. The camera bots render
# in fly mode, which applies the 1.1x movement multiplier to the default 70, so
# the effective vertical fov is 77 degrees. ``calibrate_fov.py`` refits this from
# the recorded camera actions on ground-truth video as a check.
VERTICAL_FOV_DEG = 77.0

# One Minecraft tick is one rendered frame here; the clips are 20 fps.
TICKS_PER_SECOND = 20.0


@dataclass
class Episode:
    """One generated clip, joined to its ground-truth action stream."""

    dataset: str
    video_id: int
    episode: str
    instance: str
    frame1: int
    alpha_json: Path
    bravo_json: Path
    alpha_video: Path
    bravo_video: Path
    sbs: Path

    def actions(self, player: str) -> List[dict]:
        path = self.alpha_json if player == "alpha" else self.bravo_json
        with open(path) as fh:
            return json.load(fh)


def generated_dir(model: str, dataset: str) -> Optional[Path]:
    base = GENERATIONS_BASE / model
    if not base.is_dir():
        return None
    try:
        return find_generated_video_subdir(base, dataset)
    except ValueError:
        return None


def _video_ids(subdir: Path) -> Dict[int, Path]:
    out = {}
    for f in subdir.glob("video_*_side_by_side.mp4"):
        m = re.match(r"video_(\d+)_side_by_side\.mp4", f.name)
        if m:
            out[int(m.group(1))] = f
    return out


class _Quiet:
    """The handlers print a yaw line per episode. Keep the reports readable."""

    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = open("/dev/null", "w")
        return self

    def __exit__(self, *exc):
        sys.stdout.close()
        sys.stdout = self._old
        return False


def build_index(dataset: str, model: str) -> List[Episode]:
    """Join generated clips to episodes, following ``run_eval.py``'s ordering."""
    with _Quiet():
        return _build_index(dataset, model)


def _build_index(dataset: str, model: str) -> List[Episode]:
    subdir = generated_dir(model, dataset)
    if subdir is None:
        return []
    clips = _video_ids(subdir)
    if not clips:
        return []

    # The `_long` variants are the same episodes with a longer generated
    # horizon. They share one ground-truth folder and one handler.
    base_name = dataset[: -len("_long")] if dataset.endswith("_long") else dataset
    folder = DATASET_BASE / base_name / "test"
    pairs = find_mc_video_pairs(folder)
    # run_eval.py truncates the pair list to the number of generated clips
    # before it extracts any query.
    pairs = pairs[: len(clips)]

    handler = identify_handler(base_name)
    handler_name = handler.__class__.__name__

    episodes: List[Episode] = []
    video_id = 0
    for pair in pairs:
        try:
            queries = handler.extract_keyframes(pair)
        except (ValueError, KeyError, FileNotFoundError):
            queries = []
        if not queries:
            continue
        # Mirror run_eval.py's per-handler query filtering. It removes queries
        # but never removes a whole episode, so it cannot shift the ids; it is
        # replicated only so `frame1` is read off the same surviving query.
        if "Rotation" in handler_name and "Both" not in handler_name:
            keep = queries[0].metadata["rotating_bot"]
            queries = [q for q in queries if q.metadata["variant"] == keep] or queries
        elif "LooksAway" in handler_name and "Both" not in handler_name:
            keep = queries[0].metadata["rotating_bot"]
            queries = [q for q in queries if q.metadata["variant"] == keep] or queries

        if video_id in clips:
            episodes.append(
                Episode(
                    dataset=dataset,
                    video_id=video_id,
                    episode=pair.episode_num,
                    instance=pair.instance_num,
                    frame1=int(queries[0].metadata["frame1"]),
                    alpha_json=pair.alpha_json,
                    bravo_json=pair.bravo_json,
                    alpha_video=pair.alpha_video,
                    bravo_video=pair.bravo_video,
                    sbs=clips[video_id],
                )
            )
        video_id += 1
    return episodes


# --- pixel access -----------------------------------------------------------

# The hotbar, hearts and hunger bar sit in a band across the bottom of the
# render, and the held item occupies the lower right. Both move with the HUD
# and not with the world, so they are masked out of every flow statistic.
HUD_BOTTOM_FRAC = 0.22
HAND_RIGHT_FRAC = 0.62


def view_mask(h: int, w: int) -> np.ndarray:
    """True where a pixel shows world geometry rather than HUD or held item."""
    mask = np.ones((h, w), dtype=bool)
    hud_top = int(round(h * (1.0 - HUD_BOTTOM_FRAC)))
    mask[hud_top:, :] = False
    hand_left = int(round(w * HAND_RIGHT_FRAC))
    mask[int(round(h * 0.55)) :, hand_left:] = False
    return mask


def read_quadrant(path: Path, player: str, source: str) -> np.ndarray:
    """Read one quadrant of a side-by-side clip as a uint8 grayscale stack.

    Args:
        player: ``"alpha"`` (top) or ``"bravo"`` (bottom).
        source: ``"gt"`` (left) or ``"gen"`` (right).
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        r0, r1 = (0, h // 2) if player == "alpha" else (h // 2, h)
        c0, c1 = (0, w // 2) if source == "gt" else (w // 2, w)
        frames.append(cv2.cvtColor(frame[r0:r1, c0:c1], cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 0, 0), np.uint8)


def read_all_quadrants(path: Path) -> Dict[str, np.ndarray]:
    """Decode a side-by-side clip once and return all four quadrants.

    Decoding dominates the cost of this study, so it is done once per clip
    rather than once per (player, source) pair.
    """
    cap = cv2.VideoCapture(str(path))
    buckets: Dict[str, list] = {k: [] for k in
                                ("alpha_gt", "alpha_gen", "bravo_gt", "bravo_gen")}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        hh, hw = h // 2, w // 2
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buckets["alpha_gt"].append(g[:hh, :hw])
        buckets["alpha_gen"].append(g[:hh, hw:])
        buckets["bravo_gt"].append(g[hh:, :hw])
        buckets["bravo_gen"].append(g[hh:, hw:])
    cap.release()
    return {k: (np.stack(v) if v else np.zeros((0, 0, 0), np.uint8))
            for k, v in buckets.items()}


def read_gt_video(path: Path) -> np.ndarray:
    """Read a raw ground-truth ``*_camera.mp4`` as a grayscale stack."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 0, 0), np.uint8)


def focal_px(height: int) -> float:
    """Pinhole focal length in pixels for Minecraft's 70 degree vertical fov."""
    return (height / 2.0) / np.tan(np.radians(VERTICAL_FOV_DEG) / 2.0)
