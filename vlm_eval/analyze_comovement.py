#!/usr/bin/env python3
"""Derive apparent on-screen motion geometrically, and validate the method.

For co-movement both bots move at once, so the answer cannot be read off one
bot's action label the way translationEval does — what an observer sees is the
*relative* displacement projected into their own camera frame. This computes
that from the recorded x/z/yaw.

Camera convention, derived empirically from the recorded actions rather than
assumed (see the docstring of ``camera_axes``):

    forward = (-sin(yaw), -cos(yaw))
    right   = ( cos(yaw), -sin(yaw))

``classify`` projects the other bot's displacement onto those axes and returns
the dominant component. ``validate_on_translation`` checks the whole approach
against translationEval, where expected answers are already known from action
labels — if the geometry reproduces those, it can be trusted for co-movement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import find_mc_video_pairs  # noqa: E402
from handlers.camera_utils import find_end_of_first_sneak_chunk  # noqa: E402

DIRS = ("forward", "back", "left", "right")

# Displacement smaller than this in both axes reads as no motion (blocks).
DEADZONE = 0.75
# How far the dominant axis must beat the other for the call to be clean.
DOMINANCE = 1.5


def movement_chunks(data: list[dict]) -> list[tuple[int, str, int]]:
    """Contiguous runs of a held direction key, as (start, direction, end)."""
    out: list[list] = []
    for i, frame in enumerate(data):
        action = frame.get("action", {})
        for d in DIRS:
            if action.get(d):
                if not out or out[-1][1] != d or i > out[-1][2] + 1:
                    out.append([i, d, i])
                else:
                    out[-1][2] = i
    return [tuple(c) for c in out]


def camera_axes(yaw: float):
    """Forward and right-hand unit vectors in (x, z) for a given yaw.

    Measured from the data rather than assumed: holding a direction key and
    comparing travel angle ``atan2(dx, dz)`` against yaw over translationEval
    gives, to within a fraction of a degree,

        forward -> yaw + 180 deg      left  -> yaw + 270 deg
        back    -> yaw +   0 deg      right -> yaw +  90 deg

    which yields the vectors below. Guessing the sign convention here produced
    a classifier that agreed with translationEval only 18/64 times.
    """
    return (-math.sin(yaw), -math.cos(yaw)), (math.cos(yaw), -math.sin(yaw))


def classify(obs: list[dict], other: list[dict], f1: int, f2: int) -> dict:
    """Apparent motion of ``other`` in ``obs``'s camera between two frames."""
    fwd, right = camera_axes(obs[f1]["yaw"])

    def rel(f):
        return other[f]["x"] - obs[f]["x"], other[f]["z"] - obs[f]["z"]

    x1, z1 = rel(f1)
    x2, z2 = rel(f2)
    depth1 = x1 * fwd[0] + z1 * fwd[1]
    depth2 = x2 * fwd[0] + z2 * fwd[1]
    lat1 = x1 * right[0] + z1 * right[1]
    lat2 = x2 * right[0] + z2 * right[1]
    d_depth, d_lat = depth2 - depth1, lat2 - lat1

    if abs(d_depth) < DEADZONE and abs(d_lat) < DEADZONE:
        answer, confident = "no motion", True
    elif abs(d_depth) >= abs(d_lat):
        answer = "closer" if d_depth < 0 else "farther"
        confident = abs(d_depth) >= DOMINANCE * abs(d_lat)
    else:
        answer = "right" if d_lat > 0 else "left"
        confident = abs(d_lat) >= DOMINANCE * abs(d_depth)

    return {"answer": answer, "confident": confident,
            "d_depth": round(d_depth, 2), "d_lateral": round(d_lat, 2),
            "dist1": round(math.hypot(x1, z1), 2),
            "dist2": round(math.hypot(x2, z2), 2)}


def validate_on_translation(limit: int = 32) -> bool:
    """Reproduce translationEval's action-derived answers geometrically."""
    from handlers.mc_multiplayer_handler_translation import MinecraftTranslationHandler

    base = Path("mc_multiplayer_v2_eval_new_sneak_combined/translationEval/test")
    handler = MinecraftTranslationHandler()
    agree = total = 0
    bad = []

    for pair in find_mc_video_pairs(base)[:limit]:
        queries = handler.extract_keyframes(pair)
        if not queries:
            continue
        alpha = json.load(open(pair.alpha_json))
        bravo = json.load(open(pair.bravo_json))
        for q in queries:
            f1, f2 = q.frame_index, q.second_frame_index
            obs, other = ((alpha, bravo) if q.metadata["variant"] == "alpha"
                          else (bravo, alpha))
            if f2 >= min(len(obs), len(other)):
                continue
            got = classify(obs, other, f1, f2)
            total += 1
            if got["answer"] == q.expected_answer:
                agree += 1
            else:
                bad.append((pair.episode_num, pair.instance_num,
                            q.metadata["variant"], q.expected_answer, got))

    print(f"geometry vs translationEval action labels: {agree}/{total} agree")
    for ep, inst, var, exp, got in bad[:10]:
        print(f"   ep{ep}/{inst} {var:6s} labels={exp:8s} geometry={got['answer']:8s}"
              f"  ddepth={got['d_depth']:6}  dlat={got['d_lateral']:6}")
    return bool(total) and agree == total


def survey(dataset: str, limit: int = 32, offset: int = 40) -> None:
    """Report the co-movement structure and derived answers for a dataset."""
    base = (Path("mc_multiplayer_v2_eval_new_sneak_combined_simplified_naming")
            / dataset / "test")
    pairs = find_mc_video_pairs(base)[:limit]
    print(f"\n=== {dataset}: {len(pairs)} pairs ===")

    combos, answers, shaky = Counter(), Counter(), []
    for pair in pairs:
        alpha = json.load(open(pair.alpha_json))
        bravo = json.load(open(pair.bravo_json))
        ca, cb = movement_chunks(alpha), movement_chunks(bravo)
        if len(ca) < 2 or len(cb) < 2:
            print(f"  ep{pair.episode_num}/{pair.instance_num}: "
                  f"only {len(ca)}/{len(cb)} chunks — skipped")
            continue

        # chunk 0 is the shared approach; chunk 1 is the tested co-movement
        f1 = min(ca[1][0], cb[1][0])
        f2 = f1 + offset
        if f2 >= min(len(alpha), len(bravo)):
            f2 = min(len(alpha), len(bravo)) - 1
        combos[(ca[1][1], cb[1][1])] += 1

        for var, obs, other in (("alpha", alpha, bravo), ("bravo", bravo, alpha)):
            got = classify(obs, other, f1, f2)
            answers[got["answer"]] += 1
            if not got["confident"]:
                shaky.append((pair.episode_num, pair.instance_num, var, got))

    print("  tested action pair (alpha, bravo):")
    for k, v in combos.most_common():
        print(f"     {k[0]:8s} + {k[1]:8s}  x{v}")
    print(f"  derived answers: {dict(answers)}")
    if shaky:
        print(f"  ambiguous (dominant axis < {DOMINANCE}x the other): {len(shaky)}")
        for ep, inst, var, g in shaky[:8]:
            print(f"     ep{ep}/{inst} {var:6s} -> {g['answer']:8s} "
                  f"ddepth={g['d_depth']:6} dlat={g['d_lateral']:6}")
    else:
        print("  ambiguous: none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offset", type=int, default=40,
                    help="frames after the tested chunk starts (default 40, "
                         "matching translationEval)")
    args = ap.parse_args()

    ok = validate_on_translation()
    print("VALIDATED\n" if ok else "MISMATCHES — geometry not yet trustworthy\n")
    for ds in ("coMovementEval", "coMovementWithDividerEval"):
        survey(ds, offset=args.offset)
