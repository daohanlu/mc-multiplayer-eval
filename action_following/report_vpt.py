#!/usr/bin/env python3
"""Pool the VPT IDM runs and compare them with the analytic camera estimator.

``vpt_idm.py`` writes one ``.npz`` per run. This reads them together and reports
mouse action-following the way Matrix-Game does, plus the two things that make
the numbers readable:

* **Relative to ground truth.** The IDM saturates at 10 deg/frame and our bots
  command 8.594, so most ground-truth turn frames read at the ceiling. Absolute
  gain is therefore compressed at the top: the IDM can see a model turning
  *less* than ground truth, but not one turning more. Every magnitude figure is
  divided by the ground-truth row for that player.
* **Against the analytic estimator.** The two share no code, no training data
  and no assumptions -- one is closed-form geometry with a known camera, the
  other a 1,962-hour neural network. Where they agree, the finding is not an
  artifact of either.

    python3 action_following/report_vpt.py action_following/results/vpt_raw*.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vpt_idm import YAW_COL, YAW_SIGN  # noqa: E402

GT = "GROUND TRUTH"

# One threshold for everything: what counts as a turn, in the mouse metric, in
# the no-false-turns column and in the Magnitude slope. Matrix-Game leaves theirs
# unspecified ("a pre-defined threshold"); 1 deg/frame sits far below our
# commanded 8.594 and far above the estimator noise floor, and nothing is
# sensitive to it -- see the sweep in RESULTS.md.
THRESHOLD_DEG = 1.0

LABEL = {
    "flagship": "Solaris",
    "no_player_attn_sf": "Independent",
    "concat_c": "Frame Concat",
    "from_scratch": "Solaris w/o pretrain",
    "causvid_regression": "ODE Reg",
    "causvid_dmd": "Causal FT Pre-DMD",
    "no_kv_cache_backprop": "Causal FT no KV-BP",
}

# Gain from report_camera.py, same 6 eval sets, for the cross-check. Regenerate
# with `python3 action_following/report_camera.py`.
ANALYTIC_GAIN = {
    GT: (0.863, 0.872),
    "flagship": (0.614, 0.671),
    "no_player_attn_sf": (0.937, 0.928),
    "concat_c": (0.817, 0.479),
    "from_scratch": (0.752, 0.745),
    "causvid_regression": (0.534, 0.620),
    "causvid_dmd": (0.275, 0.387),
    "no_kv_cache_backprop": (0.683, 0.711),
}


# Matrix-Game's shipped camera metric, ported from their GameWorldScore repo
# (GameWorld/third_party/IDM/IDM_bench.py, compute_camera_precision +
# camera_direction). Every frame is mapped to one of 9 classes -- still plus 8
# directions, each camera axis thresholded at 1e-2 deg/frame -- and the score
# is the fraction of frames whose class matches, one score per clip, averaged
# over clips. Despite their function's name this is per-frame *accuracy*, not
# precision: a frame with nothing commanded and nothing read counts as a hit,
# and still frames dominate, which is why published GameWorld numbers sit at
# 0.89-0.95.
MG_DELTA_DEG = 1e-2


def _dirs(yaw, pitch, thr):
    """Their 9 direction codes per frame; 0 is still."""
    xs = np.where(np.abs(yaw) < thr, 0, np.sign(yaw)).astype(int)
    ys = np.where(np.abs(pitch) < thr, 0, np.sign(pitch)).astype(int)
    return ys * 3 + xs


def load_cam(paths):
    """Raw (pred_yaw, pred_pitch, cmd_yaw, cmd_pitch) per clip, per (model, player)."""
    acc = defaultdict(list)
    for path in paths:
        d = np.load(path)
        for k in d.files:
            parts = k.split("|")
            if len(parts) == 4:
                if parts[0] != "cam":
                    continue
                _, name, player, _ = parts
            else:
                name, player, _ = parts
            pred, cmd = d[k]
            acc[(name, player)].append((YAW_SIGN * pred[:, YAW_COL],
                                        pred[:, 1 - YAW_COL], cmd[:, 0], cmd[:, 1]))
    return acc


def mg_camera_accuracy(clips) -> tuple:
    """Matrix-Game's metric, matching their code: per-clip mean over clips."""
    per = [100 * float((_dirs(py, pp, MG_DELTA_DEG) == _dirs(cy, cp, MG_DELTA_DEG)).mean())
           for py, pp, cy, cp in clips]
    return float(np.mean(per)), len(per)


# Our stricter variant, NOT what Matrix-Game ships: precision over the frames
# where a turn is *predicted*, frames pooled, 1 deg/frame threshold. It removes
# the still-frame floor, so it separates models the shipped metric cannot, and
# it charges a model for turns it invents -- the failure mode an accuracy over
# mostly-still frames barely notices.
def strict_mouse_precision(clips) -> tuple:
    """Precision over positive predictions, and the count of them."""
    pb = np.concatenate([_dirs(py, pp, THRESHOLD_DEG) for py, pp, _, _ in clips])
    cb = np.concatenate([_dirs(cy, cp, THRESHOLD_DEG) for _, _, cy, cp in clips])
    pos = pb != 0
    if not pos.sum():
        return float("nan"), 0
    return 100 * float((pb[pos] == cb[pos]).mean()), int(pos.sum())


def load(paths):
    acc = defaultdict(list)
    for p in paths:
        d = np.load(p)
        for k in d.files:
            # Runs before button prediction was added wrote "<model>|<player>|<i>";
            # later runs prefix each entry with "cam|" or "key|".
            parts = k.split("|")
            if len(parts) == 4:
                if parts[0] != "cam":
                    continue
                _, name, player, _ = parts
            else:
                name, player, _ = parts
            pred, cmd = d[k]
            acc[(name, player)].append((YAW_SIGN * pred[:, YAW_COL], cmd[:, 0]))
    return acc


def stats(pairs) -> dict:
    p = np.concatenate([a for a, _ in pairs])
    c = np.concatenate([b for _, b in pairs])
    # "A turn was commanded" uses the same 1 deg/frame threshold as the mouse
    # metric, so both columns agree on what counts as a turn. It makes almost no
    # difference to the slope -- 2.7% of turn frames fall below it and the
    # largest is 1 deg against a median command of 8.594 -- but a single
    # definition beats two.
    moving = np.abs(c) >= THRESHOLD_DEG
    still = np.abs(c) <= 1e-9
    hit = (np.abs(p) >= THRESHOLD_DEG) & (np.sign(p) == np.sign(c))
    return {
        "dir": 100 * float(hit[moving].mean()),
        "still": 100 * float((np.abs(p[still]) < THRESHOLD_DEG).mean()),
        # A single least-squares slope over the commanded frames, not a mean of
        # per-frame ratios: nothing is ever divided by a near-zero command.
        "gain": float(np.sum(p[moving] * c[moving]) / np.sum(c[moving] ** 2)),
        "n": int(moving.sum()),
        "sat": 100 * float((np.abs(p[moving]) >= 9.5).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    args = ap.parse_args()
    acc = load(args.npz)
    s = {k: stats(v) for k, v in acc.items()}

    print("=" * 90)
    print("MOUSE ACTION-FOLLOWING, VPT INVERSE DYNAMICS MODEL (the one Matrix-Game uses)")
    print("=" * 90)
    print(f"deadband {THRESHOLD_DEG} deg/frame; commanded rate 8.594; IDM saturates at 10\n")
    print(f"{'model':<24}{'dir acc A/B':>16}{'still A/B':>16}{'gain A/B':>16}"
          f"{'vs GT A/B':>16}")
    order = [GT] + [m for m in LABEL if (m, "alpha") in s]
    for name in order:
        a, b = s.get((name, "alpha")), s.get((name, "bravo"))
        if not (a and b):
            continue
        ga = a["gain"] / s[(GT, "alpha")]["gain"]
        gb = b["gain"] / s[(GT, "bravo")]["gain"]
        label = "ground truth (ceiling)" if name == GT else LABEL[name]
        print(f"{label:<24}{a['dir']:7.1f} /{b['dir']:6.1f}"
              f"{a['still']:8.1f} /{b['still']:6.1f}"
              f"{a['gain']:8.2f} /{b['gain']:6.2f}"
              f"{ga:8.2f} /{gb:6.2f}")
    print(f"\nturn frames: {s[(GT,'alpha')]['n']} alpha, {s[(GT,'bravo')]['n']} bravo")
    print(f"ground-truth frames reading at the IDM ceiling: "
          f"{s[(GT,'alpha')]['sat']:.0f}% alpha, {s[(GT,'bravo')]['sat']:.0f}% bravo.")
    print("Absolute gain is compressed at the top because of that, so read the")
    print("'vs GT' columns for magnitude, not the raw gain.")

    b = load_cam(args.npz)
    print("\n--- Matrix-Game's camera accuracy, as their code computes it ---")
    print(f"9 classes at {MG_DELTA_DEG} deg/frame, every frame counted (still")
    print("frames included), one score per clip, averaged over clips.")
    print(f"Next to it our strict variant: precision over predicted-turn frames")
    print(f"only, pooled, {THRESHOLD_DEG} deg/frame threshold.\n")
    print(f"{'model':<24}{'MG acc A/B':>16}{'strict prec A/B':>18}"
          f"{'pos preds A/B':>18}")
    for name in [GT] + [m for m in LABEL if (m, "alpha") in s]:
        mg = [mg_camera_accuracy(b[(name, p)]) for p in ("alpha", "bravo")]
        st = [strict_mouse_precision(b[(name, p)]) for p in ("alpha", "bravo")]
        lab = "ground truth" if name == GT else LABEL[name]
        print(f"{lab:<24}{mg[0][0]:8.1f} /{mg[1][0]:6.1f}"
              f"{st[0][0]:10.1f} /{st[1][0]:6.1f}"
              f"{st[0][1]:10d} /{st[1][1]:6d}")
    print("\nThe MG column is dominated by correct stillness, so every model lands")
    print("high. The strict column only scores frames where a turn is predicted,")
    print("so a model that invents turns is charged for each wrong one.")

    print("\n--- Cross-check against the analytic estimator ---")
    print("Both normalised by their own ground-truth row, so the two scales meet.\n")
    print(f"{'model':<24}{'VPT A/B':>16}{'analytic A/B':>16}")
    xs, ys = [], []
    for name, (aa, ab) in ANALYTIC_GAIN.items():
        if name == GT or (name, "alpha") not in s:
            continue
        va = s[(name, "alpha")]["gain"] / s[(GT, "alpha")]["gain"]
        vb = s[(name, "bravo")]["gain"] / s[(GT, "bravo")]["gain"]
        na = aa / ANALYTIC_GAIN[GT][0]
        nb = ab / ANALYTIC_GAIN[GT][1]
        xs += [va, vb]
        ys += [na, nb]
        print(f"{LABEL[name]:<24}{va:8.2f} /{vb:6.2f}{na:8.2f} /{nb:6.2f}")
    r = float(np.corrcoef(xs, ys)[0, 1])
    print(f"\nPearson r between the two estimators over {len(xs)} model-player "
          f"cells: {r:+.2f}")
    print("They share no code, no training data and no assumptions, so agreement")
    print("here is evidence about the models rather than about either estimator.")


if __name__ == "__main__":
    main()
