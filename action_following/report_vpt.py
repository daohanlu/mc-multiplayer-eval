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

    python3 action_following/report_vpt.py action_following/vpt_raw*.npz
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
DEADBAND_DEG = 1.0

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
    moving = np.abs(c) > 1e-9
    still = ~moving
    hit = (np.abs(p) >= DEADBAND_DEG) & (np.sign(p) == np.sign(c))
    return {
        "dir": 100 * float(hit[moving].mean()),
        "still": 100 * float((np.abs(p[still]) < DEADBAND_DEG).mean()),
        "gain": float(np.sum(p * c) / np.sum(c ** 2)),
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
    print(f"deadband {DEADBAND_DEG} deg/frame; commanded rate 8.594; IDM saturates at 10\n")
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
