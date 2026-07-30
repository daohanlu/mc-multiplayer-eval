#!/usr/bin/env python3
"""Check the camera estimator against real Minecraft video, and refit the fov.

The estimator turns pixels into viewing rays, so it needs the field of view. The
render bots run Minecraft's default 70 degrees with the 1.1x fly multiplier, so
77 degrees vertical is the nominal value. This script does not take that on
trust. It re-runs the estimator over a sweep of assumed fields of view on
ground-truth video, where the commanded turn rate is known exactly, and reports
the gain at each one. The fov whose gain is 1.0 is the one the renderer used.

The same run gives the ceiling for the whole study. A generated clip cannot be
scored above what this estimator achieves on real video of the same episodes.

    python3 action_following/calibrate_fov.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import af_data  # noqa: E402
from af_data import build_index, read_all_quadrants  # noqa: E402

DATASETS = ["rotationEval", "turnToLookEval", "turnToLookOppositeEval"]
DEADBAND = 0.02


def clips(datasets, n_episodes):
    out = []
    for dataset in datasets:
        for ep in build_index(dataset, "flagship")[:n_episodes]:
            quads = read_all_quadrants(ep.sbs)
            for player in ("alpha", "bravo"):
                gt = quads[f"{player}_gt"]
                if gt.shape[0] >= 8:
                    out.append((gt, ep.actions(player), ep.frame1 + 1))
    return out


def score(data, fov):
    import importlib

    af_data.VERTICAL_FOV_DEG = fov
    af_flow = importlib.import_module("af_flow")
    importlib.reload(af_flow)
    est, cmd = [], []
    for gt, actions, start in data:
        m = af_flow.motion_track(gt)
        c = af_flow.commanded_camera(actions, start, len(m))
        est.append(m[:, 0])
        cmd.append(c[:, 0])
    est = np.concatenate(est)
    cmd = np.concatenate(cmd)
    moving = np.abs(cmd) > 1e-9
    gain = float(np.sum(est * cmd) / np.sum(cmd ** 2))
    r = float(np.corrcoef(est, cmd)[0, 1])
    hit = (np.abs(est) >= DEADBAND) & (np.sign(est) == np.sign(cmd))
    return {
        "gain": gain,
        "r": r,
        "turn_recall": 100 * float(hit[moving].mean()),
        "still_recall": 100 * float((np.abs(est[~moving]) < DEADBAND).mean()),
        "still_noise": float(np.abs(est[~moving]).mean()),
        "n_turn": int(moving.sum()),
        "n_still": int((~moving).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4, help="per dataset")
    ap.add_argument("--fovs", type=float, nargs="*",
                    default=[66, 68, 70, 72, 73, 74, 75, 77, 80, 84])
    args = ap.parse_args()

    data = clips(DATASETS, args.episodes)
    n_frames = sum(len(g) for g, _, _ in data)
    print(f"ground-truth views: {len(data)}, frames: {n_frames}")
    print(f"datasets: {', '.join(DATASETS)}\n")
    print(f"{'vfov deg':>9}{'gain':>9}{'r':>8}{'turn%':>8}{'still%':>9}"
          f"{'still noise rad':>17}")
    best = None
    for fov in args.fovs:
        s = score(data, fov)
        print(f"{fov:9.1f}{s['gain']:9.3f}{s['r']:8.3f}{s['turn_recall']:8.1f}"
              f"{s['still_recall']:9.1f}{s['still_noise']:17.5f}")
        if best is None or abs(s["gain"] - 1) < abs(best[1]["gain"] - 1):
            best = (fov, s)
    print(f"\ncommanded turn rate: 0.15 rad/frame at 20 fps = 3 rad/s")
    print(f"deadband: {DEADBAND} rad/frame")
    print(f"turn frames {best[1]['n_turn']}, still frames {best[1]['n_still']}")
    print(f"\nbest fov {best[0]:.1f} deg, gain {best[1]['gain']:.3f}")
    print("The estimator is unbiased there, so it neither flatters nor")
    print("penalises a generated clip that turns the commanded amount.")


if __name__ == "__main__":
    main()
