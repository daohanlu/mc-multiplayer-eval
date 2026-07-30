#!/usr/bin/env python3
"""Measure how long the rendered view takes to answer a command.

The eval sets are recorded, not simulated, so a key or mouse command does not
change the picture on the same frame. The tolerance the metrics allow should be
set by that delay rather than guessed. This measures it two ways:

* **Global lag.** Shift the estimated yaw against the commanded yaw by k frames
  and take the k that maximises the correlation. Run on ground-truth video this
  is the recorder's own delay; run on generated video it is the recorder's delay
  plus whatever the model adds.
* **Per-event onset.** For each turn event, the first frame at which the
  rendered view passes the deadband, relative to the frame the command arrived.
  The spread of that, not its mean, is what a tolerance has to cover.

    python3 action_following/measure_latency.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from af_data import ROOT  # noqa: E402
from report_camera import (  # noqa: E402
    CACHE,
    CAMERA_DATASETS,
    DEADBAND,
    MODEL_LABEL,
    MODEL_ORDER,
)

MAX_LAG = 6


def load_pairs(datasets, model, source):
    out = []
    for dataset in datasets:
        d = CACHE / dataset / model
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            out.append((z[source][:, 0], z["cam"][:, 0]))
    return out


def global_lag(pairs):
    """Correlation at each integer shift. Positive k means the view is late."""
    scores = {}
    for k in range(-MAX_LAG, MAX_LAG + 1):
        num = den_e = den_c = 0.0
        for est, cmd in pairs:
            if k > 0:
                e, c = est[k:], cmd[:-k]
            elif k < 0:
                e, c = est[:k], cmd[-k:]
            else:
                e, c = est, cmd
            if len(e) < 8:
                continue
            e = e - e.mean()
            c = c - c.mean()
            num += float((e * c).sum())
            den_e += float((e * e).sum())
            den_c += float((c * c).sum())
        scores[k] = num / np.sqrt(den_e * den_c) if den_e and den_c else 0.0
    return scores


def onset_delays(pairs):
    """Frames from command onset to the view first passing the deadband."""
    out = []
    for est, cmd in pairs:
        on = np.abs(cmd) > 1e-9
        if not on.any():
            continue
        edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            window = range(max(0, s - MAX_LAG), min(len(est), e + MAX_LAG))
            hit = [i for i in window
                   if abs(est[i]) >= DEADBAND and np.sign(est[i]) == np.sign(cmd[s:e].sum())]
            if hit:
                out.append(hit[0] - s)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=CAMERA_DATASETS)
    args = ap.parse_args()
    datasets = [d for d in args.datasets if (CACHE / d).is_dir()]

    print("=" * 78)
    print("HOW LATE IS THE RENDERED VIEW?")
    print("=" * 78)
    print(f"datasets: {', '.join(datasets)}")
    print("Frame alignment already removes the recorder's fixed one-frame lag,")
    print("so what is left below is jitter around that.\n")

    rows = [("ground truth", "flagship", "gt")]
    rows += [(MODEL_LABEL[m], m, "gen") for m in MODEL_ORDER]

    print(f"{'source':<24}{'best lag':>10}{'r at best':>11}{'r at 0':>9}"
          f"{'onset med':>11}{'onset p90':>11}{'n events':>10}")
    for label, model, source in rows:
        pairs = load_pairs(datasets, model, source)
        if not pairs:
            continue
        scores = global_lag(pairs)
        best = max(scores, key=lambda k: scores[k])
        d = onset_delays(pairs)
        med = float(np.median(d)) if len(d) else float("nan")
        p90 = float(np.percentile(d, 90)) if len(d) else float("nan")
        print(f"{label:<24}{best:10d}{scores[best]:11.3f}{scores[0]:9.3f}"
              f"{med:11.1f}{p90:11.1f}{len(d):10d}")

    print("\nGround-truth onset delay distribution (frames):")
    d = onset_delays(load_pairs(datasets, "flagship", "gt"))
    for v in range(-2, 6):
        n = int((d == v).sum())
        print(f"  {v:+d}: {n:5d}  {100 * n / max(len(d), 1):5.1f}%  {'#' * (n // 8)}")
    print(f"  within +/-3 frames: {100 * float((np.abs(d) <= 3).mean()):.1f}% of events")

    print("\nGenerated onset delay, pooled over the 7 models (frames):")
    dg = np.concatenate([onset_delays(load_pairs(datasets, m, "gen"))
                         for m in MODEL_ORDER])
    for v in range(-2, 6):
        n = int((dg == v).sum())
        print(f"  {v:+d}: {n:5d}  {100 * n / max(len(dg), 1):5.1f}%  {'#' * (n // 40)}")
    print(f"  within +/-3 frames: {100 * float((np.abs(dg) <= 3).mean()):.1f}% of events")


if __name__ == "__main__":
    main()
