#!/usr/bin/env python3
"""Camera (mouse) action-following, per player, from the cached motion.

Definitions, fixed once here so every table means the same thing.

``event%``      The headline. Commanded frames are grouped into contiguous turn
                events, and an event counts as followed when the rendered
                rotation over that window goes the commanded way and reaches at
                least a quarter of the commanded size. The window is widened by
                3 frames each side, because a generated clip often starts its
                turn a frame late and that is not a control failure.
``ev ratio``    Median over events of rendered rotation divided by commanded
                rotation. 1.0 means the view turns as far as it was told to;
                below 1.0 means it under-rotates.
``frame%``      The same left/right decision judged frame by frame, with no
                slack. This is Matrix-Game's mouse accuracy in its strictest
                form. It is reported for transparency, but a one-frame timing
                shift costs a model twice here, so ``event%`` is the fairer read.
``still%``      Of the frames that carry no camera command, the fraction where
                the rendered view stays inside the deadband. One minus this is
                the rate of hallucinated camera motion.
``gain``        Least-squares slope of the rendered turn rate on the commanded
                turn rate, over every frame.
``RotErr deg``  Mean over episodes of the absolute difference between the total
                rendered and total commanded yaw, in degrees. This is the
                camera-control literature's rotation error, over a whole clip,
                so it accumulates both missed turns and invented ones.
``track%``      Fraction of tracked points that survived the forward-backward
                check. A model whose frames were too corrupt to measure would
                show a low value here, and none of them do.
``deadband``    0.02 rad per frame, about 1.1 degrees. The commanded rate is
                0.15 rad per frame, and the estimator's noise on still
                ground-truth video is under 0.001, so the band separates the two
                by a wide margin either way.

Every number is reported for the ground-truth view as well. The ground-truth row
is the ceiling: it is the same estimator on real Minecraft video of the same
episode, so it separates estimator error from model error. Its ``ev ratio`` sits
slightly above 1.0, which is the estimator's own residual bias at the assumed
field of view; read a generated ratio against that row, not against 1.0.

    python3 action_following/report_camera.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from af_data import ROOT  # noqa: E402

CACHE = ROOT / "action_following" / "cache"
DEADBAND = 0.02  # radians per frame
# Frames of timing slack each side of a turn event. Set from measurement, not
# taste: `measure_latency.py` puts the ground-truth onset delay at a median of 0
# frames and a 90th percentile of 1, with generated clips about a frame later.
# The metric sums rotation across a whole event rather than testing one frame, so
# it barely depends on this: sweeping 1 to 8 moves any model by under 0.6 points
# and reorders nothing.
EVENT_SLACK = 3

# 0 is yaw, 1 is pitch. The bots drive yaw in every eval set; only structureEval
# also drives pitch, and it does so on about a fifth as many frames.
AXIS = 0

# The common set. `no_player_attn_sf` has no clips for the short bothLookAway
# and oneLooksAway variants, so including those would score it on a different
# number of events from every other model.
CAMERA_DATASETS = [
    "rotationEval",
    "turnToLookEval",
    "turnToLookOppositeEval",
    "bothLookAwayEval_long",
    "oneLooksAwayEval_long",
    "structureEval",
]

# Cross-player specificity needs eval sets where only ONE bot moves its camera.
# Where both turn at once their commands correlate, and a view that follows its
# own command would then appear to follow the other player's as well.
SINGLE_ACTOR_DATASETS = ["rotationEval", "oneLooksAwayEval_long"]

MODEL_LABEL = {
    "flagship": "Solaris",
    "no_player_attn_sf": "Independent",
    "concat_c": "Frame Concat",
    "from_scratch": "Solaris w/o pretrain",
    "causvid_regression": "ODE Reg",
    "causvid_dmd": "Causal FT Pre-DMD",
    "no_kv_cache_backprop": "Causal FT no KV-BP",
}
MODEL_ORDER = list(MODEL_LABEL)


class Acc:
    """Accumulates the pieces every metric is built from."""

    def __init__(self):
        self.est, self.cmd, self.inl = [], [], []
        self.ep_est, self.ep_cmd = [], []

    def add(self, est: np.ndarray, cmd: np.ndarray, inl: np.ndarray = None):
        self.est.append(est)
        self.cmd.append(cmd)
        if inl is not None:
            self.inl.append(inl)
        self.ep_est.append(float(est.sum()))
        self.ep_cmd.append(float(cmd.sum()))

    def events(self) -> dict:
        """Turn-event metrics, which a one-frame timing shift does not punish.

        A generated clip is under no obligation to start its turn on the exact
        frame the command arrived, and inspection shows it often starts one
        frame late. A per-frame comparison scores that as a miss twice over,
        once at the start and once at the end. So the commanded frames are
        grouped into contiguous turn events, and the rendered yaw is summed over
        the event window widened by ``EVENT_SLACK`` frames each side. An event
        counts as followed when the summed rotation goes the commanded way and
        reaches at least a quarter of the commanded size.
        """
        hits, ratios = [], []
        for est_c, cmd_c in zip(self.est, self.cmd):
            on = np.abs(cmd_c) > 1e-9
            if not on.any():
                continue
            edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            for j, (s, e) in enumerate(zip(starts, ends)):
                # The slack must not reach into a neighbouring event. Several
                # eval sets turn a player away and straight back, and without
                # this the two windows overlap and the rotations cancel, which
                # reads as a total control failure on ground-truth video too.
                lo = 0 if j == 0 else (ends[j - 1] + s) // 2
                hi = len(est_c) if j + 1 == len(starts) else (e + starts[j + 1]) // 2
                w0 = max(lo, s - EVENT_SLACK)
                w1 = min(hi, e + EVENT_SLACK)
                ct = float(cmd_c[s:e].sum())
                et = float(est_c[w0:w1].sum())
                if abs(ct) < 1e-9:
                    continue
                hits.append(np.sign(et) == np.sign(ct) and abs(et) >= 0.25 * abs(ct))
                ratios.append(et / ct)
        if not hits:
            return {}
        return {
            "ev_acc": 100 * float(np.mean(hits)),
            "ev_ratio": float(np.median(ratios)),
            "n_ev": len(hits),
        }

    def metrics(self) -> dict:
        if not self.est:
            return {}
        est = np.concatenate(self.est)
        cmd = np.concatenate(self.cmd)
        moving = np.abs(cmd) > 1e-9
        still = ~moving
        if moving.sum() == 0:
            return {}
        hit = (np.abs(est) >= DEADBAND) & (np.sign(est) == np.sign(cmd))
        turn_recall = float(hit[moving].mean())
        still_recall = float((np.abs(est[still]) < DEADBAND).mean()) if still.sum() else float("nan")
        gain = float(np.sum(est * cmd) / np.sum(cmd ** 2))
        r = float(np.corrcoef(est, cmd)[0, 1]) if est.std() > 0 else float("nan")
        ep_err = np.abs(np.array(self.ep_est) - np.array(self.ep_cmd))
        return {
            "turn_recall": 100 * turn_recall,
            "still_recall": 100 * still_recall,
            "balanced": 100 * (turn_recall + still_recall) / 2,
            "gain": gain,
            "r": r,
            "turn_err_deg": float(np.degrees(ep_err).mean()),
            "track": 100 * float(np.concatenate(self.inl).mean()) if self.inl else float("nan"),
            "n_turn": int(moving.sum()),
            "n_still": int(still.sum()),
            "n_ep": len(self.ep_est),
            **self.events(),
        }


def collect(datasets, model, source):
    """Per-player accumulators, plus the cross-player one.

    Alpha and Bravo of one clip are grouped so the cross-player comparison can
    hold a view against the *other* player's command stream.
    """
    per = defaultdict(Acc)
    cross = defaultdict(Acc)
    cmd_pairs = []
    for dataset in datasets:
        files = defaultdict(dict)
        d = CACHE / dataset / model
        for f in sorted(d.glob("*.npz")) if d.is_dir() else []:
            stem, player = f.stem.rsplit("_", 1)
            files[stem][player] = np.load(f, allow_pickle=True)
        for players in files.values():
            for player, z in players.items():
                est = z[source][:, AXIS]
                cmd = z["cam"][:, AXIS]
                per[player].add(est, cmd, z[source][:, 3])
                other = "bravo" if player == "alpha" else "alpha"
                if other in players:
                    cmd_other = players[other]["cam"][:, AXIS]
                    n = min(len(est), len(cmd_other))
                    cross[player].add(est[:n], cmd_other[:n])
            if "alpha" in players and "bravo" in players:
                ca = players["alpha"]["cam"][:, AXIS]
                cb = players["bravo"]["cam"][:, AXIS]
                n = min(len(ca), len(cb))
                cmd_pairs.append((ca[:n], cb[:n]))
    # How correlated the two players' own commands are. If they were highly
    # correlated the cross-player table would be confounded, because a view that
    # follows its own command would appear to follow the other one too.
    if cmd_pairs:
        ca = np.concatenate([a for a, _ in cmd_pairs])
        cb = np.concatenate([b for _, b in cmd_pairs])
        cmd_r = float(np.corrcoef(ca, cb)[0, 1]) if ca.std() and cb.std() else float("nan")
    else:
        cmd_r = float("nan")
    return per, cross, cmd_r


def fmt(m: dict, keys) -> str:
    if not m:
        return "".join(f"{'-':>12}" for _ in keys)
    out = ""
    for k in keys:
        v = m[k]
        out += (f"{v:12.1f}" if k.endswith(("recall", "balanced", "deg", "track", "acc"))
                else f"{v:12.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=CAMERA_DATASETS)
    ap.add_argument("--axis", choices=["yaw", "pitch"], default="yaw")
    args = ap.parse_args()
    global AXIS
    AXIS = 0 if args.axis == "yaw" else 1
    datasets = [d for d in args.datasets if (CACHE / d).is_dir()]

    cols = ["ev_acc", "ev_ratio", "turn_recall", "still_recall", "gain",
            "turn_err_deg", "track"]
    head = ["event%", "ev ratio", "frame%", "still%", "gain", "RotErr deg",
            "track%"]

    print("=" * 96)
    print(f"CAMERA ACTION-FOLLOWING, PER PLAYER  (axis: {args.axis})")
    print("=" * 96)
    print(f"datasets: {', '.join(datasets)}")
    print(f"deadband: {DEADBAND} rad/frame; commanded rate 0.15 rad/frame (3 rad/s at 20 fps)\n")

    for player in ("alpha", "bravo"):
        print(f"--- {player.upper()} ---")
        print(f"{'model':<24}" + "".join(f"{h:>12}" for h in head) + f"{'n event':>9}{'n turn':>9}")
        gt_per, _, _ = collect(datasets, "flagship", "gt")
        m = gt_per[player].metrics()
        print(f"{'ground truth (ceiling)':<24}" + fmt(m, cols) +
              (f"{m.get('n_ev', 0):9d}{m['n_turn']:9d}" if m else ""))
        for model in MODEL_ORDER:
            per, _, _ = collect(datasets, model, "gen")
            m = per[player].metrics()
            print(f"{MODEL_LABEL[model]:<24}" + fmt(m, cols) +
                  (f"{m.get('n_ev', 0):9d}{m['n_turn']:9d}" if m else ""))
        print()

    xds = [d for d in SINGLE_ACTOR_DATASETS if (CACHE / d).is_dir()]
    print("--- CROSS-PLAYER SPECIFICITY ---")
    print(f"datasets: {', '.join(xds)}  (only one bot turns in these)")
    print("Response of one player's view to the OTHER player's camera command.")
    print("A view that is driven by its own action stream and not the other one")
    print("scores near zero gain and near zero correlation here.\n")
    print(f"{'model':<24}{'own gain':>11}{'cross gain':>12}{'own r':>9}{'cross r':>9}")
    shown_cmd_r = None
    for model in ["__gt__"] + MODEL_ORDER:
        src = "gt" if model == "__gt__" else "gen"
        key = "flagship" if model == "__gt__" else model
        per, cross, cmd_r = collect(xds, key, src)
        own_g, cr_g, own_r, cr_r = [], [], [], []
        for player in ("alpha", "bravo"):
            a, b = per[player].metrics(), cross[player].metrics()
            if a and b:
                own_g.append(a["gain"]); cr_g.append(b["gain"])
                own_r.append(a["r"]); cr_r.append(b["r"])
        if not own_g:
            continue
        label = "ground truth" if model == "__gt__" else MODEL_LABEL[model]
        og, cg = float(np.mean(own_g)), float(np.mean(cr_g))
        print(f"{label:<24}{og:11.3f}{cg:12.3f}{np.mean(own_r):9.3f}"
              f"{np.mean(cr_r):9.3f}")
        shown_cmd_r = cmd_r
    if shown_cmd_r is not None:
        print(f"\nAlpha's and Bravo's own camera commands correlate at "
              f"r = {shown_cmd_r:+.3f} on these eval sets.\nThe cross column "
              f"means what it says only because that number is near zero.")

    print("\n--- PER DATASET, Solaris, both players pooled ---")
    print(f"{'dataset':<26}" + "".join(f"{h:>12}" for h in head))
    for dataset in datasets:
        for label, model, src in (("gt", "flagship", "gt"), ("gen", "flagship", "gen")):
            per, _, _ = collect([dataset], model, src)
            acc = Acc()
            for p in ("alpha", "bravo"):
                acc.est += per[p].est
                acc.cmd += per[p].cmd
                acc.inl += per[p].inl
                acc.ep_est += per[p].ep_est
                acc.ep_cmd += per[p].ep_cmd
            m = acc.metrics()
            print(f"{dataset + ' [' + label + ']':<26}" + fmt(m, cols))


if __name__ == "__main__":
    main()
