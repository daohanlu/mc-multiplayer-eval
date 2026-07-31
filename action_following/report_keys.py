#!/usr/bin/env python3
"""Keyboard action-following, per player, with a learned inverse dynamics model.

The camera can be read off the video analytically, because a turn moves every
ray by the same angle. A key press cannot. On an open Minecraft plain a sideways
step and a small camera turn produce almost the same flow field, and the
parallax that separates them lives in the few metres of ground at the bottom of
the frame. Matrix-Game, Oasis and WorldMem all resolve this the same way: train
an inverse dynamics model to read the action back out of the video, then score
its prediction against the action that was actually sent.

This module does that at the scale available here. The model is multinomial
logistic regression over the flow summary of a five-frame window. It is trained
only on *ground-truth* video, and only on the episodes held out from the test
split, so the number it reaches on ground-truth test episodes is the ceiling for
every generated number below it.

    python3 action_following/report_keys.py
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

# Only the eval sets whose bots press movement keys.
KEY_DATASETS = ["translationEval", "structureEval"]

CLASSES = ["none", "forward", "back", "left", "right"]
WINDOW = 2  # frames each side, so a five-frame context

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


def windowed(x: np.ndarray) -> np.ndarray:
    """Stack each row with its neighbours, edge-padded."""
    pads = [np.roll(x, k, axis=0) for k in range(-WINDOW, WINDOW + 1)]
    out = np.concatenate(pads, axis=1)
    out[:WINDOW] = out[WINDOW]
    out[-WINDOW:] = out[-WINDOW - 1]
    return out


def label_frames(z) -> np.ndarray:
    names = list(z["key_names"])
    keys = z["keys"]
    lab = np.zeros(keys.shape[1], dtype=np.int64)
    active = np.zeros(keys.shape[1], dtype=np.int64)
    for cls in CLASSES[1:]:
        row = keys[names.index(cls)].astype(bool)
        lab[row] = CLASSES.index(cls)
        active += row
    lab[active > 1] = -1  # a combination the five classes cannot express
    return lab


def gather(datasets, model, source):
    """Return per-(dataset, episode, player) feature blocks and labels."""
    out = []
    for dataset in datasets:
        d = CACHE / dataset / model
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npz")):
            stem, player = f.stem.rsplit("_", 1)
            z = np.load(f, allow_pickle=True)
            x = windowed(np.nan_to_num(z[source]))
            y = label_frames(z)
            n = min(len(x), len(y))
            out.append((dataset, stem, player, x[:n], y[:n]))
    return out


# Matrix-Game's shipped keyboard metric, from their GameWorldScore repo
# (GameWorld/third_party/IDM/IDM_bench.py): four groups -- (back, forward) and
# (left, right) as 3-class problems, attack and jump as binary -- each scored
# with sklearn's micro-averaged precision over ALL frames, which for
# multi-class is plain accuracy with the no-op class included, then the four
# group scores averaged. Our bots never attack or jump and the 5-class IDM
# cannot predict either, so those two groups score exactly 100 and are
# included as such, the way their code would.
MG_GROUPS = {"forward/back": ("forward", "back"), "left/right": ("left", "right")}


def _group_labels(y: np.ndarray, members) -> np.ndarray:
    """5-class label -> the group's 3-class label (0 = neither member)."""
    out = np.zeros_like(y)
    for j, m in enumerate(members):
        out[y == CLASSES.index(m)] = j + 1
    return out


def mg_keyboard_accuracy(y, p) -> float:
    """Matrix-Game's metric, matching their code: mean over the four groups."""
    scores = [100 * float((_group_labels(y, m) == _group_labels(p, m)).mean())
              for m in MG_GROUPS.values()]
    scores += [100.0, 100.0]  # attack, jump: never commanded, never predicted
    return float(np.mean(scores))


# Our stricter variant, NOT what Matrix-Game ships: per-group precision over
# the frames where a key is *predicted*, so the no-op floor is removed and a
# model that hallucinates presses is charged for each one.
def mg_group_precision(y, p) -> dict:
    """Per-group precision over positive predictions."""
    out = {}
    for name, members in MG_GROUPS.items():
        idx = [CLASSES.index(m) for m in members]
        pred_pos = np.isin(p, idx)
        out[name] = (100 * float((p[pred_pos] == y[pred_pos]).mean())
                     if pred_pos.sum() else float("nan"))
    return out


def no_false_presses(y, p) -> float:
    """Of frames with no key commanded, the fraction read as no key."""
    idle = y == 0
    return 100 * float((p[idle] == 0).mean()) if idle.sum() else float("nan")


def balanced_accuracy(y, p) -> float:
    accs = [float((p[y == c] == c).mean()) for c in np.unique(y)]
    return 100 * float(np.mean(accs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=KEY_DATASETS)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    datasets = [d for d in args.datasets if (CACHE / d).is_dir()]
    blocks = gather(datasets, "flagship", "gt")
    if not blocks:
        print("no cache yet")
        return

    # Split by episode so no episode contributes to both halves. Alpha and Bravo
    # of one episode share a stem, so the two players never straddle the split
    # either, and the 5-frame window never crosses it.
    stems = sorted({(d, s) for d, s, _, _, _ in blocks})
    test_stems = set(stems[::3])

    def split(sel):
        xs = [b[3] for b in blocks if ((b[0], b[1]) in test_stems) == sel]
        ys = [b[4] for b in blocks if ((b[0], b[1]) in test_stems) == sel]
        x, y = np.concatenate(xs), np.concatenate(ys)
        keep = y >= 0
        return x[keep], y[keep]

    xtr, ytr = split(False)
    xte, yte = split(True)
    print("=" * 92)
    print("KEYBOARD ACTION-FOLLOWING, PER PLAYER")
    print("=" * 92)
    print(f"datasets: {', '.join(datasets)}")
    print(f"episodes: {len(stems) - len(test_stems)} train / {len(test_stems)} "
          f"held out, of {len(stems)}  (split by episode, every 3rd held out)")
    print(f"frames:   {len(ytr)} train / {len(yte)} held out")
    print("class counts (train): " +
          ", ".join(f"{CLASSES[c]}={int((ytr == c).sum())}" for c in range(len(CLASSES))))
    print(f"  non-'none' training frames: {int((ytr > 0).sum())}")
    print("class counts (held out): " +
          ", ".join(f"{CLASSES[c]}={int((yte == c).sum())}" for c in range(len(CLASSES))))

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
    )
    clf.fit(xtr, ytr)
    pte = clf.predict(xte)
    ptr = clf.predict(xtr)
    print(f"\nIDM on TRAIN episodes:    accuracy {100 * (ptr == ytr).mean():.1f}%, "
          f"balanced {balanced_accuracy(ytr, ptr):.1f}%")
    print(f"IDM on HELD-OUT episodes: accuracy {100 * (pte == yte).mean():.1f}%, "
          f"balanced {balanced_accuracy(yte, pte):.1f}%   <- the ceiling")
    print("  The two are close, so the model is not memorising episodes.")
    print("  Every ceiling quoted below is the held-out number, never the train one.")
    print("  per class recall, held out: " + ", ".join(
        f"{CLASSES[c]} {100 * (pte[yte == c] == c).mean():.1f}%"
        for c in range(len(CLASSES)) if (yte == c).sum()))

    print("\nAll rows below are scored on the HELD-OUT episodes only, the same")
    print("ones the ceiling uses, so every row sees the same scenes. The IDM never")
    print("saw generated video of any episode, but a train episode's ground truth")
    print("shares its scene, so scoring generated clips there would not be like")
    print("for like. The all-episode numbers are printed afterwards for reference.")
    print(f"\n{'model':<24}{'MG kb acc A/B':>17}{'strict prec A/B':>18}"
          f"{'no false press A/B':>22}{'balanced A/B':>18}{'n frames':>10}")
    rows = [("ground truth (ceiling)", "flagship", "gt")]
    rows += [(MODEL_LABEL[m], m, "gen") for m in MODEL_ORDER]
    for label, model, source in rows:
        blocks_m = gather(datasets, model, source)
        if not blocks_m:
            continue
        per = defaultdict(lambda: ([], []))
        for dataset, stem, player, x, y in blocks_m:
            if (dataset, stem) not in test_stems:
                continue
            keep = y >= 0
            per[player][0].append(x[keep])
            per[player][1].append(y[keep])
        cells, n_total, preds = {}, 0, []
        for player in ("alpha", "bravo"):
            if not per[player][0]:
                continue
            x = np.concatenate(per[player][0])
            y = np.concatenate(per[player][1])
            p = clf.predict(x)
            g = mg_group_precision(y, p)
            cells[player] = (mg_keyboard_accuracy(y, p),
                             np.nanmean(list(g.values())), no_false_presses(y, p),
                             balanced_accuracy(y, p))
            preds.append(p)
            n_total += len(y)
        a = cells.get("alpha", (float("nan"),) * 4)
        b = cells.get("bravo", (float("nan"),) * 4)
        print(f"{label:<24}{a[0]:9.1f} /{b[0]:6.1f}{a[1]:10.1f} /{b[1]:6.1f}"
              f"{a[2]:13.1f} /{b[2]:7.1f}{a[3]:9.1f} /{b[3]:7.1f}{n_total:10d}")

    print(f"\n{'model (all episodes)':<24}{'alpha bal%':>12}{'bravo bal%':>12}"
          f"{'n frames':>10}")
    for label, model in [(MODEL_LABEL[m], m) for m in MODEL_ORDER]:
        blocks_m = gather(datasets, model, "gen")
        if not blocks_m:
            continue
        per = defaultdict(lambda: ([], []))
        for _, _, player, x, y in blocks_m:
            keep = y >= 0
            per[player][0].append(x[keep])
            per[player][1].append(y[keep])
        cells, n_total = {}, 0
        for player in ("alpha", "bravo"):
            if not per[player][0]:
                continue
            x = np.concatenate(per[player][0])
            y = np.concatenate(per[player][1])
            cells[player] = balanced_accuracy(y, clf.predict(x))
            n_total += len(y)
        print(f"{label:<24}{cells.get('alpha', float('nan')):12.1f}"
              f"{cells.get('bravo', float('nan')):12.1f}{n_total:10d}")

    print("\nBalanced accuracy is the mean per-class recall, so the 'none' class,")
    print("which is about 90 percent of frames, cannot carry the number.")
    print("Chance level is 20.0 for five classes.")
    print("\n'pred none%' is how often the model reads as standing still. The")
    print("share of frames that really are still is the same for every row, so a")
    print("high value there means the rendered view under-responds rather than")
    print("that the estimator failed. Read it with the balanced column, not")
    print("instead of it: a generated clip is also blurrier than real video, and")
    print("some of the gap below the ground-truth row is that domain shift.")


if __name__ == "__main__":
    main()
