#!/usr/bin/env python3
"""Mouse action-following with the VPT inverse dynamics model.

Matrix-Game's action-controllability numbers come from an inverse dynamics model
trained on 1,962 hours of Minecraft with 90.6% keypress accuracy and R^2 = 0.97
on mouse movement. Those are VPT's own figures (Baker et al., 2022), and OpenAI
released that IDM, so the same model can be pointed at our clips. That makes our
mouse numbers directly comparable to the published ones instead of only
internally consistent.

Setup is in ``setup_vpt_idm.sh``. Our render is 640x352 per view against VPT's
640x360, and a different Minecraft version, so the first thing this script does
is score the IDM on ground-truth video where the commanded camera is known
exactly. If it cannot read real video of our episodes, it cannot read generated
video of them either, and that is the finding.

    python3 action_following/vpt_idm.py --validate            # ground truth only
    python3 action_following/vpt_idm.py --models flagship
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from af_data import ROOT, build_index  # noqa: E402

VPT_DIR = Path(os.environ.get(
    "VPT_DIR",
    "/tmp/claude-1000/-home-dl3957-Documents-mp-eval-datasets/"
    "3b1dd859-0038-4427-a7cb-6f471db5d2ea/scratchpad/vpt",
))

# VPT renders at 640x360 with fov 70; our camera bots render 640x352 with an
# effective fov of 77. The IDM downsamples to 128x128 internally either way.
VPT_RESOLUTION = (640, 360)

# The IDM's camera head saturates here, which matters: our bots command
# 0.15 rad/frame = 8.594 deg/frame, inside the range but in its outer bins.
CAMERA_MAXVAL_DEG = 10.0

# MineRL's yaw and our recorded ``camera[0]`` point opposite ways. Established on
# ground-truth video, where the IDM reproduces the commanded rate at a gain of
# -0.99: the magnitude is right and only the sign differs. Calibrated once, on
# ground truth, and then held fixed -- refitting it per model would let each
# model choose the sign that flattered it.
YAW_SIGN = -1.0

# MineRL orders the camera action (pitch, yaw), so yaw is column 1. Confirmed
# against yaw-only eval sets rather than assumed; --validate reprints the check.
YAW_COL = 1

CHUNK = 128  # the IDM's context length

CAMERA_DATASETS = ["rotationEval", "turnToLookEval", "turnToLookOppositeEval"]

# Matrix-Game scores the keyboard as mutually exclusive groups. Only the two
# movement groups are meaningful in our eval sets; the bots almost never jump or
# attack, so those groups would be trivially near perfect.
KEY_GROUPS = {"fb": ("forward", "back"), "lr": ("left", "right")}
KEY_NAMES = ["forward", "back", "left", "right"]
MODELS = ["flagship", "no_player_attn_sf", "concat_c", "from_scratch",
          "causvid_regression", "causvid_dmd", "no_kv_cache_backprop"]


def load_agent(device: str = "cuda:0"):
    import pickle

    sys.path.insert(0, str(VPT_DIR))
    from inverse_dynamics_model import IDMAgent  # noqa: E402

    params = pickle.load(open(VPT_DIR / "4x_idm.model", "rb"))
    net_kwargs = params["model"]["args"]["net"]["args"]
    pi_head_kwargs = params["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs,
                     device=device)
    agent.load_weights(str(VPT_DIR / "4x_idm.weights"))
    return agent


def read_quadrants_rgb(path: Path):
    """Decode a side-by-side clip once into four RGB stacks at VPT resolution."""
    cap = cv2.VideoCapture(str(path))
    out = {k: [] for k in ("alpha_gt", "alpha_gen", "bravo_gt", "bravo_gen")}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        hh, hw = h // 2, w // 2
        for key, sub in (("alpha_gt", frame[:hh, :hw]), ("alpha_gen", frame[:hh, hw:]),
                         ("bravo_gt", frame[hh:, :hw]), ("bravo_gen", frame[hh:, hw:])):
            out[key].append(cv2.resize(sub, VPT_RESOLUTION)[..., ::-1])  # BGR -> RGB
    cap.release()
    return {k: (np.stack(v) if v else np.zeros((0, 0, 0, 3), np.uint8))
            for k, v in out.items()}


def predict(agent, frames: np.ndarray):
    """Per-frame camera in degrees ``(N, 2)`` and WASD buttons ``(N, 4)``.

    MineRL orders the camera action as (pitch, yaw). The clip is fed in
    context-length chunks, as the reference script does, with the hidden state
    reset once per clip.
    """
    agent.reset()
    cams, keys = [], []
    for i in range(0, len(frames), CHUNK):
        chunk = frames[i:i + CHUNK]
        if len(chunk) < 2:
            break
        pred = agent.predict_actions(chunk)
        cams.append(np.asarray(pred["camera"])[0])
        keys.append(np.stack([np.asarray(pred[k])[0].ravel()
                              for k in KEY_NAMES], axis=1))
    if not cams:
        return np.zeros((0, 2)), np.zeros((0, 4))
    return np.concatenate(cams), np.concatenate(keys)


def commanded_deg(actions: list, gen0_gt_index: int, count: int) -> np.ndarray:
    """Commanded (yaw, pitch) per frame in degrees, aligned to the clip."""
    out = np.zeros((count, 2), dtype=np.float64)
    for i in range(count):
        k = gen0_gt_index + i
        if 0 <= k < len(actions):
            cam = actions[k]["action"]["camera"]
            out[i] = [np.degrees(float(cam[0])), np.degrees(float(cam[1]))]
    return out


def commanded_wasd(actions: list, gen0_gt_index: int, count: int) -> np.ndarray:
    """Commanded WASD per frame, ``(N, 4)`` in ``KEY_NAMES`` order."""
    out = np.zeros((count, 4), dtype=np.float64)
    for i in range(count):
        k = gen0_gt_index + i
        if 0 <= k < len(actions):
            a = actions[k]["action"]
            out[i] = [float(bool(a.get(n, False))) for n in KEY_NAMES]
    return out


def episodes(datasets, model, limit):
    for dataset in datasets:
        for ep in build_index(dataset, model)[:limit or None]:
            yield dataset, ep


def score(pred_yaw: np.ndarray, cmd_yaw: np.ndarray, deadband=1.0) -> dict:
    """Direction accuracy on commanded frames, plus correlation and gain.

    ``deadband`` is in degrees. The commanded rate is 8.594 deg/frame, so 1.0
    separates a real turn from noise by a wide margin.
    """
    moving = np.abs(cmd_yaw) > 1e-9
    if moving.sum() == 0:
        return {}
    hit = (np.abs(pred_yaw) >= deadband) & (np.sign(pred_yaw) == np.sign(cmd_yaw))
    still = ~moving
    return {
        "dir_acc": 100 * float(hit[moving].mean()),
        "still": 100 * float((np.abs(pred_yaw[still]) < deadband).mean()) if still.sum() else float("nan"),
        "gain": float(np.sum(pred_yaw * cmd_yaw) / np.sum(cmd_yaw ** 2)),
        "r": float(np.corrcoef(pred_yaw, cmd_yaw)[0, 1]) if pred_yaw.std() > 0 else float("nan"),
        "n_move": int(moving.sum()),
        "n_still": int(still.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=CAMERA_DATASETS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--limit", type=int, default=0, help="episodes per dataset")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--validate", action="store_true",
                    help="ground-truth views only, to check the IDM reads our render")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    agent = load_agent(args.device)
    sources = ["gt"] if args.validate else ["gt", "gen"]
    models = ["flagship"] if args.validate else args.models

    rows = {}
    for mi, model in enumerate(models):
        # The ground-truth half of a clip is the same whichever model generated
        # the other half, so it is only predicted once.
        want = sources if mi == 0 else [s for s in sources if s != "gt"]
        for dataset, ep in episodes(args.datasets, model, args.limit):
            quads = read_quadrants_rgb(ep.sbs)
            for player in ("alpha", "bravo"):
                acts = ep.actions(player)
                for src in want:
                    frames = quads[f"{player}_{src}"]
                    if len(frames) < 4:
                        continue
                    cam, btn = predict(agent, frames)
                    n = min(len(cam), len(frames))
                    cmd = commanded_deg(acts, ep.frame1 + 1, n)
                    wasd = commanded_wasd(acts, ep.frame1 + 1, n)
                    key = (model if src == "gen" else "GROUND TRUTH", player)
                    rows.setdefault(key, []).append(
                        (cam[:n], cmd[:n], btn[:n], wasd[:n]))
            print(f"  {dataset} {model} ep{ep.episode}", flush=True)

    # Re-derive the column and sign from ground truth every run, and say so, so
    # a drift in either shows up instead of being silently absorbed.
    gt = rows.get(("GROUND TRUTH", "alpha")) or rows.get(("GROUND TRUTH", "bravo"))
    if gt:
        pr = np.concatenate([r[0] for r in gt])
        cm = np.concatenate([r[1] for r in gt])
        with np.errstate(invalid="ignore"):
            cs = [np.corrcoef(pr[:, k], cm[:, 0])[0, 1] for k in (0, 1)]
        best = 0 if abs(np.nan_to_num(cs[0])) > abs(np.nan_to_num(cs[1])) else 1
        print(f"\nground-truth check: commanded yaw vs IDM camera column 0 "
              f"r={cs[0]:+.3f}, column 1 r={cs[1]:+.3f}")
        print(f"  using column {YAW_COL} with sign {YAW_SIGN:+.0f} "
              f"(this run favours column {best}, sign "
              f"{np.sign(np.nan_to_num(cs[best])):+.0f})")

    print("\n" + "=" * 84)
    print("VPT IDM MOUSE ACTION-FOLLOWING (camera yaw), per player")
    print("=" * 84)
    print(f"datasets: {', '.join(args.datasets)}")
    print(f"commanded rate 8.594 deg/frame; IDM camera saturates at "
          f"+/-{CAMERA_MAXVAL_DEG} deg/frame\n")
    print(f"{'source':<24}{'player':<8}{'dir acc%':>10}{'still%':>9}{'gain':>8}"
          f"{'r':>8}{'n move':>9}")
    for (name, player), pairs in sorted(rows.items()):
        pred = YAW_SIGN * np.concatenate([r[0][:, YAW_COL] for r in pairs])
        cmd = np.concatenate([r[1][:, 0] for r in pairs])
        s = score(pred, cmd)
        if not s:
            continue
        print(f"{name:<24}{player:<8}{s['dir_acc']:10.1f}{s['still']:9.1f}"
              f"{s['gain']:8.3f}{s['r']:8.3f}{s['n_move']:9d}")

    if args.out:
        flat = {}
        for (n, p), pairs in rows.items():
            for i, (cam, cmd, btn, wasd) in enumerate(pairs):
                flat[f"cam|{n}|{p}|{i}"] = np.stack([cam, cmd])
                flat[f"key|{n}|{p}|{i}"] = np.stack([btn, wasd])
        np.savez_compressed(args.out, **flat)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
