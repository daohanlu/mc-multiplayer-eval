#!/usr/bin/env python3
"""Run the motion estimator over every clip and cache the result.

One ``.npz`` per (dataset, model, episode, player) holds the estimated motion of
the generated view, the estimated motion of the ground-truth view beside it in
the same clip, and the commanded actions lined up with both. Everything the
reports need is derived from these files, so the expensive tracking runs once.

    python3 action_following/extract_motion.py                 # everything
    python3 action_following/extract_motion.py --models flagship
    python3 action_following/extract_motion.py --datasets rotationEval
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# OpenCV and BLAS both thread internally. With one process per core that is
# oversubscription, and on a shared machine it is antisocial. One thread per
# worker, set before cv2 or numpy load.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from af_data import ROOT, build_index, read_all_quadrants  # noqa: E402
from af_flow import (  # noqa: E402
    MOTION_COLUMNS,
    commanded_body_velocity,
    commanded_camera,
    commanded_keys,
    motion_track,
)

CACHE = ROOT / "action_following" / "cache"

MODELS = [
    "flagship",
    "no_player_attn_sf",
    "concat_c",
    "from_scratch",
    "causvid_regression",
    "causvid_dmd",
    "no_kv_cache_backprop",
]

# Camera-driven and movement-driven eval sets. The `_long` variants reuse the
# same ground truth with a longer generated horizon; `oneLooksAwayEval_long` is
# the one the paper's Grounding column uses.
DATASETS = [
    "rotationEval",
    "turnToLookEval",
    "turnToLookOppositeEval",
    "bothLookAwayEval",
    "bothLookAwayEval_long",
    "oneLooksAwayEval",
    "oneLooksAwayEval_long",
    "translationEval",
    "structureEval",
]

KEYS = ["forward", "back", "left", "right", "jump", "sprint", "sneak",
        "attack", "use", "place_block", "mine"]


def out_path(dataset: str, model: str, episode: str, instance: str, player: str) -> Path:
    return CACHE / dataset / model / f"ep{episode}_i{instance}_{player}.npz"


def _one_episode(dataset: str, model: str, ep_fields: dict) -> str:
    import cv2

    from af_data import Episode, read_all_quadrants  # local, keeps the worker light

    cv2.setNumThreads(1)
    ep = Episode(**ep_fields)
    wanted = [p for p in ("alpha", "bravo")
              if not out_path(dataset, model, ep.episode, ep.instance, p).exists()]
    if not wanted:
        return f"{dataset}/{model}/ep{ep.episode}_i{ep.instance}: cached"
    quads = read_all_quadrants(ep.sbs)

    made = 0
    for player in wanted:
        dest = out_path(dataset, model, ep.episode, ep.instance, player)
        gen = quads[f"{player}_gen"]
        gt = quads[f"{player}_gt"]
        if gen.shape[0] < 4 or gt.shape[0] < 4:
            continue
        m_gen = motion_track(gen)
        m_gt = motion_track(gt)
        n = min(len(m_gen), len(m_gt))
        actions = ep.actions(player)
        start = ep.frame1 + 1
        cam = commanded_camera(actions, start, n)
        keys = commanded_keys(actions, start, n, KEYS)
        vel = commanded_body_velocity(actions, start, n)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            gen=m_gen[:n].astype(np.float32),
            gt=m_gt[:n].astype(np.float32),
            cam=cam[:n].astype(np.float32),
            vel=vel[:n].astype(np.float32),
            keys=np.stack([keys[k][:n] for k in KEYS]).astype(np.uint8),
            key_names=np.array(KEYS),
            columns=np.array(MOTION_COLUMNS),
            frame1=ep.frame1,
            video_id=ep.video_id,
        )
        made += 1
    return f"{dataset}/{model}/ep{ep.episode}_i{ep.instance}: {made} written"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="episodes per cell, 0 = all")
    args = ap.parse_args()

    jobs = []
    for dataset in args.datasets:
        for model in args.models:
            episodes = build_index(dataset, model)
            if args.limit:
                episodes = episodes[: args.limit]
            if not episodes:
                print(f"  (no clips) {dataset} / {model}")
                continue
            for ep in episodes:
                fields = dict(ep.__dict__)
                jobs.append((dataset, model, fields))
    print(f"{len(jobs)} episode jobs")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one_episode, d, m, f): (d, m) for d, m, f in jobs}
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception:
                print(f"FAILED {futures[fut]}")
                traceback.print_exc()
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
