#!/usr/bin/env python3
"""VLM judge for the 2-player view-memory evals (view_mem_2p_v3).

Both players look at each other (anchor view), turn 120 degrees away, pause P frames, then
  turnBackMemoryEval   turn back to the anchor  -> the view should match the anchor (positive)
  turnAwayControlEval  keep turning 120 degrees -> the view never matches the anchor (control)

For every episode and player we ask the judge whether two frames show the same view:
  frame A: the anchor, i.e. the first frame of the generated segment (the conditioning
           frame), taken from the ground-truth half of the side-by-side video;
  frame B: a frame `--offsets` frames after the player's last turn ends, taken from the
           generated half (--source generated) or the ground-truth half (--source gt).
Both frames come from the same side-by-side mp4, so they share resolution and compression.
The yes-rate on the main suite measures view memory; on the control it measures how often
the judge (or the model) says "same" when the view cannot match.

Inputs: side-by-side videos produced by solaris-dev src/inference.py with
config/inference_view_mem.yaml (video_<i> is the i-th entry of the eval ids), the flat
eval datasets (for the action JSONs) and solaris-dev's eval-id files.

Usage:
  python vlm_eval/view_memory_eval.py --gen-root generations_view_mem --models flagship \
      --source generated --out results_view_memory
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vlm_utils import VLM_MODEL_NAME, query_vlm  # noqa: E402

SUITES = {"turnBackMemoryEval": "yes", "turnAwayControlEval": "no"}
PAUSES = [12, 24, 36, 48, 64]
PROMPTS = {
    # Stricter than the turn-to-look prompt: the two views should be identical here.
    "exact_same_view": (
        "You will be shown two Minecraft screenshots. Do they show exactly the same view: "
        "the same terrain, blocks, trees and other objects, and the same players, in the same "
        "places on screen? Ignore small differences in image quality, blur or lighting. "
        'Answer with a single word: "yes" or "no".'
    ),
    # Same place and viewpoint, ignoring what legitimately changes in ground truth while the
    # camera is back in the same pose: the player's own hand, mobs that walk, distant detail.
    "same_place_ignore_cosmetic": (
        "You will be shown two Minecraft screenshots. Do they show the same place, seen from the "
        "same position and viewing direction? Ignore the player's own hand and held item, animals "
        "or mobs that may have moved, and small differences in distant terrain, lighting or image "
        'quality. Answer with a single word: "yes" or "no".'
    ),
    # As above, but a generation may end its turn slightly off the commanded angle and still
    # render the right content, so allow up to 30 degrees of difference in viewing direction.
    "same_place_allow_30deg_turn": (
        "You will be shown two Minecraft screenshots. Do they show the same place, seen from the "
        "same position and roughly the same viewing direction? The viewing direction may differ by "
        "up to 30 degrees. Ignore the player's own hand and held item, animals or mobs that may "
        "have moved, and small differences in distant terrain, lighting or image quality. "
        'Answer with a single word: "yes" or "no".'
    ),
    # The paper's turn-to-look (co-observation) prompt, for comparison.
    "same_scenery_turn_to_look": (
        "You will be shown two Minecraft screenshots. "
        "Do these two screenshots show the same scenery? Be careful and answer based on the content "
        'of the screenshots, not just the camera angles.Answer with a single word: "yes", "no".'
    ),
}


def last_turn_end(actions):
    """Index of the first frame after the player's last camera movement."""
    yaw = np.degrees(np.unwrap([a["yaw"] for a in actions]))
    moving = np.nonzero(np.abs(np.diff(yaw)) > 0.5)[0]
    return int(moving.max()) + 1


def read_half(video, index, half):
    """Return (player-1, player-2) PNG bytes of one half ('gt' left, 'gen' right) of a frame."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {index} of {video}")
    h, w = frame.shape[:2]
    cols = slice(0, w // 2) if half == "gt" else slice(w // 2, w)
    return [cv2.imencode(".png", frame[r, cols])[1].tobytes() for r in (slice(0, h // 2), slice(h // 2, h))]


def build_queries(args, model):
    queries = []
    for suite, expected in SUITES.items():
        for p in PAUSES:
            name = f"{suite}_pause{p:03d}"
            gen_dir = Path(args.gen_root) / model / name
            ids = json.load(open(Path(args.eval_ids) / f"eval_ids_{name}.json"))
            alphas = sorted(glob.glob(f"{args.data_root}/{name}/test/*_Alpha_*.mp4"))
            for vi, (ep, s1, _, s2, _) in enumerate(ids):
                video = gen_dir / f"video_{vi}_side_by_side.mp4"
                if not video.exists():
                    continue
                for player, (bot, start) in enumerate((("Alpha", s1), ("Bravo", s2))):
                    stem = Path(alphas[ep]).name.replace("_Alpha_", f"_{bot}_").replace("_camera.mp4", "")
                    actions = json.load(open(f"{args.data_root}/{name}/test/{stem}.json"))
                    turn_end = last_turn_end(actions) - start
                    for off in args.offsets:
                        queries.append(dict(model=model, suite=suite, pause=p, dataset=name, video=str(video),
                                            video_index=vi, episode=stem.rsplit(f"_{bot}_", 1)[0], player=bot,
                                            player_row=player, frame_a=0, frame_b=turn_end + off, offset=off,
                                            expected=expected))
    return queries


SWEEP_ANGLES = [0, 10, 20, 30, 40, 50, 60, 90, 120]


def build_angle_sweep_queries(args, model, pause=24):
    """Ground-truth calibration of the judge against camera angle: pair each player's anchor
    with GT frames from the turn back at known yaw offsets (8.6 deg per frame) from the anchor."""
    queries = []
    name = f"turnBackMemoryEval_pause{pause:03d}"
    gen_dir = Path(args.gen_root) / model / name
    ids = json.load(open(Path(args.eval_ids) / f"eval_ids_{name}.json"))
    alphas = sorted(glob.glob(f"{args.data_root}/{name}/test/*_Alpha_*.mp4"))
    for vi, (ep, s1, _, s2, _) in enumerate(ids):
        video = gen_dir / f"video_{vi}_side_by_side.mp4"
        for player, (bot, start) in enumerate((("Alpha", s1), ("Bravo", s2))):
            stem = Path(alphas[ep]).name.replace("_Alpha_", f"_{bot}_").replace("_camera.mp4", "")
            actions = json.load(open(f"{args.data_root}/{name}/test/{stem}.json"))
            yaw = np.degrees(np.unwrap([a["yaw"] for a in actions]))
            rel = np.abs(yaw - yaw[start])
            apex_end = int(np.nonzero(rel >= rel.max() - 1)[0].max())  # last frame of the apex hold
            back = np.arange(apex_end, last_turn_end(actions) + 20)       # the turn back, then settled
            for target in SWEEP_ANGLES:
                f = int(back[np.argmin(np.abs(rel[back] - target))])
                queries.append(dict(model=model, suite="angle_sweep", pause=pause, dataset=name, video=str(video),
                                    video_index=vi, episode=stem.rsplit(f"_{bot}_", 1)[0], player=bot,
                                    player_row=player, frame_a=0, frame_b=f - start, offset=target,
                                    angle=float(rel[f]), expected="yes" if target <= 30 else "no"))
    return queries


def run_query(q, args):
    a = read_half(q["video"], q["frame_a"], "gt")[q["player_row"]]
    b = read_half(q["video"], q["frame_b"], "gt" if args.source == "gt" else "gen")[q["player_row"]]
    try:
        ans = query_vlm(PROMPTS[args.prompt], a, b, enable_thinking=not args.no_thinking, max_retries=6)
    except Exception as e:  # recorded, counted as an error, not as an answer
        return {**q, "answer": None, "error": str(e)[:300]}
    word = ans.strip().strip(".\"' ").split()[0] if ans.strip() else ""
    return {**q, "answer": ans, "said_same": word == "yes", "parsed": word in ("yes", "no")}


def summarize(results):
    """Accuracy per suite (main: said "yes"; control: said "no") and their mean, as the paper
    reports for the turn-to-look (co-observation) Consistency eval. Errors and unparseable
    answers count as incorrect."""
    acc = {}
    for r in results:
        correct = bool(r.get("parsed")) and (r["said_same"] == (r["expected"] == "yes"))
        acc.setdefault((r["model"], r["pause"], r["offset"], r["suite"]), []).append(correct)
    out = []
    for m, p, o in sorted({k[:3] for k in acc}):
        main = acc.get((m, p, o, "turnBackMemoryEval"), [])
        ctrl = acc.get((m, p, o, "turnAwayControlEval"), [])
        row = dict(model=m, pause=p, offset=o,
                   main_acc=float(np.mean(main)) if main else None, main_n=len(main),
                   control_acc=float(np.mean(ctrl)) if ctrl else None, control_n=len(ctrl))
        row["mean_acc"] = (row["main_acc"] + row["control_acc"]) / 2 if main and ctrl else None
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True, help="<gen-root>/<model>/<dataset>/video_<i>_side_by_side.mp4")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-root", default="/mnt/data-hdd/dl3957/view_mem_2p_v3_flat")
    ap.add_argument("--eval-ids", default=os.path.expanduser("~/GitHub/solaris-dev/src/data/eval_ids"))
    ap.add_argument("--source", choices=["generated", "gt"], default="generated")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="same_place_ignore_cosmetic")
    ap.add_argument("--offsets", type=int, nargs="+", default=[10, 90],
                    help="frames after the last turn ends at which frame B is taken")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="queries per model (for smoke tests)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results_view_memory")
    ap.add_argument("--angle-sweep", action="store_true",
                    help="GT calibration: anchor vs GT frames at known yaw offsets (needs --source gt)")
    args = ap.parse_args()

    if args.angle_sweep:
        assert args.source == "gt", "--angle-sweep compares ground-truth frames"
        for model in args.models:
            qs = build_angle_sweep_queries(args, model)[: args.limit]
            print(f"{model}: {len(qs)} angle-sweep queries ({args.prompt}, judge {VLM_MODEL_NAME})", flush=True)
            with cf.ThreadPoolExecutor(args.workers) as ex:
                results = list(ex.map(lambda q: run_query(q, args), qs))
            out = Path(args.out) / f"angle_sweep_{args.prompt}"
            out.mkdir(parents=True, exist_ok=True)
            json.dump(dict(model=model, judge=VLM_MODEL_NAME, prompt=PROMPTS[args.prompt], results=results),
                      open(out / f"{model}.json", "w"), indent=1)
            print(f"  errors {sum(r.get('answer') is None for r in results)}; wrote {out / (model + '.json')}")
            print(f"  {'target':>6} {'actual':>12} | said same | n")
            for t in SWEEP_ANGLES:
                rs = [r for r in results if r["offset"] == t and r.get("parsed")]
                ang = [r["angle"] for r in rs]
                print(f"  {t:5d}° {min(ang):5.1f}-{max(ang):5.1f}° | {100 * np.mean([r['said_same'] for r in rs]):8.1f}% | {len(rs)}")
        return

    tag = f"{args.source}_{args.prompt}{'_nothink' if args.no_thinking else ''}"
    for model in args.models:
        qs = build_queries(args, model)[: args.limit]
        print(f"{model}: {len(qs)} queries ({tag}, judge {VLM_MODEL_NAME})", flush=True)
        with cf.ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(lambda q: run_query(q, args), qs))
        errors = sum(r.get("answer") is None for r in results)
        unparsed = sum(r.get("answer") is not None and not r.get("parsed") for r in results)
        out = Path(args.out) / tag
        out.mkdir(parents=True, exist_ok=True)
        summary = summarize(results)
        json.dump(dict(model=model, judge=VLM_MODEL_NAME, prompt=PROMPTS[args.prompt], source=args.source,
                       thinking=not args.no_thinking, offsets=args.offsets, errors=errors, unparsed=unparsed,
                       summary=summary, results=results),
                  open(out / f"{model}.json", "w"), indent=1)
        print(f"  errors {errors}, unparsed {unparsed}; wrote {out / (model + '.json')}")
        fmt = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"
        print(f"  {'pause':>5} {'frame B':>8} | {'turn-back':>9} {'turn-away':>9} {'mean':>6} | n")
        for s in summary:
            print(f"  {s['pause']:5d} {'+' + str(s['offset']):>8} | {fmt(s['main_acc']):>9} {fmt(s['control_acc']):>9} "
                  f"{fmt(s['mean_acc']):>6} | {s['main_n']}/{s['control_n']}")


if __name__ == "__main__":
    main()
