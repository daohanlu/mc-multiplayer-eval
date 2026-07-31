#!/usr/bin/env python3
"""Export the exact images a VLM was shown for one or more episodes.

Frames are produced by ``run_eval.extract_query_frames`` — the same helper the
evaluation itself uses — so the PNGs are pixel-identical to what was sent to
Gemini, rather than an approximation re-cropped by hand.

Note on the late-episode toggles: ``frame_extraction/`` on disk is whatever the
*last* run left behind, which for several evals is the ``--late-episode``
sweep, not the run behind the paper's numbers. This script leaves
``LATE_EPISODE_QUERY``/``LATE_EPISODE_QUERY_STRICT`` unset by default so it
reproduces the plain (non-late) queries; pass ``--late`` or ``--late-strict``
to reproduce those variants instead.

Usage
-----
::

    # Movement / flagship, episode 2 (both instances)
    python export_vlm_frames.py --dataset translationEval --model flagship \
        --episode 2

    python export_vlm_frames.py --dataset turnToLookEval --model concat_c \
        --episode 5 --late-strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATASET_BASE = REPO / "mc_multiplayer_v2_eval_new_sneak_combined"
GENERATIONS = REPO / "mc_multiplayer_v2_generations"
DEFAULT_OUT = REPO / "vlm_frames_export"
DEFAULT_VIDEO_PAIR_LIMIT = 32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True,
                    help="e.g. translationEval, turnToLookEval")
    ap.add_argument("--model", required=True, help="e.g. flagship, concat_c")
    ap.add_argument("--episode", required=True,
                    help="episode number; 2 and 000002 are both accepted")
    ap.add_argument("--instance", help="restrict to one instance (e.g. 000)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--late", action="store_true",
                    help="reproduce the --late-episode variant")
    ap.add_argument("--late-strict", action="store_true",
                    help="reproduce the --late-episode-strict variant")
    ap.add_argument("--gt", action="store_true",
                    help="also export the ground-truth frames for comparison")
    args = ap.parse_args()

    # Must be set before the handlers compute any frame index.
    os.environ.pop("LATE_EPISODE_QUERY", None)
    os.environ.pop("LATE_EPISODE_QUERY_STRICT", None)
    os.environ.pop("LATE_EPISODE_GEN_LEN", None)
    if args.late:
        os.environ["LATE_EPISODE_QUERY"] = "1"
    if args.late_strict:
        os.environ["LATE_EPISODE_QUERY_STRICT"] = "1"

    from run_eval import (extract_query_frames, find_generated_video_subdir,
                          find_mc_video_pairs, identify_handler,
                          _maybe_set_late_episode_gen_len)

    episode = args.episode.zfill(6)
    folder = DATASET_BASE / args.dataset / "test"
    if not folder.is_dir():
        raise SystemExit(f"dataset not found: {folder}")

    generated_path = GENERATIONS / args.model
    subdir = find_generated_video_subdir(generated_path, args.dataset)
    if subdir is None:
        raise SystemExit(f"no generated videos for {args.model}/{args.dataset}")
    if args.late or args.late_strict:
        _maybe_set_late_episode_gen_len(
            generated_subdir_override=None, generated_path=generated_path,
            dataset_name=args.dataset)

    handler = identify_handler(args.dataset)
    pairs = find_mc_video_pairs(folder)[:DEFAULT_VIDEO_PAIR_LIMIT]

    out = args.out / f"{args.model}_{args.dataset}_ep{episode}"
    out.mkdir(parents=True, exist_ok=True)

    manifest, found = [], 0
    video_id = -1
    last = None

    for pair in pairs:
        for query in handler.extract_keyframes(pair):
            meta = query.metadata
            ei = (meta["episode"], meta["instance"])
            if ei != last:
                video_id += 1          # mirrors run_eval's bookkeeping exactly
                last = ei
            if meta["episode"] != episode:
                continue
            if args.instance and meta["instance"] != args.instance:
                continue

            frames = extract_query_frames(
                query=query, generated_subdir=subdir,
                current_video_id=video_id, frame1_idx=meta["frame1"])
            if args.gt:
                frames.update({
                    f"gt_{k}": v for k, v in extract_query_frames(
                        query=query, generated_subdir=None,
                        current_video_id=video_id,
                        frame1_idx=meta["frame1"]).items()})

            qt = meta.get("query_type", "default")
            for suffix, data in frames.items():
                fn = (f"ep{meta['episode']}_inst{meta['instance']}"
                      f"_{meta['variant']}_{qt}_{suffix}.png")
                (out / fn).write_bytes(data)
                found += 1
            manifest.append({
                "video": f"video_{video_id}_side_by_side.mp4",
                "episode": meta["episode"], "instance": meta["instance"],
                "variant": meta["variant"], "query_type": qt,
                "expected": query.expected_answer,
                "frames": sorted(frames),
            })
            print(f"  ep{meta['episode']} inst{meta['instance']} "
                  f"{meta['variant']:6s} {qt:24s} -> {len(frames)} frame(s) "
                  f"(video_{video_id})")

    if not manifest:
        raise SystemExit(f"no queries found for episode {episode}")

    (out / "manifest.json").write_text(json.dumps({
        "model": args.model, "dataset": args.dataset, "episode": episode,
        "generated_subdir": subdir.name,
        "late_episode": bool(args.late), "late_episode_strict": bool(args.late_strict),
        "queries": manifest,
    }, indent=2) + "\n")

    print(f"\nwrote {found} PNG(s) + manifest.json to {out}")


if __name__ == "__main__":
    main()
