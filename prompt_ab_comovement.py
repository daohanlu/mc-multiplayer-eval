#!/usr/bin/env python3
"""A/B different prompts for the co-movement eval on ground truth.

Frames are extracted once and reused across every variant, and the VLM calls
run concurrently, so comparing N prompts costs about as much wall-clock as one
sequential trial.

The metric that matters is not overall accuracy — half the queries are
"no motion", so a prompt that biases toward it can gain several points while
getting worse. Per-class recall is reported for exactly that reason, along with
the balanced mean over the five classes.

Runs on ground truth by default, or on a generation with --generated-subdir.
--exclude-no-motion drops the half of the queries whose answer is "no motion",
which is what you want when comparing prompts on the directional classes alone.

    python3 prompt_ab_comovement.py --datasets coMovementEval coMovementWithDividerEval
    python3 prompt_ab_comovement.py --datasets coMovementEval \
        --generated-subdir generations_comovement/solaris/co_movement \
        --exclude-no-motion --trials 3 --variants translation_exact baseline ignore_landmarks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import extract_query_frames, find_mc_video_pairs, identify_handler  # noqa: E402
from vlm_utils import query_vlm  # noqa: E402

DATASET_BASE = Path("mc_multiplayer_v2_eval_new_sneak_combined_simplified_naming")
CLASSES = ["closer", "farther", "left", "right", "no motion"]

ANSWER_LINE = ('Answer with a single word from "closer", "farther", "left", '
               '"right", or "no motion".')

PROMPTS: dict[str, str] = {
    # translationEval's prompt, byte-for-byte — the wording behind the paper's
    # Movement column. Included so a co-movement number can be compared against
    # the Movement number without the prompt being a confound.
    "translation_exact": (
        "Here are Minecraft screenshots showing another player on the screen. "
        "Between the first frame and the second frame, did the player being shown "
        "move closer, farther, to the left, or to the right on-screen? "
        + ANSWER_LINE
    ),

    # What shipped with the handler. Asks whether the player "moved", which
    # invites a world-frame reading — and with a landmark in shot the model
    # answers relative to the landmark.
    "baseline": (
        "Here are Minecraft screenshots showing another player on the screen. "
        "Both players may be moving at the same time. "
        "Between the first frame and the second frame, did the player being shown "
        "move closer, farther, to the left, or to the right on-screen? "
        'If the other player stayed in the same place on-screen, answer "no motion". '
        + ANSWER_LINE
    ),

    # Reframes the question as comparing the player's position and size *within
    # the picture*, and warns that the camera itself may move. Says nothing
    # about the action structure, so it does not leak the answer distribution.
    "screen_relative": (
        "These are two screenshots from one player's camera, taken at two different "
        "moments. Another player is visible in both. "
        "The camera itself may have moved between the two screenshots, so the "
        "scenery may shift. Judge only the other player, not the scenery. "
        "Compare where the other player appears within the picture, and how large "
        "they appear, in the first screenshot versus the second. "
        "Did the other player appear to move closer, farther, to the left, or to "
        "the right within the picture? "
        "If the other player appears in the same place and at the same size in "
        'both screenshots, answer "no motion". '
        + ANSWER_LINE
    ),

    # As above, plus an explicit instruction to ignore fixed landmarks — aimed
    # at the divider, where a static block is the thing being tracked instead.
    "ignore_landmarks": (
        "These are two screenshots from one player's camera, taken at two different "
        "moments. Another player is visible in both. "
        "The camera itself may have moved between the two screenshots, so the "
        "ground, the sky and any blocks or structures may shift between the two "
        "images. Ignore all of that. Do not judge the other player's position "
        "relative to any block, structure or landmark. "
        "Judge only how the other player appears within the picture itself: "
        "compare their position in the frame and how large they appear, in the "
        "first screenshot versus the second. "
        "If the other player appears at the same place in the frame and at the "
        'same size in both screenshots, answer "no motion", even if the scenery '
        "around them has moved. "
        + ANSWER_LINE
    ),
}


def build_items(dataset: str, limit: int,
                generated_subdir: Path | None = None,
                exclude_no_motion: bool = False) -> list[dict]:
    """Extract every query's frames once.

    ``video_id`` counts pairs in discovery order, exactly as run_eval does, so
    generated clips line up with the ground-truth pair they were rendered from.
    """
    handler = identify_handler(dataset)
    pairs = find_mc_video_pairs(DATASET_BASE / dataset / "test")[:limit]
    items = []
    for video_id, pair in enumerate(pairs):
        for q in handler.extract_keyframes(pair):
            if exclude_no_motion and q.expected_answer == "no motion":
                continue
            frames = extract_query_frames(query=q,
                                          generated_subdir=generated_subdir,
                                          current_video_id=video_id,
                                          frame1_idx=q.metadata["frame1"])
            items.append({
                "expected": q.expected_answer,
                "frame1": frames["frame1"], "frame2": frames["frame2"],
                "combo": (q.metadata["alpha_direction"], q.metadata["bravo_direction"]),
                "episode": q.metadata["episode"], "instance": q.metadata["instance"],
                "variant": q.metadata["variant"],
            })
    return items


def run_variant(name: str, prompt: str, items: list[dict], workers: int) -> dict:
    def ask(item):
        try:
            resp = query_vlm(prompt, item["frame1"], item["frame2"],
                             enable_thinking=False)
            return resp.strip().lower()
        except Exception as exc:  # quota / transient
            return f"__error__ {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(ask, items))

    hits, tot = Counter(), Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    combo_hits, combo_tot = Counter(), Counter()
    errors = 0
    for item, got in zip(items, answers):
        if got.startswith("__error__"):
            errors += 1
            continue
        exp = item["expected"]
        tot[exp] += 1
        combo_tot[item["combo"]] += 1
        if got == exp:
            hits[exp] += 1
            combo_hits[item["combo"]] += 1
        confusion[exp][got] += 1

    n = sum(tot.values())
    recalls = {c: (100.0 * hits[c] / tot[c]) for c in CLASSES if tot[c]}
    return {
        "name": name, "n": n, "errors": errors,
        "overall": 100.0 * sum(hits.values()) / n if n else 0.0,
        "balanced": sum(recalls.values()) / len(recalls) if recalls else 0.0,
        "recalls": recalls, "confusion": confusion,
        "combo": {k: (combo_hits[k], combo_tot[k]) for k in sorted(combo_tot)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+",
                    default=["coMovementEval", "coMovementWithDividerEval"])
    ap.add_argument("--limit", type=int, default=32, help="video pairs")
    ap.add_argument("--variants", nargs="+", default=list(PROMPTS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--generated-subdir", type=Path,
                    help="score a generation instead of ground truth")
    ap.add_argument("--exclude-no-motion", action="store_true",
                    help="drop the queries whose answer is 'no motion'")
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat each variant N times and report mean +/- sd")
    ap.add_argument("--out", type=Path, default=Path("logs/prompt_ab.json"))
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set")
    if args.generated_subdir and not args.generated_subdir.exists():
        raise SystemExit(f"no such generated subdir: {args.generated_subdir}")

    source = str(args.generated_subdir) if args.generated_subdir else "ground truth"
    everything = {}
    for dataset in args.datasets:
        print(f"\nextracting frames for {dataset} ({source}) ...", flush=True)
        items = build_items(dataset, args.limit,
                            generated_subdir=args.generated_subdir,
                            exclude_no_motion=args.exclude_no_motion)
        print(f"  {len(items)} queries "
              f"({sum(i['expected'] == 'no motion' for i in items)} are 'no motion')")

        results = []
        for name in args.variants:
            trials = [run_variant(name, PROMPTS[name], items, args.workers)
                      for _ in range(args.trials)]
            res = trials[0]
            overalls = [t["overall"] for t in trials]
            mean = sum(overalls) / len(overalls)
            sd = (sum((o - mean) ** 2 for o in overalls) / len(overalls)) ** 0.5
            res["overall_mean"], res["overall_sd"] = mean, sd
            res["per_trial"] = overalls
            results.append(res)
            print(f"\n  --- {dataset} / {name}")
            print(f"      overall {mean:5.1f}% +/- {sd:.1f}   "
                  f"balanced {res['balanced']:5.1f}%   n={res['n']}"
                  + (f"   errors {res['errors']}" if res["errors"] else ""))
            if args.trials > 1:
                print(f"      per-trial {['%.1f' % o for o in overalls]}")
            for c, r in res["recalls"].items():
                top = ", ".join(f"{k}={v}" for k, v in
                                res["confusion"][c].most_common(3))
                print(f"      {c:11s} {r:6.1f}%   {top}   (trial 1)")
        everything[f"{dataset} [{source}]"] = [
            {k: v for k, v in r.items() if k != "confusion"} | {
                "confusion": {c: dict(cc) for c, cc in r["confusion"].items()},
                "combo": {f"{a}+{b}": v for (a, b), v in r["combo"].items()},
            } for r in results
        ]

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(everything, indent=2, default=str) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
