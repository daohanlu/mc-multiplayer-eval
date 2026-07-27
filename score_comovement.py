#!/usr/bin/env python3
"""Break down co-movement results by expected class and action combination.

Half the queries are "no motion" by construction, so a bare accuracy number is
easy to misread — always answering "no motion" scores 50%. This reports
per-class recall, what the model says instead, and the always-"no motion"
baseline alongside the real figure.

    python3 score_comovement.py [--results-dir results_json_comovement]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["closer", "farther", "left", "right", "no motion"]

# Datasets whose numbers should not be reported. The divider variant has an
# occlusion confound — see COMOVEMENT_EVAL.md — so its "no motion" class is
# measuring how much of the player the block hides, not the model's grasp of
# relative motion. Scored only when explicitly asked for.
UNRELIABLE = {
    "coMovementWithDividerEval":
        "occlusion confound: the static divider hides progressively more of "
        "the other player as the observer closes on it, so unchanged on-screen "
        "size still reads as 'farther'. Do not report these numbers.",
}


def load_trials(eval_dir: Path) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(eval_dir.glob("trial_*.json"),
                            key=lambda p: int(p.stem.split("_")[1]))]


def report(eval_dir: Path, include_unreliable: bool = False) -> None:
    trials = load_trials(eval_dir)
    if not trials:
        print(f"  no trials in {eval_dir}")
        return

    if eval_dir.name in UNRELIABLE and not include_unreliable:
        print(f"\n=== {eval_dir.name}  [EXCLUDED] ===")
        print(f"  {UNRELIABLE[eval_dir.name]}")
        print("  Pass --include-unreliable to score it anyway.")
        return

    print(f"\n=== {eval_dir.name} ({len(trials)} trial(s), "
          f"vlm={trials[0].get('vlm_model_name')}, "
          f"thinking={trials[0].get('thinking_enabled')}) ===")

    accs = [t["accuracy"] for t in trials]
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    print(f"  overall accuracy: {mean:.1f}% +/- {var ** 0.5:.1f}   "
          f"per-trial {['%.1f' % a for a in accs]}")

    per_class_hits: dict[str, list[int]] = defaultdict(list)
    per_class_tot: dict[str, list[int]] = defaultdict(list)
    confusion: dict[str, Counter] = defaultdict(Counter)
    per_combo_hits: dict[tuple, int] = defaultdict(int)
    per_combo_tot: dict[tuple, int] = defaultdict(int)
    baseline = []

    for t in trials:
        hits, tot = Counter(), Counter()
        no_motion = 0
        for r in t["results"]:
            exp = r["expected"]
            tot[exp] += 1
            if r["correct"]:
                hits[exp] += 1
            confusion[exp][r["response"].strip().lower()] += 1
            m = r["metadata"]
            combo = (m.get("alpha_direction"), m.get("bravo_direction"))
            per_combo_tot[combo] += 1
            per_combo_hits[combo] += bool(r["correct"])
            no_motion += (exp == "no motion")
        for c in CLASSES:
            if tot[c]:
                per_class_hits[c].append(hits[c])
                per_class_tot[c].append(tot[c])
        baseline.append(100.0 * no_motion / max(1, len(t["results"])))

    print(f"  always-\"no motion\" baseline: {sum(baseline)/len(baseline):.1f}%")
    print(f"\n  {'expected':12s} {'recall':>18s}   most common answers")
    for c in CLASSES:
        if c not in per_class_tot:
            continue
        h, n = sum(per_class_hits[c]), sum(per_class_tot[c])
        top = ", ".join(f"{k}={v}" for k, v in confusion[c].most_common(3))
        print(f"  {c:12s} {h:4d}/{n:4d} = {100.0*h/n:6.1f}%   {top}")

    print(f"\n  {'alpha + bravo':26s} {'accuracy':>14s}")
    for combo in sorted(per_combo_tot):
        h, n = per_combo_hits[combo], per_combo_tot[combo]
        label = f"{combo[0]} + {combo[1]}"
        print(f"  {label:26s} {h:4d}/{n:4d} = {100.0*h/n:6.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path,
                    default=Path("results_json_comovement"))
    ap.add_argument("--include-unreliable", action="store_true",
                    help="also score datasets flagged in UNRELIABLE")
    args = ap.parse_args()

    roots = sorted((args.results_dir / "real").glob("coMovement*"))
    roots += sorted((args.results_dir / "generated").glob("*coMovement*"))
    if not roots:
        raise SystemExit(f"no co-movement results under {args.results_dir}")
    for r in roots:
        report(r, include_unreliable=args.include_unreliable)
    print()


if __name__ == "__main__":
    main()
