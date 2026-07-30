#!/usr/bin/env python3
"""The per-sample agreement numbers reviewer nAFu asked for, and their context.

``pool_human_eval.py`` already prints each annotator against the VLM judge.
Three more numbers are needed before that table can be read, and all three are
about what a *high* score could even have been:

* the judge against itself, across its three trials, which caps any cross-judge
  agreement;
* each annotator against the other four, which is the like-for-like comparison
  with "judge against the panel";
* the mean pairwise human-human kappa, not only its range.

    python3 human-eval/agreement_extras.py
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pool_human_eval import Study, cohen_kappa  # noqa: E402

RESULTS = HERE.parent / "results_json_late_episode_strict"
MODELS = ["flagship", "no_player_attn_sf", "concat_c"]
LABEL = {"flagship": "Solaris", "no_player_attn_sf": "Independent",
         "concat_c": "Frame Concat"}


def match(a, b, scope):
    shared = [i for i in scope if i in a and i in b]
    if not shared:
        return float("nan"), float("nan"), 0
    xa = [a[i] for i in shared]
    xb = [b[i] for i in shared]
    pct = 100.0 * sum(x == y for x, y in zip(xa, xb)) / len(shared)
    return pct, cohen_kappa(xa, xb), len(shared)


def majority_excluding(panel, who, scope):
    out = {}
    for iid in scope:
        votes = [a[iid] for name, a in panel.items() if name != who and iid in a]
        if votes:
            c = Counter(votes)
            out[iid] = "yes" if c["yes"] > c["no"] else "no"
    return out


def main() -> None:
    st = Study(HERE.parent, RESULTS)
    scope = sorted(set(itertools.chain.from_iterable(
        st.item_ids(m) for m in MODELS)))
    annotators = st.annotators()
    trials = st.vlm_trials()
    maj_v = st.majority(st.vlm)

    print(f"items in scope: {len(scope)}  "
          f"({len(MODELS)} models x 128 pairs)")
    print(f"annotators: {', '.join(annotators)}")
    print(f"VLM trials: {', '.join(trials)}\n")

    print("## The judge against itself (repeatability)\n")
    print(f"  {'pair':<20}{'agree':>9}{'kappa':>9}{'n':>7}")
    vals, ks = [], []
    for a, b in itertools.combinations(trials, 2):
        pct, k, n = match(st.vlm[a], st.vlm[b], scope)
        vals.append(pct)
        ks.append(k)
        print(f"  {a + ' vs ' + b:<20}{pct:8.1f}%{k:+9.2f}{n:7d}")
    print(f"  {'mean':<20}{sum(vals) / len(vals):8.1f}%"
          f"{sum(ks) / len(ks):+9.2f}")
    print("\n  The judge is sampled, not deterministic. Two runs of the same")
    print("  judge on the same image agree at this rate, so no human-judge")
    print("  agreement can be expected above it.\n")

    print("## Human against human\n")
    print(f"  {'pair':<24}{'agree':>9}{'kappa':>9}")
    vals, ks = [], []
    for a, b in itertools.combinations(annotators, 2):
        pct, k, _ = match(st.human[a], st.human[b], scope)
        vals.append(pct)
        ks.append(k)
        print(f"  {a + ' vs ' + b:<24}{pct:8.1f}%{k:+9.2f}")
    hh_pct, hh_k = sum(vals) / len(vals), sum(ks) / len(ks)
    print(f"  {'mean pairwise':<24}{hh_pct:8.1f}%{hh_k:+9.2f}")
    print(f"  {'range':<24}{min(vals):8.1f}-{max(vals):.1f}%"
          f"   kappa {min(ks):+.2f} to {max(ks):+.2f}\n")

    print("## Each judge against a panel that does not contain it\n")
    print("  Like for like: every row is one judge scored against the same kind")
    print("  of target, the majority of the human annotators who are not that")
    print("  judge. A human row uses the other four; a VLM row uses all five,")
    print("  since the VLM is never in the panel.\n")
    print(f"  {'judge':<22}{'agree':>9}{'kappa':>9}{'n':>7}")
    for who in annotators:
        target = majority_excluding(st.human, who, scope)
        pct, k, n = match(st.human[who], target, scope)
        print(f"  {who:<22}{pct:8.1f}%{k:+9.2f}{n:7d}")
    human_panel = st.majority(st.human)
    for t in trials:
        pct, k, n = match(st.vlm[t], human_panel, scope)
        print(f"  {t:<22}{pct:8.1f}%{k:+9.2f}{n:7d}")
    pct, k, n = match(maj_v, human_panel, scope)
    print(f"  {'VLM majority of 3':<22}{pct:8.1f}%{k:+9.2f}{n:7d}")

    print("\n## Per model, for the record\n")
    print("  Kappa per model is not evidence about the judge. On Frame Concat")
    print("  both sides answer 'different' to nearly everything, so chance")
    print("  agreement is high and kappa correctly discounts the raw rate.\n")
    print(f"  {'model':<16}{'agree':>9}{'kappa':>9}{'n':>7}")
    for m in MODELS:
        sub = st.item_ids(m)
        pct, k, n = match(human_panel, maj_v, sub)
        print(f"  {LABEL[m]:<16}{pct:8.1f}%{k:+9.2f}{n:7d}")


if __name__ == "__main__":
    main()
