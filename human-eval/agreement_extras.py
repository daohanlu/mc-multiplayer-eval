#!/usr/bin/env python3
"""The per-sample agreement numbers reviewer nAFu asked for, and their context.

Two terms only, throughout: the **VLM judge** and the **human annotators**.
Nothing here calls a human annotator a judge.

The headline table compares the VLM judge with each human annotator, one at a
time, and averages; a human annotator row is that annotator against the other
four, built the same way. Two supporting numbers say what a high score could even
have been: the VLM judge against itself across its three trials, which caps any
VLM-to-human agreement, and the mean pairwise agreement between two human
annotators.

An earlier version scored everything against the *majority answer* of the human
annotators who were not that rater. That is kept below and marked not to be used:
a human annotator's panel of four is even, so 2-2 ties fall on 54 to 90 of the
384 items and are resolved by convention, while the VLM judge's panel of five
never ties. The convention moves a human row by up to 14 points and the VLM row
not at all.

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


def _pop(v):
    """Mean and population sd, matching the paper's convention across trials."""
    m = sum(v) / len(v)
    return m, (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def _votes(st, names, iid):
    return Counter(st.human[n][iid] for n in names if iid in st.human[n])


def _tied(st, names, iid) -> bool:
    c = _votes(st, names, iid)
    return c["yes"] == c["no"] and (c["yes"] or c["no"])


def _panel(st, names, scope, tie: str):
    """Per-item majority answer of ``names``, with ties resolved to ``tie``."""
    out = {}
    for iid in scope:
        c = _votes(st, names, iid)
        if not (c["yes"] or c["no"]):
            continue
        out[iid] = tie if c["yes"] == c["no"] else ("yes" if c["yes"] > c["no"] else "no")
    return out


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

    print("## VLM judge and human annotators, pairwise  <- the headline\n")
    print("  A human annotator row is that human annotator compared with each")
    print("  of the other four, one at a time, then averaged. The VLM judge row")
    print("  is the VLM judge compared with each of the five the same way. No")
    print("  consensus label is built, so no vote and no tie-breaking convention")
    print("  can move a row. This is also the literal form of the reviewer's")
    print("  question: VLM judge against human annotator, sample by sample.\n")
    print(f"  {'compared with the human annotators':<36}"
          f"{'agree':>9}{'kappa':>9}{'pairs':>7}")
    rows = []
    for who in annotators:
        ms = [match(st.human[who], st.human[o], scope) for o in annotators if o != who]
        rows.append((who, sum(m[0] for m in ms) / len(ms),
                     sum(m[1] for m in ms) / len(ms), len(ms)))
    for who, pct, k, n in sorted(rows, key=lambda r: -r[1]):
        print(f"  {'human annotator ' + who:<36}{pct:8.1f}%{k:+9.2f}{n:7d}")
    # The paper scores each VLM trial separately and averages, reporting the
    # population sd across trials. It never merges the trials into one answer,
    # so neither does this. The majority-of-3 construct is printed afterwards
    # only to show what quoting it would have bought.
    per_trial = []
    for t in trials:
        ms = [match(st.vlm[t], st.human[o], scope) for o in annotators]
        per_trial.append((sum(m[0] for m in ms) / len(ms),
                          sum(m[1] for m in ms) / len(ms)))
        print(f"  {'VLM judge, ' + t:<36}{per_trial[-1][0]:8.1f}%"
              f"{per_trial[-1][1]:+9.2f}{len(annotators):7d}")
    ma, sa = _pop([p[0] for p in per_trial])
    mk, sk = _pop([p[1] for p in per_trial])
    print(f"  {'VLM judge, 3 trials  <- quote this':<36}{ma:8.1f}%{mk:+9.2f}"
          f"{len(annotators):7d}   +/- {sa:.1f} and {sk:.2f} over trials")
    ms = [match(maj_v, st.human[o], scope) for o in annotators]
    mv_pct = sum(m[0] for m in ms) / len(ms)
    print(f"\n  For contrast only, majority of the 3 trials: {mv_pct:.1f}%, "
          f"kappa {sum(m[1] for m in ms) / len(ms):+.2f}.")
    print(f"  That is {mv_pct - ma:+.1f} points, and merging trials this way is")
    print("  not what the paper does, so it is not quoted anywhere.")

    print("\n## The same thing against a majority-vote panel, and why it is not used\n")
    print("  A human row here is scored against the majority answer of the other")
    print("  four annotators, which is an EVEN panel, so 2-2 ties happen on 54 to")
    print("  90 of the 384 items and are broken by convention. The VLM row uses")
    print("  all five, an odd panel, where no tie can occur. Switching the tie")
    print("  rule moves a human row by up to 14 points while leaving the VLM row")
    print("  untouched, so these numbers are not comparable across rows and the")
    print("  pairwise table above is the one to quote.\n")
    print(f"  {'judge':<22}{'tie->no':>9}{'tie->yes':>10}{'ties':>7}")
    for who in annotators:
        others = [a for a in annotators if a != who]
        n_tie = sum(1 for iid in scope if _tied(st, others, iid))
        a = match(st.human[who], _panel(st, others, scope, "no"), scope)[0]
        b = match(st.human[who], _panel(st, others, scope, "yes"), scope)[0]
        print(f"  {who:<22}{a:8.1f}%{b:9.1f}%{n_tie:7d}")
    human_panel = st.majority(st.human)
    pct, k, n = match(maj_v, human_panel, scope)
    print(f"  {'VLM majority of 3':<22}{pct:8.1f}%{'n/a':>9}{0:7d}   (odd panel)")

    print("\n## Panel-size control, also for the demoted table\n")
    print("  A second reason the majority-vote table is not comparable across")
    print("  rows: a human annotator is scored against the other four, while the")
    print("  VLM judge is never in the panel and so is scored against all five.")
    print("  Size matching the VLM judge to four-annotator panels:\n")
    subs = [match(maj_v, majority_excluding(
        {k: v for k, v in st.human.items() if k != drop}, None, scope), scope)
        for drop in annotators]
    pcts = [s[0] for s in subs]
    ks = [s[1] for s in subs]
    full = match(maj_v, human_panel, scope)
    print(f"  {'vs 5 annotators':<28}{full[0]:8.1f}%{full[1]:+9.2f}")
    print(f"  {'vs 4, mean of the 5':<28}{sum(pcts) / len(pcts):8.1f}%"
          f"{sum(ks) / len(ks):+9.2f}   range {min(pcts):.1f}-{max(pcts):.1f}%")
    print("\n  Neither figure is quoted anywhere. The pairwise table is.")

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
