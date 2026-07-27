#!/usr/bin/env python3
"""Pooled-across-annotators analysis of the human evaluation.

``score_human_eval.py`` reports one annotator at a time. Every figure that
compares *judges* rather than items — majority vote, agreement, panel AUROC,
the margin t-test, the artifact clean rates and their bootstrap CIs — needs the
annotators pooled, and each of those was originally computed in a throwaway
script. This module is that analysis, checked in, so the numbers can be
regenerated when new annotations land instead of re-derived.

    python3 human-eval/pool_human_eval.py                 # everything
    python3 human-eval/pool_human_eval.py --part consistency
    python3 human-eval/pool_human_eval.py --root /path/to/checkout

Output mirrors the section order of ``RESULTS.md`` so a table can be copied
across without re-sorting.

Three data sources are joined:

* ``human-eval/data/*_key.json``  — the item key, which carries the ground
  truth and the identity ``(model, eval, episode, instance, query_type)``.
  Regenerate with ``build_human_eval.py``; the directory is gitignored.
* ``human-eval/responses/*.json`` — one file per annotator per task.
* ``results_json_late_episode_strict/`` — the VLM judge's own answers, joined
  to the same items so the two judge types are scored by identical code.

Consistency follows the paper's rule throughout: each episode contributes two
queries and counts correct only if both are right (``build_vlm_tables.get_cell``).
Standard deviations are population sd, matching ``_pop_mean_std``, except where
an inference is reported (t-test, CI), which uses the sample sd.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Human "same"/"different" and the key's "yes"/"no" are the same two classes.
ANSWER_TO_EXPECTED = {"same": "yes", "different": "no"}

ARTIFACT_LABELS = ["none", "character", "building", "other"]

# Published VLM cells, for the "does this reproduce the paper" check only.
PAPER_CONSISTENCY = {
    "flagship": (56.8, 2.9),
    "concat_c": (25.5, 3.2),
    "no_player_attn_sf": (38.0, 1.5),
    "causvid_regression": (34.9, 1.5),
}

# Published VLM axes, used for the clean-rate correlation in Part 2.
PAPER_AXES = {
    # model:                 Movement, Grounding, Building, Consistency
    "causvid_dmd":          (35.4,  3.1,  2.1, 45.8),
    "causvid_regression":   (19.8,  4.2,  3.1, 34.9),
    "from_scratch":         (69.8, 25.0,  0.0, 32.8),
    "no_player_attn_sf":    (12.5, 26.0,  0.0, 38.0),
    "flagship":             (67.7, 53.1,  9.4, 56.8),
    "no_kv_cache_backprop": (88.5, 77.1,  5.2, 52.6),
    "concat_c":             (84.4, 50.0,  0.0, 25.5),
}

# Which consistency task each model was annotated under. Solaris Default and
# Frame Concat shared task 1; Independent was added later as its own task with
# the same stimuli and instructions.
CONSISTENCY_TASKS = ("consistency", "consistency_independent")

BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 0


# --- small statistics helpers ----------------------------------------------


def pop_mean_sd(xs: list[float]) -> tuple[float, float]:
    """Population mean and sd, matching ``_pop_mean_std`` in the table builder."""
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def acc(oks: list[bool]) -> float:
    return 100.0 * sum(oks) / len(oks) if oks else float("nan")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Two-rater kappa. Chance agreement from each rater's own label rates."""
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def auroc(scores: list[float], positive: list[bool]) -> float:
    """Normalized Mann-Whitney U with 0.5 credit for ties.

    P(a random positive outranks a random negative). A single binary judge
    gives one ROC point, not a curve; a *panel* gives a graded score (how many
    judges voted "same"), which is what makes this computable.
    """
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def paired_t(xs: list[float]) -> tuple[float, float, tuple[float, float]]:
    """One-sample t against 0, two-sided p, and the 95% CI on the mean.

    Sample sd here, not the population sd used in the tables — an inference
    about the mean needs the unbiased estimate.
    """
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    se = sd / n ** 0.5
    t = m / se if se else float("inf")
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), n - 1)
        crit = stats.t.ppf(0.975, n - 1)
    except ImportError:                       # keep the script runnable bare
        p = float("nan")
        crit = 2.776 if n == 5 else float("nan")
    return t, p, (m - crit * se, m + crit * se)


def sign_test_one_sided(xs: list[float]) -> float:
    """P(all n signs agree by chance), i.e. the sign test without normality."""
    n = sum(1 for x in xs if x != 0)
    k = sum(1 for x in xs if x > 0)
    if n == 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


# --- loading ----------------------------------------------------------------


class Study:
    """Every human and VLM answer, joined on item identity."""

    def __init__(self, root: Path, results_dir: Path):
        self.root = root
        self.he = root / "human-eval"
        self.results_dir = results_dir
        self.items: dict[str, dict] = {}
        self.human: dict[str, dict[str, str]] = defaultdict(dict)
        self.vlm: dict[str, dict[str, str]] = defaultdict(dict)
        self._load_consistency()

    # -- consistency
    def _load_consistency(self) -> None:
        for task in CONSISTENCY_TASKS:
            key_path = self.he / "data" / f"{task}_key.json"
            if not key_path.exists():
                continue
            for it in json.loads(key_path.read_text())["items"]:
                it["task"] = task
                self.items[it["id"]] = it

            for path in sorted((self.he / "responses").glob(f"{task}__*.json")):
                run = json.loads(path.read_text())
                if run.get("task") != task:       # consistency__ prefixes overlap
                    continue
                who = run.get("annotator", path.stem)
                for item_id, ans in run["answers"].items():
                    got = ANSWER_TO_EXPECTED.get(ans.get("value"))
                    if got is not None and item_id in self.items:
                        self.human[who][item_id] = got

        # The judge's own answers, joined to the same items.
        index: dict[tuple, str] = {
            (it["model"], it["eval"], it["episode"], it["instance"],
             it["query_type"]): iid
            for iid, it in self.items.items()
        }
        models = {it["model"] for it in self.items.values()}
        evals = {it["eval"] for it in self.items.values()}
        for model in models:
            for ev in evals:
                d = self.results_dir / "generated" / f"{model}_{ev}"
                for trial_path in sorted(d.glob("trial_*.json")):
                    trial = f"VLM t{trial_path.stem.split('_')[1]}"
                    for r in json.loads(trial_path.read_text())["results"]:
                        m = r["metadata"]
                        iid = index.get((model, ev, m["episode"], m["instance"],
                                         m["query_type"]))
                        if iid is not None:
                            self.vlm[trial][iid] = r["response"].strip().lower()

    # -- views
    def annotators(self) -> list[str]:
        return sorted(self.human)

    def vlm_trials(self) -> list[str]:
        return sorted(self.vlm)

    def models(self) -> list[str]:
        seen = {it["model"] for it in self.items.values()}
        order = ["flagship", "concat_c", "no_player_attn_sf"]
        return [m for m in order if m in seen] + sorted(seen - set(order))

    def judges(self) -> dict[str, dict[str, str]]:
        """Every judge on one footing: VLM trials first, then annotators."""
        return {**{t: self.vlm[t] for t in self.vlm_trials()},
                **{a: self.human[a] for a in self.annotators()}}

    def complete_for(self, answers: dict[str, str], model: str) -> bool:
        want = sum(1 for it in self.items.values() if it["model"] == model)
        have = sum(1 for iid in answers if self.items[iid]["model"] == model)
        return want > 0 and have == want

    def episode_acc(self, answers: dict[str, str], model: str) -> float:
        """Episode-level accuracy under the paper's AND rule."""
        eps: dict[tuple, list[bool]] = defaultdict(list)
        for iid, got in answers.items():
            it = self.items[iid]
            if it["model"] != model:
                continue
            eps[(it["eval"], it["episode"], it["instance"])].append(
                got == it["expected"])
        done = [all(v) for v in eps.values() if len(v) == 2]
        return acc(done)

    def query_acc(self, answers: dict[str, str], model: str) -> float:
        oks = [got == self.items[iid]["expected"]
               for iid, got in answers.items()
               if self.items[iid]["model"] == model]
        return acc(oks)

    def item_ids(self, model: str | None = None) -> list[str]:
        return sorted(iid for iid, it in self.items.items()
                      if model is None or it["model"] == model)

    def majority(self, answers_by: dict[str, dict[str, str]]) -> dict[str, str]:
        """Per-item majority label across a panel. Ties resolve to 'no'.

        Ties only occur with an even panel; both current panels are odd.
        """
        out = {}
        for iid in self.items:
            votes = [a[iid] for a in answers_by.values() if iid in a]
            if votes:
                c = Counter(votes)
                top = c.most_common()
                out[iid] = "yes" if c["yes"] > c["no"] else "no" if top else "no"
        return out


# --- Part 1: consistency ----------------------------------------------------


def part1(st: Study) -> None:
    models = st.models()
    annotators = st.annotators()
    trials = st.vlm_trials()

    print("\n" + "=" * 78)
    print("PART 1 — CONSISTENCY (pooled)")
    print("=" * 78)

    # Only score a model where the whole panel finished it, so a half-collected
    # set cannot quietly change a panel number.
    full = [m for m in models
            if all(st.complete_for(st.human[a], m) for a in annotators)]
    partial = [m for m in models if m not in full]
    if partial:
        print("\n  INCOMPLETE, excluded from panel rows:")
        for m in partial:
            who = [a for a in annotators if st.complete_for(st.human[a], m)]
            print(f"    {m}: {len(who)}/{len(annotators)} annotators done"
                  f" ({', '.join(who) or 'none'})")
        print("    Per-annotator rows below still show them.")

    print("\n## Headline (episode-level)\n")
    hdr = f"  {'Judge':24s}" + "".join(f"{m:>20s}" for m in models)
    print(hdr)

    def row(label, per_model):
        print(f"  {label:24s}" + "".join(
            f"{v:>20s}" for v in per_model))

    vlm_cells = {m: [st.episode_acc(st.vlm[t], m) for t in trials] for m in models}
    row(f"VLM judge, {len(trials)} trials",
        [f"{pop_mean_sd(vlm_cells[m])[0]:.1f} +/- {pop_mean_sd(vlm_cells[m])[1]:.1f}"
         for m in models])

    hum_cells = {m: [st.episode_acc(st.human[a], m) for a in annotators
                     if st.complete_for(st.human[a], m)] for m in models}
    row(f"Humans, {len(annotators)} annotators",
        [f"{pop_mean_sd(hum_cells[m])[0]:.1f} +/- {pop_mean_sd(hum_cells[m])[1]:.1f}"
         f"{'' if m in full else ' *'}" for m in models])

    maj_h = st.majority(st.human)
    maj_v = st.majority(st.vlm)
    row("Human majority vote",
        [f"{st.episode_acc(maj_h, m):.1f}{'' if m in full else ' *'}"
         for m in models])
    row("VLM majority of trials", [f"{st.episode_acc(maj_v, m):.1f}"
                                   for m in models])

    print("\n  paper (published cells): " + "  ".join(
        f"{m} {PAPER_CONSISTENCY[m][0]:.1f} +/- {PAPER_CONSISTENCY[m][1]:.1f}"
        for m in models if m in PAPER_CONSISTENCY))
    if partial:
        print("  * panel incomplete for this model — see the note above.")

    # Margins are only meaningful within a judge, so compute them per judge and
    # then aggregate. Averaging two separately-averaged columns is not the same
    # thing when a judge is missing a model.
    if len(models) >= 2:
        base = models[0]
        for other in models[1:]:
            _margin_block(st, base, other, annotators, trials)

    print("\n## Query level, for reference (no AND rule)\n")
    print(f"  {'Judge':24s}" + "".join(f"{m:>20s}" for m in models))
    for a in annotators:
        cells = []
        for m in models:
            v = st.query_acc(st.human[a], m)
            cells.append(f"{'—':>20s}" if v != v else f"{v:>20.1f}")
        print(f"  {a:24s}" + "".join(cells))
    vq = {m: [st.query_acc(st.vlm[t], m) for t in trials] for m in models}
    print(f"  {'VLM, %d trials' % len(trials):24s}" + "".join(
        f"{pop_mean_sd(vq[m])[0]:>14.1f} +/- {pop_mean_sd(vq[m])[1]:.1f}"
        for m in models))

    # Judge-vs-judge sections compare rates, so every judge must be scored over
    # the same items. A model only one annotator has finished would otherwise
    # give that annotator a different denominator from everyone else.
    _agreement(st, annotators, trials, full)
    _thresholds(st, full, annotators, trials)
    _panel_auroc(st, full, annotators, trials)


def _margin_block(st: Study, base: str, other: str,
                  annotators: list[str], trials: list[str]) -> None:
    print(f"\n## Per judge: {base} vs {other} (episode-level)\n")
    print(f"  {'Judge':12s} {base:>14s} {other:>14s} {'margin':>9s}")
    for t in trials:
        a, b = st.episode_acc(st.vlm[t], base), st.episode_acc(st.vlm[t], other)
        print(f"  {t:12s} {a:14.1f} {b:14.1f} {a - b:+9.1f}")
    margins = []
    for who in annotators:
        ans = st.human[who]
        if not (st.complete_for(ans, base) and st.complete_for(ans, other)):
            print(f"  {who:12s} incomplete for this pair — skipped")
            continue
        a, b = st.episode_acc(ans, base), st.episode_acc(ans, other)
        margins.append(a - b)
        print(f"  {who:12s} {a:14.1f} {b:14.1f} {a - b:+9.1f}")

    if len(margins) < 2:
        print(f"\n  only {len(margins)} annotator(s) have both models — "
              f"no panel inference yet")
        return

    vlm_margins = [st.episode_acc(st.vlm[t], base) - st.episode_acc(st.vlm[t], other)
                   for t in trials]
    hm, hs = pop_mean_sd(margins)
    vm, vs = pop_mean_sd(vlm_margins)
    t, p, ci = paired_t(margins)
    print(f"\n  human margin {hm:+.1f} +/- {hs:.1f} (population sd over "
          f"{len(margins)} annotators)")
    print(f"  VLM   margin {vm:+.1f} +/- {vs:.1f}")
    print(f"  paired t({len(margins) - 1}) = {t:.1f}, p = {p:.4f}, "
          f"95% CI [{ci[0]:.1f}, {ci[1]:.1f}]"
          f"{'  <- contains the VLM margin' if ci[0] <= vm <= ci[1] else ''}")
    print(f"  sign test, one-sided: p = {sign_test_one_sided(margins):.3f}")
    print(f"  annotators ranking {base} above {other}: "
          f"{sum(1 for x in margins if x > 0)}/{len(margins)}")


def _agreement(st: Study, annotators: list[str], trials: list[str],
               models: list[str]) -> None:
    scope = set(itertools.chain.from_iterable(st.item_ids(m) for m in models))
    print(f"\n## Human-VLM agreement (per item, over {len(scope)} items: "
          f"{', '.join(models)})\n")
    maj_v = st.majority(st.vlm)
    first = trials[0] if trials else None

    def match(a: dict[str, str], b: dict[str, str]) -> tuple[float, float, int]:
        shared = [i for i in scope if i in a and i in b]
        if not shared:
            return float("nan"), float("nan"), 0
        xa = [a[i] for i in shared]
        xb = [b[i] for i in shared]
        return (100.0 * sum(x == y for x, y in zip(xa, xb)) / len(shared),
                cohen_kappa(xa, xb), len(shared))

    print(f"  {'Annotator':12s} {'vs ' + (first or '?'):>12s} "
          f"{'vs VLM maj':>12s} {'kappa(maj)':>11s} {'n':>6s}")
    for who in annotators:
        m1, _k1, _n1 = match(st.human[who], st.vlm[first]) if first else (
            float("nan"), 0, 0)
        mm, km, n = match(st.human[who], maj_v)
        print(f"  {who:12s} {m1:11.1f}% {mm:11.1f}% {km:+11.2f} {n:6d}")
    maj_h = st.majority(st.human)
    mm, km, n = match(maj_h, maj_v)
    print(f"  {'majority':12s} {'—':>12s} {mm:11.1f}% {km:+11.2f} {n:6d}")

    pairs = [match(st.human[a], st.human[b])
             for a, b in itertools.combinations(annotators, 2)]
    if pairs:
        vals = [p[0] for p in pairs]
        ks = [p[1] for p in pairs]
        print(f"\n  mean pairwise human-human agreement: "
              f"{sum(vals) / len(vals):.1f}% "
              f"(range {min(vals):.1f}-{max(vals):.1f}%, "
              f"kappa {min(ks):+.2f} to {max(ks):+.2f})")
        print("  Read the two together: the judge is no further from a human "
              "than two humans are from each other.")


def _thresholds(st: Study, models: list[str], annotators: list[str],
                trials: list[str]) -> None:
    print("\n## Threshold effects (query level)\n")
    print("  The item set is balanced, so accuracy == balanced accuracy and a")
    print("  skewed threshold cannot inflate it. What a threshold can do is")
    print("  understate a judge that separates the classes but sits badly.\n")
    same_hdr = '"same" rate'
    scope = set(itertools.chain.from_iterable(st.item_ids(m) for m in models))
    print(f"  {'Judge':12s} {same_hdr:>12s} {'sens':>8s} {'spec':>8s}   "
          + "  ".join(f"{m[:12]:>12s}" for m in models))
    for label, full_ans in st.judges().items():
        ans = {i: v for i, v in full_ans.items() if i in scope}
        yes = [iid for iid in ans if ans[iid] == "yes"]
        rate = 100.0 * len(yes) / len(ans) if ans else float("nan")
        pos = [iid for iid in ans if st.items[iid]["expected"] == "yes"]
        neg = [iid for iid in ans if st.items[iid]["expected"] == "no"]
        sens = 100.0 * sum(ans[i] == "yes" for i in pos) / len(pos) if pos else float("nan")
        spec = 100.0 * sum(ans[i] == "no" for i in neg) / len(neg) if neg else float("nan")
        per_model = []
        for m in models:
            ids = [i for i in ans if st.items[i]["model"] == m]
            per_model.append(f"{100.0 * sum(ans[i] == 'yes' for i in ids) / len(ids):11.1f}%"
                             if ids else f"{'—':>12s}")
        print(f"  {label:12s} {rate:11.1f}% {sens:7.1f}% {spec:7.1f}%   "
              + "  ".join(per_model))

    print("\n  Youden's J per judge per model (J = TPR - FPR; acc = (1+J)/2)\n")
    print(f"  {'Judge':12s}" + "".join(f"{m[:18]:>26s}" for m in models))
    j_by_judge: dict[str, dict[str, float]] = defaultdict(dict)
    for label, full_ans in st.judges().items():
        ans = {i: v for i, v in full_ans.items() if i in scope}
        cells = []
        for m in models:
            ids = [i for i in ans if st.items[i]["model"] == m]
            pos = [i for i in ids if st.items[i]["expected"] == "yes"]
            neg = [i for i in ids if st.items[i]["expected"] == "no"]
            if not pos or not neg:
                cells.append(f"{'—':>26s}")
                continue
            tpr = 100.0 * sum(ans[i] == "yes" for i in pos) / len(pos)
            fpr = 100.0 * sum(ans[i] == "yes" for i in neg) / len(neg)
            j_by_judge[label][m] = tpr - fpr
            cells.append(f"{tpr:10.1f} /{fpr:6.1f} J{tpr - fpr:+6.1f}")
        print(f"  {label:12s}" + "".join(cells))

    if len(models) >= 2:
        base, other = models[0], models[1]
        hj = [j_by_judge[a][base] - j_by_judge[a][other] for a in annotators
              if base in j_by_judge[a] and other in j_by_judge[a]]
        vj = [j_by_judge[t][base] - j_by_judge[t][other] for t in trials
              if base in j_by_judge[t] and other in j_by_judge[t]]
        if hj and vj:
            hm, hs = pop_mean_sd(hj)
            vm, vs = pop_mean_sd(vj)
            print(f"\n  J margin ({base} - {other}): humans {hm:+.1f} +/- {hs:.1f}, "
                  f"VLM {vm:+.1f} +/- {vs:.1f}")
            print("  At a 50% base rate this is exactly twice the query-level "
                  "accuracy margin,")
            print("  so it restates that number on a bias-free scale rather "
                  "than adding evidence.")


def _panel_auroc(st: Study, models: list[str], annotators: list[str],
                 trials: list[str]) -> None:
    print("\n## Panel AUROC (query level, threshold-free)\n")
    print("  A single binary judge is one ROC point. A panel has a graded score")
    print("  for free — how many judges voted \"same\" — so AUROC is defined.\n")
    if not models:
        print("  no model has a complete panel yet")
        return

    def panel_auroc(panel: list[dict[str, str]], model: str) -> float:
        ids = st.item_ids(model)
        ids = [i for i in ids if all(i in p for p in panel)]
        if not ids:
            return float("nan")
        scores = [sum(p[i] == "yes" for p in panel) for i in ids]
        pos = [st.items[i]["expected"] == "yes" for i in ids]
        return auroc(scores, pos)

    print(f"  {'Panel':40s}" + "".join(f"{m[:16]:>18s}" for m in models))
    hp = [st.human[a] for a in annotators]
    print(f"  {'%d human annotators' % len(hp):40s}"
          + "".join(f"{panel_auroc(hp, m):>18.3f}" for m in models))

    k = 3
    if len(hp) > k:
        subsets = list(itertools.combinations(range(len(hp)), k))
        cells = []
        for m in models:
            vals = [panel_auroc([hp[i] for i in s], m) for s in subsets]
            mean, sd = pop_mean_sd(vals)
            cells.append(f"{mean:11.3f} +/-{sd:5.3f}")
        print(f"  {'%d of %d humans, mean over %d subsets' % (k, len(hp), len(subsets)):40s}"
              + "".join(cells))

    vp = [st.vlm[t] for t in trials]
    print(f"  {'VLM, %d trials' % len(vp):40s}"
          + "".join(f"{panel_auroc(vp, m):>18.3f}" for m in models))
    print("\n  Compare the size-matched human row against the VLM row: a panel "
          "with more\n  members has more score levels, and ties score 0.5, so "
          "the full human panel\n  is granularity-flattered.")


# --- Part 2: artifacts ------------------------------------------------------


def part2(st: Study) -> None:
    key_path = st.he / "data" / "artifacts_key.json"
    if not key_path.exists():
        return
    key = {it["id"]: it for it in json.loads(key_path.read_text())["items"]}
    runs = {}
    for path in sorted((st.he / "responses").glob("artifacts__*.json")):
        run = json.loads(path.read_text())
        runs[run.get("annotator", path.stem)] = run["answers"]
    if not runs:
        return

    print("\n" + "=" * 78)
    print("PART 2 — ARTIFACTS (pooled)")
    print("=" * 78)
    print(f"\n  {len(runs)} annotators x {len(key)} clips = "
          f"{sum(len(a) for a in runs.values())} judgements")

    # Judgements of one clip are not independent, so the interval resamples
    # clips, not judgements.
    rng = random.Random(BOOTSTRAP_SEED)
    by_model_clip: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list))
    for who, answers in runs.items():
        for iid, ans in answers.items():
            it = key.get(iid)
            label = ans.get("value")
            if it and label in ARTIFACT_LABELS:
                by_model_clip[it["model"]][iid].append(label)

    print(f"\n  CI: percentile bootstrap over the 9 clips, "
          f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}. With 9 "
          f"clips the\n  endpoints land on a coarse grid, so a different seed "
          f"can move them one step (~2 pts).")
    print(f"\n  {'Model':22s}" + "".join(f"{l:>10s}" for l in ARTIFACT_LABELS)
          + f"{'clean %':>10s}{'95% CI (clip)':>20s}{'maj-clean':>11s}")
    clean_rate = {}
    for model in sorted(by_model_clip, key=lambda m: -_clean_pct(by_model_clip[m])):
        clips = by_model_clip[model]
        counts = Counter(l for labels in clips.values() for l in labels)
        n = sum(counts.values())
        pct = 100.0 * counts["none"] / n if n else float("nan")
        clean_rate[model] = pct
        per_clip = [100.0 * sum(l == "none" for l in labels) / len(labels)
                    for labels in clips.values()]
        boots = sorted(
            sum(rng.choice(per_clip) for _ in per_clip) / len(per_clip)
            for _ in range(BOOTSTRAP_RESAMPLES))
        lo = boots[int(0.025 * len(boots))]
        hi = boots[int(0.975 * len(boots))]
        maj = sum(1 for labels in clips.values()
                  if sum(l == "none" for l in labels) * 2 > len(labels))
        print(f"  {model:22s}" + "".join(f"{counts[l]:10d}" for l in ARTIFACT_LABELS)
              + f"{pct:9.1f}%[{lo:8.1f},{hi:6.1f}]{maj:8d}/{len(clips)}")

    # Head-to-head is per annotator: it is the same person judging both models,
    # which removes their private strictness from the comparison.
    order = sorted(clean_rate, key=lambda m: -clean_rate[m])
    if "flagship" in clean_rate and "concat_c" in clean_rate:
        diffs = []
        for who, answers in runs.items():
            def rate(model):
                got = [answers[i]["value"] for i in answers
                       if key.get(i, {}).get("model") == model
                       and answers[i].get("value") in ARTIFACT_LABELS]
                return 100.0 * sum(g == "none" for g in got) / len(got) if got else float("nan")
            diffs.append(rate("flagship") - rate("concat_c"))
        m, s = pop_mean_sd(diffs)
        print(f"\n  flagship - concat_c clean rate, per annotator: "
              + ", ".join(f"{d:+.1f}" for d in diffs))
        print(f"  mean {m:+.1f} +/- {s:.1f}, favouring flagship "
              f"{sum(1 for d in diffs if d > 0)}/{len(diffs)}")

    # Whether two annotators agree a clip is clean, against the rate they would
    # agree by chance given how often each says "clean".
    per_clip_clean: dict[str, dict[str, bool]] = defaultdict(dict)
    for who, answers in runs.items():
        for iid, ans in answers.items():
            if ans.get("value") in ARTIFACT_LABELS:
                per_clip_clean[who][iid] = ans["value"] == "none"
    names = sorted(per_clip_clean)
    obs, tot = 0, 0
    for a, b in itertools.combinations(names, 2):
        shared = set(per_clip_clean[a]) & set(per_clip_clean[b])
        obs += sum(per_clip_clean[a][i] == per_clip_clean[b][i] for i in shared)
        tot += len(shared)
    p_clean = sum(v for d in per_clip_clean.values() for v in d.values()) / max(
        1, sum(len(d) for d in per_clip_clean.values()))
    pe = p_clean ** 2 + (1 - p_clean) ** 2
    po = obs / tot if tot else float("nan")
    print(f"\n  clean/not-clean agreement: {100 * po:.1f}% observed, "
          f"{100 * pe:.1f}% by chance (clean rate {100 * p_clean:.1f}%), "
          f"kappa {(po - pe) / (1 - pe):+.2f}")

    axes = ["Movement", "Grounding", "Building", "Consistency"]
    have = [m for m in order if m in PAPER_AXES]
    if len(have) >= 3:
        print("\n  Spearman rho, clean rate vs the published VLM axes "
              f"(n={len(have)} models):")
        xs = [clean_rate[m] for m in have]
        for i, axis in enumerate(axes):
            ys = [PAPER_AXES[m][i] for m in have]
            print(f"    {axis:14s} {spearman(xs, ys):+.2f}")
        print("  A near-static generator is artifact-free and useless, so a "
              "clean rate is\n  not a quality score. Report it beside the "
              "capability axes or not at all.")


def _clean_pct(clips: dict[str, list[str]]) -> float:
    labels = [l for v in clips.values() for l in v]
    return 100.0 * sum(l == "none" for l in labels) / len(labels) if labels else 0.0


# --- entry point ------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=HERE.parent,
                    help="Repository root holding human-eval/ (default: this checkout)")
    ap.add_argument("--results-dir", type=Path,
                    help="Strict VLM results tree "
                         "(default: <root>/results_json_late_episode_strict)")
    ap.add_argument("--part", choices=["consistency", "artifacts", "all"],
                    default="all")
    args = ap.parse_args()

    results_dir = args.results_dir or args.root / "results_json_late_episode_strict"
    st = Study(args.root, results_dir)
    if not st.items:
        raise SystemExit(
            f"no consistency key under {args.root / 'human-eval' / 'data'} — "
            f"run build_human_eval.py first")

    if args.part in ("consistency", "all"):
        part1(st)
    if args.part in ("artifacts", "all"):
        part2(st)
    print()


if __name__ == "__main__":
    main()
