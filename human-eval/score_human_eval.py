#!/usr/bin/env python3
"""Score collected human answers against the answer key.

    python human-eval/score_human_eval.py                    # everything in responses/
    python human-eval/score_human_eval.py responses/consistency__fred.json
    python human-eval/score_human_eval.py --per-annotator

**Consistency** is scored the same way the paper's Consistency column is: with
the STRICT toggle each episode contributes two queries (the original turn-end
frame and the late-horizon duplicate), and the episode counts correct only if
*both* are right. turnToLookEval and turnToLookOppositeEval are then pooled into
one 64-episode number per model — matching ``build_vlm_tables.get_cell``. The
paper's VLM numbers are printed alongside for reference.

**Artifacts** is a label distribution, broken down by model and by category.
There is no ground truth, so nothing is scored as right or wrong.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESPONSES = HERE / "responses"

# What the paper reports for these two rows (Consistency column, Table 3).
PAPER_CONSISTENCY = {
    "flagship": (56.8, 2.9),            # Tables 2 and 3
    "concat_c": (25.5, 3.2),            # Table 2, Frame concat
    "no_player_attn_sf": (38.0, 1.5),   # Table 2, Independent
    "causvid_regression": (34.9, 1.5),  # Table 3, ODE Reg (kept for old runs)
}

ANSWER_TO_EXPECTED = {"same": "yes", "different": "no"}

ARTIFACT_LABELS = ["none", "character", "building", "other"]

# Wording version each task's pages currently serve. A run recorded under an
# older one is still valid — the options did not change — but is flagged so a
# mixed-guidance set is visible rather than silent.
CURRENT_INSTRUCTION_VERSION = {"consistency": 1, "artifacts": 2}


def _instruction_note(run: dict) -> None:
    task = run.get("task")
    seen = run.get("instruction_version")
    current = CURRENT_INSTRUCTION_VERSION.get(task)
    if seen is None:
        print("  NOTE: no instruction_version recorded — collected before the "
              "field existed.")
    elif current is not None and seen < current:
        print(f"  NOTE: collected under instruction wording v{seen}; pages now "
              f"serve v{current}.")
    if run.get("instruction_version_note"):
        print(f"        {run['instruction_version_note']}")


def load_key(task: str) -> dict[str, dict]:
    path = DATA / f"{task}_key.json"
    if not path.exists():
        raise SystemExit(f"missing {path} — run build_human_eval.py first")
    return {it["id"]: it for it in json.loads(path.read_text())["items"]}


def load_responses(paths: list[Path]) -> dict[str, list[dict]]:
    """Group response files by task."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        data = json.loads(p.read_text())
        task = data.get("task")
        if task not in {"consistency", "consistency_independent", "artifacts"}:
            print(f"  skipping {p.name}: unknown task {task!r}")
            continue
        data["_file"] = p.name
        by_task[task].append(data)
    return by_task


# --- consistency -----------------------------------------------------------


def score_consistency(runs: list[dict], key: dict[str, dict],
                      title: str = "CONSISTENCY — human vs VLM") -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    for run in runs:
        answers = run["answers"]
        who = run.get("annotator", "?")
        total = len(key)
        print(f"\nannotator: {who}   ({len(answers)}/{total} answered"
              f"{'' if len(answers) == total else ' — INCOMPLETE'})")
        _instruction_note(run)

        # query level, and episode buckets keyed by (model, eval, episode, instance)
        q_right = defaultdict(int)
        q_total = defaultdict(int)
        episodes: dict[tuple, list[bool]] = defaultdict(list)

        for item_id, ans in answers.items():
            k = key.get(item_id)
            if k is None:
                continue
            got = ANSWER_TO_EXPECTED.get(ans.get("value"))
            if got is None:
                continue
            ok = got == k["expected"]
            model = k["model"]
            q_total[model] += 1
            q_right[model] += ok
            episodes[(model, k["eval"], k["episode"], k["instance"])].append(ok)

        by_model_ep: dict[str, list[bool]] = defaultdict(list)
        for (model, _ev, _ep, _inst), oks in episodes.items():
            # AND-semantics: every query in the episode must be correct. Only
            # count episodes the annotator finished, so partial runs are not
            # silently penalised.
            if len(oks) == 2:
                by_model_ep[model].append(all(oks))

        print(f"  {'model':22s} {'query-level':>12s} {'episode-level':>14s} "
              f"{'n_ep':>5s}   paper (VLM)")
        for model in sorted(q_total):
            qa = 100.0 * q_right[model] / q_total[model] if q_total[model] else 0.0
            eps = by_model_ep[model]
            ea = 100.0 * sum(eps) / len(eps) if eps else float("nan")
            pm, ps = PAPER_CONSISTENCY.get(model, (float("nan"), float("nan")))
            flag = "" if len(eps) == 64 else "  (partial)"
            print(f"  {model:22s} {qa:11.1f}% {ea:13.1f}% {len(eps):5d}   "
                  f"{pm:.1f} +/- {ps:.1f}{flag}")

        # Only rank models that actually have completed episodes — otherwise a
        # partial run reports a comparison against nan. Handles any model count,
        # so adding a third model to CONSISTENCY_MODELS still prints a ranking.
        ranked = {m: v for m, v in by_model_ep.items() if v}
        if len(ranked) >= 2:
            order = sorted(ranked, key=lambda m: -_acc(ranked[m]))
            print("  -> humans rank " + " > ".join(
                f"{m} ({_acc(ranked[m]):.1f}%)" for m in order))
        else:
            print("  -> not enough completed episodes to rank yet")


def score_pooled_independent(by_task: dict[str, list[dict]]) -> None:
    """The 56.8 vs 38.0 comparison R3 asks about.

    ``flagship`` lives in task 1 and ``no_player_attn_sf`` in task 3, so the two
    sides come from different tasks. The stimuli and instructions are identical,
    which is what makes the pooling legitimate — but the two sides were
    collected in different sittings, so an annotator's threshold may have moved
    between them. Read the per-annotator margins, not just the mean.
    """
    print("\n" + "=" * 72)
    print("POOLED: Solaris Default (task 1) vs Independent (task 3)")
    print("=" * 72)

    keys = {"consistency": load_key("consistency"),
            "consistency_independent": load_key("consistency_independent")}
    by_annotator: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list))
    episodes: dict[tuple, list[bool]] = defaultdict(list)

    for task, runs in by_task.items():
        if task not in keys:
            continue
        for run in runs:
            who = run.get("annotator", "?")
            for item_id, ans in run["answers"].items():
                k = keys[task].get(item_id)
                got = ANSWER_TO_EXPECTED.get(ans.get("value"))
                if k is None or got is None:
                    continue
                episodes[(who, k["model"], k["eval"], k["episode"],
                          k["instance"])].append(got == k["expected"])

    for (who, model, _ev, _ep, _inst), oks in episodes.items():
        if len(oks) == 2:                      # AND over both timestamps
            by_annotator[who][model].append(all(oks))

    print(f"  {'annotator':10s} {'Solaris Default':>16s} {'Independent':>12s} "
          f"{'margin':>8s}   n_ep")
    margins = []
    for who in sorted(by_annotator):
        row = by_annotator[who]
        fl, ind = row.get("flagship", []), row.get("no_player_attn_sf", [])
        if not fl or not ind:
            print(f"  {who:10s} incomplete — "
                  f"{len(fl)} flagship, {len(ind)} Independent episodes")
            continue
        m = _acc(fl) - _acc(ind)
        margins.append(m)
        print(f"  {who:10s} {_acc(fl):15.1f}% {_acc(ind):11.1f}% "
              f"{m:+7.1f}   {len(fl)}/{len(ind)}")

    if margins:
        mean = sum(margins) / len(margins)
        var = sum((x - mean) ** 2 for x in margins) / len(margins)
        print(f"\n  mean margin {mean:+.1f} +/- {var ** 0.5:.1f} "
              f"(pop sd over {len(margins)} annotators)")
        print(f"  annotators ranking Solaris Default above Independent: "
              f"{sum(1 for x in margins if x > 0)}/{len(margins)}")
        pm, ps = PAPER_CONSISTENCY["flagship"]
        im, _is = PAPER_CONSISTENCY["no_player_attn_sf"]
        print(f"  paper (VLM): {pm:.1f} vs {im:.1f}, margin {pm - im:+.1f}")


def _acc(oks: list[bool]) -> float:
    return 100.0 * sum(oks) / len(oks) if oks else float("nan")


# --- artifacts -------------------------------------------------------------


def score_artifacts(runs: list[dict], key: dict[str, dict]) -> None:
    print("\n" + "=" * 72)
    print("ARTIFACTS — label distribution")
    print("=" * 72)

    for run in runs:
        answers = run["answers"]
        who = run.get("annotator", "?")
        total = len(key)
        print(f"\nannotator: {who}   ({len(answers)}/{total} answered"
              f"{'' if len(answers) == total else ' — INCOMPLETE'})")
        _instruction_note(run)

        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        by_cat: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        stale = 0

        for item_id, ans in answers.items():
            k = key.get(item_id)
            if k is None:
                continue
            label = ans.get("value")
            if label not in ARTIFACT_LABELS:
                # A value from a superseded option set (see TASK_VERSION).
                stale += 1
                continue
            counts[k["model"]][label] += 1
            by_cat[(k["model"], k["category"])][label] += 1

        if stale:
            print(f"  WARNING: {stale} answer(s) use retired options and were "
                  f"skipped — collected before the options changed?")

        # Any clip used as a worked example in the guide is shown to annotators
        # with its answer, so it is primed rather than blind. Surface it so the
        # bias is visible instead of buried in the totals.
        for k in key.values():
            if k.get("guide_example") and k["id"] in answers:
                print(f"  NOTE: {k['id']} ({k['category']}/{k['model']}) is the "
                      f"guide's worked example, so it is primed, not blind — "
                      f"answered {answers[k['id']].get('value')!r}. "
                      f"Consider excluding it.")

        # Width follows the longest label so renaming a category cannot silently
        # push the columns out of alignment.
        w = max(len(l) for l in ARTIFACT_LABELS) + 2
        print("  {:22s}".format("model")
              + "".join(f"{l:>{w}s}" for l in ARTIFACT_LABELS)
              + f"{'clean %':>10s}")
        for model in sorted(counts):
            row = counts[model]
            n = sum(row.values())
            clean = 100.0 * row["none"] / n if n else 0.0
            print("  {:22s}".format(model)
                  + "".join(f"{row[l]:{w}d}" for l in ARTIFACT_LABELS)
                  + f"{clean:9.1f}%")

        print("\n  by category (clean %):")
        cats = sorted({c for _m, c in by_cat})
        print("  {:22s}".format("model") + "".join(f"{c:>12s}" for c in cats))
        for model in sorted(counts):
            cells = []
            for c in cats:
                row = by_cat[(model, c)]
                n = sum(row.values())
                cells.append(f"{100.0 * row['none'] / n:11.1f}%" if n else f"{'-':>12s}")
            print("  {:22s}".format(model) + "".join(cells))



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="*", type=Path,
                        help="Response JSONs. Default: everything in responses/")
    args = parser.parse_args()

    paths = args.files or sorted(RESPONSES.glob("*.json"))
    if not paths:
        raise SystemExit(f"no response files found in {RESPONSES}")

    by_task = load_responses(paths)
    if not by_task:
        raise SystemExit("no usable response files")

    if "consistency" in by_task:
        score_consistency(by_task["consistency"], load_key("consistency"))
    if "consistency_independent" in by_task:
        score_consistency(by_task["consistency_independent"],
                          load_key("consistency_independent"),
                          title="CONSISTENCY (task 3, Independent) — human vs VLM")
    if {"consistency", "consistency_independent"} <= set(by_task):
        score_pooled_independent(by_task)
    if "artifacts" in by_task:
        score_artifacts(by_task["artifacts"], load_key("artifacts"))
    print()


if __name__ == "__main__":
    main()
