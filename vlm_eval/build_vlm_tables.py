#!/usr/bin/env python3
"""Build LaTeX Tables 2 and 3 with fresh VLM accuracy numbers.

This script reads per-(model, eval) ``stats.json`` files produced by the
evaluation pipeline and renders Tables 2 and 3 from the paper using a
``string.Template``. FID numbers are hard-coded in the template (those did
not change). Only the VLM cells are recomputed here.

Output
------
A LaTeX file (``vlm_tables.tex`` by default) next to this script. The
output paths are added to ``.gitignore`` because they are regenerated
from JSON.

Column → eval mapping
---------------------
The four LaTeX columns always map onto the same eval handlers:

+----------------+----------------------------------+------------------------------------+--------------+
| LaTeX column   | eval handler                     | results tree                       | late-episode |
+================+==================================+====================================+==============+
| Movement       | translationEval                  | ``results_json/`` (always)         | NO           |
| Grounding      | oneLooksAwayEval_long            | ``--late-tree`` (CLI arg)          | YES          |
| Building       | structureEval                    | ``--late-tree``                    | YES          |
| Consistency    | turnToLookEval +                 | ``--late-tree``                    | YES          |
|                | turnToLookOppositeEval combined  |                                    |              |
+----------------+----------------------------------+------------------------------------+--------------+

translationEval is pinned to the OLD (non-late-episode) results tree
because the dataset reverses bot actions in the latter half of each
episode, so a late-horizon query no longer matches the prompt's intent.
The translationEval cells were collected with **thinking OFF** (handler
default).

Per-handler "thinking" config differs between supported late trees
-------------------------------------------------------------------

``--late-tree results_json_late_episode_mixed_thinking`` (default):
  Per-handler thinking config that won the GT-validation ablation in
  ``results_json_late_episode_mixed_thinking/comparison_late_vs_current.md``:

  =======================  ========
  handler                  thinking
  =======================  ========
  oneLooksAwayEval_long    OFF
  structureEval            ON
  turnToLookEval           ON
  turnToLookOppositeEval   ON
  =======================  ========

``--late-tree results_json_late_episode_strict``:
  STRICT toggle (``LATE_EPISODE_QUERY_STRICT=1``) — every handler emits
  both the original "last" query and a duplicate at the late-horizon
  frame, with AND-semantics for episode-level accuracy. Thinking config
  in the trial files (matches the mixed-thinking tree after the
  structureEval handler default was flipped to thinking ON):

  =======================  ========
  handler                  thinking
  =======================  ========
  oneLooksAwayEval_long    OFF
  structureEval            ON
  turnToLookEval           ON
  turnToLookOppositeEval   ON
  =======================  ========

(``thinking_enabled`` is recorded per-trial in every ``trial_*.json`` so
the config is verifiable post-hoc.)

Consistency std
---------------
For the Consistency column we treat ``turnToLookEval`` and
``turnToLookOppositeEval`` as one eval with 64 episodes per trial: per
trial we sum ``fully_correct_episodes`` and ``total_episodes`` from the
two handlers' ``trial_*.json`` files, get a combined per-trial
**episode-level** accuracy, and then take mean / population-std across
the 3 trials.

We use ``fully_correct_episodes`` / ``total_episodes`` rather than the
top-level ``correct`` / ``total_queries`` because in the STRICT
late-episode tree each handler emits 2 queries per episode with
AND-semantics, so the latter pair gives a query-level accuracy rather
than the episode-level one we want. (In the mixed-thinking tree the
two are equal, so this is purely a correctness fix for the strict
case.)

This combined-trials method is **not** the same as
``sqrt(s1**2 + s2**2) / 2`` (that formula assumes the two evals are
independent samples, but in fact they share trial seeds), and it differs
slightly from the ``turnToLook avg`` column in the trees'
``accuracy_table.md`` which uses the independent-std approximation.

Usage
-----
::

    # Default: mixed-thinking late-episode tree.
    python build_vlm_tables.py

    # Strict late-episode tree:
    python build_vlm_tables.py \\
        --late-tree results_json_late_episode_strict \\
        --out vlm_tables_strict.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LATE_TREE = REPO_ROOT / "results_json_late_episode_mixed_thinking"
OLD_TREE = REPO_ROOT / "results_json" / "generated"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "vlm_tables.tex"

# Table 2 model order: (latex_label_for_doc_only, json_model_name, var_prefix).
TABLE2_MODELS = [
    ("Independent",          "no_player_attn_sf", "ind"),
    ("Frame concat",         "concat_c",          "fc"),
    ("Solaris w/o pretrain", "from_scratch",      "swp"),
    ("Solaris",              "flagship",          "sol"),
]

# Table 3 row order.
TABLE3_ROWS = [
    ("ODE Reg",                    "causvid_regression",   "oder"),
    ("Causal FT (Pre-DMD)",        "causvid_dmd",          "dmd"),
    ("Causal FT (no KV-BP)",       "no_kv_cache_backprop", "nokvbp"),
    ("Causal FT (flagship)",       "flagship",             "flag"),
]

# Column key -> short id used in template variable names.
COLUMNS = ["mov", "gnd", "bld", "con"]


def _read_stats(path: Path) -> tuple[float, float]:
    """Return (mean, std) from a stats.json file."""
    with path.open() as f:
        data = json.load(f)
    return float(data["mean"]), float(data["std"])


def _pop_mean_std(values: list[float]) -> tuple[float, float]:
    """Population mean and std (ddof=0), matching what stats.json uses."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def _read_trial_episode_counts(eval_dir: Path) -> list[tuple[int, int]]:
    """Return ``[(fully_correct_episodes, total_episodes)]`` for each
    ``trial_*.json``, sorted by trial number.

    We deliberately use *episode-level* counts (not the top-level
    ``correct`` / ``total_queries``) because in the STRICT late-episode
    tree each handler emits 2 queries per episode with AND-semantics, so
    ``correct`` / ``total_queries`` is a query-level accuracy rather than
    the desired episode-level one.
    """
    trials = sorted(
        eval_dir.glob("trial_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    out: list[tuple[int, int]] = []
    for p in trials:
        with p.open() as f:
            d = json.load(f)
        ep = d["episode_level_accuracy"]
        out.append((int(ep["fully_correct_episodes"]), int(ep["total_episodes"])))
    return out


def get_cell(model: str, column: str, late_tree: Path) -> tuple[float, float]:
    """Return (mean, std) for a (model, column) pair following the mapping
    in the module docstring. ``late_tree`` is the ``generated/`` directory
    of the chosen late-episode results tree."""
    if column == "mov":
        return _read_stats(OLD_TREE / f"{model}_translationEval" / "stats.json")
    if column == "gnd":
        return _read_stats(late_tree / f"{model}_oneLooksAwayEval_long" / "stats.json")
    if column == "bld":
        return _read_stats(late_tree / f"{model}_structureEval" / "stats.json")
    if column == "con":
        # Combine the two turnToLook evals into one 64-episode-per-trial
        # eval: per trial, sum fully_correct_episodes / total_episodes
        # across the two handlers; then take population mean/std of the
        # per-trial accuracy across the 3 trials. This is robust to the
        # STRICT toggle (where total_queries != total_episodes).
        a = _read_trial_episode_counts(late_tree / f"{model}_turnToLookEval")
        b = _read_trial_episode_counts(late_tree / f"{model}_turnToLookOppositeEval")
        if len(a) != len(b):
            raise ValueError(
                f"trial count mismatch for {model}: "
                f"turnToLookEval={len(a)} vs turnToLookOppositeEval={len(b)}"
            )
        per_trial_acc = [
            100.0 * (ca + cb) / (ta + tb)
            for (ca, ta), (cb, tb) in zip(a, b)
        ]
        return _pop_mean_std(per_trial_acc)
    raise ValueError(f"unknown column {column!r}")


def fmt_cell(mean: float, std: float, *, bold: bool) -> str:
    body = f"{mean:.1f} $\\pm$ {std:.1f}"
    return f"\\textbf{{{body}}}" if bold else body


def build_subs_for_table(
    rows: list[tuple[str, str, str]],
    late_tree: Path,
) -> dict[str, str]:
    """Compute every cell, mark the per-column max as bold, and return a
    flat substitution dict ``{f"{prefix}_{col}": formatted_cell}``."""
    cells: dict[tuple[str, str], tuple[float, float]] = {}
    for _label, model, prefix in rows:
        for col in COLUMNS:
            cells[(prefix, col)] = get_cell(model, col, late_tree)

    best_prefix_per_col: dict[str, str] = {}
    for col in COLUMNS:
        best_prefix_per_col[col] = max(
            (prefix for _l, _m, prefix in rows),
            key=lambda p: cells[(p, col)][0],
        )

    subs: dict[str, str] = {}
    for _label, _model, prefix in rows:
        for col in COLUMNS:
            mean, std = cells[(prefix, col)]
            bold = best_prefix_per_col[col] == prefix
            subs[f"{prefix}_{col}"] = fmt_cell(mean, std, bold=bold)
    return subs


# ---------------------------------------------------------------------------
# LaTeX template. FID numbers are baked in. Every VLM cell is a $-placeholder
# whose name is "<row_prefix>_<column_key>", e.g. ``$sol_mov``.
# ---------------------------------------------------------------------------
# NOTE: ``string.Template`` treats every ``$`` as the start of a placeholder.
# Because LaTeX uses ``$`` for inline math (e.g. ``$\uparrow$``) and ``$\pm$``,
# every *literal* dollar sign in the template body is escaped as ``$$`` and
# is converted back to a single ``$`` by ``Template.substitute``. Only the
# 32 ``${name}`` placeholders below are real substitutions.
TEMPLATE = Template(
    r"""% Auto-generated by build_vlm_tables.py. Do NOT edit by hand.
% VLM cells come from JSON; FID cells are hard-coded.

% ===== Table 2 =====
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{Movement}}
 & \multicolumn{2}{c}{\textbf{Grounding}}
 & \multicolumn{2}{c}{\textbf{Building}}
 & \multicolumn{2}{c}{\textbf{Consistency}} \\
\cmidrule(lr){2-3}
\cmidrule(lr){4-5}
\cmidrule(lr){6-7}
\cmidrule(lr){8-9}
\textbf{Method}
& \textbf{VLM} $$\uparrow$$ & \textbf{FID} $$\downarrow$$
& \textbf{VLM} $$\uparrow$$ & \textbf{FID} $$\downarrow$$
& \textbf{VLM} $$\uparrow$$ & \textbf{FID} $$\downarrow$$
& \textbf{VLM} $$\uparrow$$ & \textbf{FID} $$\downarrow$$ \\
\midrule
Independent
& ${ind_mov} & 58.9
& ${ind_gnd} & 67.1
& ${ind_bld} & 77.8
& ${ind_con} & 104.8 \\

Frame concat
& ${fc_mov} & 69.5
& ${fc_gnd} & 66.6
& ${fc_bld} & 103.2
& ${fc_con} & 129.4 \\

\solaris w/o pretrain
& ${swp_mov} & 37.11
& ${swp_gnd} & 49.9
& ${swp_bld} & 86.6
& ${swp_con} & 121.4 \\

\solaris
& ${sol_mov} & \textbf{36.84}
& ${sol_gnd} & \textbf{38.0}
& ${sol_bld} & \textbf{83.6}
& ${sol_con} & \textbf{99.4} \\
\bottomrule
\end{tabular}%

% ===== Table 3 =====
\begin{tabular}{lcccccccccc}
\toprule
\multicolumn{3}{c}{\textbf{Components}} &
\multicolumn{2}{c}{\textbf{Movement}} &
\multicolumn{2}{c}{\textbf{Grounding}} &
\multicolumn{2}{c}{\textbf{Building}} &
\multicolumn{2}{c}{\textbf{Consistency}} \\
\cmidrule(r){1-3}
\cmidrule(lr){4-5}
\cmidrule(lr){6-7}
\cmidrule(lr){8-9}
\cmidrule(l){10-11}

\textbf{Init.} & \textbf{Pre-DMD} & \textbf{KV-BP} &
\textbf{VLM} $$\uparrow$$ & \textbf{FID} $$\downarrow$$ &
\textbf{VLM} $$\uparrow$$ & \textbf{FID} $$\downarrow$$ &
\textbf{VLM} $$\uparrow$$ & \textbf{FID} $$\downarrow$$ &
\textbf{VLM} $$\uparrow$$ & \textbf{FID} $$\downarrow$$ \\
\midrule

ODE Reg & \texttimes & \checkmark
& ${oder_mov} & 77.42
& ${oder_gnd} & 56.6
& ${oder_bld} & 95.7
& ${oder_con} & 142.3 \\

Causal FT & \checkmark & \checkmark
& ${dmd_mov} & 54.36
& ${dmd_gnd} & 40.4
& ${dmd_bld} & 90.5
& ${dmd_con} & 160.1 \\

Causal FT & \texttimes & \texttimes
& ${nokvbp_mov} & 53.76
& ${nokvbp_gnd} & 55.2
& ${nokvbp_bld} & 87.4
& ${nokvbp_con} & 105.1 \\

\rowcolor{gray!15}
Causal FT & \texttimes & \checkmark
& ${flag_mov} & \textbf{36.84}
& ${flag_gnd} & \textbf{38.0}
& ${flag_bld} & \textbf{83.6}
& ${flag_con} & \textbf{99.4} \\

\bottomrule
\end{tabular}%
"""
)


def _resolve(p: str | Path) -> Path:
    """Resolve a user-provided path: absolute paths kept as-is, relative
    paths resolved against the repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--late-tree",
        default=str(DEFAULT_LATE_TREE),
        help=(
            "Late-episode results tree (the directory containing "
            "'generated/' and 'real/'). Default: "
            "results_json_late_episode_mixed_thinking"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output .tex file. Default: vlm_tables.tex",
    )
    args = parser.parse_args()

    late_tree_root = _resolve(args.late_tree)
    late_tree_generated = late_tree_root / "generated"
    if not late_tree_generated.is_dir():
        raise SystemExit(f"--late-tree generated/ not found: {late_tree_generated}")

    subs: dict[str, str] = {}
    subs.update(build_subs_for_table(TABLE2_MODELS, late_tree_generated))
    subs.update(build_subs_for_table(TABLE3_ROWS, late_tree_generated))

    rendered = TEMPLATE.substitute(subs)
    out_path = _resolve(args.out)
    out_path.write_text(rendered)
    try:
        rel = out_path.relative_to(REPO_ROOT)
    except ValueError:
        rel = out_path
    print(f"wrote {rel} (late-tree: {late_tree_root.name})")


if __name__ == "__main__":
    main()
