#!/usr/bin/env python3
"""
Build the late-episode vs current-pattern comparison artifact.

Reads ``stats.json`` for every ``{model}_{eval}`` pair under
``results_json/generated/`` and ``results_json_late_episode/generated/`` and
writes a Markdown report to (by default)
``results_json_late_episode/comparison_late_vs_current.md``.

The report contains:

1. Three wide tables (rows = models, columns = evals) showing per-cell
   episode accuracy under the OLD (current) pattern, the NEW (late-episode)
   pattern, and the per-cell delta ``Δ = NEW − OLD``.
2. A discriminativeness summary per eval (spread = max − min of model means,
   plus population std across models) for both patterns.
3. A "biggest swings" section that ranks (model, eval) pairs by the
   absolute size of the late-vs-current delta.
4. A GT-validation table that compares OLD vs NEW accuracy on real (GT)
   videos to confirm the late-episode question is still well-posed; an
   ``OK``/``WARN``/``FAIL`` verdict flags evals whose late-frame choice
   makes the question ambiguous on real footage.

Special handling:

- ``translationEval`` is rendered with the OLD value in the NEW column too
  (Δ pinned to 0). The eval reverses bot actions in the second half of the
  episode, so the late-horizon question is ill-posed for it; we keep the
  column for completeness but do not draw conclusions from a late-vs-current
  comparison.
- A synthetic ``turnToLook avg`` column is added that averages
  ``turnToLookEval`` and ``turnToLookOppositeEval`` per (model, pattern).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_OLD = Path("results_json/generated")
DEFAULT_NEW = Path("results_json_late_episode/generated")
DEFAULT_OLD_GT = Path("results_json/real")
DEFAULT_NEW_GT = Path("results_json_late_episode/real")
DEFAULT_OUT = Path("results_json_late_episode/comparison_late_vs_current.md")

# Verdict thresholds for the GT validation table. The 90% floor is a soft
# target — `bothLookAwayEval_long` was already at 92.7% on the OLD pattern
# (see plan), so a few-point dip there is within baseline noise.
GT_VERDICT_OK = 90.0
GT_VERDICT_WARN = 75.0

# Evals we evaluated under the late-episode toggle. (structureNoPlace / short
# look-away variants weren't part of the late-episode run.)
DEFAULT_EVALS = [
    "translationEval",
    "rotationEval",
    "structureEval",
    "turnToLookEval",
    "turnToLookOppositeEval",
    "oneLooksAwayEval_long",
    "bothLookAwayEval_long",
]

# Evals where the late-episode pattern is not meaningful and the NEW value
# should be pinned to OLD (Δ = 0). Currently only translation, because the
# dataset reverses actions in the second half of the episode.
PINNED_TO_OLD: set = {"translationEval"}

# Synthetic aggregate columns: name -> list of base eval names to average.
TURN_TO_LOOK_AVG = "turnToLook avg"
AGGREGATE_COLUMNS: Dict[str, List[str]] = {
    TURN_TO_LOOK_AVG: ["turnToLookEval", "turnToLookOppositeEval"],
}


def _short_label(eval_name: str) -> str:
    """Compact column label for wide tables."""
    return (
        eval_name.replace("Eval_long", "_long")
        .replace("Eval", "")
        .replace("turnToLookOpposite", "turnToLookOpp")
    )


def _load_mean(root: Path, model: str, eval_name: str) -> Optional[float]:
    p = root / f"{model}_{eval_name}" / "stats.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text()).get("mean"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _discover_models(new_root: Path, evals: List[str]) -> List[str]:
    found = set()
    for d in new_root.iterdir():
        if not d.is_dir():
            continue
        for ev in evals:
            suffix = "_" + ev
            if d.name.endswith(suffix):
                found.add(d.name[: -len(suffix)])
                break
    return sorted(found)


def _column_value(
    model: str,
    column: str,
    pattern_old: bool,
    old_root: Path,
    new_root: Path,
) -> Optional[float]:
    """Compute the cell value for ``model`` × ``column`` for either pattern.

    Handles three cases:
    - Aggregate columns (e.g. ``turnToLook avg``): mean of the underlying
      base evals' values for the requested pattern.
    - Pinned-to-OLD evals: returns the OLD value regardless of pattern.
    - Regular base evals: looks up directly from the requested root.
    """
    if column in AGGREGATE_COLUMNS:
        parts = [
            _column_value(model, ev, pattern_old, old_root, new_root)
            for ev in AGGREGATE_COLUMNS[column]
        ]
        present = [v for v in parts if v is not None]
        if not present:
            return None
        return sum(present) / len(present)

    if column in PINNED_TO_OLD:
        return _load_mean(old_root, model, column)

    root = old_root if pattern_old else new_root
    return _load_mean(root, model, column)


def _build_wide_table(
    title: str,
    columns: List[str],
    models: List[str],
    cell_fn: Callable[[str, str], Optional[float]],
    *,
    signed: bool = False,
    note: Optional[str] = None,
) -> List[str]:
    header = "| model | " + " | ".join(_short_label(c) for c in columns) + " |"
    sep = "|---|" + "---:|" * len(columns)
    lines: List[str] = [f"### {title}", ""]
    if note:
        lines.append(note)
        lines.append("")
    lines.append(header)
    lines.append(sep)
    for m in models:
        cells: List[str] = []
        for c in columns:
            v = cell_fn(m, c)
            if v is None:
                cells.append("—")
            elif signed:
                cells.append(f"{v:+.2f}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _build_discriminativeness_table(
    columns: List[str],
    models: List[str],
    old_root: Path,
    new_root: Path,
) -> List[str]:
    lines = [
        "## Discriminativeness across models",
        "",
        "Spread = max(model mean) − min(model mean). Std = population stddev "
        "across models. Larger spread/std → the test more cleanly separates "
        "models. `translation*` rows reuse the OLD value for both patterns "
        "(see top of report), so their Δ is 0 by construction.",
        "",
        "| eval | OLD spread | NEW spread | Δ spread | OLD std | NEW std | Δ std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for col in columns:
        olds = [
            v
            for v in (_column_value(m, col, True, old_root, new_root) for m in models)
            if v is not None
        ]
        news = [
            v
            for v in (_column_value(m, col, False, old_root, new_root) for m in models)
            if v is not None
        ]
        if not olds or not news:
            continue
        os_, ns_ = max(olds) - min(olds), max(news) - min(news)
        ostd = statistics.pstdev(olds) if len(olds) > 1 else 0.0
        nstd = statistics.pstdev(news) if len(news) > 1 else 0.0
        marker = "*" if col in PINNED_TO_OLD else ""
        lines.append(
            f"| `{_short_label(col)}{marker}` | {os_:.2f} | {ns_:.2f} | {ns_ - os_:+.2f} "
            f"| {ostd:.2f} | {nstd:.2f} | {nstd - ostd:+.2f} |"
        )
    lines.append("")
    return lines


def _build_swings_table(
    columns: List[str],
    models: List[str],
    old_root: Path,
    new_root: Path,
    top_n: int,
) -> List[str]:
    rows: List[Tuple[str, str, float, float, float]] = []
    for col in columns:
        if col in PINNED_TO_OLD or col in AGGREGATE_COLUMNS:
            # Pinned columns have Δ=0 by construction; aggregate columns are
            # already covered by the underlying base evals. Skip both.
            continue
        for m in models:
            old = _column_value(m, col, True, old_root, new_root)
            new = _column_value(m, col, False, old_root, new_root)
            if old is None or new is None:
                continue
            rows.append((col, m, old, new, new - old))
    rows.sort(key=lambda r: abs(r[4]), reverse=True)

    lines = [
        f"## Biggest accuracy swings (|Δ|), top {top_n}",
        "",
        "Excludes `translation*` (pinned) and the synthetic `turnToLook avg` column.",
        "",
        "| eval | model | OLD | NEW | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for ev, m, old, new, delta in rows[:top_n]:
        lines.append(f"| `{_short_label(ev)}` | `{m}` | {old:.2f} | {new:.2f} | {delta:+.2f} |")
    lines.append("")
    return lines


def _load_gt_mean(root: Path, eval_name: str) -> Optional[float]:
    """Load the GT (real-video) mean accuracy for ``eval_name`` under ``root``.

    GT result trees use ``<root>/<eval_name>/stats.json`` (no model component),
    in contrast to the model-keyed ``<root>/<model>_<eval_name>/`` layout.
    """
    p = root / eval_name / "stats.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text()).get("mean"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _gt_verdict(new_gt: Optional[float]) -> str:
    if new_gt is None:
        return "—"
    if new_gt >= GT_VERDICT_OK:
        return "OK"
    if new_gt >= GT_VERDICT_WARN:
        return "WARN"
    return "FAIL"


def _build_gt_validation_table(
    evals: List[str],
    old_gt_root: Path,
    new_gt_root: Path,
) -> List[str]:
    """Build the GT-validation section appended to the comparison report.

    For each eval (translation pinned), compares the OLD-pattern GT mean from
    ``old_gt_root/<eval>/stats.json`` against the NEW-pattern (late-episode)
    GT mean from ``new_gt_root/<eval>/stats.json``. The verdict column flags
    whether the late-episode question is well-posed on real videos:
    a small drop is fine, a large drop means the chosen frame is ambiguous
    for that eval and the handler's late-frame logic needs adjustment.
    """
    lines = [
        "## GT validation: is the late-episode question well-posed?",
        "",
        f"OLD GT: `{old_gt_root}/<eval>/stats.json` — current-pattern accuracy on real videos.\n"
        f"NEW GT: `{new_gt_root}/<eval>/stats.json` — late-episode toggle on real videos "
        f"(generated via `validate_late_episode_gt.sh`).",
        "",
        f"Verdict: `OK` if NEW GT ≥ {GT_VERDICT_OK:.0f}%, `WARN` in "
        f"[{GT_VERDICT_WARN:.0f}%, {GT_VERDICT_OK:.0f}%), `FAIL` below {GT_VERDICT_WARN:.0f}%. "
        f"A `WARN`/`FAIL` row points back to the handler's late-frame choice "
        f"for that eval (see `frame_extraction/<eval>/_real/`). "
        f"`translationEval` is `n/a (pinned)` because the dataset reverses "
        f"actions in the latter half of the episode.",
        "",
        "| eval | OLD GT (current) | NEW GT (late) | Δ | verdict |",
        "|---|---:|---:|---:|:---:|",
    ]
    for ev in evals:
        if ev in PINNED_TO_OLD:
            old = _load_gt_mean(old_gt_root, ev)
            old_str = f"{old:.2f}" if old is not None else "—"
            lines.append(
                f"| `{_short_label(ev)}` | {old_str} | n/a (pinned) | n/a | n/a |"
            )
            continue
        old = _load_gt_mean(old_gt_root, ev)
        new = _load_gt_mean(new_gt_root, ev)
        old_str = f"{old:.2f}" if old is not None else "—"
        new_str = f"{new:.2f}" if new is not None else "—"
        delta_str = f"{new - old:+.2f}" if (old is not None and new is not None) else "—"
        lines.append(
            f"| `{_short_label(ev)}` | {old_str} | {new_str} | {delta_str} "
            f"| {_gt_verdict(new)} |"
        )
    lines.append("")
    return lines


def _columns_for_wide_tables(evals: List[str]) -> List[str]:
    """Insert the synthetic turnToLook avg column right after the two base evals."""
    cols: List[str] = []
    inserted = False
    for ev in evals:
        cols.append(ev)
        if (
            not inserted
            and ev == "turnToLookOppositeEval"
            and "turnToLookEval" in evals
        ):
            cols.append(TURN_TO_LOOK_AVG)
            inserted = True
    if not inserted and {"turnToLookEval", "turnToLookOppositeEval"}.issubset(evals):
        cols.append(TURN_TO_LOOK_AVG)
    return cols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW)
    parser.add_argument(
        "--old-gt-root",
        type=Path,
        default=DEFAULT_OLD_GT,
        help="Root containing OLD-pattern GT stats (<root>/<eval>/stats.json).",
    )
    parser.add_argument(
        "--new-gt-root",
        type=Path,
        default=DEFAULT_NEW_GT,
        help="Root containing NEW-pattern (late-episode) GT stats.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evals", nargs="+", default=DEFAULT_EVALS)
    parser.add_argument("--models", nargs="+", help="Override discovered model list.")
    parser.add_argument("--top-swings", type=int, default=15)
    args = parser.parse_args()

    if not args.new_root.exists():
        raise SystemExit(f"NEW root not found: {args.new_root}")
    if not args.old_root.exists():
        raise SystemExit(f"OLD root not found: {args.old_root}")

    models = args.models or _discover_models(args.new_root, args.evals)
    if not models:
        raise SystemExit(f"No models discovered under {args.new_root}")

    columns = _columns_for_wide_tables(args.evals)

    out_lines: List[str] = []
    out_lines.append("# Late-episode vs current-pattern comparison\n")
    out_lines.append(f"- OLD (current pattern): `{args.old_root}`")
    out_lines.append(f"- NEW (late-episode toggle): `{args.new_root}`")
    out_lines.append(f"- Models ({len(models)}): {', '.join(f'`{m}`' for m in models)}")
    out_lines.append(f"- Evals: {', '.join(f'`{e}`' for e in args.evals)}")
    out_lines.append("")
    out_lines.append(
        "Notes:\n"
        f"- `translation*` is pinned to the OLD value in the NEW table (and Δ=0). "
        f"The dataset reverses bot actions in the second half of the episode, so "
        f"a late-horizon comparison frame is ill-posed.\n"
        f"- `turnToLook avg` averages `turnToLookEval` and `turnToLookOppositeEval` "
        f"per (model, pattern)."
    )
    out_lines.append("")

    out_lines.append("## Per-model accuracy across evals\n")
    out_lines.extend(
        _build_wide_table(
            "Mean episode accuracy (%) — OLD pattern (current)",
            columns,
            models,
            cell_fn=lambda m, c: _column_value(m, c, True, args.old_root, args.new_root),
        )
    )
    out_lines.extend(
        _build_wide_table(
            "Mean episode accuracy (%) — NEW pattern (late-episode toggle)",
            columns,
            models,
            cell_fn=lambda m, c: _column_value(m, c, False, args.old_root, args.new_root),
            note=(
                "`translation*` reuses the OLD value; the late-episode pattern is "
                "not meaningful when the dataset reverses actions in the latter "
                "half of the episode."
            ),
        )
    )

    def _delta(m: str, c: str) -> Optional[float]:
        old = _column_value(m, c, True, args.old_root, args.new_root)
        new = _column_value(m, c, False, args.old_root, args.new_root)
        if old is None or new is None:
            return None
        return new - old

    out_lines.extend(
        _build_wide_table(
            "Δ (NEW − OLD) — late-episode toggle effect",
            columns,
            models,
            cell_fn=_delta,
            signed=True,
        )
    )

    out_lines.extend(_build_discriminativeness_table(columns, models, args.old_root, args.new_root))
    out_lines.extend(_build_swings_table(columns, models, args.old_root, args.new_root, args.top_swings))

    if args.new_gt_root.exists() and args.old_gt_root.exists():
        out_lines.extend(
            _build_gt_validation_table(args.evals, args.old_gt_root, args.new_gt_root)
        )
    else:
        missing = [
            str(p) for p in (args.old_gt_root, args.new_gt_root) if not p.exists()
        ]
        print(
            "Skipping GT validation table; missing root(s): "
            + ", ".join(missing)
            + ". Run `validate_late_episode_gt.sh` first."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines))
    print(f"Wrote comparison report to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
