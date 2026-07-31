#!/usr/bin/env python3
"""
Generate bar plots comparing visual similarity metrics between Look-Same
(turnToLookEval) and Look-Opposite (turnToLookOppositeEval) conditions.

Reads per-episode JSON results from results_consistency_metrics/ and produces:
  - A multi-panel bar plot (one subplot per model, ordered by avg thresholding accuracy)
  - Per-metric best 1-D thresholding accuracy annotations

Usage:
  python plot_consistency_metrics.py [--results-dir DIR] [--output PATH]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRIC_NAMES = ["lpips", "clip_cosine_sim", "dinov2_cosine_sim", "dinov3_cosine_sim"]
METRIC_LABELS = ["1−LPIPS", "CLIP", "DINOv2", "DINOv3"]
EVAL_TYPES = ["turnToLookEval", "turnToLookOppositeEval"]
EVAL_LABELS = {"turnToLookEval": "Look-Same", "turnToLookOppositeEval": "Look-Opposite"}
BAR_COLORS = {"turnToLookEval": "#4C72B0", "turnToLookOppositeEval": "#DD8452"}

MODEL_DISPLAY_NAMES = {
    "flagship": "Solaris",
    "no_kv_cache_backprop": "Solaris w/o KV-BP",
    "causvid_dmd": "Solaris w/ Pre-DMD",
    "causvid_regression": "ODE Reg",
    "from_scratch": "Solaris w/o pretrain",
    "concat_c": "Frame concat",
}

PLOT_GROUPS = [
    {
        "title": "Table 2",
        "suffix": "table2",
        "models": ["flagship", "no_kv_cache_backprop", "causvid_dmd", "causvid_regression"],
    },
    {
        "title": "Table 3",
        "suffix": "table3",
        "models": ["flagship", "from_scratch", "concat_c"],
    },
]


def best_1d_thresholding_accuracy(
    vals_high: np.ndarray, vals_low: np.ndarray
) -> float:
    """Best thresholding accuracy with a fixed direction.

    Assumes vals_high should be >= threshold (positive class) and
    vals_low should be < threshold (negative class).  This prevents
    inflated numbers when the metric is anti-correlated with the
    hypothesis — accuracy will simply drop below 50%.
    """
    all_vals = np.concatenate([vals_high, vals_low])
    labels = np.concatenate([np.ones(len(vals_high)), np.zeros(len(vals_low))])
    unique = np.unique(all_vals)
    thresholds = np.concatenate(
        [[unique[0] - 1], (unique[:-1] + unique[1:]) / 2, [unique[-1] + 1]]
    )
    return max(np.mean((all_vals >= t) == labels) for t in thresholds)


def load_all_results(results_dir: Path):
    """Load per-episode JSON results and discover models."""
    models = sorted(
        d.name.removesuffix("_turnToLookEval")
        for d in results_dir.iterdir()
        if d.is_dir() and d.name.endswith("_turnToLookEval")
    )
    all_data = {}
    for mdl in models:
        for et in EVAL_TYPES:
            mf = results_dir / f"{mdl}_{et}" / "metrics.json"
            if not mf.exists():
                raise FileNotFoundError(f"Missing results: {mf}")
            all_data[(mdl, et)] = json.loads(mf.read_text())
    return models, all_data


def compute_lc_accuracies(models, all_data):
    """Compute per-(model, metric) and average-per-model thresholding accuracies.

    Direction is fixed to the hypothesis: Look-Same should have higher
    similarity (or lower LPIPS distance) than Look-Opposite.
    """
    lc_accs = {}
    for mdl in models:
        for m in METRIC_NAMES:
            vs = np.array([ep[m] for ep in all_data[(mdl, EVAL_TYPES[0])]["per_episode"]])
            vo = np.array([ep[m] for ep in all_data[(mdl, EVAL_TYPES[1])]["per_episode"]])
            if m == "lpips":
                # LPIPS is a distance: Look-Same should be LOWER.
                # Flip so vals_high = Look-Opposite, vals_low = Look-Same
                # (i.e., threshold: predict "opposite" if lpips >= t)
                lc_accs[(mdl, m)] = best_1d_thresholding_accuracy(vo, vs)
            else:
                # Similarity metrics: Look-Same should be HIGHER
                lc_accs[(mdl, m)] = best_1d_thresholding_accuracy(vs, vo)

    avg_lc = {mdl: np.mean([lc_accs[(mdl, m)] for m in METRIC_NAMES]) for mdl in models}
    return lc_accs, avg_lc


def plot(models_sorted, all_data, lc_accs, avg_lc, output_path: Path,
         group_title: str = ""):
    n_models = len(models_sorted)
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = np.atleast_1d(axes).flatten()

    bar_width = 0.35
    x = np.arange(len(METRIC_NAMES))

    for idx, mdl in enumerate(models_sorted):
        ax = axes[idx]
        display_name = MODEL_DISPLAY_NAMES.get(mdl, mdl)
        for j, et in enumerate(EVAL_TYPES):
            data = all_data[(mdl, et)]
            means = [data["stats"][m]["mean"] for m in METRIC_NAMES]
            stds = [data["stats"][m]["std"] for m in METRIC_NAMES]
            means[0] = 1.0 - means[0]
            offset = -bar_width / 2 + j * bar_width
            ax.bar(
                x + offset, means, bar_width, yerr=stds,
                label=EVAL_LABELS[et], color=BAR_COLORS[et],
                capsize=4, edgecolor="white", linewidth=0.5,
                error_kw=dict(lw=1.2),
            )

        for i, m in enumerate(METRIC_NAMES):
            acc = lc_accs[(mdl, m)]
            vals = []
            for et in EVAL_TYPES:
                mean_v = all_data[(mdl, et)]["stats"][m]["mean"]
                std_v = all_data[(mdl, et)]["stats"][m]["std"]
                if m == "lpips":
                    mean_v = 1.0 - mean_v
                vals.append(mean_v + std_v)
            y_max = max(vals)
            ax.text(
                i, min(y_max + 0.03, 1.05), f"Thresh acc.: {acc * 100:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_LABELS, fontsize=11)
        ax.set_title(
            f"{display_name}  (avg thresh acc.: {avg_lc[mdl] * 100:.1f}%)",
            fontsize=13, fontweight="bold",
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Similarity")
        ax.legend(fontsize=9, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    suptitle = group_title or "Consistency Metrics: Look-Same vs Look-Opposite"
    fig.suptitle(
        f"{suptitle}\n"
        "(bars = mean, whiskers = 1 std dev, thresh acc. = best 1-D thresholding accuracy; "
        "models ordered by avg thresh acc.)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results_consistency_metrics"),
        help="Directory containing per-model metric JSONs (default: results_consistency_metrics)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output image path (default: <results-dir>/consistency_barplots_<suffix>.png)",
    )
    args = parser.parse_args()

    models, all_data = load_all_results(args.results_dir)
    lc_accs, avg_lc = compute_lc_accuracies(models, all_data)

    for group in PLOT_GROUPS:
        group_models = [m for m in group["models"] if m in models]
        if not group_models:
            continue
        group_sorted = sorted(group_models, key=lambda m: avg_lc[m], reverse=True)
        out = args.output or (args.results_dir / f"consistency_barplots_{group['suffix']}.png")
        plot(group_sorted, all_data, lc_accs, avg_lc, out, group_title=group["title"])

        print(f"\n--- {group['title']}: Avg Thresholding Accuracy (descending) ---")
        for mdl in group_sorted:
            display = MODEL_DISPLAY_NAMES.get(mdl, mdl)
            per_metric = "  ".join(
                f"{ml}={lc_accs[(mdl, m)] * 100:.1f}%"
                for ml, m in zip(METRIC_LABELS, METRIC_NAMES)
            )
            print(f"  {display:25s}  avg={avg_lc[mdl] * 100:.1f}%  ({per_metric})")


if __name__ == "__main__":
    main()
