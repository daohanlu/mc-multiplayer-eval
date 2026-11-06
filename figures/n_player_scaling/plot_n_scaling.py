"""N-player memory and time scaling plots (NeurIPS 2026, Fig. 9 / Sec. 7).

Compares splash_attention with jax.nn.dot_product_attention ("Standard Attention")
across both model widths (full and 25%):

- splash_attention            -> BLUE
- dot_product_attention       -> RED
- full width                  -> solid line
- 25% width                   -> dashed line
- v5p chip HBM (95.74 GiB)    -> dashed gray horizontal line on the memory plot
- first OOM N                 -> filled marker at the OOM N, connected to the last fit by
                                 a segment in the curve's own line style, labeled "OOM"

Outputs in plots/ (next to this script):
  plot_scaling.{pdf,png}    -- the paper figure: (A) memory and (B) time side by side. Each
                               panel is drawn at the original 13 x 7.5 in size and scaled down
                               by LaTeX to the text width, so it looks like the original slide
                               figure, as vector graphics.
  plot_memory.{pdf,png,svg} -- standalone peak HBM vs. N
  plot_time.{pdf,png,svg}   -- standalone time per training iteration vs. N

Inputs (CSVs in data/), measured on TPU v5p with random inputs, which cost the same
memory and time as real gameplay data:
  np_experiments_data.csv                          full-width splash_attention
  np_experiments_no_splash_data.csv                full-width dot_product_attention
  np_experiments_quarter_width_splash_data.csv     25% width splash_attention
  np_experiments_quarter_width_no_splash_data.csv  25% width dot_product_attention

Usage:
  python figures/n_player_scaling/plot_n_scaling.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "plots"

HBM_LIMIT_GIB = 95.74     # v5p chip HBM (nominal)
MEM_YLIM = (0, 128)       # memory y-axis cap (above HBM line)
HBM_LABEL_X = 9.5         # HBM label starts here, below the line, clear of every curve

BLUE = "#4C72B0"          # splash_attention
RED = "#C44E52"           # dot_product_attention
GRAY = "#8C8C8C"

PANEL_SIZE = (13, 7.5)    # inches, per standalone panel
PAPER_PANEL_HEIGHT = 8.7  # inches; a little taller, matching the proportions of the original figure
PAPER_LEFT = 0.05         # left edge of panel A's plot area, as a fraction of the figure width
PAPER_WSPACE = 0.14       # gap between the panels, as a fraction of the plot width
PAPER_TAG_X = -0.085      # (A)/(B) start above the y-axis label, in axes coordinates
LINE_WIDTH = 3
MARKER_SIZE = 8
FONT_SIZE_TITLE = 28
FONT_SIZE_AXIS = 24
FONT_SIZE_LEGEND = 18

# Avenir is the original font. Without it, DejaVu Sans (TrueType) is the closest in width;
# OpenType CFF fonts such as Nimbus Sans do not embed cleanly as TrueType.
FONT_FAMILY = ["Avenir", "DejaVu Sans", "Liberation Sans"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "pdf.fonttype": 42,        # embed TrueType, not Type 3
    "ps.fonttype": 42,
    "svg.fonttype": "none",    # keep SVG text as text
    "font.size": FONT_SIZE_AXIS,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.titleweight": "normal",  # the original figure rendered its titles in regular weight
    "axes.labelsize": FONT_SIZE_AXIS,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": FONT_SIZE_AXIS - 6,
    "ytick.labelsize": FONT_SIZE_AXIS - 6,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#dddddd",
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def _load_curves():
    """Returns list of (df, color, linestyle, label) tuples for all 4 curves."""
    fw_sa = pd.read_csv(DATA / "np_experiments_data.csv")
    fw_sa = fw_sa[fw_sa["per_chip_batch"] == 1]
    fw_dpa = pd.read_csv(DATA / "np_experiments_no_splash_data.csv")
    qw_sa = pd.read_csv(DATA / "np_experiments_quarter_width_splash_data.csv")
    qw_dpa = pd.read_csv(DATA / "np_experiments_quarter_width_no_splash_data.csv")
    return [
        (fw_dpa, RED, "-",  "Standard Attention (full width)"),
        (fw_sa,  BLUE, "-", "Splash Attention (full width)"),
        (qw_dpa, RED, "--", "Standard Attention (25% width)"),
        (qw_sa,  BLUE, "--", "Splash Attention (25% width)"),
    ]


def _all_n(curves):
    """Union of all N values across all curves -- both fits and first OOM."""
    n = set()
    for df, *_ in curves:
        fits = df[df["status"] == "fit"]
        ooms = df[df["status"] == "OOM"].sort_values("N")
        n.update(int(x) for x in fits["N"])
        if not ooms.empty:
            n.add(int(ooms["N"].iloc[0]))
    return sorted(n)


def _oom_label(ax, x, y, color):
    ax.annotate(
        "OOM", xy=(x, y), xytext=(8, 8), textcoords="offset points",
        color=color, fontsize=FONT_SIZE_LEGEND, fontweight="bold",
    )


def draw_memory(ax):
    curves = _load_curves()
    x_values = _all_n(curves)

    ax.axhline(HBM_LIMIT_GIB, color=GRAY, linestyle="--", linewidth=2)
    ax.text(
        HBM_LABEL_X, HBM_LIMIT_GIB - 4,
        f"v5p chip HBM ({HBM_LIMIT_GIB:.1f} GiB)",
        color=GRAY, fontsize=FONT_SIZE_LEGEND - 2, style="italic",
        ha="left", va="top",
    )

    for df, color, linestyle, label in curves:
        fits = df[df["status"] == "fit"].sort_values("N")
        ax.plot(fits["N"], fits["peak_hbm_gib"], marker="o", color=color,
                linestyle=linestyle, label=label)

        ooms = df[df["status"] == "OOM"].sort_values("N")
        if not ooms.empty and not fits.empty:
            n_oom = float(ooms["N"].iloc[0])
            n_last = float(fits["N"].iloc[-1])
            peak_last = float(fits["peak_hbm_gib"].iloc[-1])
            # connector segment in the curve's own line style, not dotted
            ax.plot([n_last, n_oom], [peak_last, HBM_LIMIT_GIB], color=color,
                    linestyle=linestyle, zorder=4)
            # filled marker at the chip HBM line (no quantitative claim)
            ax.plot([n_oom], [HBM_LIMIT_GIB], marker="o", color=color, linestyle="None",
                    markersize=MARKER_SIZE + 2, zorder=5)
            _oom_label(ax, n_oom, HBM_LIMIT_GIB, color)

    ax.set_xlabel("Number of players (N)")
    ax.set_ylabel("Peak Memory (GiB)")
    ax.set_xticks(x_values)
    ax.set_ylim(*MEM_YLIM)
    ax.set_title("Peak HBM Memory vs. Number of Players")
    ax.tick_params(length=0)
    ax.legend(loc="lower right")


def draw_time(ax):
    curves = _load_curves()
    x_values = _all_n(curves)

    for df, color, linestyle, label in curves:
        fits = df[df["status"] == "fit"].sort_values("N")
        ax.plot(fits["N"], fits["sec_per_step"], marker="o", color=color,
                linestyle=linestyle, label=label)

        ooms = df[df["status"] == "OOM"].sort_values("N")
        if not ooms.empty and len(fits) >= 2:
            n_oom = float(ooms["N"].iloc[0])
            # extrapolate time at the OOM N from the last two fitting points
            x = fits["N"].astype(float).iloc[-2:].values
            y = fits["sec_per_step"].astype(float).iloc[-2:].values
            slope = (y[1] - y[0]) / (x[1] - x[0]) if x[1] != x[0] else 0.0
            time_oom = float(y[1] + slope * (n_oom - x[1]))
            n_last = float(fits["N"].iloc[-1])
            time_last = float(fits["sec_per_step"].iloc[-1])
            # connector in the curve's own line style
            ax.plot([n_last, n_oom], [time_last, time_oom], color=color,
                    linestyle=linestyle, zorder=4)
            # filled marker at the extrapolated time
            ax.plot([n_oom], [time_oom], marker="o", color=color, linestyle="None",
                    markersize=MARKER_SIZE + 2, zorder=5)
            _oom_label(ax, n_oom, time_oom, color)

    ax.set_xlabel("Number of players (N)")
    ax.set_ylabel("Time per training iter (s)")
    ax.set_xticks(x_values)
    ax.set_title("Time per Training Iteration vs. Number of Players")
    ax.tick_params(length=0)
    ax.legend(loc="upper left")


def _save(fig, name, formats):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / name}.{{{','.join(formats)}}}")


def plot_paper():
    """Both panels side by side at the original panel size, labeled (A) and (B)."""
    fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=(2 * PANEL_SIZE[0], PAPER_PANEL_HEIGHT))
    # Fixed margins instead of constrained layout, which left a wider left margin and a
    # wider gap between the panels than the original figure (measured at print width:
    # plot areas ~44% of the width each, a ~6% gap between them).
    fig.subplots_adjust(left=PAPER_LEFT, right=0.995, bottom=0.12, top=0.9, wspace=PAPER_WSPACE)
    draw_memory(ax_mem)
    draw_time(ax_time)
    for ax, tag in ((ax_mem, "(A)"), (ax_time, "(B)")):
        ax.text(PAPER_TAG_X, 1.04, tag, transform=ax.transAxes, fontsize=FONT_SIZE_TITLE,
                ha="left", va="bottom")
    _save(fig, "plot_scaling", ("pdf", "png"))


def plot_standalone():
    """The two plots as separate figures."""
    for name, draw in (("plot_memory", draw_memory), ("plot_time", draw_time)):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        draw(ax)
        _save(fig, name, ("png", "pdf", "svg"))


if __name__ == "__main__":
    plot_paper()
    plot_standalone()
