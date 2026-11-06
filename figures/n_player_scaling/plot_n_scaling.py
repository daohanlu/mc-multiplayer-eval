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
  plot_scaling.{pdf,png}    -- the paper figure: (A) memory and (B) time side by side,
                               drawn at the paper's text width (5.5 in) with print-size fonts
  plot_memory.{pdf,png,svg} -- standalone peak HBM vs. N (13 x 7.5 in, slide-size fonts)
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

BLUE = "#4C72B0"          # splash_attention
RED = "#C44E52"           # dot_product_attention
GRAY = "#8C8C8C"

# Avenir was the original font; the rest are fallbacks for machines without it.
# Liberation Sans (TrueType, Arial metrics) comes before Nimbus Sans, whose OpenType CFF
# outlines do not embed cleanly as TrueType.
FONT_FAMILY = ["Avenir", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]

# Standalone slide-size figures (the original layout).
SLIDE = dict(figsize=(13, 7.5), line=3, marker=8, title=28, axis=24, tick=18, legend=18)
# Paper figure: two panels at the NeurIPS text width, fonts close to the 9 pt caption.
PAPER = dict(figsize=(5.5, 2.15), line=1.2, marker=3, title=8, axis=7.5, tick=6.5, legend=5.5)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "pdf.fonttype": 42,        # embed TrueType, not Type 3
    "ps.fonttype": 42,
    "svg.fonttype": "none",    # keep SVG text as text
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#dddddd",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
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


def _style_axes(ax, s, x_values, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=s["axis"])
    ax.set_ylabel(ylabel, fontsize=s["axis"])
    ax.set_xticks(x_values)
    ax.tick_params(length=0, labelsize=s["tick"])
    ax.set_title(title, fontsize=s["title"])


def draw_memory(ax, s, title="Peak HBM Memory vs. Number of Players", legend=True, hbm_label_x=None):
    curves = _load_curves()
    x_values = _all_n(curves)

    ax.axhline(HBM_LIMIT_GIB, color=GRAY, linestyle="--", linewidth=s["line"] * 2 / 3)
    ax.text(
        min(x_values) + 0.05 if hbm_label_x is None else hbm_label_x, HBM_LIMIT_GIB - 4,
        f"v5p chip HBM ({HBM_LIMIT_GIB:.1f} GiB)",
        color=GRAY, fontsize=s["legend"] - 1, style="italic", ha="left", va="top",
    )

    for df, color, linestyle, label in curves:
        fits = df[df["status"] == "fit"].sort_values("N")
        ax.plot(fits["N"], fits["peak_hbm_gib"], marker="o", color=color,
                linestyle=linestyle, linewidth=s["line"], markersize=s["marker"], label=label)

        ooms = df[df["status"] == "OOM"].sort_values("N")
        if not ooms.empty and not fits.empty:
            n_oom = float(ooms["N"].iloc[0])
            n_last = float(fits["N"].iloc[-1])
            peak_last = float(fits["peak_hbm_gib"].iloc[-1])
            # connector segment in the curve's own line style, not dotted
            ax.plot([n_last, n_oom], [peak_last, HBM_LIMIT_GIB], color=color,
                    linestyle=linestyle, linewidth=s["line"], zorder=4)
            # filled marker at the chip HBM line (no quantitative claim)
            ax.plot([n_oom], [HBM_LIMIT_GIB], marker="o", color=color, linestyle="None",
                    markersize=s["marker"] * 1.25, zorder=5)
            ax.annotate("OOM", xy=(n_oom, HBM_LIMIT_GIB), xytext=(3, 3),
                        textcoords="offset points", color=color,
                        fontsize=s["legend"], style="italic", fontweight="bold")

    _style_axes(ax, s, x_values, "Number of players (N)", "Peak Memory (GiB)", title)
    ax.set_ylim(*MEM_YLIM)
    if legend:
        ax.legend(loc="lower right", fontsize=s["legend"])


def draw_time(ax, s, title="Time per Training Iteration vs. Number of Players", legend=True):
    curves = _load_curves()
    x_values = _all_n(curves)

    for df, color, linestyle, label in curves:
        fits = df[df["status"] == "fit"].sort_values("N")
        ax.plot(fits["N"], fits["sec_per_step"], marker="o", color=color,
                linestyle=linestyle, linewidth=s["line"], markersize=s["marker"], label=label)

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
                    linestyle=linestyle, linewidth=s["line"], zorder=4)
            # filled marker at the extrapolated time
            ax.plot([n_oom], [time_oom], marker="o", color=color, linestyle="None",
                    markersize=s["marker"] * 1.25, zorder=5)
            ax.annotate("OOM", xy=(n_oom, time_oom), xytext=(3, 3),
                        textcoords="offset points", color=color,
                        fontsize=s["legend"], style="italic", fontweight="bold")

    _style_axes(ax, s, x_values, "Number of players (N)", "Time per training iter (s)", title)
    ax.set_ylim(bottom=0)
    if legend:
        ax.legend(loc="upper left", fontsize=s["legend"])


def _save(fig, name, formats):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / name}.{{{','.join(formats)}}}")


def plot_paper():
    """Both panels side by side, sized for the paper's text width."""
    s = PAPER
    fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=s["figsize"], constrained_layout=True)
    # One shared legend below both panels (the four curves are the same in each), and
    # the HBM label moved right, under the dashed line, where no curve crosses it.
    draw_memory(ax_mem, s, title="(A) Peak HBM Memory vs. Number of Players",
                legend=False, hbm_label_x=9.5)
    draw_time(ax_time, s, title="(B) Time per Training Iteration vs. Number of Players",
              legend=False)
    handles, labels = ax_mem.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=s["legend"] + 0.5,
               columnspacing=1.2, handlelength=1.8, handletextpad=0.5, borderpad=0.3)
    _save(fig, "plot_scaling", ("pdf", "png"))


def plot_standalone():
    """The original two slide-size figures."""
    for name, draw in (("plot_memory", draw_memory), ("plot_time", draw_time)):
        fig, ax = plt.subplots(figsize=SLIDE["figsize"])
        draw(ax, SLIDE)
        _save(fig, name, ("png", "pdf", "svg"))


if __name__ == "__main__":
    plot_paper()
    plot_standalone()
