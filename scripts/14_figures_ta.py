"""Figures for the title + abstract measure: share per year and change.

fig4: per topic, share 2024 (hollow) -> 2025 (small) -> 2026 (filled), log x, numbers at right; colour = trend call.
fig5: size vs change: x = share 2025, y = annual odds multiplier (95% CI), colour = trend call.
Colours: diverging pair of the reference palette (blue rising, red declining) + neutral gray (unclear).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
Y = config.YEARS

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"rising": "#2a78d6", "declining": "#e34948", "unclear": "#7c7b77"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "font.family": "DejaVu Sans"})
d = pd.read_csv(os.path.join(ROOT, "results", "shares_and_change_ta.csv"))
TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30]


def fig_shares():
    t = d.sort_values(f"share_{config.REF_YEAR}")
    fig, ax = plt.subplots(figsize=(11, 14))
    for i, r in enumerate(t.itertuples()):
        c = COL[r.call]
        vals = [getattr(r, f"share_{y}") for y in Y]
        ax.plot([vals[0], vals[-1]], [i, i], color=c, lw=1.8, alpha=0.9)
        for v in vals[1:-1]:
            ax.plot(v, i, "o", ms=3, color=c, alpha=0.8)
        ax.plot(vals[0], i, "o", ms=6, mfc=SURF, mec=c, mew=1.6)
        ax.plot(vals[-1], i, "o", ms=7, color=c, mec=SURF, mew=0.8)
        ax.text(1.01, i, f"{vals[0]:4.1f} → {getattr(r, 'share_2024'):4.1f} → {vals[-1]:4.1f}%   ({r.change_pp:+.1f})",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.5, color=INK2, family="DejaVu Sans Mono")
    ax.set_xscale("log"); ax.set_xlim(0.03, 40)
    ax.set_xticks([0.05] + TICKS); ax.set_xticklabels([f"{x:g}%" for x in [0.05] + TICKS])
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.topic, fontsize=8); ax.set_ylim(-0.8, len(t) - 0.2)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.text(1.01, len(t) - 0.1, f"{Y[0]} → 2024 → {Y[-1]}   (change, points)", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xlabel("Share of papers whose title or abstract names the topic (log scale)")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, mew=1.6, ms=6, label=str(Y[0])),
               plt.Line2D([], [], marker="o", ls="", color=INK2, ms=3, label=f"{Y[1]}–{Y[-2]}"),
               plt.Line2D([], [], marker="o", ls="", color=INK2, ms=7, label=str(Y[-1])),
               plt.Line2D([], [], color=COL["rising"], lw=2, label="rising (q<0.05)"),
               plt.Line2D([], [], color=COL["declining"], lw=2, label="declining (q<0.05)"),
               plt.Line2D([], [], color=COL["unclear"], lw=2, label="unclear")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(0.0, -0.075))
    n = len(pd.read_csv(os.path.join(ROOT, "data", "topic_matrix_ta.csv")))
    fig.suptitle(f"Topic shares at NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV {Y[0]}–{Y[-1]} (title + abstract, {n:,} papers)\n"
                 f"shares standardised to the 2024+2025 venue mix; {Y[-1]} estimated from ICLR, CVPR and ECCV; trend over {Y[0]}–{Y[-1]}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.34, right=0.70, top=0.945, bottom=0.075)
    out = os.path.join(ROOT, "figures", "fig4_shares_title_abstract.png"); fig.savefig(out, dpi=150); print(out)


def fig_size_change():
    fig, ax = plt.subplots(figsize=(13, 9.5))
    for r in d.itertuples():
        c = COL[r.call]
        x = getattr(r, f"share_{config.REF_YEAR}")
        ax.plot([x, x], [r.annual_OR_lo, r.annual_OR_hi], color=c, lw=1, alpha=0.35)
        ax.plot(x, r.annual_OR, "o", ms=7, color=c, mfc=c if r.call != "unclear" else SURF, mew=1.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    lo, hi = d.annual_OR.min(), d.annual_OR.max()
    ax.set_xlim(0.25, 40); ax.set_ylim(min(0.5, lo / 1.3), max(10, hi * 1.3))   # axis follows the point estimates
    ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5%", "1%", "2%", "5%", "10%", "20%"])
    yt = [v for v in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 15, 30, 60) if min(0.5, lo / 1.3) <= v <= max(10, hi * 1.3)]
    ax.set_yticks(yt); ax.set_yticklabels([f"×{v:g}" for v in yt])
    ax.axhline(1, color=INK2, lw=1)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # greedy label placement: try a few offsets, keep the first that does not overlap earlier labels
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); placed = []
    order = d.assign(k=d[f"share_{config.REF_YEAR}"]).sort_values("k", ascending=False)
    order = order[(order.call != "unclear") | (order.k >= 10)]   # selective labels
    for r in order.itertuples():
        for dx, dy, ha, va in ((6, 0, "left", "center"), (-6, 0, "right", "center"), (0, 7, "center", "bottom"),
                               (0, -7, "center", "top"), (6, 8, "left", "bottom"), (6, -8, "left", "top"),
                               (-6, 8, "right", "bottom"), (-6, -8, "right", "top")):
            txt = ax.annotate(r.topic, (r.k, r.annual_OR), xytext=(dx, dy), textcoords="offset points",
                              ha=ha, va=va, fontsize=7, color=INK)
            bb = txt.get_window_extent(rend).expanded(1.02, 1.08)
            if not any(bb.overlaps(p) for p in placed):
                placed.append(bb); break
            txt.remove()
        else:
            txt = ax.annotate(r.topic, (r.k, r.annual_OR), xytext=(6, 0), textcoords="offset points",
                              ha="left", va="center", fontsize=6.5, color=INK2)
            placed.append(txt.get_window_extent(rend))
    ax.set_xlabel(f"Share of papers in {config.REF_YEAR} (title + abstract, all six venues; log scale)")
    ax.set_ylabel(f"Change in share per year {Y[0]}–{Y[-1]} (odds multiplier, within venue, 95% CI; log scale)")
    ax.set_title(f"Size and change of ML / CV topics, {Y[0]}–{Y[-1]} (filled: significant after FDR; blue rising, red declining; hollow gray: unclear)\n"
                 "labels: every significant topic and every topic above 10%; unlabelled hollow points are the remaining topics, listed in Fig. 4",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig5_size_vs_change.png"); fig.savefig(out, dpi=150); print(out)


if __name__ == "__main__":
    fig_shares()
    fig_size_change()
