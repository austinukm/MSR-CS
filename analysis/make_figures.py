"""Figures for the IEEE Access manuscript (vector PDF, IEEE column widths, Times-like font).

fig_rel.pdf      paired NDCG@10 difference of every mixer to the selective Mamba mixer, one panel per dataset
fig_use.pdf      (a) test-time placebos: loss under permutation vs. loss without the last item
                 (b) NDCG@10 by history-length tertile on Sports and MovieLens-1M
fig_dim.pdf      embedding size 64 -> 128 on Sports and MovieLens-1M
Run: python analysis/make_figures.py   (reads results/runs/*.json and results/results.json)
"""
import glob, json, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
D = os.environ.get("S1_RESULTS", os.path.join(ROOT, "results"))
OUT = f"{D}/figures"
os.makedirs(OUT, exist_ok=True)

# ---- style: IEEE widths (3.5 in single column, 7.16 in double), Times-like text, recessive chrome
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.0, "xtick.major.size": 2.5,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SSM, SIMPLE, REF = "#2a78d6", "#eb6834", "#8c8b86"        # palette slots 1, 2 and a neutral grey
FAMILY = {"last": SIMPLE, "mean": SIMPLE, "ema": SIMPLE, "lti": SSM, "mamba": SSM, "sasrec": REF, "gru": REF}
MARK = {"last": "v", "mean": "s", "ema": "o", "lti": "D", "mamba": "*", "sasrec": "^", "gru": "P"}
NAME = {"last": "Last item", "mean": "Last + mean", "ema": "EMA (recency)", "lti": "Mamba w/o selectivity",
        "mamba": "Mamba (selective)", "sasrec": "SASRec", "gru": "GRU4Rec"}
DSN = {"baby": "Baby", "sports": "Sports", "clothing": "Clothing", "ml1m": "MovieLens-1M", "elec": "Electronics"}
MEDLEN = {"baby": 6, "sports": 6, "clothing": 6, "ml1m": 96}

R = collections.defaultdict(dict)
for f in glob.glob(f"{D}/runs/final_*.json"):
    r = json.load(open(f)); R[(r["dataset"], r["model"])][r["seed"]] = r
RES = json.load(open(f"{D}/results.json"))


def clean(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Fig. rel: forest plot
rows = ["ema", "mean", "lti", "last", "sasrec", "gru"]            # top to bottom
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, ds in zip(axes, ["baby", "sports", "clothing", "ml1m"]):
    clean(ax)
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    for y, m in enumerate(rows):
        e = RES["per_dataset_vs_mamba"][f"{ds}|{m}"]
        mu, lo, hi = 100 * e["mean_rel"], 100 * e["ci"][0], 100 * e["ci"][1]
        ax.plot([lo, hi], [y, y], color=FAMILY[m], lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(mu, y, MARK[m], color=FAMILY[m], ms=4.2, mec="white", mew=0.6, zorder=3)
        if m in ("ema", "mean") or ds == "ml1m" and m in ("ema", "lti"):
            ax.text(hi + 1.5, y, f"{mu:+.1f}", va="center", ha="left", fontsize=6.5, color=INK2)
    ax.set_xlim(-44, 16)
    ax.set_xticks([-40, -20, 0])
    ax.set_title(f"{DSN[ds]}", color=INK, pad=11)
    ax.text(0.5, 1.015, f"median sequence length {MEDLEN[ds]}", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.5, color=INK2)
    ax.set_ylim(len(rows) - 0.5, -1.05) if ds == "baby" else None
    ax.text(1.0, -0.78, "Mamba = 0", fontsize=6, color=INK2, ha="left", va="center")
axes[0].set_yticks(range(len(rows)))
axes[0].set_yticklabels([NAME[m] for m in rows])
fig.text(0.56, -0.06, "NDCG@10 difference to the selective Mamba mixer (%), mean and 95% interval over five seeds",
         ha="center", color=INK)
fig.savefig(f"{OUT}/fig_rel.pdf"); fig.savefig(f"{OUT}/fig_rel.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------------ Fig. use: placebos + history length
fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.3})
clean(a, xgrid=False); a.grid(color=GRID, lw=0.5, zorder=0)
a.spines["left"].set_visible(True)
a.plot([0, 90], [0, 90], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
for ds, col in (("baby", SSM), ("sports", SSM), ("clothing", SSM), ("ml1m", SIMPLE)):
    for m in ["last", "mean", "ema", "lti", "mamba", "sasrec", "gru"]:
        rs = R[(ds, m)].values()
        base = np.mean([r["test"]["NDCG@10"] for r in rs])
        px = 100 * (1 - np.mean([r["test"]["placebo_shuffle"]["NDCG@10"] for r in rs]) / base)
        py = 100 * (1 - np.mean([r["test"]["placebo_droplast"]["NDCG@10"] for r in rs]) / base)
        a.plot(px, py, MARK[m], color=col, ms=4.0 if m != "mamba" else 6.2, mec="white", mew=0.5, zorder=3, alpha=0.95)
a.set_xlim(0, 90); a.set_ylim(0, 50); a.set_aspect("auto")
a.set_xlabel("NDCG@10 loss with permuted order (%)")
a.set_ylabel("NDCG@10 loss without last item (%)")
a.text(53, 48, "Amazon (three datasets)", color=SSM, fontsize=7.5, va="top")
a.text(53, 45.2, "near the diagonal: order matters\nmainly through the last item", color=INK2, fontsize=6.3, va="top")
a.text(89, 17.5, "MovieLens-1M", color=SIMPLE, fontsize=7.5, ha="right", va="top")
a.text(89, 14.7, "far right: the order of the\nwhole history matters", color=INK2, fontsize=6.3, ha="right", va="top")
a.text(7.0, 13.5, "equal loss", color=INK2, fontsize=6, rotation=57, ha="center", va="center")
a.set_title("(a) Test-time placebos (no retraining)", loc="left", color=INK, pad=3)
mk = [Line2D([], [], ls="", marker=MARK[m], color=INK2, ms=3.6 if m != "mamba" else 5.2, label=NAME[m])
      for m in ["mamba", "lti", "ema", "mean", "last", "sasrec", "gru"]]
a.legend(handles=mk, fontsize=5.8, frameon=False, loc="upper left", ncol=1, handletextpad=0.2, borderaxespad=0.2,
         labelspacing=0.22)

clean(b, xgrid=False); b.grid(axis="y", color=GRID, lw=0.5, zorder=0); b.spines["left"].set_visible(True)
b.tick_params(axis="y", width=0.6, length=2.5)
series = [("mamba", "-"), ("lti", (0, (3, 1.5))), ("ema", "-"), ("sasrec", "-")]
xs = np.arange(3)
for off, ds in ((0, "sports"), (4, "ml1m")):
    for m, ls in series:
        rs = R[(ds, m)].values()
        v = [np.mean([r["test"]["by_len"][g]["NDCG@10"] for r in rs]) for g in ("short", "medium", "long")]
        norm = [np.mean([r["test"]["by_len"][g]["NDCG@10"] for r in R[(ds, "mean")].values()]) for g in ("short", "medium", "long")]
        rel = [100 * (x / n - 1) for x, n in zip(v, norm)]
        b.plot(xs + off, rel, ls=ls, color=FAMILY[m], lw=1.3, marker=MARK[m], ms=3.6 if m != "mamba" else 5.6, mec="white", mew=0.5, zorder=3)
        if ds == "ml1m":
            b.text(xs[-1] + off + 0.15, rel[-1], NAME[m].replace(" (selective)", "").replace(" (recency)", ""),
                   fontsize=6, color=INK2, va="center")
b.axhline(0, color=SIMPLE, lw=0.9, ls=(0, (3, 1.5)), zorder=1)
b.text(8.55, 1.3, "Last + mean = 0", fontsize=6, color=INK2, ha="right", va="bottom")
b.set_xticks(list(xs) + list(xs + 4))
b.set_xticklabels(["short", "medium", "long"] * 2)
b.set_xlim(-0.4, 8.6)
b.text(1, -14, "Sports", ha="center", color=INK, fontsize=7.5)
b.text(5, -14, "MovieLens-1M", ha="center", color=INK, fontsize=7.5)
b.set_ylim(-17, 48)
b.set_ylabel("NDCG@10 vs. Last + mean (%)")
b.set_title("(b) Gain over order-free pooling by history-length tertile", loc="left", color=INK, pad=3)
fig.savefig(f"{OUT}/fig_use.pdf"); fig.savefig(f"{OUT}/fig_use.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------------ Fig. dim: d = 64 -> 128
rob = RES["robustness"]
rows = ["ema", "mean", "lti", "mamba", "sasrec"]
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, ds in zip(axes, ["sports", "ml1m"]):
    clean(ax)
    for y, m in enumerate(rows):
        e = rob[f"dim128|{ds}|{m}"]
        x0, x1 = e["d64"], e["d128"]
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=FAMILY[m], lw=1.1,
                    shrinkA=2.2, shrinkB=1.2, mutation_scale=6), zorder=2)
        ax.plot(x0, y, MARK[m], mfc="white", mec=FAMILY[m], ms=3.8, mew=0.9, zorder=3)
        ax.plot(x1, y, MARK[m], color=FAMILY[m], ms=3.8, mec="white", mew=0.4, zorder=3)
    ax.set_title(DSN[ds], color=INK, pad=3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([NAME[m] for m in rows], fontsize=6.8)
axes[0].invert_yaxis()
axes[0].set_xlim(0.0245, 0.0300); axes[1].set_xlim(0.145, 0.192)
fig.text(0.6, -0.08, r"test NDCG@10   ($\circ$ $d$=64 $\rightarrow$ $\bullet$ $d$=128, mean of seeds 0–2)", ha="center", color=INK, fontsize=7)
fig.savefig(f"{OUT}/fig_dim.pdf"); fig.savefig(f"{OUT}/fig_dim.png", dpi=220); plt.close(fig)
print("figures written to", OUT)
