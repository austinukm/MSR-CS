"""Strata numbers quoted in Sections V-D and V-E of the IEEE Access manuscript (exploratory, computed after the main analysis).

  * target popularity (training interactions of the target: <=5 cold, 6-20 mid, >20 head):
    paired relative NDCG@10 difference of EMA and last + mean to Mamba, per Amazon dataset and pooled,
    95% percentile bootstrap over (dataset, seed) pairs (20,000 resamples, seed 0)
  * history-length tertiles: EMA and last + mean relative to Mamba (mean over seeds)
Run: python analysis/extra_strata.py   (reads results/runs/final_*.json)
"""
import collections, glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
D = os.environ.get("S1_RESULTS", os.path.join(ROOT, "results"))
R = collections.defaultdict(dict)
for f in glob.glob(f"{D}/runs/final_*.json"):
    r = json.load(open(f)); R[(r["dataset"], r["model"])][r["seed"]] = r
rng = np.random.default_rng(0)
AMZ = ["baby", "sports", "clothing"]


def ci(x):
    b = rng.choice(x, (20000, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])


def rel(ds, m, key, g, s):
    return R[(ds, m)][s]["test"][key][g]["NDCG@10"] / R[(ds, "mamba")][s]["test"][key][g]["NDCG@10"] - 1


print("== target popularity: NDCG@10 relative to Mamba, mean [95% interval]")
for m in ["ema", "mean"]:
    for ds in AMZ + ["pooled"]:
        dss = AMZ if ds == "pooled" else [ds]
        cells = []
        for g in ["cold", "mid", "head"]:
            x = np.array([rel(d, m, "by_pop", g, s) for d in dss for s in range(5)])
            lo, hi = ci(x)
            cells.append(f"{g} {x.mean():+.1%} [{lo:+.1%}, {hi:+.1%}]")
        print(f"  {m:5s} {ds:8s} " + " | ".join(cells))
for ds in AMZ:
    t = R[(ds, "mamba")][0]["test"]
    n = {g: t["by_pop"][g]["n"] for g in ["cold", "mid", "head"]}
    print(f"  share of test users {ds}: " + ", ".join(f"{g} {v / sum(n.values()):.0%}" for g, v in n.items()))

print("== history-length tertiles: NDCG@10 relative to Mamba (mean over seeds)")
for ds in AMZ + ["ml1m"]:
    cuts = R[(ds, "mamba")][0]["test"]["len_cuts"]
    cells = []
    for g in ["short", "medium", "long"]:
        v = {m: np.mean([R[(ds, m)][s]["test"]["by_len"][g]["NDCG@10"] for s in range(5)]) for m in ["ema", "mean", "mamba"]}
        cells.append(f"{g}: EMA {v['ema'] / v['mamba'] - 1:+.1%}, last+mean {v['mean'] / v['mamba'] - 1:+.1%}")
    print(f"  {ds:8s} cuts {cuts}  " + " | ".join(cells))
