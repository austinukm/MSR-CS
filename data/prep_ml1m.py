"""MovieLens-1M -> the same leave-one-out csv layout as the Amazon datasets.
Every rating is an implicit interaction; per user, interactions sorted by time (ties: file order);
last -> test, second last -> val, the rest -> train.  Run with python -I (reads a downloaded file).
Usage: python -I data/prep_ml1m.py ml-1m.zip data/splits/ml1m
Input used in the paper: ml-1m.zip from https://grouplens.org/datasets/movielens/1m/,
SHA-256 a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20. Output files are written with
LF line endings, so their SHA-256 values match the ones printed by verify_splits.py on every platform."""
import sys, os
import numpy as np, pandas as pd

src, dst = sys.argv[1], sys.argv[2]
import zipfile, io
with zipfile.ZipFile(src) as z:
    raw = z.read("ml-1m/ratings.dat").decode("latin-1")
r = pd.read_csv(io.StringIO(raw), sep="::", engine="python", names=["user_id", "item_id", "rating", "timestamp"])
r["order"] = np.arange(len(r))
r = r.sort_values(["user_id", "timestamp", "order"])
r["rk"] = r.groupby("user_id").cumcount(ascending=False)          # 0 = last
os.makedirs(dst, exist_ok=True)
cols = ["user_id", "item_id", "rating", "timestamp"]
r[r.rk == 0][cols].to_csv(f"{dst}/test.csv", index=False, lineterminator="\n")
r[r.rk == 1][cols].to_csv(f"{dst}/val.csv", index=False, lineterminator="\n")
r[r.rk >= 2][cols].to_csv(f"{dst}/train.csv", index=False, lineterminator="\n")
n = r.groupby("user_id").size()
print(f"users {r.user_id.nunique()} items {r.item_id.nunique()} interactions {len(r)} "
      f"| per-user length mean {n.mean():.1f} median {n.median():.0f} min {n.min()}")
