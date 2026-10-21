"""Check every split used in the paper and print the statistics of Table I.

For each dataset: one validation and one test interaction per user; test not earlier than validation and
validation not earlier than the last training interaction; users/items/interactions, mean and median
sequence length; SHA-256 of the three split files (compare with data/SHA256SUMS.txt).
Usage: python data/verify_splits.py [DATA_DIR]     (default: data/splits)
"""
import hashlib, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R0 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "splits")
NAMES = {"baby": "Baby", "sports": "Sports", "clothing": "Clothing", "ml1m": "MovieLens-1M", "elec": "Electronics"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ok_all, seen = True, 0
for key, name in NAMES.items():
    d = os.path.join(R0, key)
    if not os.path.exists(os.path.join(d, "train.csv")):
        print(f"{name:13s} not found in {d} (skipped)"); continue
    seen += 1
    tr, va, te = (pd.read_csv(os.path.join(d, f"{s}.csv")) for s in ("train", "val", "test"))
    al = pd.concat([tr, va, te])
    one = va.user_id.is_unique and te.user_id.is_unique and set(va.user_id) == set(te.user_id) == set(al.user_id)
    last_tr = tr.groupby("user_id").timestamp.max()
    v = va.set_index("user_id").timestamp; t = te.set_index("user_id").timestamp
    order = bool(((v >= last_tr.reindex(v.index)) | last_tr.reindex(v.index).isna()).all() and (t >= v.reindex(t.index)).all())
    ln = al.groupby("user_id").size().values
    print(f"{name:13s} users {al.user_id.nunique():7,d} items {al.item_id.nunique():6,d} interactions {len(al):9,d} "
          f"mean len {ln.mean():6.1f} median {np.median(ln):4.0f} | one val/test per user {one} | time order {order}")
    for s in ("train", "val", "test"):
        print(f"    {sha(os.path.join(d, s + '.csv'))}  splits/{key}/{s}.csv")
    ok_all &= one and order
print("ALL CHECKS PASSED" if ok_all and seen else "SOME CHECK FAILED")
sys.exit(0 if ok_all and seen else 1)
