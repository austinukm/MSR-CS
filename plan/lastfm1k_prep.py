"""B15 data preparation for Last.fm-1K (20_B15复购复制能力_查新与门槛_20261023.md, section 3).
  python -I lfm_prep.py extract RAW_DIR          safely extract the TSV from lastfm-dataset-1K.tar.gz into RAW_DIR/extracted
  python -I lfm_prep.py inspect RAW_DIR          statistics only (no output files): events, users, tracks, repeat shares,
                                                 tracks / events kept for several minimum-play thresholds
  python -I lfm_prep.py build RAW_DIR K OUT_NPZ  build the cache: song = (artist name, track name); consecutive duplicate
                                                 plays removed; songs with >= K plays kept; consecutive duplicates removed
                                                 again; per-user chronological sequences of item ids 1..N
The raw file is treated as data only (read with pandas, no code from the archive is executed)."""
import os, sys, tarfile, csv
import numpy as np, pandas as pd

mode, raw = sys.argv[1], sys.argv[2]
TSV = os.path.join(raw, "extracted", "lastfm-dataset-1K", "userid-timestamp-artid-artname-traid-traname.tsv")

if mode == "extract":
    dst = os.path.abspath(os.path.join(raw, "extracted")); os.makedirs(dst, exist_ok=True)
    with tarfile.open(os.path.join(raw, "lastfm-dataset-1K.tar.gz"), "r:gz") as tf:
        for m in tf.getmembers():
            target = os.path.abspath(os.path.join(dst, m.name))
            if not target.startswith(dst + os.sep) or not (m.isfile() or m.isdir()):
                raise SystemExit(f"refusing unsafe member {m.name!r}")
            print(m.name, m.size)
        tf.extractall(dst)
    sys.exit(0)


def load():
    df = pd.read_csv(TSV, sep="\t", header=None, names=["u", "ts", "aid", "art", "tid", "tra"], usecols=["u", "ts", "art", "tra"],
                     quoting=csv.QUOTE_NONE, on_bad_lines="skip", dtype=str, encoding="utf-8", encoding_errors="replace")
    n0 = len(df); df = df.dropna()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce"); df = df.dropna(subset=["ts"])
    df["song"] = df["art"] + "\x1f" + df["tra"]
    df = df.sort_values(["u", "ts"], kind="stable")
    return df[["u", "ts", "song"]].reset_index(drop=True), n0


def dedup(df):
    keep = (df["song"] != df["song"].shift()) | (df["u"] != df["u"].shift())
    return df[keep].reset_index(drop=True)


df, n0 = load()
print(f"rows read {n0:,}; valid {len(df):,}; users {df.u.nunique()}; songs {df.song.nunique():,}")
df = dedup(df)
cnt = df["song"].value_counts()
print(f"after removing consecutive duplicates: events {len(df):,}; songs {len(cnt):,}")
if mode == "inspect":
    for k in (5, 10, 20, 50, 100, 200):
        keep = cnt[cnt >= k]
        print(f"  min plays {k:4d}: songs {len(keep):8,d}  events kept {int(keep.sum()):11,d} ({keep.sum() / len(df):.1%})")
    # repeat share of next-song events (before filtering): target listened to before by the same user
    seen = df.groupby("u")["song"].transform(lambda s: s.duplicated())
    print(f"share of events that are repeats of an earlier song (any distance): {seen.mean():.1%}")
    sys.exit(0)

K, out = int(sys.argv[3]), sys.argv[4]
keep = set(cnt[cnt >= K].index)
df = dedup(df[df["song"].isin(keep)].reset_index(drop=True))
songs = sorted(df["song"].unique()); idx = {s: i + 1 for i, s in enumerate(songs)}
seqs = np.empty(df.u.nunique(), dtype=object); users = []
for j, (u, g) in enumerate(df.groupby("u", sort=True)):
    seqs[j] = np.array([idx[s] for s in g["song"]], dtype=np.int32); users.append(u)
L = np.array([len(s) for s in seqs])
print(f"built: users {len(seqs)}, items {len(songs):,}, events {L.sum():,}, median len {np.median(L):.0f}, min {L.min()}, max {L.max()}")
np.savez(out, seqs=seqs, n_items=len(songs), users=np.array(users), min_plays=K)
print("saved", out)
