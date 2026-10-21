"""S1 runner.  One JSON per run in s1_paper/runs/, skipped when it already exists and parses.

  python s1_run.py tune  <dataset> [models...]   seed 0, grid lr x dropout, select by validation NDCG@10
  python s1_run.py final <dataset> [models...]   seeds 0-4 with the selected configuration
"""
import sys, os, json, itertools, time
import numpy as np, torch
import s1_lib as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
RUNS = os.environ.get("S1_RUNS", os.path.join(ROOT, "results", "runs"))
os.makedirs(RUNS, exist_ok=True)
GRID = {"lr": [1e-3, 3e-3], "drop": [0.2, 0.5]}
SEEDS = [int(s) for s in os.environ.get("S1_SEEDS", "0,1,2,3,4").split(",")]


def ok(p):
    try:
        json.load(open(p)); return True
    except Exception:
        return False


def cfg_name(lr, drop):
    return f"lr{lr:g}_dr{drop:g}"


def run_one(data, kind, lr, drop, seed, phase, d=64, shuffle_train=False):
    p = f"{RUNS}/{phase}_{data['name']}_{kind}_{cfg_name(lr, drop)}_s{seed}.json"
    if os.path.exists(p) and ok(p):
        return json.load(open(p))
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    if S.DEV.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        model, info = S.train(data, kind, seed, lr=lr, drop=drop, log=log if seed == 0 and phase == "final" else None,
                              d=d, shuffle_train=shuffle_train)
    except torch.cuda.OutOfMemoryError as e:                          # capped by S1_MEMFRAC: skip, do not crash
        print(f"[{phase}] {data['name']} {kind} s{seed}: OUT OF MEMORY under the cap, skipped ({str(e)[:120]})", flush=True)
        torch.cuda.empty_cache()
        return None
    res = {"dataset": data["name"], "model": kind, "lr": lr, "drop": drop, "seed": seed, "phase": phase,
           "maxlen": data["L"], **info}
    res["val"] = S.evaluate(model, data, "val")
    res["test"] = S.evaluate(model, data, "test", full=(phase != "tune"))
    res["d"], res["shuffle_train"] = d, shuffle_train
    if phase != "tune":
        res["latency_ms_per_user"] = S.latency(model, data)
    if S.DEV.type == "cuda":
        res["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    del model
    if S.DEV.type == "cuda":
        torch.cuda.empty_cache()
    tmp = p + ".tmp"
    json.dump(res, open(tmp, "w")); os.replace(tmp, p)                # atomic: never a half-written json
    print(f"[{phase}] {data['name']} {kind:6s} {cfg_name(lr, drop)} s{seed}: val {res['val']['NDCG@10']:.4f} "
          f"test NDCG@10 {res['test']['NDCG@10']:.4f} HR@10 {res['test']['HR@10']:.4f} ep {info['best_ep']} "
          f"{time.time() - t0:.0f}s", flush=True)
    return res


def best_cfg(ds, kind):
    best = None
    for lr, drop in itertools.product(GRID["lr"], GRID["drop"]):
        p = f"{RUNS}/tune_{ds}_{kind}_{cfg_name(lr, drop)}_s0.json"
        if os.path.exists(p) and ok(p):
            v = json.load(open(p))["val"]["NDCG@10"]
            if best is None or v > best[0]:
                best = (v, lr, drop)
    return best


if __name__ == "__main__":
    phase, ds = sys.argv[1], sys.argv[2]
    kinds = sys.argv[3:] or list(S.MIXERS)
    data = S.load_ds(ds)
    print(f"{ds}: users {data['nU']} items {data['nI']} maxlen {data['L']} "
          f"mean train length {np.mean([len(s) for s in data['seqs']]):.1f} device {S.DEV}", flush=True)
    for kind in kinds:
        if phase == "tune":
            for lr, drop in itertools.product(GRID["lr"], GRID["drop"]):
                run_one(data, kind, lr, drop, 0, "tune")
        else:
            b = best_cfg(ds, kind)
            if b is None:
                print(f"no tuning result for {ds} {kind}, skipped", flush=True); continue
            for s in SEEDS:
                if phase == "final":
                    run_one(data, kind, b[1], b[2], s, "final")
                elif phase == "dim128":                        # robustness: wider model, configuration from d=64 tuning
                    run_one(data, kind, b[1], b[2], s, "dim128", d=128)
                elif phase == "shuftrain":                     # robustness: train on randomly ordered histories
                    run_one(data, kind, b[1], b[2], s, "shuftrain", shuffle_train=True)
