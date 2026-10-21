"""Run every remaining S1 step inside ONE long-lived CUDA process.

The 0x113 crashes clustered at the moment a new GPU process started after a period of sustained
load (10-18 15:41, 10-19 12:54), so the remaining steps no longer start a process each:
  * one process, one CUDA context, all steps in sequence (skip-if-exists as before);
  * before each unfinished run: wait until the GPU is at or below COOL_C (at most COOL_MAX s);
  * a pause after every batch on every dataset (S1_DUTY), ML-1M / Electronics also batch 128.
"""
import os, sys, time, subprocess, json, datetime
import numpy as np, torch
import s1_lib as S
import s1_run as Rn

COOL_C = int(os.environ.get("S1_COOL_C", "999"))     # 999 = no cool-down wait (default)
COOL_MAX = float(os.environ.get("S1_COOL_MAX", "1800"))
DUTY = os.environ.get("S1_DUTY", "0")  # pause after every batch = DUTY x its compute time (no effect on results)
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "queue.log")
STEPS = [  # phase, dataset, models, seeds
    ("shuftrain", "baby", ["mean", "ema", "lti", "mamba", "sasrec"], [0, 1, 2]),
    ("shuftrain", "sports", ["mean", "ema", "lti", "mamba", "sasrec"], [0, 1, 2]),
    ("shuftrain", "clothing", ["mean", "ema", "lti", "mamba", "sasrec"], [0, 1, 2]),
    ("dim128", "sports", ["mean", "ema", "lti", "mamba", "sasrec"], [0, 1, 2]),
    ("dim128", "ml1m", ["mean", "ema", "lti", "mamba", "sasrec"], [0, 1, 2]),
    ("final", "elec", ["mean", "ema", "lti", "mamba", "sasrec"], [0]),
]


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{datetime.datetime.now():%m-%d %H:%M:%S}] {msg}\n")
    print(msg, flush=True)


def gpu_temp():
    try:
        out = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def cool_down():
    t0, last = time.monotonic(), None
    while True:
        last = gpu_temp()
        waited = time.monotonic() - t0
        if last is not None and last <= COOL_C:
            return last, waited
        if waited >= COOL_MAX:
            raise RuntimeError(f"GPU did not reach <= {COOL_C} C within {COOL_MAX:.0f}s "
                               f"(last reading: {last!r}); stopping safely")
        time.sleep(min(15, max(0, COOL_MAX - waited)))


log("runall start (single process, per-run cool-down, duty on every dataset)")
for phase, ds, models, seeds in STEPS:
    heavy = ds in ("ml1m", "elec")
    os.environ["S1_BS"] = "128" if heavy else "256"
    os.environ["S1_DUTY"] = DUTY                     # fixed pause after each batch, set in s1_queue3.sh
    log(f"start {phase} {ds} (in-process) bs={os.environ['S1_BS']} duty={os.environ['S1_DUTY']} "
        f"with per-run cool-down at <= {COOL_C} C")
    data = S.load_ds(ds)
    for kind in models:
        b = Rn.best_cfg(ds, kind)
        if b is None:
            log(f"  no tuning result for {ds} {kind}, skipped"); continue
        for s in seeds:
            kw = {"d": 128} if phase == "dim128" else ({"shuffle_train": True} if phase == "shuftrain" else {})
            p = f"{Rn.RUNS}/{phase}_{ds}_{kind}_{Rn.cfg_name(b[1], b[2])}_s{s}.json"
            if os.path.exists(p) and Rn.ok(p):
                continue
            t, waited = cool_down()
            log(f"  ready {phase} {ds} {kind} s{s}: gpu {t} C after {waited:.0f}s cool-down")
            Rn.run_one(data, kind, b[1], b[2], s, phase, **kw)
    log(f"done  {phase} {ds}")
    del data
    torch.cuda.empty_cache()
log("#runall done")
