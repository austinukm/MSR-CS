"""S1 full study -- what drives Mamba-based sequential recommenders?

Frozen code for the paper (the pilot code in ../seq_common.py stays untouched).
Identical wrapper for every model; only the sequence mixer changes:
    item embedding -> dropout -> LayerNorm -> B x [ mixer -> +residual -> LN -> FFN -> +residual -> LN ]
    representation = last position; scores = representation . item table (tied); full-softmax CE.
Mixers:
    last   : no mixing (the mixer outputs zeros, so only the current item passes through)
    mean   : causal running mean of the inputs
    ema    : h_t = a*h_{t-1} + (1-a)*x_t, decay a learned per channel (no input dependence)
    lti    : Mamba block with dt, B, C learned constants (linear time-invariant)
    mamba  : Mamba block, dt, B, C computed from the input (selective) == Mamba4Rec's mixer
    sasrec : causal multi-head self-attention block (SASRec), learned positions
    gru    : GRU (GRU4Rec), one recurrent layer
The SSM scan uses the mamba_ssm CUDA kernel on GPU (checked against the pure-PyTorch recurrence:
max abs diff 1e-5) and the pure-PyTorch loop on CPU.
"""
import os, time, json
import numpy as np, pandas as pd, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F

DEV = torch.device(os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu"))
CE_ELEMS = 5e7                                       # max logits materialised at once (~200 MB in float32)
if DEV.type == "cuda":
    # hard cap: a run that needs more fails with an out-of-memory error instead of pushing the
    # card into oversubscription (Windows/WSL pages VRAM, which coincided with the 0x113 crashes)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("S1_MEMFRAC", "0.7")))
torch.set_num_threads(int(os.environ.get("NT", "4")))
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:                                  # CPU-only environments
    selective_scan_fn = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
DATA = os.environ.get("S1_DATA", os.path.join(ROOT, "data", "splits"))     # <DATA>/<dataset>/{train,val,test}.csv
PATHS = {k: os.path.join(DATA, k) for k in ("baby", "sports", "clothing", "elec", "ml1m")}
MAXLEN = {"ml1m": 200}                             # everything else: 50
L = 50


def load_ds(ds):
    global L
    L = MAXLEN.get(ds, 50)
    d = PATHS[ds]
    tr, va, te = (pd.read_csv(f"{d}/{s}.csv") for s in ("train", "val", "test"))
    users = np.unique(np.concatenate([x.user_id.values for x in (tr, va, te)]))
    items = np.unique(np.concatenate([x.item_id.values for x in (tr, va, te)]))
    um = pd.Series(np.arange(len(users)), index=users)
    im = pd.Series(np.arange(1, len(items) + 1), index=items)          # 0 = padding
    nU, nI = len(users), len(items)
    tu, ti, tt = um[tr.user_id.values].values, im[tr.item_id.values].values, tr.timestamp.values
    order = np.lexsort((np.arange(len(tu)), tt, tu))
    seqs = [[] for _ in range(nU)]
    for k in order:
        seqs[tu[k]].append(int(ti[k]))
    vu, vi = um[va.user_id.values].values, im[va.item_id.values].values
    eu, ei = um[te.user_id.values].values, im[te.item_id.values].values
    vmap = dict(zip(vu, vi))

    def csr(u, i):
        m = sp.csr_matrix((np.ones(len(u), np.float32), (u, i - 1)), shape=(nU, nI)); m.sum_duplicates(); m.data[:] = 1
        return m
    pop = np.bincount(ti, minlength=nI + 1)[1:]
    return dict(name=ds, nU=nU, nI=nI, L=L, seqs=seqs, val=(vu, vi), test=(eu, ei), vmap=vmap,
                seen_val=csr(tu, ti), seen_test=csr(np.concatenate([tu, vu]), np.concatenate([ti, vi])), pop=pop)


def pad_left(rows, length=None, cap=None):
    cap = L if cap is None else cap
    if length is None:
        length = max(1, min(cap, max((len(s) for s in rows), default=1)))
    out = np.zeros((len(rows), length), np.int64)
    for r, s in enumerate(rows):
        s = s[-length:]
        if len(s):
            out[r, length - len(s):] = s
    return torch.from_numpy(out)


# ------------------------------------------------------------------ mixers
class SSM(nn.Module):
    def __init__(self, d, selective=True, d_state=16, d_conv=4, expand=2):
        super().__init__()
        di = expand * d; self.di, self.N, self.selective = di, d_state, selective
        self.dt_rank = max(1, d // 16)
        self.in_proj = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        dt = torch.exp(torch.rand(di) * (np.log(0.1) - np.log(0.001)) + np.log(0.001))
        inv = dt + torch.log(-torch.expm1(-dt))
        if selective:
            self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, di)
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv)
        else:
            self.dt = nn.Parameter(inv)
            self.Bp = nn.Parameter(torch.randn(d_state) * 0.5)
            self.Cp = nn.Parameter(torch.randn(d_state) * 0.5)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)

    def forward(self, x, mask):
        Bn, Ln, _ = x.shape
        xi, z = self.in_proj(x).chunk(2, -1)
        xi = xi * mask
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :Ln].transpose(1, 2)) * mask      # (B, L, di)
        A = -torch.exp(self.A_log.float())
        if self.selective:
            dt, Bm, Cm = torch.split(self.x_proj(xi), [self.dt_rank, self.N, self.N], -1)
            dt = F.softplus(self.dt_proj(dt))                                               # (B, L, di)
        else:
            dt = F.softplus(self.dt).expand(Bn, Ln, -1)
            Bm = self.Bp.expand(Bn, Ln, -1); Cm = self.Cp.expand(Bn, Ln, -1)
        if x.is_cuda and selective_scan_fn is not None:
            y = selective_scan_fn(xi.transpose(1, 2).contiguous().float(), dt.transpose(1, 2).contiguous().float(), A,
                                  Bm.transpose(1, 2).contiguous().float(), Cm.transpose(1, 2).contiguous().float(),
                                  self.D.float(), delta_softplus=False).transpose(1, 2)
        else:
            dA = torch.exp(dt.unsqueeze(-1) * A)
            dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xi.unsqueeze(-1)
            h = x.new_zeros(Bn, self.di, self.N); ys = []
            for t in range(Ln):
                h = dA[:, t] * h + dBx[:, t]
                ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(ys, 1) + xi * self.D
        return self.out_proj(y * F.silu(z))


class EMA(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d) + 1.0)

    def forward(self, x, mask):
        a = torch.sigmoid(self.w)
        h = x.new_zeros(x.shape[0], x.shape[2]); ys = []
        for t in range(x.shape[1]):
            m = mask[:, t]
            h = m * (a * h + (1 - a) * x[:, t]) + (1 - m) * h
            ys.append(h)
        return torch.stack(ys, 1)


class Mean(nn.Module):
    def forward(self, x, mask):
        return (x * mask).cumsum(1) / mask.cumsum(1).clamp(min=1)


class Zero(nn.Module):
    def forward(self, x, mask):
        return torch.zeros_like(x)


class Attn(nn.Module):
    """Causal multi-head self-attention sublayer (SASRec); padding keys blocked, diagonal always open."""
    def __init__(self, d, heads=2, drop=0.2):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.heads = heads

    def forward(self, x, mask):
        Ln = x.shape[1]
        pad = mask.squeeze(-1) == 0
        causal = torch.triu(torch.ones(Ln, Ln, dtype=torch.bool, device=x.device), 1)
        m = (causal.unsqueeze(0) | pad.unsqueeze(1)) & ~torch.eye(Ln, dtype=torch.bool, device=x.device).unsqueeze(0)
        y, _ = self.att(x, x, x, attn_mask=m.repeat_interleave(self.heads, 0), need_weights=False)
        return y


MIXERS = ("last", "mean", "ema", "lti", "mamba", "sasrec", "gru")


class Seq(nn.Module):
    def __init__(self, nI, kind, d=64, blocks=2, drop=0.2):
        super().__init__()
        self.kind = kind
        self.item = nn.Embedding(nI + 1, d, padding_idx=0)
        nn.init.normal_(self.item.weight, std=0.02)
        self.drop = nn.Dropout(drop); self.ln0 = nn.LayerNorm(d)
        if kind == "gru":
            self.gru = nn.GRU(d, d, batch_first=True)
            return
        if kind == "sasrec":
            self.pos = nn.Embedding(L, d)
        mk = {"last": Zero, "mean": Mean, "ema": lambda: EMA(d), "lti": lambda: SSM(d, False),
              "mamba": lambda: SSM(d, True), "sasrec": lambda: Attn(d, 2, drop)}[kind]
        self.mix = nn.ModuleList(mk() for _ in range(blocks))
        self.ln1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(blocks))
        self.ffn = nn.ModuleList(nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
                                 for _ in range(blocks))
        self.ln2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(blocks))

    def forward(self, seq):
        E = self.item.weight
        mask = (seq > 0).float().unsqueeze(-1)
        x = self.ln0(self.drop(E[seq])) * mask
        if self.kind == "gru":
            # run the GRU on right-aligned, packed sequences so leading padding never touches the state,
            # then shift the outputs back to the left-padded layout used everywhere else
            Ln = seq.shape[1]
            lens = (seq > 0).sum(1)
            p = (Ln - lens).unsqueeze(1)
            ar = torch.arange(Ln, device=seq.device).unsqueeze(0)
            xr = torch.gather(x, 1, ((ar + p) % Ln).unsqueeze(-1).expand_as(x))
            packed = nn.utils.rnn.pack_padded_sequence(xr, lens.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
            yr, _ = self.gru(packed)
            yr, _ = nn.utils.rnn.pad_packed_sequence(yr, batch_first=True, total_length=Ln)
            return torch.gather(yr, 1, ((ar - p) % Ln).unsqueeze(-1).expand_as(yr)) * mask
        if self.kind == "sasrec":
            x = x + self.pos.weight[L - seq.shape[1]:].unsqueeze(0) * mask
        for mix, l1, ffn, l2 in zip(self.mix, self.ln1, self.ffn, self.ln2):
            x = l1(x + self.drop(mix(x, mask)))
            x = l2(x + self.drop(ffn(x))) * mask
        return x


# ------------------------------------------------------------------ evaluation
def _ranks(model, data, split, variant=None, bs=None):
    # evaluation batch shrinks with the catalogue so the (users x items) score matrix stays ~120 MB
    bs = bs or int(os.environ.get("S1_EVAL_BS", min(1024, max(128, int(3e7 // data["nI"])))))
    """variant: None | 'shuffle' (history order permuted) | 'droplast' (most recent item removed)."""
    model.eval()
    E = model.item.weight[1:]
    us, ts = data[split]
    seen = data["seen_val"] if split == "val" else data["seen_test"]
    hist = [data["seqs"][u] + ([data["vmap"][u]] if split == "test" else []) for u in us]
    if variant == "droplast":
        hist = [h[:-1] if len(h) > 1 else h for h in hist]
    order = np.argsort([min(len(h), L) for h in hist], kind="stable")
    rng = np.random.default_rng(12345)
    ranks = np.zeros(len(us), np.int64)
    with torch.no_grad():
        for a in range(0, len(us), bs):
            idx = order[a:a + bs]
            rows = [hist[k][-L:] for k in idx]
            if variant == "shuffle":
                rows = [list(rng.permutation(r)) for r in rows]
            h = model(pad_left(rows).to(DEV))[:, -1]
            s = torch.nan_to_num(h @ E.T, nan=-float("inf"))
            sub = seen[us[idx]]
            rr = torch.as_tensor(np.repeat(np.arange(len(idx)), np.diff(sub.indptr)), device=DEV)
            s[rr, torch.as_tensor(sub.indices, device=DEV, dtype=torch.long)] = -float("inf")
            t = torch.as_tensor(ts[idx] - 1, device=DEV)
            tg = s[torch.arange(len(idx), device=DEV), t]
            ranks[idx] = ((s >= tg[:, None]).sum(1) - 1).cpu().numpy()
    return ranks, hist


def metrics(ranks):
    out = {}
    for k in (10, 20):
        hit = ranks < k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(np.where(hit, 1 / np.log2(ranks + 2), 0).mean())
    return out


def evaluate(model, data, split="test", full=False):
    r, hist = _ranks(model, data, split)
    res = metrics(r)
    if not full:
        return res
    res["ranks"] = r.tolist()                                   # per-user ranks: any later stratification offline
    pop = data["pop"][data[split][1] - 1]
    res["by_pop"] = {nm: metrics(r[m]) | {"n": int(m.sum())} for nm, m in
                     (("cold", pop <= 5), ("mid", (pop > 5) & (pop <= 20)), ("head", pop > 20))}
    ln = np.array([len(h) for h in hist])
    q = np.quantile(ln, [1 / 3, 2 / 3])
    res["by_len"] = {nm: metrics(r[m]) | {"n": int(m.sum())} for nm, m in
                     (("short", ln <= q[0]), ("medium", (ln > q[0]) & (ln <= q[1])), ("long", ln > q[1]))}
    res["len_cuts"] = q.tolist()
    res["placebo_shuffle"] = metrics(_ranks(model, data, split, "shuffle")[0])
    res["placebo_droplast"] = metrics(_ranks(model, data, split, "droplast")[0])
    return res


# ------------------------------------------------------------------ training
def train(data, kind, seed=0, lr=1e-3, drop=0.2, bs=256, max_epochs=200, eval_every=2, patience=5, log=None,
          d=64, shuffle_train=False):
    max_epochs = int(os.environ.get("S1_MAXEP", max_epochs))          # smoke tests only
    bs = int(os.environ.get("S1_BS", bs))                              # protocol deviation 1: ML-1M / Elec use 128
    duty = float(os.environ.get("S1_DUTY", "0"))                       # pause after each batch = duty x its compute time
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    model = Seq(data["nI"], kind, d=d, drop=drop).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seqs = data["seqs"]
    users = np.array([u for u, s in enumerate(seqs) if len(s) >= 2])
    best, bad, t0, ep_times = (-1.0, None, 0), 0, time.time(), []
    for ep in range(1, max_epochs + 1):
        model.train(); te = time.time()
        perm = rng.permutation(users)
        batches = []
        for a in range(0, len(perm), bs * 50):
            ch = perm[a:a + bs * 50]
            ch = ch[np.argsort([min(len(seqs[u]), L + 1) for u in ch], kind="stable")]
            batches += [ch[b:b + bs] for b in range(0, len(ch), bs)]
        paused = 0.0
        for bi in rng.permutation(len(batches)):
            tb = time.time()
            rows = [seqs[u][-(L + 1):] for u in batches[bi]]
            if shuffle_train:                                          # robustness: order destroyed during training
                rows = [list(rng.permutation(r)) for r in rows]
            full = pad_left(rows, None, L + 1).to(DEV)
            inp, tgt = full[:, :-1], full[:, 1:]
            h = model(inp)
            m = (tgt > 0) & (inp > 0)
            hm, tm = h[m], tgt[m] - 1
            n, W = len(tm), model.item.weight[1:]
            chunk = max(256, int(CE_ELEMS // data["nI"]))
            opt.zero_grad()
            if n <= chunk:
                F.cross_entropy(hm @ W.T, tm).backward()
            else:
                # identical gradient to the full-softmax mean loss, but the (positions x items) logits
                # are materialised one chunk at a time: bounded GPU memory on large catalogues
                hd = hm.detach().requires_grad_()
                for a in range(0, n, chunk):
                    (F.cross_entropy(hd[a:a + chunk] @ W.T, tm[a:a + chunk], reduction="sum") / n).backward()
                hm.backward(hd.grad)
            opt.step()
            if duty > 0:                                               # keep the GPU below sustained full load
                if DEV.type == "cuda":
                    torch.cuda.synchronize()
                pz = (time.time() - tb) * duty
                time.sleep(pz); paused += pz
        ep_times.append(time.time() - te - paused)                     # compute time only, pauses excluded
        if ep % eval_every == 0:
            v = evaluate(model, data, "val")["NDCG@10"]
            if log:
                log(f"    ep{ep} val NDCG@10={v:.4f} ({time.time() - t0:.0f}s)")
            if v > best[0]:
                best = (v, {k: t.detach().clone() for k, t in model.state_dict().items()}, ep); bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(best[1])
    return model, dict(best_ep=best[2], val_ndcg10=best[0], epochs_run=ep, train_s=time.time() - t0,
                       sec_per_epoch=float(np.mean(ep_times)), params=sum(p.numel() for p in model.parameters()))


def latency(model, data, n=2048, bs=512, reps=5):
    """Mean wall time to score n test users (full catalogue), after one warm-up pass."""
    us = data["test"][0][:n]
    rows = [(data["seqs"][u] + [data["vmap"][u]])[-L:] for u in us]
    model.eval()
    with torch.no_grad():
        def run():
            for a in range(0, len(rows), bs):
                h = model(pad_left(rows[a:a + bs]).to(DEV))[:, -1]
                (h @ model.item.weight[1:].T).max(1)
            if DEV.type == "cuda":
                torch.cuda.synchronize()
        run(); t = time.time()
        for _ in range(reps):
            run()
    return (time.time() - t) / reps / len(rows) * 1000.0             # ms per user
