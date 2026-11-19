"""
E4b: GRU latch vs query-time content addressing (CPU smoke).

Stream (T=96): m_t = [flag, id_onehot(16), value_onehot(16)] = 33 dims.
K markers per sequence, distinct ids 0..15, random positions/order, non-marker = 0.
At query time t_q: one-hot query id; target = value of the marker with that id.
t_q is after the last marker and before T-1, so future tokens exist.

Arms:
  cross    : Q = Wq([query, h_tq]); K/V = m_{<=tq}; learned rel-position bias; causal.
  gru128   : GRU(33,128), [h_tq, query] -> MLP.
  fir16    : last 16 m vectors (<= t_q) + query -> MLP.
  tap      : oracle read of m at the marker with query id.

Controls (same trained models):
  id-shuffle     : relabel marker ids inside each sequence (id<->value pairing changes;
                   query label kept, target recomputed).
  absent id      : query id not in the sequence.
  multi-query    : every valid query id on the same sequence; prediction must change.
  future perturb : perturb m_{>tq}; logits must be bit-identical.

CPU smoke defaults: K=1/4/8, T=96, B=16, 1000 steps, seed 0.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(min(8, os.cpu_count() or 8))
D_IN, N_ID, N_VAL, N_CLS = 33, 16, 16, 16
OFF_ID, OFF_VAL = 1, 17


# ---------------------------------------------------------------- data
def make_dataset(n, T, K, rng, seq_id_base=0, absent_frac=0.0):
    m = np.zeros((n, T, D_IN), dtype=np.float32)
    q = np.zeros((n, N_ID), dtype=np.float32)
    tq = np.zeros(n, dtype=np.int64)
    pos = np.full((n, K), -1, dtype=np.int64)
    ids = np.zeros((n, K), dtype=np.int64)
    vals = np.zeros((n, K), dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    for b in range(n):
        p = rng.choice(T - 8, size=K, replace=False)  # leaves >=7 future slots
        p = np.sort(p)
        a = rng.permutation(N_ID)[:K]
        v = rng.integers(0, N_VAL, size=K)
        order = rng.permutation(K)  # random stream order == position order
        p, a, v = p[order], a[order], v[order]
        t = int(rng.integers(int(p.max()) + 1, T))
        for j in range(K):
            m[b, p[j], 0] = 1.0
            m[b, p[j], OFF_ID + a[j]] = 1.0
            m[b, p[j], OFF_VAL + v[j]] = 1.0
        if rng.random() < absent_frac:
            cand = [i for i in range(N_ID) if i not in set(a.tolist())]
            q[b, int(rng.choice(cand))] = 1.0
            y[b] = -1
        else:
            qi = int(rng.integers(0, K))
            q[b, a[qi]] = 1.0
            y[b] = v[qi]
        tq[b] = t
        pos[b], ids[b], vals[b] = p, a, v
    return dict(m=m, q=q, tq=tq, pos=pos, ids=ids, vals=vals, y=y)


def relabel_ids(ds, rng):
    """Shuffle id labels among positions within each sequence; query label fixed."""
    m2 = ds["m"].copy()
    ids2 = ds["ids"].copy()
    y2 = ds["y"].copy()
    n, K = ds["ids"].shape
    for b in range(n):
        perm = rng.permutation(K)  # new id at position j = old ids[perm[j]]
        ids2[b] = ds["ids"][b][perm]
        # value channel stays at position; rewrite only id channels
        for j in range(K):
            p = ds["pos"][b, j]
            m2[b, p, OFF_ID:OFF_ID + N_ID] = 0.0
            m2[b, p, OFF_ID + ids2[b, j]] = 1.0
        qid = int(np.argmax(ds["q"][b]))
        j_star = int(np.where(ids2[b] == qid)[0][0])
        y2[b] = ds["vals"][b][j_star]
    out = dict(ds)
    out.update(m=m2, ids=ids2, y=y2)
    return out


def absent_query(ds, rng):
    """Replace query with an id absent from the sequence (no target)."""
    q2 = np.zeros_like(ds["q"])
    for b in range(ds["q"].shape[0]):
        present = set(ds["ids"][b].tolist())
        cand = [i for i in range(N_ID) if i not in present]
        q2[b, int(rng.choice(cand))] = 1.0
    out = dict(ds)
    out["q"] = q2
    return out


def multi_query(ds):
    """Yield one dataset copy per marker slot, with query = that marker's id."""
    out = []
    K = ds["ids"].shape[1]
    for j in range(K):
        c = dict(ds)
        q2 = np.zeros_like(ds["q"])
        q2[np.arange(len(ds["y"])), ds["ids"][:, j]] = 1.0
        c["q"] = q2
        c["y"] = ds["vals"][:, j]
        out.append(c)
    return out


# ---------------------------------------------------------------- models
def presence(m, q):
    """1 if query id occurs in m (causal), else 0. Same feature for every arm."""
    mid = m[:, :, OFF_ID:OFF_ID + N_ID]
    return (q[:, None, :] * mid).sum(-1).max(-1).values.unsqueeze(-1)


class GRUArm(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(D_IN, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + N_ID + 1, 128), nn.ReLU(),
                                  nn.Linear(128, N_CLS))

    def forward(self, m, q, tq):
        h, _ = self.gru(m)
        ht = h[torch.arange(m.shape[0], device=m.device), tq]
        return self.head(torch.cat([ht, q, presence(m, q)], -1))


class CrossArm(nn.Module):
    def __init__(self, T, d=96, hidden=96):
        super().__init__()
        self.d = d
        self.gru = nn.GRU(D_IN, hidden, batch_first=True)
        self.qh = nn.Linear(N_ID + hidden, d)
        self.wk = nn.Linear(D_IN, d)
        self.wv = nn.Linear(D_IN, d)
        self.rel = nn.Parameter(torch.zeros(T))
        self.match_scale = nn.Parameter(torch.tensor(4.0))
        self.out = nn.Linear(d + N_ID + 1, N_CLS)
        self.t = torch.arange(T)

    def forward(self, m, q, tq):
        B, T, _ = m.shape
        h, _ = self.gru(m)
        ht = h[torch.arange(B, device=m.device), tq]
        qpart = self.qh(torch.cat([q, ht], -1))
        # explicit query-id <-> marker-id matching channel, plus learned content
        Q = torch.cat([q, qpart], -1)[:, None, :]              # [B,1,16+d]
        mid = m[:, :, OFF_ID:OFF_ID + N_ID]
        K = torch.cat([mid, self.wk(m)], -1)         # [B,T,16+d]
        V = self.wv(m)
        diff = (tq[:, None] - self.t[None, :].to(m.device)).clamp(0, T - 1)
        bias = self.rel[diff]
        score = (Q * K).sum(-1) / math.sqrt(N_ID + self.d) + bias
        score = score + self.match_scale * (q[:, None, :] * mid).sum(-1)
        mask = self.t[None, :].to(m.device) <= tq[:, None]
        score = score.masked_fill(~mask, -1e9)
        att = torch.softmax(score, -1)
        ctx = (att[:, :, None] * V).sum(1)
        return self.out(torch.cat([ctx, q, presence(m, q)], -1))


class FIR16(nn.Module):
    def __init__(self, T, win=16):
        super().__init__()
        self.win = win
        self.net = nn.Sequential(nn.Linear(win * D_IN + N_ID + 1, 256), nn.ReLU(),
                                 nn.Linear(256, N_CLS))

    def forward(self, m, q, tq):
        B, T, _ = m.shape
        pad = F.pad(m, (0, 0, self.win, 0))            # left pad
        idx = (tq + self.win)[:, None] + torch.arange(-self.win + 1, 1)[None, :].to(m.device)
        x = pad[torch.arange(B, device=m.device)[:, None], idx].reshape(B, -1)
        return self.net(torch.cat([x, q, presence(m, q)], -1))


class TapOracle(nn.Module):
    def forward(self, m, q, tq):
        B, T, _ = m.shape
        out = torch.zeros(B, N_CLS, device=m.device)
        for b in range(B):
            qid = int(q[b].argmax())
            seg = m[b, :tq[b] + 1]
            hit = (seg[:, OFF_ID + qid] > 0) & (seg[:, 0] > 0)
            if hit.any():
                j = int(torch.where(hit)[0][0])
                out[b] = seg[j, OFF_VAL:OFF_VAL + N_VAL]
            else:
                out[b] = 1.0 / N_CLS
        return out


# ---------------------------------------------------------------- helpers
def to_t(ds, dev):
    return (torch.from_numpy(ds["m"]).to(dev), torch.from_numpy(ds["q"]).to(dev),
            torch.from_numpy(ds["tq"]).to(dev), torch.from_numpy(ds["y"]).to(dev))


@torch.no_grad()
def predict(model, ds, dev, bs=128):
    model.eval()
    m, q, tq, y = to_t(ds, dev)
    logits = []
    for i in range(0, len(y), bs):
        logits.append(model(m[i:i + bs], q[i:i + bs], tq[i:i + bs]))
    logits = torch.cat(logits)
    return logits, y.to(dev)


@torch.no_grad()
def evaluate(model, ds, dev, rng=None, K=1):
    logits, y = predict(model, ds, dev)
    pred = logits.argmax(-1)
    acc = (pred == y).float().mean().item()
    res = {"acc": acc, "n": int(len(y))}
    if K >= 2:
        preds, accs = [], []
        for c in multi_query(ds):
            lg, yy = predict(model, c, dev)
            preds.append(lg.argmax(-1))
            accs.append((lg.argmax(-1) == yy).float().mean().item())
        P = torch.stack(preds, 1)  # [n,K]
        res["multi_acc"] = float(np.mean(accs))
        res["query_sensitivity"] = float((P.max(1).values != P.min(1).values).float().mean())
        res["pairwise_disagree"] = float(
            (P[:, :, None] != P[:, None, :]).float().mean())
    if rng is not None:
        ab = absent_query(ds, rng)
        lg, _ = predict(model, ab, dev)
        p = torch.softmax(lg, -1)
        pred = lg.argmax(-1)
        present = []
        for b in range(len(ds["y"])):
            present.append(pred[b].item() in set(ds["vals"][b].tolist()))
        res["absent_maxprob"] = float(p.max(-1).values.mean())
        res["absent_hits_present_value"] = float(np.mean(present))
        res["absent_entropy"] = float((-(p * (p + 1e-12).log()).sum(-1)).mean())
    return res


def causal_check(model, ds, dev):
    model.eval()
    m, q, tq, _ = to_t(ds, dev)
    with torch.no_grad():
        a = model(m, q, tq)
        m2 = m.clone()
        for b in range(m2.shape[0]):
            t = int(tq[b])
            if t + 1 < m2.shape[1]:
                # strong perturbation of every future slot
                m2[b, t + 1:, 0] = 1.0
                m2[b, t + 1:, OFF_ID + (b % N_ID)] = 1.0
                m2[b, t + 1:, OFF_VAL + ((b * 7) % N_VAL)] = 1.0
        b2 = model(m2, q, tq)
    return float((a - b2).abs().max().item())


# ---------------------------------------------------------------- train
def train_one(arm, K, T, B, steps, seed, dev, eval_ds, rng_eval, lr=2e-3):
    torch.manual_seed(seed * 1000 + K * 10 + {"cross": 1, "gru128": 2, "fir16": 3}[arm])
    np.random.seed(seed * 1000 + K * 10 + 7)
    rng = np.random.default_rng(seed * 1000 + K * 100 + 3)
    model = {"cross": lambda: CrossArm(T), "gru128": lambda: GRUArm(),
             "fir16": lambda: FIR16(T)}[arm]().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        ds = make_dataset(B, T, K, rng, absent_frac=0.2)
        m, q, tq, y = to_t(ds, dev)
        logits = model(m, q, tq)
        valid = y >= 0
        lp = F.cross_entropy(logits[valid], y[valid]) if valid.any() else logits.sum() * 0
        if (~valid).any():
            # no matching id: drive output to uniform (chance)
            la = -F.log_softmax(logits[~valid], -1).mean()
            loss = (lp * valid.sum() + la * (~valid).sum()) / B
        else:
            loss = lp
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    tr_time = time.time() - t0
    metrics = evaluate(model, eval_ds, dev,
                       rng=np.random.default_rng(30_000 + seed * 100 + K), K=K)
    metrics["causal_maxdiff"] = causal_check(model, eval_ds, dev)
    metrics["train_s"] = tr_time
    metrics["params"] = int(sum(p.numel() for p in model.parameters()))
    metrics["final_loss"] = float(loss.item())
    return model, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--K", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--arms", nargs="+", default=["cross", "gru128", "fir16", "tap"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="e4b_results.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    all_res = {"config": vars(args), "runs": []}
    for seed in args.seeds:
        for K in args.K:
            rng_eval = np.random.default_rng(10_000 + seed * 100 + K)
            eval_ds = make_dataset(args.eval_n, args.T, K, rng_eval)
            print(f"\n=== seed={seed} K={K} T={args.T} B={args.B} steps={args.steps} ===", flush=True)
            for arm in args.arms:
                if arm == "tap":
                    model, met = TapOracle(), {}
                elif arm == "tap_oracle":
                    model, met = TapOracle(), {}
                else:
                    model, met = train_one(arm, K, args.T, args.B, args.steps,
                                           seed, dev, eval_ds, rng_eval)
                if arm.startswith("tap"):
                    met = evaluate(model, eval_ds, dev,
                                   rng=np.random.default_rng(30_000 + seed * 100 + K), K=K)
                    met["causal_maxdiff"] = causal_check(model, eval_ds, dev)
                    met.update(params=0, train_s=0.0)
                # control: eval-time id relabel
                rng_sh = np.random.default_rng(20_000 + seed * 100 + K)
                sh = relabel_ids(eval_ds, rng_sh)
                met["idshuffle_acc"] = float(
                    (predict(model, sh, dev)[0].argmax(-1) == torch.from_numpy(sh["y"]).to(dev)).float().mean())
                met.update(arm=arm, K=K, seed=seed)
                all_res["runs"].append(met)
                print(f"{arm:8s} acc={met['acc']:.4f} multi={met.get('multi_acc', float('nan')):.4f} "
                      f"qsens={met.get('query_sensitivity', float('nan')):.3f} "
                      f"idshuf={met['idshuffle_acc']:.4f} "
                      f"abs_maxp={met.get('absent_maxprob', float('nan')):.3f} "
                      f"abs_hit={met.get('absent_hits_present_value', float('nan')):.3f} "
                      f"causal={met['causal_maxdiff']:.1e} t={met['train_s']:.0f}s", flush=True)
            # save incrementally
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(all_res, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()