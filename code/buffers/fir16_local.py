# FIR-16 本地版：确认 L=15 是延迟线，L=30 应回 chance。
import numpy as np, torch, torch.nn as nn

rng = np.random.default_rng(0)
N = 300000
c = rng.choice([-1.0, 1.0], size=N).astype(np.float32)
z = rng.standard_normal(N).astype(np.float32)
m = np.stack([c, z], 1)

def build(L, W=16):
    idx = np.arange(W-1, N)
    feats = np.concatenate([m[idx - k] for k in range(W)], 1)
    y = c[idx - L]
    return feats, y

def ridge_eval(X, y, split=0.7):
    n = len(X); ntr = int(n*split)
    Xtr, Xte = X[:ntr], X[ntr:]; ytr, yte = y[:ntr], y[ntr:]
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]; B = np.c_[np.ones(len(Xte)), (Xte-mu)/sd]
    lam = 10.0; M = A.T@A + lam*np.eye(A.shape[1]); M[0,0] -= lam
    beta = np.linalg.solve(M, A.T@ytr)
    return float((np.sign(B@beta) == yte).mean()), beta

class MLP(nn.Module):
    def __init__(s, d, h1=1024, h2=512):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d,h1), nn.GELU(), nn.Linear(h1,h2), nn.GELU(), nn.Linear(h2,1))
    def forward(s,x): return s.net(x).squeeze(-1)

def mlp_eval(X, y, split=0.7, epochs=6, bs=4096):
    n = len(X); ntr = int(n*split)
    Xtr = torch.tensor(X[:ntr]); ytr = torch.tensor(y[:ntr])
    Xte = torch.tensor(X[ntr:]); yte = torch.tensor(y[ntr:])
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-8
    Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd
    net = MLP(X.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            j = perm[i:i+bs]
            opt.zero_grad(); loss = lossf(net(Xtr[j]), ytr[j]); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Xte); acc = float(((pred>0).float() == (yte>0).float()).mean())
    return acc, sum(p.numel() for p in net.parameters())

for L in [1, 5, 15, 30]:
    X, y = build(L)
    acc, beta = ridge_eval(X, y)
    coef = beta[1:]; ccoef = [abs(coef[2*k]) for k in range(16)]
    top = int(np.argmax(ccoef))
    print(f"L={L:2d} ridge acc={acc:.4f} top|c-coef| @ lag {top} (target {L})", flush=True)
    if L in (15, 30):
        acc_m, params = mlp_eval(X, y)
        print(f"      MLP-16 acc={acc_m:.4f} params={params}", flush=True)