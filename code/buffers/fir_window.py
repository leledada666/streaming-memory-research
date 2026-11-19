import numpy as np
rng = np.random.default_rng(0)
N = 300000
c = rng.choice([-1.0, 1.0], size=N).astype(np.float32)
z = rng.standard_normal(N).astype(np.float32)
m = np.stack([c, z], 1)
W = 16
idx = np.arange(W-1, N)
X = np.concatenate([m[idx - k] for k in range(W)], 1)
for L in [15, 30, 60]:
    y = c[idx - L]
    n = len(X); ntr = int(n*0.7)
    Xtr, Xte, ytr, yte = X[:ntr], X[ntr:], y[:ntr], y[ntr:]
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]; B = np.c_[np.ones(len(Xte)), (Xte-mu)/sd]
    lam = 10.0; M = A.T@A + lam*np.eye(A.shape[1]); M[0,0] -= lam
    beta = np.linalg.solve(M, A.T@ytr)
    acc = float((np.sign(B@beta) == yte).mean())
    print(f"FIR-16 L={L} ridge acc={acc:.4f}", flush=True)