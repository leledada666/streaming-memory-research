# -*- coding: utf-8 -*-
"""Step3: modality dropout on DIRECT. Target y = 1{c_{t-L} + 0.6*d_t > 0} with m_t=[c_t,d_t].
Variants: nodrop, drop50 (zero enc_head(m) p=0.5 in train), twostage (train bypass 2k steps with drop=1 on head, then unfreeze).
Memory reliance: at test, zero all bypass deltas for positions < t (ablate past injections) and measure acc drop.
Uses multis-style residual bypass (param-matched) as the memory path.
"""
from __future__ import annotations
import argparse, json, time, sys, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos
OUT = ROOT / "runs" / "roadmap_2026-10-20"
DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
TOK = ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz"
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["nodrop","drop50","twostage"], required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lag", type=int, default=10)
ap.add_argument("--cap", type=int, default=8000); ap.add_argument("--min_steps", type=int, default=1500)
ap.add_argument("--patience", type=int, default=800); ap.add_argument("--B", type=int, default=32); ap.add_argument("--T", type=int, default=96)
a = ap.parse_args(); dev="cuda"; torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed); T0=time.time()
df=pd.read_csv(DATA,usecols=["ts"]); N=len(df); i1,i2=int(N*0.70),int(N*0.85)
model=Kronos.from_pretrained(str(ROOT/"models"/"Kronos-small")).to(dev).eval()
for p in model.parameters(): p.requires_grad=False
z=np.load(TOK); ts=pd.to_datetime(df["ts"],unit="s",utc=True)
STAMP=np.stack([ts.dt.minute,ts.dt.hour,ts.dt.weekday,ts.dt.day,ts.dt.month],1).astype(np.float32)
S1t,S2t,STt=torch.tensor(z["S1"],device=dev),torch.tensor(z["S2"],device=dev),torch.tensor(STAMP,device=dev)
H=model.d_model; NL=len(model.transformer); sites=list(range(NL)); W=64
def mlp(w): return nn.Sequential(nn.Linear(2,w), nn.GELU(), nn.Linear(w,H))
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc=nn.ModuleList([mlp(W) for _ in sites]); s.g=nn.Parameter(torch.full((len(sites),),-2.0))
        s.enc_head=mlp(H); s.pi=nn.Sequential(nn.Linear(2*H,128),nn.GELU(),nn.Linear(128,1))
        s.drop_p = 0.0; s.ablate_past=False
    def forward(s, s1,s2,st,m):
        x=model.embedding([s1,s2])+model.time_emb(st)
        B,T,_=m.shape
        for i,layer in enumerate(model.transformer):
            if i in sites:
                k=sites.index(i); u=s.enc[k](m); u=u/(u.pow(2).mean(-1,keepdim=True).sqrt()+1e-6)
                if s.ablate_past:
                    # zero injection for all positions (memory path off)
                    inj = torch.zeros_like(u)
                else:
                    inj = torch.sigmoid(s.g[k])*u
                x=x+inj
            x=layer(x)
        h=model.norm(x)
        hm=s.enc_head(m)
        if s.training and s.drop_p>0:
            mask=(torch.rand(B,1,1,device=m.device)>=s.drop_p).float()
            hm=hm*mask
        return s.pi(torch.cat([h/(h.norm(dim=-1,keepdim=True)+1e-6)*8, hm],-1)).squeeze(-1)
net=Net().to(dev)
opt=torch.optim.Adam(net.parameters(), lr=1e-3); lossf=nn.BCEWithLogitsLoss()
def batch(lo,hi,B,g):
    st=g.integers(lo,hi-a.T,B); idx=torch.tensor(st[:,None]+np.arange(a.T),device=dev)
    c=torch.tensor(g.choice([-1.,1.],size=(B,a.T)).astype(np.float32),device=dev)
    d=torch.tensor(g.choice([-1.,1.],size=(B,a.T)).astype(np.float32),device=dev)
    # y_t = 1{c_{t-L}+0.6*d_t > 0}
    y=((c[:,:a.T-a.lag] + 0.6*d[:,a.lag:])>0).float()
    return S1t[idx],S2t[idx],STt[idx],torch.stack([c,d],-1),y
vg=np.random.default_rng(999); VAL=[batch(i1,i2,64,vg) for _ in range(4)]
tg=np.random.default_rng(777); TEST=[batch(i2,N,64,tg) for _ in range(8)]
def evaluate(S, ablate=False):
    net.eval(); net.ablate_past=ablate; cor=n=0
    with torch.no_grad():
        for s1,s2,st,m,y in S:
            lg=net(s1,s2,st,m)[:,a.lag:]; cor+=((lg>0).float()==y).sum().item(); n+=y.numel()
    net.ablate_past=False; net.train(); return cor/n
def train_loop(steps, drop_p, freeze_head=False):
    net.drop_p=drop_p
    if freeze_head:
        for p in net.enc_head.parameters(): p.requires_grad=False
        for p in net.pi.parameters(): p.requires_grad=False
    else:
        for p in net.parameters(): p.requires_grad=True
    params=[p for p in net.parameters() if p.requires_grad]
    opt_local=torch.optim.Adam(params, lr=1e-3)
    curve=[]; best=-1; best_step=0; best_state=copy.deepcopy(net.state_dict()); first60=None
    for it in range(1, steps+1):
        s1,s2,st,m,y=batch(0,i1,a.B,rng)
        loss=lossf(net(s1,s2,st,m)[:,a.lag:], y); opt_local.zero_grad(); loss.backward(); opt_local.step()
        if it%100==0:
            acc=evaluate(VAL); curve.append({"step":it,"val_acc":acc,"loss":float(loss)})
            if first60 is None and acc>=0.60: first60=it
            if acc>best+0.001: best,best_step,best_state=acc,it,copy.deepcopy(net.state_dict())
            print(json.dumps(curve[-1]), flush=True)
            if it>=a.min_steps and it-best_step>=a.patience: return curve,best,best_step,best_state,first60,"early_stop",it
    return curve,best,best_step,best_state,first60,"cap",steps

curve=[]; stop="cap"; steps_run=0; first60=None; best_step=0; best=-1; best_state=None
if a.mode=="nodrop":
    curve,best,best_step,best_state,first60,stop,steps_run=train_loop(a.cap, 0.0)
elif a.mode=="drop50":
    curve,best,best_step,best_state,first60,stop,steps_run=train_loop(a.cap, 0.5)
else:
    # stage1: head frozen effectively by drop_p=1.0 (always zero head features), train bypass
    c1,b1,bs1,st1,f1,s1,r1=train_loop(min(2000,a.cap), 1.0)
    net.load_state_dict(st1)
    c2,best,best_step,best_state,first60,stop,steps_run=train_loop(a.cap, 0.0)
    curve=c1+[{"stage2_start":True}]+c2; steps_run=r1+steps_run; best_step=best_step  # best from stage2

net.load_state_dict(best_state)
test=evaluate(TEST, ablate=False); test_ablate=evaluate(TEST, ablate=True)
res={"mode":a.mode,"seed":a.seed,"lag":a.lag,"test_acc":test,"test_acc_ablate_memory":test_ablate,
     "memory_reliance_drop":test-test_ablate,"best_val_acc":best,"best_step":best_step,"steps_run":steps_run,
     "stop_reason":stop,"steps_to_60_val":first60,"curve":curve,"runtime_s":round(time.time()-T0,1),
     "trainable_params":int(sum(p.numel() for p in net.parameters()))}
json.dump(res, open(OUT/f"step3_{a.mode}_L{a.lag}_s{a.seed}.json","w"), indent=1)
print("DONE", json.dumps({k:res[k] for k in ["mode","seed","lag","test_acc","memory_reliance_drop","best_step","steps_run","stop_reason","runtime_s"]}))