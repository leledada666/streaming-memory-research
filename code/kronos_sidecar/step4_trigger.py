# -*- coding: utf-8 -*-
"""Step4 minimal conditional semantic trigger.
Frozen Qwen2.5-1.5B-Instruct fp16. Plan text fed once (prefix cache). Then stream synthetic bars via emb-bypass only
(no text re-read). Head on last hidden -> logit close. Labels: close at first bar matching plan condition.
Baselines logged: no-plan (empty plan), rule-engine upper bound, text-refeed TTFT latency sample.
"""
from __future__ import annotations
import argparse, json, time, sys, copy, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(r"E:\rwkv7"); OUT = ROOT / "runs" / "roadmap_2026-10-20"; MDIR = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--B", type=int, default=4); ap.add_argument("--T", type=int, default=32)
ap.add_argument("--min_steps", type=int, default=800); ap.add_argument("--patience", type=int, default=600)
ap.add_argument("--latency_only", type=int, default=0)
a = ap.parse_args(); dev="cuda"; torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); T0=time.time()

PLANS = [
    {"id":"flow_shrink_low","n":3,"kind":"shrink","brk":"low",
     "text":"In a long position. If buy-side order flow shrinks for {n} consecutive bars AND price breaks the previous low, close immediately; otherwise ignore small pullbacks."},
    {"id":"flow_surge_high","n":3,"kind":"surge","brk":"high",
     "text":"In a short position. If buy-side order flow surges for {n} consecutive bars AND price breaks the previous high, close immediately; otherwise ignore."},
    {"id":"flow_shrink_low_5","n":5,"kind":"shrink","brk":"low",
     "text":"Long. Close only if flow shrinks for {n} bars in a row and the prior low is broken; else ignore."},
    {"id":"flow_surge_high_2","n":2,"kind":"surge","brk":"high",
     "text":"Short. Close if flow surges {n} consecutive bars and prior high breaks; else ignore."},
]
HOLD_OUT = "flow_surge_high_2"  # never seen in train

def make_episode(plan, g):
    T=a.T; ret=g.normal(0,1,T).astype(np.float32)
    flow=g.normal(0,1,T).astype(np.float32)
    # plant a trigger window
    t0=int(g.integers(plan["n"]+2, T-2))
    if plan["kind"]=="shrink":
        flow[t0-plan["n"]+1:t0+1] = -abs(g.normal(1.5,0.2,plan["n"])).astype(np.float32)
    else:
        flow[t0-plan["n"]+1:t0+1] = abs(g.normal(1.5,0.2,plan["n"])).astype(np.float32)
    # prior low/high flags: 1 at break bar
    brk=np.zeros(T,np.float32); brk[t0]=1.0
    # HAR-like vol proxy
    vol=np.abs(ret).astype(np.float32)
    # m_t = [ret, flow, vol, brk]  (brk is oracle feature "prior-low/high broken this bar")
    m=np.stack([ret,flow,vol,brk],1)
    # label close only at t0
    y=np.zeros(T,np.float32); y[t0]=1.0
    return m, y, t0

def rule_engine(plan, m):
    # upper bound using features
    flow=m[:,1]; brk=m[:,3]; n=plan["n"]; pred=np.zeros(len(m))
    for t in range(n-1,len(m)):
        window=flow[t-n+1:t+1]
        ok = (window< -0.5).all() if plan["kind"]=="shrink" else (window>0.5).all()
        if ok and brk[t]>0.5: pred[t]=1; break
    return pred

tok=AutoTokenizer.from_pretrained(str(MDIR), trust_remote_code=True)
base=AutoModelForCausalLM.from_pretrained(str(MDIR), torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
for p in base.parameters(): p.requires_grad=False
H=base.config.hidden_size; E=base.get_input_embeddings()

class Bridge(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(4, H), nn.GELU(), nn.Linear(H, H))
        s.g=nn.Parameter(torch.tensor(-2.0))
        s.head=nn.Sequential(nn.Linear(H, 128), nn.GELU(), nn.Linear(128, 1))
    def bar_embed(s, m):
        # m (B,T,4) fp32 -> emb (B,T,H) fp16
        u=s.enc(m.float()); u=u/(u.pow(2).mean(-1,keepdim=True).sqrt()+1e-6)
        return (torch.sigmoid(s.g)*u).to(dtype=next(base.parameters()).dtype)

bridge=Bridge().to(dev)
opt=torch.optim.Adam(bridge.parameters(), lr=1e-3); lossf=nn.BCEWithLogitsLoss()

def plan_prefix(plan, use_plan=True):
    text = (plan["text"].format(n=plan["n"]) if use_plan else "No special plan. Decide ignore or close from market features alone.")
    msgs=[{"role":"system","content":"You are a trading trigger. Output is handled by a head; just encode the plan."},
          {"role":"user","content":text}]
    ids=tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out=base(ids, use_cache=True)
    return out.past_key_values, ids.shape[1]

def forward_episode(m, past, use_plan_past=True):
    # m: (B,T,4)
    B,T,_=m.shape
    emb=bridge.bar_embed(m)  # B,T,H
    # run one forward with inputs_embeds, past_key_values from plan
    # For training simplicity: concatenate by running full bar sequence with cache from plan once
    out=base(inputs_embeds=emb, past_key_values=past if use_plan_past else None, use_cache=False, output_hidden_states=True)
    h=out.hidden_states[-1].float()  # B,T,H
    return bridge.head(h).squeeze(-1)

def batch(plans, g, use_plan=True):
    ms=[]; ys=[]; ps=[]
    for _ in range(a.B):
        plan=plans[int(g.integers(0,len(plans)))]
        m,y,_=make_episode(plan,g)
        ms.append(m); ys.append(y); ps.append(plan)
    m=torch.tensor(np.stack(ms),device=dev); y=torch.tensor(np.stack(ys),device=dev)
    # one plan prefix for the batch (same plan text approx - use first); for proper multi-plan would need B prefixes
    # Minimal: encode each uniquely is expensive; use plan of item 0 for all (still teaches plan conditioning weakly)
    # Better: loop B=1 style inside - keep B small
    return m,y,ps

train_plans=[p for p in PLANS if p["id"]!=HOLD_OUT]
val_plans=train_plans; test_plans=PLANS  # includes holdout

def eval_f1(plans, use_plan=True, n_ep=64):
    bridge.eval(); g=np.random.default_rng(42)
    tp=fp=fn=0; hits=0; ntrig=0
    with torch.no_grad():
        for _ in range(n_ep):
            plan=plans[int(g.integers(0,len(plans)))]
            m,y,t0=make_episode(plan,g)
            past,plen=plan_prefix(plan, use_plan=use_plan)
            mt=torch.tensor(m[None],device=dev)
            lg=forward_episode(mt, past)
            pred=(lg[0]>0).cpu().numpy(); gold=y
            # exact-bar hit
            if pred[t0]==1: hits+=1
            ntrig+=1
            for t in range(len(y)):
                if pred[t]==1 and gold[t]==1: tp+=1
                elif pred[t]==1 and gold[t]==0: fp+=1
                elif pred[t]==0 and gold[t]==1: fn+=1
    bridge.train()
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)
    return {"f1":f1,"exact_bar_hit":hits/ntrig,"tp":tp,"fp":fp,"fn":fn}

def rule_f1(plans, n_ep=64):
    g=np.random.default_rng(42); tp=fp=fn=0; hits=0
    for _ in range(n_ep):
        plan=plans[int(g.integers(0,len(plans)))]
        m,y,t0=make_episode(plan,g); pred=rule_engine(plan,m)
        if pred[t0]==1: hits+=1
        for t in range(len(y)):
            if pred[t]==1 and y[t]==1: tp+=1
            elif pred[t]==1 and y[t]==0: fp+=1
            elif pred[t]==0 and y[t]==1: fn+=1
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)
    return {"f1":f1,"exact_bar_hit":hits/64}

def latency(n=20):
    plan=train_plans[0]; g=np.random.default_rng(0)
    # streaming step: plan cached once, then one bar
    past,plen=plan_prefix(plan, True)
    m,_,_=make_episode(plan,g); mt=torch.tensor(m[None,:1],device=dev)
    # warmup
    for _ in range(3):
        torch.cuda.synchronize(); forward_episode(mt, past)
    ts=[]
    for _ in range(n):
        torch.cuda.synchronize(); t0=time.perf_counter()
        forward_episode(mt, past); torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
    # text refeed TTFT: tokenize plan+last 10 bars text each step
    bars_txt=" | ".join(f"r={m[i,0]:.2f},f={m[i,1]:.2f}" for i in range(10))
    msgs=[{"role":"user","content":plan["text"].format(n=plan["n"])+"\nRecent: "+bars_txt+"\nSay CLOSE or IGNORE."}]
    ids=tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(dev)
    for _ in range(3):
        torch.cuda.synchronize(); base.generate(ids, max_new_tokens=1, do_sample=False)
    tt=[]
    for _ in range(n):
        torch.cuda.synchronize(); t0=time.perf_counter()
        base.generate(ids, max_new_tokens=1, do_sample=False); torch.cuda.synchronize(); tt.append((time.perf_counter()-t0)*1000)
    return {"stream_step_ms_p50":float(np.median(ts)),"stream_step_ms_p95":float(np.percentile(ts,95)),
            "text_refeed_ttft_ms_p50":float(np.median(tt)),"text_refeed_ttft_ms_p95":float(np.percentile(tt,95))}

if a.latency_only:
    lat=latency(); json.dump(lat, open(OUT/f"step4_latency_s{a.seed}.json","w"), indent=1); print("DONE", lat); sys.exit(0)

# train
curve=[]; best=-1; best_step=0; best_state=copy.deepcopy(bridge.state_dict()); g=np.random.default_rng(a.seed)
for it in range(1, a.cap+1):
    # B episodes possibly different plans — process one-by-one to keep correct past
    loss_acc=0.0
    opt.zero_grad()
    for b in range(a.B):
        plan=train_plans[int(g.integers(0,len(train_plans)))]
        m,y,_=make_episode(plan,g)
        past,_=plan_prefix(plan, True)
        mt=torch.tensor(m[None],device=dev); yt=torch.tensor(y[None],device=dev)
        lg=forward_episode(mt, past)
        loss=lossf(lg, yt)/a.B; loss.backward(); loss_acc+=float(loss)*a.B
    opt.step()
    if it%100==0:
        ev=eval_f1(val_plans, True, n_ep=32)
        curve.append({"step":it,"val_f1":ev["f1"],"val_hit":ev["exact_bar_hit"],"loss":loss_acc})
        print(json.dumps(curve[-1]), flush=True)
        if ev["f1"]>best+0.001: best,best_step,best_state=ev["f1"],it,copy.deepcopy(bridge.state_dict())
        if it>=a.min_steps and it-best_step>=a.patience: break

bridge.load_state_dict(best_state)
with_plan=eval_f1(test_plans, True, 64)
no_plan=eval_f1(test_plans, False, 64)
hold=eval_f1([p for p in PLANS if p["id"]==HOLD_OUT], True, 64)
rule=rule_f1(test_plans, 64)
lat=latency(15)
# optional compile
compile_note="not_tried"
try:
    # measure stream with compile on bridge only (base frozen)
    compile_note="skipped_windows_risk"
except Exception as e:
    compile_note=str(e)

res={"seed":a.seed,"with_plan":with_plan,"no_plan":no_plan,"holdout_plan":hold,"rule_engine":rule,"latency":lat,
     "best_val_f1":best,"best_step":best_step,"steps_run":it,"curve":curve,"compile":compile_note,
     "runtime_s":round(time.time()-T0,1),"hold_out_id":HOLD_OUT,
     "trainable_params":int(sum(p.numel() for p in bridge.parameters()))}
json.dump(res, open(OUT/f"step4_trigger_s{a.seed}.json","w"), indent=1)
print("DONE", json.dumps({"seed":a.seed,"with_f1":with_plan["f1"],"no_f1":no_plan["f1"],"hold_f1":hold["f1"],
    "rule_f1":rule["f1"],"stream_ms":lat["stream_step_ms_p50"],"text_ms":lat["text_refeed_ttft_ms_p50"],"runtime_s":res["runtime_s"]}))