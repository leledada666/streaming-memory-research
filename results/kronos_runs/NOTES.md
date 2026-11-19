# Roadmap run 2026-10-20 (steps 1-4)  — started 03:33 Asia/Shanghai
Safety: Price bot (gui.server 7180/7900, run_gold_live 576/15416) never touched; watchdog every ~60 min kills only our jobs if any Price python vanishes.
nvidia-smi -pl unsupported on this laptop (logged 01:17); BelowNormal priority; one GPU job at a time.

## STEP 1  E6 long-train  (PRE-REGISTERED 01:16 in runs\e6_long_2026-10-20\NOTES.md; criteria unchanged)
Task y_t=c_{t-L}; frozen Kronos-small; cap 15000 steps, early stop patience 8 evals (800 steps), min 2000.
PASS: (c) multis mean test at L=15 >= .58 AND (c)-(b) mean >= .02 with all 3 seeds positive at L=15.
State at 03:33: L=15 emb x3, multis x3, none x1 done (01:17-01:46). Runner died ~01:47 during emb L10 s0 (no file). Remaining now: direct L15 s0, L10 emb x3, multis x3.


## STEP 1 progress 03:39
L=15 complete earlier (FAIL primary). Remaining: direct L15 done 03:36; emb L10 s0 running (~step 1900). Price=4. Watchdog now hourly.

## STEP 2  KV Delta-K/V injection  (PRE-REGISTERED 03:39, before any real Step2 train run)
Script scripts\step2_kv_inject.py. Layers 0-5; frozen W_k/W_v; dK/dV = gate * einsum(MLP(m), U) after RoPE on current tokens, written into KV (streaming-compatible). mid=128, rank=12, trainable_params=622477 (within 10% of multis 663305). DIRECT head included.
Task: same y_t=c_{t-L}. L=15 seeds 0,1,2 then L=10 seeds 0,1 if time. Cap 15000, patience 800.
Consistency: incremental cache vs full causal, max abs logit diff < 1e-4 (smoke: 2.2e-8 PASS).
PASS: mean test_acc(KV) at L=15 >= mean test_acc(multis_long L=15) + 0.03.
(multis_long L=15 mean = 0.512 from Step1 -> threshold 0.542.)
Smoke 100-step file moved to _smoke_step2_kv.json; not counted.
