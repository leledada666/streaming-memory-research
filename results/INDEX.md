# Result index

All files here are raw experiment output, not curated tables. Column names follow the scripts.
`synthetic` = iid ±1 stream, no market data; `Kronos` = frozen Kronos-small carrier on real
BTCUSDT 1m tokens.

## `m1/` — attention reach (zero training)

| File | What |
|---|---|
| `k96.json`, `k192.json` | Kronos T=96/192: per-layer, per-head `α(t+L→t)` ratio vs uniform, head-max, entropy |
| `kr96.json`, `kr192.json` | Kronos random-token control (same geometry, no market content) |
| `q96.json`, `q192.json`, `q192_16.json` | Qwen2.5-1.5B T=96/192, 12/16 layers |
| `q96_full.json`, `q192_full.json` | Qwen full-depth runs |

Read: ratio crosses 1.0 near L≈14 at T=96 for Kronos (~0.1T); Qwen ~0.05T. Used for Fig. 1
(left axis).

## `m1b/` — natural read decay and permutation null (Kronos, 3 seeds)

| File | What |
|---|---|
| `m1b_clean_results.csv` | per-site/head logistic probe accuracy at L=0,1,5,10,15,30,60; random unit direction, amplitude matched |
| `m1b_perm_real.csv` | real-label probe accuracy, L=0/10 |
| `m1b_perm_null.csv` | 200-permutation null distribution |
| `m1b_perm_crossfit.csv` | cross-fit accuracy by period |
| `m1b_perm_summary.csv` | null quantiles and corrected decision |
| `m1b_scaled_results.csv` | amplitude scale sweep |

Read: `.632 @L1 → .546 @L5 → .516 @L10` (cross-fit); null p99 `.523`, so L≥10 is chance after
selection correction.

## `m12/` — storage vs address

| File | What |
|---|---|
| `m12_mass_vs_address_results.json` | **self-contained CPU reconstruction**: force / wrong / uniform / natural / shuffled-target, layer 0/7, L=0/15/30/60, amplitude scan |
| `m12_e3.json` | Kronos weak-write grid: `a∈{0.004,0.01,0.02,1.0}`, L∈{15,60}, layer∈{0,7}, 3 seeds |
| `m12_e3.csv` | flattened E3 table |
| `m12_sweep.csv` | amplitude × lag sweep |
| `m12_scaled.csv` | scaled-injection follow-up |
| `m12_cross.csv` | cross-layer read |

Read: at `a=0.004` (0.1σ/dim), force `.984/.944` at L=15/60; natural/uniform/wrong `.50–.58`.
Mass alone (`uniform`) does not recover the bit; the address does.

## `e3/` and `fir/` and `stag/` — delay-line controls

| File | What |
|---|---|
| `e3/e3_weak_uniform.csv` | weak-write uniform-attention control |
| `fir/fir16.json` | tap oracle, FIR-16 ridge (33 params), param-matched MLP (69,633), cue-{5,10} arm |
| `stag/stag.json`, `stag.csv` | all-same / staggered / site {0,7} / site 0 / site 7 / none / wrong-shift |

Read: FIR-16 `1.000 @L15`, `.501 @L30`; MLP `1.000/.500`; staggered `.489`; all-current `.505`;
answer-at-layer-7 `1.000`.

## `e4/` — pointer benchmark (synthetic, self-contained)

| File | What |
|---|---|
| `e4_full_merged.json` | merged 3-seed full run, all arms/conditions |
| `e4_full_report.md` | rendered table + pre-registered verdicts |
| `e4_full.csv` | flat table |
| `e4_cross_full.json`, `e4_cross_controls.json` | cross-attention real + zero-shot shuffle/no-marker |
| `e4_gru_fir_full.json` | GRU-128 and FIR-16 arms |
| `e4_gate_reference.json`, `gate/e4_gate.json` | gate/interface reference run |
| `e4b_*.json` | E4b cross-only query interface (asymmetric; not a clean mechanism test) |
| `smoke/*.json` | diagnostics and short runs (not paper evidence) |

Read: cross `1.000` at 1–64 in 3/3 seeds; GRU also `1.000`; FIR-16 `.571` overall; shuffle /
no-marker `.47–.53`. **Verdict: addressability, not learned content addressing.**

## `m3/` — trainable cross-memory with/without position

| File | What |
|---|---|
| `v2_smoke_table.json` | full v2 arm × lag × position table |
| `v2_smoke_summary.json` | summary |
| `smoke_summary.json` | v1 summary |
| `m3_longsmoke*.json` | longer smoke runs (L10/L60) |

Read: with relative position, cross-memory lifts long-lag recall; without position it cannot
select the lag. Single-seed smokes; the paper treats this as directional, not final.

## `e7/` — lag cue, capacity, retention

| File | What |
|---|---|
| `e7_crossattn_1bit_T32_s0.json` | cross-attention with an explicit lag cue |
| `e7_gru_S16/S64/S256_*.json` | GRU state-size sweep |
| `e7_bypass_1bit_T32_s0.json` | bounded bypass control |
| `e7_l0_*.json` | L=0 positive control |
| `e7_e3ctl_*.json`, `e7_e3fix_*.json` | E3 control / fix |
| `e7_try_*.json` | hyperparameter tries (not evidence) |

## `ca_vs_pa/` — content vs position addressing (M10)

`smoke_ca_pa_T32_B4_*.json`, `smoke_600steps.json`, `smoke_bypass_direct.json`, `smoke_log.txt`.
Prefix-mean baseline is confounded; E4 is the clean test. Design figure only.

## `rate_phase/` — observations per bar

`rate_gru_A_r4_L10_T16_none_s0.json`, `rate_bypass_mean_A_r4_L5_T16_none_s0.json`.
Smoke-scale; the phase diagram is partly predicted.

## `kronos_runs/` — E5/E6, streaming, RWKV, toy trainers

| File | What |
|---|---|
| `e5_*.json`, `e5diag_*.json` | E5 bypass-depth (emb / multi / direct) |
| `e6_*.json` | E6 clean bypass-depth (none/direct/emb/multi/multis, L5/10/15, 3 seeds) |
| `e6long_*.json` | E6-long (15k cap) |
| `e1_results.json`, `e2_results.json`, `e3_results.json`, `e3_extract_meta.json` | earlier activation/attention extraction |
| `results_kronos_fp32*.json`, `results_rwkv_fp32.json`, `results_qwen_fp16.json` | streaming vs re-feed latency + action consistency |
| `probe_s0.json`, `probe_codecheck.json` | RWKV7 state read (L=0..60, amplitude/random-w) |
| `metrics.json`, `decision_step_test.json`, `one_step_test.json` | RWKV bypass toy trainers |
| `summary.json` | KV-cache + bypass smoke |
| `_smoke_step2_kv.json` | 100-step Step2 smoke — **not citable** |
| `NOTES.md`, `PLAN.md`, `runner.log` | run notes and job plan |

Read: E6-long L10 multis `.605` vs emb `.524` (deduped, 4k grid); L15 no gain. The paper's
`multis>emb` is redundancy, not "layer = hop" (see staggered control).
