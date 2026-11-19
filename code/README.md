# Code map

Scripts are archived **as run**. They execute at import, use absolute paths, and write fixed
output files. Treat them as provenance, not as a package.

## `probes/` — is the read path local?

| Script | Needs | Produces |
|---|---|---|
| `lookback_probe.py` | Kronos or Qwen checkpoint | attention ratio vs lag (`k*.json`, `q*.json`) |
| `m1_attention_reach.py` | Kronos + E6 tokens | M1 summary |
| `m1b_clean.py` | Kronos + E6 tokens | 3-seed natural-read decay |
| `m1b_perm.py` | Kronos + E6 tokens | permutation null, cross-fit |
| `m1b_site_probe.py` | Kronos | per-site probe |
| `m1b_scale_sweep.py` | Kronos | amplitude sweep |
| `m1b_diag.py`, `m1b_diag2.py` | Kronos | diagnostics |
| `probe_env.py` | — | environment/version check |

## `addressing/` — storage vs address, forced vs natural, pointers

| Script | Needs | Produces |
|---|---|---|
| `m12_mass_vs_address.py` | **nothing** (CPU, synthetic) | `m12_mass_vs_address_results.json` |
| `m12_storage_vs_address.py` | Kronos | storage/address probe |
| `m12_e3.py` | Kronos + tokens | weak-write grid (`m12_e3.json`) |
| `m12_sweep.py`, `m12_scaled.py`, `m12_cross.py` | Kronos | amplitude/layer/cross-read sweeps |
| `e4_addressing.py` | **nothing** (synthetic, CPU/GPU) | pointer benchmark |
| `e4_gate.py` | **nothing** | gate/interface control |
| `e4_to_csv.py`, `e4_final_report.py` | E4 JSON | flat table + report |
| `e4b_query_addressing.py` | **nothing** | E4b cross-only interface |

## `buffers/` — is fixed-lag recall just a delay line?

| Script | Needs | Produces |
|---|---|---|
| `fir16.py` | **nothing** | tap/FIR/MLP/cue arms (`fir16.json`) |
| `fir16_local.py`, `fir_window.py` | **nothing** | FIR variants |
| `stag.py` | Kronos + tokens | staggered/all-same/site controls (`stag.json`) |
| `e3_weak_uniform.py` | Kronos | weak-write uniform control |
| `dedup_e6.py` | E6 JSON | de-duplicated E6 table |

## `memory/` — adapters on a frozen carrier

| Script | Question |
|---|---|
| `m2_memory.py` | gated bypass vs summary |
| `m2b_causal.py`, `m2b_ctx.py` | causal context / depth (E6 design) |
| `m3_crossmem.py`, `m3_crossmem_v2.py` | cross-attention read; v2 adds relative position |
| `m3_runner.py`, `m3_runner_v2.py`, `m3_summarize.py`, `m3_long2.py` | batch runs + tables |
| `m4_append.py`, `m4_poscheck.py`, `m4_smoke.py` | token append vs bypass |
| `e7_lagcue.py` | explicit lag cue, unseen lags, capacity |
| `e7_adapter_retention.py` | retention control |
| `rate_phase.py` | r observations per bar phase diagram |
| `ca_pa_probe.py` | content vs position addressing (M10) |

All need frozen Kronos-small and the E6 token stream (`e5_tokens.npz`).

## `kronos_sidecar/` — KV cache, ΔK/ΔV injection, streaming, RWKV

| Script | What |
|---|---|
| `_patch_kronos_kv.py` | **rewrites** `kronos_src/model/module.py` for grow-only KV + RoPE offset |
| `_patch_kronos_forward_hidden.py` | **rewrites** `kronos_src/model/kronos.py`, adds `forward_hidden` (+ `.bak`) |
| `_probe_kronos_attn.py` | ΔK/ΔV injection probe at layers 0–5 |
| `step2_kv_inject.py` | Step2 ΔK/ΔV trainer (smoke only, **not citable**) |
| `step3_moddrop.py` | Step3 modality dropout (smoke only, **not citable**) |
| `step4_trigger.py` | Step4 conditional trigger on Qwen (smoke only, **not citable**) |
| `e5_bypass_depth.py`, `e6_bypass_depth.py`, `e6_long_bypass.py` | bypass-depth E5/E6 |
| `train_bypass.py` | RWKV7 + gated-residual toy trainer |
| `decision_step_injector_test.py` | decision-step injector toy test |
| `kronos_cache_bypass_smoke.py` | KV-cache equivalence + bypass smoke |
| `bench_stream_vs_refeed.py` | streaming injection vs re-feed latency/consistency |
| `rwkv_state_probe.py` | linear read of a bit from RWKV7 recurrent state |

The patch scripts modify a third-party checkout in place. Back it up first. No third-party
source is included in this repository.
