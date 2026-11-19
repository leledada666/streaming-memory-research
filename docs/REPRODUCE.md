# Reproduction guide

The code is archived **as run**. It is not a library: most scripts execute at import, write a
fixed output path, and contain absolute paths from the original machine. Two groups exist:

- **Self-contained (no model, no data):** M12d+, E4, E4-gate, E4b. Synthetic iid ±1 streams on
  CPU. These should run anywhere with the dependencies below.
- **Model-dependent:** M1/M1b/M12/E3/FIR/staggered/M2–M4/E7/rate/CA-vs-PA and the Kronos
  sidecar. These need a frozen Kronos-small checkpoint, a tokenised BTCUSDT 1m stream, and
  (for a few) a Qwen2.5 or RWKV7 checkpoint. None of those assets are in this repo.

## Environment used

- Python 3.12, PyTorch 2.x (CUDA for Kronos probes, CPU for E4/M12/FIR)
- numpy, pandas, scikit-learn, scipy
- For Kronos/Qwen/RWKV: their own modelling code and checkpoints, loaded read-only

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install torch numpy pandas scikit-learn scipy
```

## Path assumptions

Most model-dependent scripts begin with:

```python
ROOT = Path(r"E:\rwkv7")
```

and expect:

```
E:\rwkv7\kronos_src\                 # Kronos source (NOT included)
E:\rwkv7\data\binance\klines_1m\bars_1m_of_3m.csv
E:\rwkv7\runs\e6_long_2026-10-20\e5_tokens.npz
E:\rwkv7\runs\<experiment>\           # outputs
```

To rerun elsewhere, change `ROOT` (and the `OUT` constant where present) at the top of each
script. Output paths for the self-contained CPU scripts are the original temp directory
(`C:\Users\lele\AppData\Local\Temp\opencode\...`); change `OUT` to redirect them.

## 1. Storage vs addressing (M12d+) — self-contained

The cleanest entry point and the basis for Fig. 2. Frozen random 8-layer / 8-head / d=256
transformer, T=96, B=96, iid ±1 stream, injection at layer 0, read at layer 0/7, per-head ridge
probe. No model download.

```bash
python code/addressing/m12_mass_vs_address.py
# -> m12_mass_vs_address_results.json
```

Expected: `force` 1.000 at L=0/15/30/60 and both read layers; `wrong` ≈ .50; `uniform` ≈ .56–.58
at L≥15; `natural` .99 at L=0 but .55–.57 at L≥15. The amplitude scan shows force is flat from
0.05× to 1.0× activation norm, while uniform does not move with amplitude.

The Kronos version of the same design (weaker write, natural SNR, 3 seeds) is
`code/addressing/m12_e3.py`; it needs the Kronos assets and writes `m12_e3.json`. Result:
force `.984/.944` at L=15/60 with `a=0.004` (0.1σ/dim), natural/uniform/wrong `.50–.58`.

## 2. Pointer benchmark (E4) — self-contained

Task: `m_t=[c_t, marker_t]`, exactly one random marker at `s∈[0,t−1]`, target `c_s`; the lag is
never given. Cross-attention gets relative position and a strict causal mask. FIR-16 and a
causal GRU-128 are parameter-aligned controls.

```bash
# Full run used in the paper: T=96, B=32, 4000 steps, 3 seeds
python code/addressing/e4_addressing.py --T 96 --B 32 --steps 4000 \
    --arms fir16,gru128,cross --conds real --seeds 0,1,2 \
    --out results/e4 --tag full

# Zero-shot controls are produced by the same script with --conds shuffle,nomarker
# (see the archived JSON: e4_cross_controls.json, e4_gru_fir_full.json)
```

Expected (3 seeds): cross `1.000` at every bucket 1–8 / 9–16 / 17–32 / 33–64; gru128 `1.000`
except overall `.994`; fir16 `1.000 / .929 / .500 / .527`; shuffle and no-marker `.47–.53`.

**Important negative result:** the GRU also reaches 1.000 on 33–64, so the pre-registered
"GRU ≤ .55" criterion fails. E4 supports *addressability*, not a cross-attention mechanism.
See `results/e4/e4_full_report.md`.

`code/addressing/e4_gate.py` is the gate/interface control; `code/addressing/e4b_query_addressing.py`
is the asymmetric cross-only follow-up (not a clean mechanism test).

## 3. Delay-line controls — self-contained data, model only for staggered

`code/buffers/fir16.py` is pure synthetic: 16 taps of `m_t=[c_t,z_t]`, closed-form ridge.

```bash
python code/buffers/fir16.py     # -> fir16.json
```

Expected: tap-oracle 1.000; FIR-16 1.000 at L15 and .501 at L30; param-matched MLP 1.000/.500;
cue trained only on {5,10} .498 at L15.

`code/buffers/stag.py` needs frozen Kronos-small and writes `stag.json`. Injection at layer
inputs with shifts `d_i = L` (all-same), `d_i = L−i` (staggered), or single sites; readout is a
ridge probe on the final residual stream.

## 4. Attention reach and natural read (M1 / M1b)

`code/probes/lookback_probe.py` measures normalized attention `α(q_{t+L}→k_t)` versus lag for
Kronos and Qwen:

```bash
python code/probes/lookback_probe.py --which kronos --T 96  --bs 32 --out k96.json
python code/probes/lookback_probe.py --which kronos --T 192 --bs 32 --out k192.json
python code/probes/lookback_probe.py --which qwen   --T 96  --bs 4  --qwen_layers 16 --out q96.json
python code/probes/lookback_probe.py --which qwen   --T 192 --bs 4  --qwen_layers 16 --out q192_16.json
```

Results are in `results/m1/`. Crossover `R̂` is where the ratio hits 1.0: ≈14 at T=96 for
Kronos, ≈0.1T; Qwen ≈0.05T.

`code/probes/m1b_*.py` are the natural-read probes (random unit direction, amplitude matched,
L=0 positive control, residual-stream sites, 3 seeds). `m1b_perm.py` builds the permutation
null; `m1b_clean.py` is the 3-seed decay; `m1b_site_probe.py` and `m1b_scale_sweep.py` are the
site/amplitude follow-ups. Outputs are the CSVs in `results/m1b/`.

## 5. Memory adapters (M2–M4, E7, rate, content-vs-position)

All need the frozen Kronos carrier and the E6 token file.

| Script | Question | Output |
|---|---|---|
| `code/memory/m2_memory.py` | gated bypass vs summary | console/JSON |
| `code/memory/m2b_causal.py`, `m2b_ctx.py` | causal context / depth (E6) | `results/kronos_runs/` |
| `code/memory/m3_crossmem.py`, `m3_crossmem_v2.py` | cross-attention read with/without position | `results/m3/` |
| `code/memory/m3_runner*.py`, `m3_summarize.py` | batch runner + table | `results/m3/` |
| `code/memory/m4_append.py`, `m4_poscheck.py`, `m4_smoke.py` | token-append vs bypass | JSON |
| `code/memory/e7_lagcue.py` | capacity + unseen-lag generalisation | `results/e7/` |
| `code/memory/e7_adapter_retention.py` | adapter retention control | `results/e7/` |
| `code/memory/rate_phase.py` | observations-per-bar phase diagram | `results/rate_phase/` |
| `code/memory/ca_pa_probe.py` | content vs position addressing (M10) | `results/ca_vs_pa/` |

Example:

```bash
python code/memory/e7_lagcue.py --variant crossattn --T 96 --bits 1 --steps 4000
python code/memory/rate_phase.py --grid full --arm crossattn --task A --r 4 --lag 10 --T 16 --B 4
python code/memory/ca_pa_probe.py --T 32 --B 4 --seeds 0,1,2
```

## 6. Kronos sidecar (KV cache, ΔK/ΔV, streaming, RWKV)

These scripts assume a local, already-patched Kronos checkout. `_patch_kronos_kv.py` and
`_patch_kronos_forward_hidden.py` rewrite `model/module.py` and `model/kronos.py` **in place**
(the latter saves a `.bak`). Do not run them on a checkout you care about without a backup.

- `_probe_kronos_attn.py` / `step2_kv_inject.py`: low-rank ΔK/ΔV injection at layers 0–5.
- `step3_moddrop.py`, `step4_trigger.py`: modality dropout and a conditional trigger on Qwen.
- `e5_bypass_depth.py`, `e6_bypass_depth.py`, `e6_long_bypass.py`: bypass-depth (E5/E6).
- `kronos_cache_bypass_smoke.py`: KV-cache equivalence + bypass smoke.
- `bench_stream_vs_refeed.py`: streaming state injection vs re-feed latency/consistency.
- `rwkv_state_probe.py`: linear read of a bit from RWKV7 recurrent state.

```bash
python code/kronos_sidecar/rwkv_state_probe.py --n 768 --lags 0,1,5,10,15,30,60
python code/kronos_sidecar/bench_stream_vs_refeed.py --part kronos --dtype fp32
python code/kronos_sidecar/_probe_kronos_attn.py --lag 15 --sites 0,1,2,3,4,5 --rank 16
```

The Step2/Step3/Step4 numbers are **not citable**: only 100-step smokes exist. The scripts are
included for completeness, not as evidence.

## 7. Result map

| Claim | Primary files |
|---|---|
| Storage sufficient at natural SNR | `results/m12/m12_e3.json`, `results/kronos_runs/*` |
| Mass ≠ address | `results/m12/m12_mass_vs_address_results.json` |
| Fixed lag is a buffer | `results/fir/fir16.json` |
| Layers cannot hop/stagger | `results/stag/stag.json` |
| Pointer recall is addressable | `results/e4/e4_full_merged.json`, `e4_full_report.md` |
| Natural read is chance past L≈10 | `results/m1b/m1b_clean_results.csv`, `m1b_perm_summary.csv` |
| Bypass depth / E6 | `results/kronos_runs/e6long_*.json`, `e6_*.json` |
| Cross-memory with/without position | `results/m3/v2_smoke_table.json` |
| Lag cue / capacity | `results/e7/*.json` |
| Content vs position | `results/ca_vs_pa/*.json` |
| Rate phase | `results/rate_phase/*.json` |
| RWKV state read | `results/kronos_runs/probe_s0.json` |
| Streaming ≈ re-feed consistency | `results/kronos_runs/results_*.json` |
