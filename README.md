# Address, Not Depth

**A given pointer, not more layers, sets the recall horizon.**

Research artifact for a mechanistic study of memory in frozen autoregressive backbones with a
per-step external stream injection. The claim is narrow and measured:

> Storage is not the binding constraint. A frozen backbone can hold a bit written at natural
> signal-to-noise for 60+ steps, yet its natural read is at chance beyond ~10 steps. What is
> missing is an **address**, not capacity and not depth.

This repository contains the paper draft, the full research log, the experiment code, and the
raw result JSON/CSV for that study. It does **not** contain trading strategy code, third-party
model source, model weights, market data, or virtual environments.

- Paper draft: [`docs/paper-address-not-depth.md`](docs/paper-address-not-depth.md)
- Full research log (Chinese, chronological, including negative results): [`docs/research-log-zh.md`](docs/research-log-zh.md)
- M12 reconstruction note: [`docs/m12-mass-vs-address.md`](docs/m12-mass-vs-address.md)
- How to rerun: [`docs/REPRODUCE.md`](docs/REPRODUCE.md)
- Result index: [`results/INDEX.md`](results/INDEX.md)

---

## Headline results

| # | Claim | Measurement | Evidence |
|---|---|---|---|
| 1 | Storage is sufficient at natural write SNR | forced read `.984 / .944` at L=15/60; natural/uniform/wrong `.50–.58` | `results/m12/`, `results/kronos_runs/` |
| 2 | Fixed-lag recall is a delay line, not memory | 1-param tap `1.000`; FIR-16 (33 params) `1.000` at L15, `.501` at L30; 69.6k MLP `1.000 / .500` | `results/fir/fir16.json` |
| 3 | Layers cannot manufacture an address or accumulate delay | 1/2/8 sites all `1.000` only when cue = target; all-current `.505`; staggered `.489`; answer at layer 7 `1.000` | `results/stag/stag.json` |
| 4 | Given a unique pointer, arbitrary-lag recall becomes addressable | cross-attn `1.000` at 1–64 in 3/3 seeds; GRU128 also `1.000`; shuffle/no-marker `.47–.53` | `results/e4/` |
| 5 | Natural read is at chance beyond L≈10 | best-head `.632 → .546 (L5) → .516 (L10 cross-fit)`; permutation null p99 `.523` | `results/m1/`, `results/m1b/` |

The negative controls are part of the result: a GRU latch reaches `1.000` on the same
single-marker task, so **E4 does not isolate cross-attention**. E4 establishes
*addressability* — a given pointer makes the target readable — not a content-addressing
mechanism and not address invention from no cue.

## What this is not

- Not a claim that the model invents addresses. Every positive E4 arm receives an explicit
  marker at the target position.
- Not a universal depth law. The depth claim assumes a single pass through the layers per step;
  multi-step recurrence / adaptive computation is untested.
- Not a trading result. This is the mechanism study. No alpha, no PnL, no execution claims.
- Not a reproduction of third-party code. Kronos, Qwen and RWKV are used as frozen black boxes;
  their source is not redistributed.

## Layout

```
docs/                     paper draft, research log, M12 note, reproduction guide
code/
  probes/                 attention-reach and natural-read probes (M1, M1b)
  addressing/             forced vs natural addressing, pointer benchmark (M12, E3, E4, E4b)
  buffers/                tap / FIR / staggered / weak-write controls
  memory/                 bypass, cross-memory, summary, append, lag-cue, content-vs-position
  kronos_sidecar/         KV-cache patches, ΔK/ΔV injection, streaming benchmark, RWKV state probe
results/
  m1/ m1b/                attention reach, natural-read decay, permutation nulls
  m12/ e3/                storage-vs-addressing grids, weak-write sweeps
  fir/ stag/              delay-line and stagger controls
  e4/                     pointer benchmark (full, gate, smoke, E4b)
  m3/ m16/ e7/            cross-memory, LR scan, lag-cue capacity
  ca_vs_pa/ rate_phase/   content-vs-position, rate phase diagram
  kronos_runs/            E5/E6 bypass-depth, streaming-vs-refeed, RWKV probe, toy trainers
```

## Reproduction status

The scripts are archived **as run**, including hard-coded paths from the original machine
(`E:\rwkv7\...` and a temp output directory). They are not packaged as a library and are not
parameterised for arbitrary machines. `docs/REPRODUCE.md` lists what each script needs and
what it produced.

The two primary artifacts — the M12 storage/addressing grid and the E4 pointer benchmark —
run on CPU with a frozen random transformer and on a frozen Kronos-small checkpoint
respectively. M12 is self-contained. E4 needs a Kronos-small checkpoint; weights and the
BTCUSDT 1m token stream are **not** included here.

## Data and third-party components

- Synthetic streams are iid ±1 bits, generated in-script (seeded).
- Real-context experiments use BTCUSDT 1m bars, tokenised by a local Kronos tokenizer.
  Neither the bars nor the token files are redistributed.
- Kronos, Qwen2.5 and RWKV7 are referenced by name only. No third-party source, weights or
  environment is included.

## Status

Frozen 2026-11-15. The research log is intentionally left in its original chronological form,
including abandoned branches and corrections (the "common-command audit" section lists which
earlier numbers are **not** citable).

## License

Code and text in this repository are released under the MIT License (see `LICENSE`).
Third-party models and data remain under their own licenses and are not included.
