# Draft — "Address, Not Depth"

> Status: final (2026-11-15). E4: cross 3/3 1.000; GRU 3/3 1.000; FIR window-limited; trained/zero-shot shuffle/no-marker .47–.53. Addressability established; content-addressing mechanism **not** established.

## Title

**Address, Not Depth: A Given Pointer, Not More Layers, Sets the Recall Horizon**

## Abstract (≤200 words)

Streaming sequence models are optimized for storage: larger KV caches, faster appends, longer windows. We show storage is not the binding constraint. Using a frozen autoregressive financial-sequence backbone and a synthetic recall task, we separate **storage**, **addressing**, and **read**. At natural write SNR (0.1σ per dimension), forcing attention onto the target recovers a bit with .984/.944 accuracy at lags 15/60, while unmodified attention stays at .53: storage is sufficient. Yet additive per-layer bypass fails beyond ≈L10, and layers cannot manufacture an address or accumulate delay: writing the current cue everywhere yields .505, staggering c_{t−i} across layers yields .489, while putting the answer at layer 7 (L=7) returns 1.000. A one-parameter tap, a 33-parameter FIR, and a 69k MLP solve fixed-lag recall only inside their window (1.000 at L15, .50 at L30). Given a unique explicit pointer, however, 1–64 lag recall becomes a 1-bit write plus a 1-slot read: cross-attention reaches 1.000 at every lag, a GRU latch also reaches 1.000, and a fixed 16-tap window fails outside its span. Depth cannot create or accumulate an address. We do not claim the model invents an address from no cue; we release the pointer benchmark.

## Contributions

1. **Storage is sufficient at natural SNR.** Weak-write forced addressing (a=0.004, 0.1σ/dim, 3 seeds) reads .980–.992 at L15/L60; natural/uniform/wrong stay at .52. *Measured.*
2. **Fixed-lag recall is a buffer.** 1-param tap oracle 1.000; 33-param FIR-16 1.000 at L15/.501 at L30; 69.6k param-matched MLP 1.000/.500; cue trained only on {5,10} gives .498 at L15. *Measured.*
3. **Layers cannot manufacture an address or accumulate delay.** 1/2/8-site all 1.000 only when the cue is the target; all-current .505; staggered-cur .489; answer at layer 7 (L=7) 1.000. `multis>emb` is redundancy (.512 vs .509), not “layer=hop”. *Measured.*
4. **Given a pointer, the target becomes addressable; mechanism open.** Without a cue, natural read at L≥10 is chance after selection correction (.516 cross-fit, null p99 .523) and L5 test .545 equals zero-training .546. Given a fixed lag, an external cross-attention with relative position learns the address (L10 .893 vs .496 without position, 1 seed; the .998/.893 val-test gap still needs a period holdout). With random markers at arbitrary lags, cross-attention reaches 1.000 at 1–8/9–16/17–32/33–64 in 3/3 seeds; shuffle/no-marker .47–.53; tap-fixed15 ≤.544; FIR16 .48–.54 outside its window. **A GRU latch also reaches 1.000 on this single-marker/single-query task**, so E4 does not isolate cross-attention; E4b's cross-only interface is asymmetric and not a clean test. We claim addressability, not address invention.
5. **Boundary and benchmark.** Using a given pointer ≠ inventing an address from no cue; multi-hop/compositional depth and real streams are untested. We release the pointer/marker benchmark for both. *Design + negative boundary.*

## Results

### R1. Storage is sufficient; fixed-lag recall is a buffer

Weak-write forced addressing at natural SNR (a=0.004, 0.1σ/dim, 3 seeds) reads `.980–.992` at L=15/60, while natural/uniform/wrong stay at `.52`. Storage and readout are not the binding constraint.

| Arm | Params | L=15 | L=30 |
|---|---:|---:|---:|
| tap-only oracle | 1 | 1.000 | — |
| FIR-16 ridge | 33 | 1.000 | .501 |
| MLP-16, param-matched to GRU | 69,633 | 1.000 | .500 |
| cue trained only on {5,10} | 528 | .498 | — |

The fixed-lag task is solved by a 16-tap buffer; a recurrent summary matching it is a delay line, not memory.

### R2. Layers cannot manufacture an address or accumulate delay

| Injection | Target | Acc |
|---|---|---:|
| 1 / 2 / 8 sites, same lag | 15 | 1.000 |
| all sites, current cue | 15 | .505 |
| staggered c_{t−i} across layers | 15 | .489 |
| answer at layer 7 (L=7) | 7 | 1.000 |
| none | 15 | .481 |

A single site suffices when the cue is the target; staggering and current-cue injection fail. `multis>emb` (E6-long: `.512 vs .509`) is redundancy, not “layer=hop”.

### R3. A given pointer makes arbitrary-lag recall addressable (E4)

Task: `m_t=[c_t, marker_t]`; one random marker at s∈[0,t−1]; target c_s; the lag is never given. Cross-attention uses relative position, K/V from m_{≤t}, strict causal mask.

| Arm / condition | 1–8 | 9–16 | 17–32 | 33–64 | overall |
|---|---:|---:|---:|---:|---:|
| cross, real (3 seeds) | 1.000±0 | 1.000±0 | 1.000±0 | 1.000±0 | 1.000±0 |
| gru128, real (3 seeds) | 1.000±0 | 1.000±0 | 1.000±0 | 1.000±0 | .994±.008 |
| cross, shuffle trained (3) | .503±.021 | .492±.027 | .479±.023 | .498±.024 | .528±.020 |
| cross, nomarker trained (3) | .501±.021 | .500±.023 | .512±.013 | .514±.027 | .494±.014 |
| cross, shuffle zero-shot (3) | .466±.034 | .503±.016 | .487±.028 | .513±.026 | .493±.020 |
| cross, nomarker zero-shot (3) | .518±.035 | .475±.018 | .493±.003 | .508±.030 | .521±.010 |
| tap-oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tap-fixed15 | .501 | .544 | .491 | .488 | .502 |
| fir16, real (3 seeds) | 1.000±0 | .929±.003 | .500±.021 | .527±.029 | .571±.009 |
| fir16, shuffle zero-shot (3) | .496±.026 | .498±.046 | .503±.034 | .486±.024 | .494±.007 |
| fir16, nomarker zero-shot (3) | .506±.035 | .504±.066 | .507±.004 | .486±.021 | .500±.006 |
| gru128, shuffle zero-shot (3) | .467±.036 | .502±.017 | .493±.032 | .514±.030 | .494±.021 |
| gru128, nomarker zero-shot (3) | .482±.035 | .525±.018 | .507±.003 | .492±.030 | .478±.010 |

Causal checks: future perturbation max|Δlogit|=0; cross future attention mass=0. **Cross and GRU both replicate 3/3 at 1.000 (GRU overall .994); trained and zero-shot shuffle/no-marker are all chance (.47–.53); FIR real .571 with long buckets .50/.53.** Pre-registered “learned content addressing” is **not established**: the GRU also reaches 1.000, so the single-marker task does not isolate cross-attention. E4b's cross-only query-key interface is asymmetric and not a clean mechanism test. We claim **addressability**: a given pointer makes the target addressable; we do not claim the model invents an address from no cue.

## Figures

**Fig. 1 — Attention reach and the natural read boundary.**
x: lag L ∈ {1,2,3,5,8,10,12,15,20,30,60} (linear to 20, then compressed); y (left): normalized attention α(q_t→k_{t−L}) / uniform, log scale. Curves: mean over 8x8 heads with ±1 sd band; max single head. Horizontal reference at 1.0, vertical marker at crossover R̂≈10. y (right): amplitude-aligned linear-probe accuracy at the best site, chance at .5. Expected shape: monotone decay crossing 1.0 near L≈14 at T=96 (≈0.1T); probe .632 at L=1, .546 at L=5, .516 at L=10 (cross-fit). Inset: the invalid unit-‖w‖ probe (.507 at L=0, 2% of activation norm) as a cautionary control.

**Fig. 2 — Forced vs natural addressing (upper bound).**
x: L ∈ {0,1,10,15,30,60}; y: test accuracy. Three curves: `force` (flat 1.000, including the layer-0-write/layer-7-read variant with open markers), `natural` (.50–.60, slight decline), `wrong-lag` control (overlays natural); chance band at .5. Secondary panel: accuracy vs injection amplitude ratio {0.004, 0.01, 0.02, 1.0}; force .984–.998 at a=.004/.01, natural .50–.56, uniform/wrong .50–.58. Expected shape: force flat at ceiling across lags and depths — a storage-sufficient, addressing-deficient signature.

**Fig. 3 — Interface phase diagram.**
x: side-memory capacity S (bits/slots, log); y: achievable horizon L_max (bars, log). Regions: bypass (horizontal at L≈R̂, *independent of S* — the key visual); token append (L≈R/r, three diagonals for r=1/4/16, dashed, `[pending]`); recurrent summary (L≈S, diagonal); cross-attention + relative position (L≈S, diagonal above summary); position-free KV edit (horizontal at R̂, since amplitude-only edits cannot select a lag). Solid = measured, dashed = predicted. Single measured points for bypass at L=10 and L=15; an arrow from the force ceiling down to the bypass line marks the "addressing gap". Expected shape: bypass flat despite rising S; summary/cross-attention rising until capacity binds.

**Fig. 4 — Content vs position addressing.**
Two panels sharing y (accuracy) and x (retrieval distance, 1–60). Left: position query (target c_{t−L}); right: content query (query key drawn from the past at random distance). Each panel: bypass, cross-attention, GRU summary, plus shuffled-key and absent-key controls. Smoke (M10, 200–600 步, prefix-mean baseline confounded): cross content .664 vs baseline .594 (+.07); position PE .699@600 vs no-PE .583; bypass below baseline. The baseline is too high (task shortcut), so E4 is the clean test; Fig. 4 remains a design figure.

## Related work

**Streaming LLMs and KV-cache management.** These optimize storage and assume addressing is free — the assumption we test.
- *Efficient Memory Management for LLM Serving with PagedAttention* — Kwon et al., 2023. Block-level KV allocation; reduces fragmentation, not read reach.
- *H2O: Heavy-Hitter Oracle for Efficient Generative Inference* — Zhang et al., 2023. Eviction policy over KV entries; a capacity/interference story, orthogonal to addressing.
- *StreamingLLM: Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2024. Keeps sink plus recent tokens; its horizon is defined by recency, exactly the regime our L-sweep measures.
- *Leave No Context Behind: Infini-attention* — Munkhdalai et al., 2024. Compressive memory with a learned read; closest to our summary arm, but evaluated on perplexity, not on a lag-resolved read metric.

**Memory, retrieval, and adapters.** These supply storage and read mechanisms; we ask which ones cross the wall and why.
- *Memorizing Transformers* — Wu et al., 2022. kNN retrieval over past activations; content addressing, the case our Fig. 4 tests.
- *Flamingo: a Visual Language Model for Few-Shot Learning* — Alayrac et al., 2022. Frozen backbone plus cross-attention into a resampled memory; the architectural template for our cross-attention arm.
- *Recurrent Memory Transformer* — Bulatov et al., 2022. Learned recurrent summary tokens; the GRU/summary arm, with capacity as the binding constraint.
- *Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks* — Zhang et al., 2020. Additive side network on a frozen backbone — structurally our bypass arm, and therefore the arm our data predicts will cap at L≈R̂.

**Time-series foundation models.** The substrate; all are evaluated on forecast error, which cannot reveal a read-path limit.
- *Chronos: Learning the Language of Time Series* — Ansari et al., 2024. Tokenized TS pretraining; same backbone family, no lag-resolved memory diagnostic.
- *A Decoder-only Foundation Model for Time-Series Forecasting (TimesFM)* — Das et al., 2024. Long-context decoder; context length treated as capability, not measured read reach.
- *Kronos: A Foundation Model for the Language of Financial Markets* — Shi et al., 2025. The model we probe; our results are a property of this checkpoint, not of the family (unverified venue).

## Limitations

1. **The “depth” claim rests on a single-step layered update assumption.** Each step passes through the layers once; we have not tested multi-step recurrence, cross-layer loops, or adaptive computation. The claim is that layer count cannot substitute for temporal depth in this regime—not a universal law.
2. **Synthetic task.** iid ±1 bits, no regime shifts, autocorrelation, or heteroskedasticity. It isolates the mechanism at the cost of external validity.
3. **One primary model/asset.** Kronos-small on BTCUSDT 1m. Qwen/RWKV probes cover attention reach/state only, not the full task. Depth, pretraining corpus, and tokenizer may move the boundary.
4. **Period confound and seed variance.** L5 test .545 equals zero-training .546; L10 multis .605 (n=4, range .566–.622) and the 15k s0 continuation drops test to .566 while val rises. All L≥10 effects must be reported with period-level holdouts.
5. **The 1.000 is forced/oracle addressing at natural SNR.** Storage is sufficient, but no adapter learns the address; the positive readouts use a given target. The only learned arm (GRU .981) is a 16-tap delay line.
6. **Pointer benchmark, not a mechanism test.** E4 gives every sequence exactly one marker, so lag, capacity, collision, and noise are removed. Both cross-attention and a GRU latch replicate 3/3 at 1.000; the pre-registered content-addressing criterion (GRU ≤.55) fails. We claim only that a given pointer makes the target addressable; we do not claim the model invents an address from no cue, and we do not attribute the result to cross-attention.
7. **No economic evaluation.** We claim no trading alpha; prior measurement in this program found no 1m taker-direction edge and an infeasible cost/volatility ratio.

## One-line answer: why is this not another ControlNet / Side-Tuning?

"ControlNet and Side-Tuning add capacity to a frozen backbone and assume the read path already works; we show the read path is the binding constraint — storage decodes at 1.000 while natural recall sits at chance beyond L≈10 — so Side-Tuning's additive bypass is precisely the arm our measurements predict will cap at L≈R̂, and our contribution is the diagnosis plus the interface taxonomy that says which adapters can escape it."

---

### Internal notes (not for submission)
- Numbers verified against `e6long_*.json`: L=10 n=5 mean .553 (range .521–.622), best-val up to .743; L=15 n=8 mean .508 (range .499–.512), best-val up to .549.
- The note "L=10 .62, L=15 .51" in the research log is best-seed at L=10 and rounds up at L=15; the paper must use the ranges above.
- Gates at L=15 reach .75–.95 with no accuracy gain (e.g. multis L15 s0, step 1800) — usable as evidence of "capacity spent, readout flat".
- Do not cite any Step2/Step3/Step4 number: no reproducible files exist (only a 100-step smoke at .5017).