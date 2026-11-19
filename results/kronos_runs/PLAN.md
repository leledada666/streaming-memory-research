# Next plan — 2026-10-20 (bypass / injection roadmap)

Status: design only until E6-long finishes. Hard constraints: never touch Price bot; GPU BelowNormal + power-cap; synthetic / supervised only for the injection tests (no RL until the mechanism works).

## Honest caveats from E0–E6 (must not be forgotten)

| finding | implication |
|---|---|
| E0: Kronos S vs R0 only **2.1×** at bar 2000 (not 10×); launch/Python overhead dominates at B=1 | Latency narrative must be honest; `torch.compile` / CUDA graphs may help, but do not claim 10×. |
| E0: inverted SDPA mask bug (fixed 2026-10-18); only hit multi-token append-to-cache | Any new cache path (ΔK/ΔV) needs the same bool-mask convention: True = ALLOWED. |
| E1 FAIL: h encodes RV but **hurts** HAR; mkt adds ~0 | Do not sell "Kronos h as volatility feature". Overlay (old E4) stays shelved. |
| E2 FAIL: direction at chance | Direction from frozen h alone is dead. |
| E3 FAIL: trained bypass barely moves h (gate 0.12, mean \|dh\| 0.003); h_post worse than h_pre+mkt | Current emb bypass does **not** write market info into the backbone on real data. |
| E5 FAIL (lag-30 insensitive); E5b weak PASS (capacity-confounded) | Need param-matched depth test → E6. |
| E6 formal **FAIL** (c−b mean >.02 at L=5 and L=10, but seed0 at L=5 was −.001); strong L=10 signal (+.09 every seed); L=15 collapsed to chance; runs hit 4k cap (not plateaued); large val→test gap | Underfitting at L=15 is the first thing to rule out. Formal fail stands; do not re-tune criteria. |

---

## 1. E6 long-train (10k–20k) — rule out underfitting at L=15

**Hypothesis.** The 4000-step cap left (b)/(c) underfit at L=15; with longer train + early stop, (c) multis recovers recall and still beats (b) beyond capacity.

**Setup.** Same synthetic cue as E6 (`m_t=[c_t,z_t]`, `y_t=c_{t-L}`), frozen Kronos-small, Adam 1e-3, B=32, T=96, gate init −2. `max_steps=15000`, eval every 100, early stop patience **8** val-evals (~800 steps) with min_delta 0.001 after min 2000 steps. Test at best-val weights.

**Priority cells.** (b) emb, (c) multis (±10% params), (a) none control. Skip full-size (d) unless time. (e) direct only at L=15, 1 seed. L∈{5,10,15}; seeds 0,1,2 for (b)(c); (a) 1 seed/L.

**PASS (pre-registered, do not re-tune).**  
Primary: (c) mean test acc at L=15 ≥ **0.58** AND (c)−(b) mean ≥ **0.02** with **all 3 seeds** positive at L=15.  
Secondary: reproduce L=10 (c)>(b) every seed after longer train.

**If PASS → stop.** Do **not** start ΔK/ΔV yet; write up and wait for user.  
**If FAIL after plateau → go to §2.**

---

## 2. Cross-Attention / ΔK,ΔV KV-cache injection (only if §1 FAILs)

**Motivation.** E6 gates open in layers 0–5 and close late; residual-on-x may be the wrong write path for long memory. Streamed Prefix-tuning style: at each bar, from `m_t` predict low-rank ΔK,ΔV and **add** them into the attention KV of selected layers (0–5 preferred), with a learned gate. `W_k`/`W_v` stay frozen.

**Minimal prototype (Part C).** Same L=15 task, 2 seeds, vs (c) multis. Trainable: MLP(m)→ΔK/ΔV (rank r≈8–16) + gate per site. Streaming-compatible (one step of KV per bar).

**PASS (pre-register before first run):** ΔKV mean test ≥ (c)+**0.03** at L=15 (both seeds).

**Caveats.** Must not break S/R consistency (re-run a tiny consistency check). Respect the SDPA True=ALLOWED convention. Leave Price ≥2 GB free; BelowNormal + power cap.

---

## 3. Modality Dropout on DIRECT (Kronos→2B bridge)

When building the bridge that feeds market features into a larger LM head / 2B model: randomly drop the market stream (`m`) during training so the backbone path cannot rely on "cheat" current-`m` and must use injected memory. Rate schedule TBD (e.g. 0.3→0.5). Applies only after an injection path (§1 or §2) shows synthetic memory works.

**PASS sketch (later):** with dropout on, held-out lag recall stays within 0.02 of no-dropout; without injection, dropout collapses to chance (sanity).

---

## 4. Conditional semantic trigger task

Natural-language plan (short) + streaming `m_t`; model must fire / hold / cancel when market conditions match the plan. HAR / order-flow features are **inputs**, not prediction targets (E1 already showed h loses to HAR on RV). This is the product-shaped test for the injection mechanism, not a training objective for volatility.

**Gate to start:** §1 PASS or §2 PASS on synthetic memory. Do not mix this with RL until the trigger accuracy clears a pre-registered bar on a frozen backbone + trained injection + small head.

---

## 5. Latency narrative + `torch.compile` + Price GPU red line

- **Honest story:** S is ~2× R0 at 2000 bars on this laptop, not 10×; Qwen prefix-cache TTFT 60 ms vs fresh 501 ms is the clearer win for text. Kronos R fills 8 GB by ~step 800 (grow-only `torch.cat` KV).
- **Next engineering (non-blocking on injection science):** try `torch.compile` on Kronos step; measure S p95 before/after, pre-register "≥1.3×" as a soft win. Optional: fixed-size KV ring to keep R under VRAM.
- **Price GPU red line (always):**
  - Never kill / restart / inspect Price processes.
  - Before GPU work: `PriorityClass=BelowNormal`; try `nvidia-smi -pl 50` (log success/fail); leave ≈2 GB free if possible.
  - Every ~10 min: confirm Price python patterns still exist; if any vanish → kill **only** our e6/kronos children, restore power limit, INCIDENT note, stop agenda. Do not restart Price.
  - If GPU util >40% or temp >75 °C before start → wait / report, do not start.

---

## Suggested order tonight

1. Write this PLAN + Chinese brief.  
2. Pre-register E6-long in NOTES → run §1 (one GPU job at a time).  
3. If §1 FAIL and ≥2 h remain and Price healthy → minimal §2 prototype.  
4. Cleanup: restore power limit, no leftover processes, final report.
