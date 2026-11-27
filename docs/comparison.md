# Raijin against prior attention accelerators

This note places Raijin next to the best-known published attention accelerators, normalises their numbers to Raijin's process (SkyWater 130 nm), and states what Raijin contributes that they do not.

## Published figures

| Design | Venue | Node | Clock | Area | Throughput | Energy efficiency | Status of numbers |
|---|---|---|---|---|---|---|---|
| A3 | HPCA 2020 | 40 nm | 1 GHz | 2.08 mm² | 221 GOP/s | 269 GOP/J | Synthesis (figures as tabulated by SpAtten, Table III) |
| SpAtten 1/8 | HPCA 2021 | 40 nm | 1 GHz | 1.55 mm² | 360 GOP/s | 382 GOP/J | Synthesis; SRAM from CACTI |
| SpAtten (full) | HPCA 2021 | 40 nm | not stated | 18.71 mm² | 1.61 TFLOPS (BERT), 0.43 TFLOPS (GPT-2) | 8.30 W total, about 70% of it DRAM | Synthesis; SRAM from CACTI, HBM from Ramulator |
| ELSA (one unit) | ISCA 2021 | 40 nm | 1 GHz | 1.255 mm² (2.15 mm² with its 36 KB memories) | 1.088 TOPS peak | about 730 GOPS/W (1.49 W with memories) | Post-layout (Design Compiler, place and route) |
| **Raijin, P = 2** | this work | 130 nm (sky130) | 49.3 MHz (tt) | 4.84 mm² die; 1.146 mm² std cells + 4 × 1 KB SRAM | 4.67 GOPS (N = 16 fused), 6.3 GOPS peak | 67 GOPS/W (70.3 mW during a fused pass, gate-level activity) | **Post-layout, LVS clean, open PDK, open flow** |
| Raijin, P = 8 (model) | this work | 130 nm | 45 MHz | not laid out | 23.0 GOPS peak, 22.9 GOPS at N = 128 | — | Cycle model validated against RTL |

Sources: SpAtten (arXiv 2012.09852, Tables I–III, Section V); ELSA (ISCA 2021, Table I, Sections V-C and V-D); A3 (arXiv 2002.10941), whose hardware figures here are the ones SpAtten reports for it.

## Normalised to 130 nm

First-order (constant-field) scaling from 40 nm to 130 nm, factor s = 130/40 = 3.25:
- clock divides by s (1 GHz becomes about 308 MHz);
- area multiplies by s² = 10.6;
- energy per operation multiplies by s × (V₁₃₀/V₄₀)² ≈ 3.25 × (1.8/1.1)² ≈ 8.7.

These factors are rough. Real 130 nm logic is usually slower than 308 MHz, so the normalised prior-work numbers are, if anything, optimistic for them.

| Design (at 130 nm) | Clock | Area | GOP/s | GOP/s/mm² | GOP/J |
|---|---|---|---|---|---|
| A3 | 308 MHz | 22.0 mm² | 68 | 3.1 | 31 |
| SpAtten 1/8 | 308 MHz | 16.4 mm² | 111 | 6.8 | 44 |
| ELSA (with memories) | 308 MHz | 22.7 mm² | 335 | 14.8 | 84 |
| Raijin P = 2 (measured clock) | 49.3 MHz | 4.84 mm² die | 4.67 | 0.96 | 67 |
| Raijin P = 2 at the same 308 MHz | 308 MHz | 4.84 mm² die | 29 | 6.0 | — |

How to read this honestly:
- On raw throughput the prior designs are 2–16x larger machines (128 to 1,024 multipliers against Raijin's 64 MAC lanes at P = 2), and run at a far higher clock. Raijin's peak per multiplier per cycle (2 ops) is the same as theirs; the gap is size and clock, not architecture.
- Area efficiency at equal clock is in the same range as A3 and SpAtten 1/8 even though Raijin's figure uses the whole padded die at 40% utilisation and real OpenRAM macros, and that it is a post-layout number rather than synthesis.
- Energy efficiency (67 GOP/J, measured from gate-level activity) is above the normalised A3 and SpAtten 1/8 figures and below ELSA's. This comparison is the least certain one: the prior figures are synthesis or CACTI estimates scaled by a rough voltage factor, and Raijin's SRAM power is a constant from the OpenRAM library.
- ELSA's lead comes from approximation (hash-based candidate selection skips most keys before any dot product) and from a 1 GHz post-layout clock in a commercial 40 nm library. Raijin computes every score exactly and gets its skipping after the score (key skip, feature 8), which is accuracy-safe but saves memory traffic rather than MACs.
- Raijin's 49.3 MHz is limited by the open sky130 standard cells and OpenRAM macros (the SRAM read path is on the critical path). The layout is 1.4% short of 50 MHz at tt.

## What Raijin contributes

The prior accelerators were designed for BERT-style encoders: a whole sequence arrives at once, Q, K and V are produced together, and attention is a dense batch problem. Edge LLM inference is different: the model generates one token at a time against a growing KV cache, so decode is memory-bound and the cache is the dominant cost. None of A3, SpAtten or ELSA has a KV cache, a decode mode or a positional encoding path.

Raijin's claims, each backed by a bit-exact test and a measured result in `docs/performance.md`:

1. **One datapath, three dataflows.** The same dot-product units and softmax lanes run unfused (baseline), fused streaming online softmax (the FlashAttention recurrence in hardware, with base-2 exponent and a deferred single reciprocal per row) and KV-cache decode, under one controller. Group pipelining hides the softmax tail and reaches 99–100% MAC utilisation for N ≥ 64.
2. **MX block arithmetic end to end.** Q, K, V and O are MXINT8 rows sharing an E8M0 exponent, and the exponent is carried through the score, the online maximum and the output requantiser. This costs +0.04% to +0.15% perplexity against FP32 in models trained for the project (`results/lm_perplexity.txt`, `results/lm_wikitext2.txt`) and +4.3% on pretrained TinyStories-8M with no retraining (`results/lm_eval.md`). SpAtten and ELSA use fixed-point formats chosen per model.
3. **A KV cache designed for decode.** Causal prefill, multi-query and grouped-query decode (P heads in the time of one), speculative-decode verification, streaming attention sinks, a paged cache with a page table, and per-page mixed INT8/INT4 precision. INT4 V costs +0.44% to +0.81% perplexity for half the V storage; keeping the sink page and worst page in INT8 recovers a quarter of the full-INT4 loss, and MXFP4 elements (E2M1, feature 16) cut it from 3.3% to 2.6% at the same 4 bits.
4. **Exact, accuracy-safe skipping.** Keys whose softmax weight underflows to zero are detected from the score's exponent and their V reads are gated off. At SKIPTHR = 6 this skips 25% of keys with no measurable perplexity change; a max-first decode mode (feature 17) finds the true maximum before any V work and skips 64% to 69% of V reads in the same models for +0.05% to +0.06% perplexity, with no approximation of the scores themselves.
5. **RoPE in the write path.** Keys are rotated as they enter the cache, with a quarter-wave table, so the host sends raw K and positional encoding costs no extra pass.
6. **A complete, reproducible chip.** SPI/QSPI host port with a token window that moves one decode token in one 80-byte burst (27.7k tokens/s over QSPI), idle clock gating, and a post-layout, LVS-clean sky130 implementation built entirely with open tools (OpenLane 2, OpenRAM). The RTL, golden model, testbenches, layout configurations and evaluation scripts are all in this repository.

In one sentence: Raijin is, to our knowledge, the first open-source, post-layout attention accelerator that combines MX-format arithmetic, a fused online-softmax dataflow and a decode-oriented KV cache (paged, mixed-precision, RoPE-on-write, with exact key skipping) in a single datapath, and it shows that this costs under 1% perplexity in transformers trained alongside it (Tiny Shakespeare, WikiText-2) and about 4% on an unmodified pretrained model (TinyStories-8M, no retraining or calibration).

The phrase "to our knowledge" should be kept: the claim rests on the papers surveyed here and has not been checked against every recent workshop paper.
