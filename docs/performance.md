# Raijin performance

This file tracks throughput, area efficiency and energy efficiency as each
feature of the November 2026 upgrade lands. Numbers marked *estimate* come from
the cycle model below. Each is replaced with a measured number once its feature
is merged and hardened.

## Metrics

All figures are for N = d = 16 and one head, with MXINT8 inputs.

- **Work per fused pass**: QKᵀ is N·N·d = 4,096 MACs and P·V is another
  4,096 MACs. That is 8,192 MACs, or 16,384 ops.
- **Peak rate**: P DPUs × 16 MACs, plus P × 16 shift-add accumulate lanes,
  gives 32·P MACs per cycle. At P = 2 an ideal pass takes 128 cycles; at
  P = 4 it takes 64.
- **Utilization**: ideal cycles divided by measured cycles.
- **GOPS**: 16,384 ops divided by (cycles / f_clk).
- **GOPS/mm²** uses the whole die area. **GOPS/W** uses the total power at
  the typical corner.

## Baseline (sram-kv, P = 2 layout `sram_p2_v2`)

| Metric | Value |
| --- | --- |
| Fused pass | 357 cycles (RTL) |
| Clock, typical corner | about 42 MHz (−3.75 ns at 50 MHz) |
| Time per pass | 8.5 µs |
| Throughput | 1.93 GOPS |
| Utilization | 36% |
| Die | 4.84 mm² |
| Area efficiency | 0.40 GOPS/mm² |
| Power | 37 mW |
| Energy efficiency | 52 GOPS/W |
| Decode step, LEN = 16 | 48 cycles for one token |

### Where the fused cycles go (P = 2)

Each group of P query rows passes through these controller states:

| State | Cycles | Useful MAC work |
| --- | --- | --- |
| F_CLR | 1 | no |
| Q_LOAD | P = 2 | no |
| F_STREAM | N = 16 | yes |
| F_DRAIN | 7 | no |
| RCP_START, RCP_WAIT | about 15 | no |
| OUT_ISSUE, OUT_WAIT | P + 1 = 3 | no |

That is about 44 cycles per group across 8 groups. Groups run strictly one
after another, so the MAC arrays sit idle for about 64% of the pass.

## Feature log

| # | Feature | Status | Measured effect |
| --- | --- | --- | --- |
| 1 | Timing closure: four copies each of 1/l and the output shift amount (`rtl/raijin_dupreg.v`) | RTL merged; layout running | Cycles unchanged (357). The clock target is 50 MHz, which would give 2.29 GOPS (estimate) |
| 2 | Group pipelining in the fused dataflow | RTL merged | Fused pass 357 → **173** cycles at P = 2 (2.06×, 74% utilization) and 197 → **111** at P = 4 (1.77×). That is 3.9 GOPS at 42 MHz, or 4.7 GOPS at 50 MHz. Unfused and decode are unchanged |
| 3 | Causal prefill (CTRL bit 5) | RTL merged | One causal fused pass: **153** cycles at P = 2 and **97** at P = 4. Building the same outputs from 16 decode steps takes about 650 cycles at P = 2 and 680 at P = 4, so prefill is 4.2× to 7× faster, before counting the host traffic each decode step needs |
| 4 | Multi-query decode: GQA/MQA and speculative verify (CTRL bits 6, 7) | RTL merged | One step serves P query rows at LEN = 16: **49** cycles for 2 rows at P = 2, against 96 for 2 single steps (1.96×), and **53** cycles for 4 rows at P = 4, against 200 (3.77×). Decode throughput per cycle scales with P |
| 6 | MXINT4 KV cache (KVFMT 0x5012: bit 0 K, bit 1 V) | RTL merged | With K and V both INT4, the same SRAM holds **2N = 32 tokens** instead of 16, and every step is bit-exact. Accuracy cost is mean cosine 0.965 to 0.983 against 0.9996+ for INT8. V-only INT4 keeps 0.991 |
| 7 | Streaming KV cache with attention sinks (StreamingLLM) | Test and host policy, no RTL change | Generation runs past the 16-row cache. Over 48 tokens with 4 sinks, every step is bit-exact; mean cosine vs FP32 is 0.99983 overall and 0.99979 for tokens past the cache size |
| 8 | Hardware key skip and V-read gating (SKIPTHR 0x5013, VSKIPS 0x5014) | RTL merged | Cycles unchanged; this is an energy feature. With the gate alone (SKIPTHR = 0), a fused pass on sharp attention skips 15 of 64 V-row reads (23%), and 10 of 64 (16%) on outlier-heavy data. SKIPTHR = 6 raises that to 25 of 64 (39%) with cosine 0.99984 against 0.99986. Lower thresholds save more but cost accuracy on flat attention |
| 9 | RoPE on write (ROPE 0x5018, ROPEPOS 0x5019, `rtl/raijin_rope.v`) | RTL merged | No cycle or datapath-timing cost: rows are rotated on their way into SRAM. Bit-exact; mean cosine against FP32 RoPE attention is 0.9989 to 0.9999 across fused, unfused, causal and decode passes, including decode at absolute positions 200 to 215 |
| 10 | Paged KV cache (PAGED 0x501A, PTAB 0x501C..F) | RTL merged | No cycle cost: the page table sits on the K and V read addresses. Two sequences share one cache and one prompt page, and a 32-token INT4 cache is scattered over a shuffled page order. Every decode step is bit-exact with decode on the gathered logical sequence |
| 11 | Chip top: SPI host port and idle clock gating (`rtl/raijin_spi.v`, `rtl/raijin_chip.v`) | RTL merged; layout config added, not yet run | The whole register and memory map is reachable through 4 SPI pins. Passes over SPI are bit-exact. During a decode loop with 200-cycle idle gaps between tokens, the core clock is off **90.8%** of the time. SPI bandwidth (0.78 MB/s), not compute, now limits the end-to-end token rate |

### 1. Timing closure

The worst paths in `sram_p2_v2` are two broadcast nets: 1/l (13 bits into 16
multipliers) and the rounding shift amount (6 bits into 16 barrel shifters).
The lanes are spread across a 2.2 mm die. An extra pipeline stage did not help
(run v4, −4.3 ns), because the delay is in the wires, not the logic.

Both values are now held in RG = 4 copies, and each copy feeds four lanes
(`raijin_dupreg`). Yosys merges identical flip-flops even when they carry
`(* keep *)`, so the sky130 build defines `RAIJIN_SKY130_CELLS` and builds the
copies from `sky130_fd_sc_hd__dfxtp_1` instances. `opt_merge` only merges
built-in cell types, so those copies survive. A synthesis check of
`raijin_out` keeps all 24 shift-amount flops. Simulation uses a behavioural
register instead. Latency is unchanged, and every test is still bit-exact.

### 2. Group pipelining

Before this change, each fused query group ran its whole tail before the next
group could load: pipeline drain, reciprocal, output normalise and O write,
about 28 of its 44 cycles. Now group g+1 loads its Q rows and streams its keys
right behind group g. The only gap between two groups is the P cycles of the
Q load, which uses SRAM port 1.

| Mechanism | Where | What it does |
| --- | --- | --- |
| `first` / `last` key flags | `raijin_softmax_lane`, `raijin_acc_array` | On the first key of a group, the lane restarts m = −256 and l = 0 and the accumulators restart, without a clear cycle. On the last key, the lane latches the final l into its own divisor register and starts the reciprocal itself, so the next group's keys can update l while the division runs |
| `acc_snap` | `raijin_core` | The finished group's P accumulator rows are copied one cycle after its last update lands (`al4`). The reciprocal (14 cycles) and the output stage then read the copy while the live accumulators already serve the next group |
| Pipelined output issue | `raijin_core` | On the rising edge of the lane's `r_valid`, P output rows are issued from `acc_snap`, each with its own lane's 1/l |
| O write queue (2P entries, with bypass) | `raijin_core` | During streaming, SRAM port 0 carries a V read every cycle. O rows are written on the P free cycles each group leaves, straight through when the queue is empty. A simulation check flags queue overflow and port-0 conflicts; neither occurs |
| `P_WAIT` state | `raijin_core` | After the last group streams, the controller waits until all N O rows are written |

The Q-row register needs no second buffer. The next group's Q rows reach
`qreg` two cycles after their read, which is one cycle after the DPU has used
the previous group's last key.

**Cost (estimate):** `acc_snap` is P × 16 × 24 bits (768 flops at P = 2) and
the queue is 2P × 132 bits (528 flops). That is about 1,300 flops, roughly 25%
more than the 5,158 of `sram_p2_v2`. Measured area comes from the next
hardening run.

**Check:** every core test is bit-exact at P = 2 and P = 4. Decode mode (one
group) and the unfused dataflow keep their old control path. Setting
`RAIJIN_TRACE=1` for `tb/run_tests.py` prints a per-cycle trace of the
pipelined controller.

### 3. Causal prefill

Decoder models (GPT- and Llama-class) need causal attention: row i may only
attend to keys 0..i. Before this change, a full pass was non-causal, so a
prompt's KV cache could only be built one decode step at a time. Setting CTRL
bit 5 (fused, non-decode) now makes the pass causal.

- **Work is skipped, not just masked.** Group g streams only keys
  0 .. gP+P−1, so the pass processes about N²/2 key-row pairs instead of N².
- **Per-lane mask.** Inside a group's last P keys, lane r masks key j when
  j > gP + r. A masked key leaves m and l untouched and carries p = 0, so the
  accumulators skip it. The lane flags it on `out_mask`, so it isn't counted
  as an underflow skip.
- **Minimum group spacing.** Causal groups can be only 2P cycles long, which
  is shorter than the reciprocal (14 cycles) plus the output issue. A group
  may issue its last key only L_MIN = 14 + P cycles after the previous
  group's last key. Non-causal groups are 16 + P cycles long, so they never
  stall. That spacing is what holds causal at 153 cycles instead of about 110.
  A two-deep reciprocal and snapshot would remove the stall; that is noted as
  a follow-up.
- **Numerics.** e_vmax is still taken over all N V rows, as in a full pass.
  The golden model is `attention(..., fused=True, causal=True)`.

Measured accuracy against FP32 causal attention, as the mean per-row cosine
(`results/causal_N16_D16.txt`): 0.99984 random, 0.99992 sharp and 0.99981
with outliers in the exact mode, and 0.9992 to 0.9999 in MF2 mode.

The test `test_causal_prefill` covers three tensor families, all three exp
modes and two requant offsets. It is bit-exact with the model, and it checks
that a non-causal pass straight after a causal one is unaffected.

### 4. Multi-query decode (GQA / MQA and speculative-decoding verification)

A single-row decode step keeps only one of the P DPUs and accumulator rows
busy, yet it pays for every K and V read. CTRL bit 6 (with decode) runs all P
rows of QROW's group against the LEN cached keys in one step, so every K and
V row read from SRAM is used P times. The datapath is unchanged; only the lane
enables and the output issue differ.

| Use | How the host lays it out | Mask |
| --- | --- | --- |
| Grouped-query / multi-query attention | The P query heads that share one KV head go in rows gP .. gP+P−1 | none: every row sees keys 0..LEN−1 |
| Several sequences with one shared prompt (prefix sharing) | One query per sequence in rows gP .. gP+P−1 | none |
| Speculative decoding verification (bit 7) | P draft tokens: their K/V are the last P cached rows, and their queries go in rows gP .. gP+P−1 | lane r sees keys 0..LEN−P+r (causal among the drafts) |

Speculative decoding is how a small draft model's guesses get checked. With
bit 7, Raijin scores a whole P-token draft in the time of one decode step.

Measured at LEN = 16 (`results/multi_decode_N16_D16.txt`, P = 4 build):

| Build | One multi-query step | The same rows as single steps | Speed-up |
| --- | --- | --- | --- |
| P = 2 | 49 cycles | 96 cycles | 1.96× |
| P = 4 | 53 cycles | 200 cycles | 3.77× |

The golden model is `attention_decode_multi`. e_vmax is taken over the LEN
cached V rows for every row, as in a single-row step, so row r of a GQA step
is bit-identical to a single-row step on that row. `test_multi_query_decode`
covers GQA and speculative verify, three tensor families, LEN ∈ {P, N/2, N}
and the first and last groups, all bit-exact.

### 7. Streaming KV cache with attention sinks

Without this, a KV cache of N rows stops generation at N tokens.
StreamingLLM (Xiao et al., 2023) showed that keeping the first few tokens
(the "attention sinks") plus a window of the most recent tokens lets a model
generate indefinitely with fixed memory.

On Raijin this needs no new hardware. Attention does not depend on the order
of its keys (positions are already folded into Q and K by RoPE on the host).
The online softmax takes any key order, and decode mode takes any LEN and
QROW. The host's cache policy is `rm.streaming_slot(t, N, sinks)`: rows
0..sinks−1 stay pinned, and token t ≥ N overwrites row
sinks + (t − sinks) mod (N − sinks). Each decode step then attends over
LEN = N rows.

`test_streaming_kv_cache` generates 3N = 48 tokens with 4 sinks through the
real decode path. Every step is bit-exact with the model run on the mirrored
cache contents. The FP32 reference `reference_streaming` uses the same
sink-plus-window key set (`results/streaming_N16_D16.txt`):

| Tokens | Mean cosine | Min cosine |
| --- | --- | --- |
| All 48 | 0.99983 | 0.99900 |
| The 32 past the cache size | 0.99979 | — |

To be clear about the claim: this is a host-side policy that Raijin's
order-free dataflow supports at no cost. It is not a hardware block. The
hardware part of the claim is that a decode step costs the same whichever
rows are live.

### 6. MXINT4 KV cache

In LLM decode the KV cache, not the arithmetic, sets how long a context fits
on chip. The KVFMT register (0x5012) stores K (bit 0) and/or V (bit 1) as
MXINT4: 16 four-bit elements with one shared E8M0 exponent per token.

- **Layout.** An INT4 region packs two tokens per 128-bit SRAM word. Token t's
  eight bytes sit at byte offset 8t of the region, so the host writes them at
  0x1000 + 8t + b (K) or 0x2000 + 8t + b (V). Element 2b is the low nibble of
  byte b. With both bits set, the 16-word K and V regions hold 32 tokens, and
  decode takes LEN up to 32.
- **Datapath.** The K read uses word j >> 1 and the V read uses word
  acc_j >> 1. A nibble mux picks the token's half and widens each element to
  INT8 as x4 << 4. The MAC arrays, softmax and accumulators are unchanged,
  so this costs no extra cycles. With 2 fraction bits, x4 / 4 equals
  (x4 << 4) / 64, so the host's e_k and e_v keep their MXINT8 meaning.
- **Index widening.** The key index and the tags grew by one bit, to LK =
  log2(N) + 1. e_k, e_v and the dv shifts now have 2N entries, and the e_vmax
  tree spans 2N rows. The unfused key counters wrap explicitly at N. The extra
  state is about 340 flops (estimate).
- **Model.** `rm.mx_quantize4`, `rm.pack4` and `rm.prepare_kv4`. The INT8
  golden model applies unchanged because it is given x4 << 4.

`test_int4_kv_cache` checks full unfused, fused and causal passes with an INT4
K and V, a 32-token decode cache for three tensor families, and a causal pass
with INT8 K and INT4 V. All are bit-exact (`results/kv4_N16_D16.txt`).

**Accuracy.** Mean cosine against FP32 causal attention over a 32-token decode:

| Tensor family | INT8 K, INT8 V (model) | INT4 K, INT4 V (RTL) |
| --- | --- | --- |
| random | 0.99984 | 0.98307 |
| sharp (scale 2.5) | 0.99975 | 0.96525 |
| outliers | 0.99960 | 0.97689 |

A model-only sweep over 4 × 32 tokens per family shows where the loss
comes from:

| Family | K8 V8 | K4 V8 | K8 V4 | K4 V4 |
| --- | --- | --- | --- | --- |
| random | 0.99985 | 0.99504 | 0.99140 | 0.98683 |
| sharp | 0.99970 | 0.96558 | 0.99118 | 0.95747 |
| outliers | 0.99971 | 0.97452 | 0.99190 | 0.96656 |

Keys are the sensitive half when attention is sharp, because a 4-bit key
error moves the score inside the exponential. That matches what KIVI and
KVQuant report for LLM caches. In practice:

- Use KVFMT = 2 (INT8 K, INT4 V) when accuracy comes first. This shrinks V
  traffic but not the token count.
- Use KVFMT = 3 for twice the context.
- Feature 9 (adaptive precision) targets the gap: it would keep INT8 for
  tokens whose exponent spread shows outliers.

### 8. Hardware key skip and V-read gating

In decode, and in any pass over a long context, most keys get a softmax
weight that rounds to nothing. Before this feature, the hardware still
read each of those keys' V rows from SRAM and ran its accumulate beat with a
zero weight. Fetching a 128-bit SRAM word costs much more energy than the
add it feeds, so those reads were pure waste. (SpAtten and A3 make the same
argument for pruning keys.)

Two mechanisms, both in the fused, unfused, causal and decode dataflows:

- **V-read gating (always on).** The V row for a key is read only if at
  least one lane of the current P-row group will add it (`v_rd = acc_valid
  && |acc_en` in `raijin_core`). A key that underflowed (p = 0) in every row
  of the group costs no SRAM read and no accumulator toggle. The beat still
  takes its cycle, so timing and bit-exactness don't change. A saved beat
  also leaves SRAM port 0 free, so the O-write queue drains sooner.
- **Key-skip threshold (SKIPTHR, 0x5013, 4 bits, 0 = off).** A key whose
  score sits SKIPTHR or more powers of two below the running max gets p = 0
  and no multiplier-free terms. Its weight is below 2^-SKIPTHR of the
  largest one. The test is one 11-bit compare on t = 16·m − s, which the lane
  already computes for the exponential (`prune` in `raijin_softmax_lane`).
  Pruned keys count in SKIPS like underflows. In the fused dataflow the
  running max only rises, so a key pruned against the running max would also
  be pruned against the final max.

**Counters.** VSKIPS (0x5014..7, 32 bits) counts V-row reads saved in the
last pass. SKIPS still counts (row, key) pairs with p = 0.

**Model.** `rm.exp2(..., thr)`, plus `thr` on `attention`,
`attention_decode` and `attention_decode_multi`. `PassResult.used` records
which (row, key) pairs contributed. `rm.v_reads_saved(used, P, causal)` gives
the VSKIPS value the RTL must report.

`test_key_skip` covers SKIPTHR = 0, 6, 4 and 2 on three tensor families. Each
combination runs unfused, fused and causal passes in exact and MF2 modes,
plus two decode steps. Outputs, exponents, SKIPS and VSKIPS all match the
model exactly (`results/keyskip_N16_D16.txt`). The full suite passes at P = 2
and P = 4.

**Measured at P = 4, fused pass, exact mode** (64 V-row beats: 4 groups × 16 keys):

| Family | SKIPTHR | Zero-weight keys | V reads saved | Mean row cosine vs FP32 |
| --- | --- | --- | --- | --- |
| random | 0 | 0 / 256 | 0 | 0.99980 |
| random | 6 | 5 / 256 | 0 | 0.99979 |
| random | 4 | 42 / 256 | 0 | 0.99693 |
| sharp | 0 | 165 / 256 | 15 (23%) | 0.99986 |
| sharp | 6 | 182 / 256 | 25 (39%) | 0.99984 |
| sharp | 4 | 191 / 256 | 27 (42%) | 0.99971 |
| outliers | 0 | 115 / 256 | 10 (16%) | 0.99846 |
| outliers | 6 | 132 / 256 | 12 (19%) | 0.99719 |
| outliers | 4 | 155 / 256 | 19 (30%) | 0.93286 |

**What this shows, honestly.**

- The gate is free and always helps. It needs no threshold and changes no
  output.
- SKIPTHR = 6 (weights below 1/64) costs nothing measurable on random or sharp
  attention, but costs 0.0013 cosine on outlier-heavy data. SKIPTHR ≤ 4 is
  only safe when attention is known to be sharp.
- A V read is saved only when all P rows of a group skip the key, so savings
  shrink as P grows. Decode rows and P = 2 builds save more per key. Real LLM
  attention over long contexts is much sparser than these 16-key tests, so
  these figures are a lower bound on what a long-context decode would save.
- Cycles are unchanged. Turning saved beats into saved cycles would mean
  compacting the key stream ahead of the accumulators. That is left for later,
  because it would break the fixed schedule that the O-write queue relies on.

### 9. RoPE on write

Rotary position embedding (RoPE) is the position scheme of almost every
current decoder LLM, including LLaMA, Mistral, Qwen and Gemma. Q and K are
rotated pairwise by an angle proportional to the token position before QKᵀ,
so the score depends only on the relative position. On an edge system the
host is a small MCU, and computing sin/cos for every element of every Q and
K row is real work for it. Raijin now does the rotation itself.

**Where.** Real models cache K after RoPE, so the right place to rotate is on
the way into the KV cache, not on every read. With ROPE bit 0 set, writes to
the Q and K regions go through `raijin_rope`. The host writes raw rows
exactly as before.

- An **even byte** (element 2i) is latched. In the same cycle, the phase
  pos · ω_i is formed, and cos/sin come from a quarter-wave table. The phase
  is in turns, Q0.20, rounded to 10 bits, and the table holds 256 Q1.8
  entries.
- An **odd byte** (element 2i+1) completes the pair. The rotated pair is
  `x' = sat8((x·c − y·s + 128) >> 8)` and `y' = sat8((x·s + y·c + 128) >> 8)`.
  It goes to SRAM as one two-byte write in place of the host's single-byte
  write, so there is no queue, no extra cycle and no port conflict.
- The position is the row index, or the ROPEPOS register when ROPE bit 1 is
  set. ROPEPOS lets a streaming or paged cache place a token at its true
  absolute position, which can be anywhere in 0..255.
- Rows keep their MXINT8 exponent. A rotation preserves each pair's length,
  so saturation clips only pairs whose length is above 127.

The rotator is off the pass datapath entirely. The SRAM → DPU path, the one
the timing work in feature 1 is fighting, is untouched. It costs four 8 × 10
multipliers, one 8 × 18 phase multiplier, a 256-entry 9-bit table and about
30 flops. That is roughly 2 to 3 k cells (*estimate*, before synthesis).

**Conventions.**
- Pairs are interleaved (2i, 2i+1), with θ_i = 10000^(−2i/16). Models trained
  with the "half-split" layout (i, i + d/2) work unchanged after the same
  column permutation is applied to the Q and K projection weights, because
  QKᵀ is invariant to a shared permutation.
- The host writes each pair's even byte before its odd byte (natural order).
- RoPE does not apply to an MXINT4 K region.

**Design choice, measured.** Before fixing the arithmetic, I swept four
variants in the model over 24 random tensors. Each variant either halved the
result (no saturation, exponent + 1) or saturated it (exponent unchanged),
with Q1.7, Q1.8 or Q1.9 trig and 9 to 12 phase bits. Trig precision mattered
more than the output format. The halved Q1.7 version lost up to 0.002
cosine on outlier-heavy data. Saturating Q1.8 with a 10-bit phase was the
best trade for its table size, and it keeps the host contract simple: no
exponent adjustment.

**Verification.**
- `test_rope_rotator` (block level) compares the rotator with `rm.rope_row`
  for every position 0..255 and every pair, with random and extreme values:
  4,192 pairs.
- `test_rope` (core) writes raw Q and K with ROPE on. It runs unfused, fused
  and causal passes in exact and MF2 modes, plus a 16-token decode where each
  token is written at absolute position 200 + t through ROPEPOS.
- Everything is bit-exact with the model. The core suite passes 11/11 at
  P = 2 and P = 4.

**Accuracy** (P = 4, exact mode, mean row cosine against FP32 attention with
exact float RoPE, `results/rope_N16_D16.txt`):

| Family | Unfused | Fused | Causal | Decode, positions 200..215 |
| --- | --- | --- | --- | --- |
| random | 0.99983 | 0.99984 | 0.99990 | 0.99986 |
| sharp | 0.99941 | 0.99946 | 0.99981 | 0.99823 |
| outliers | 0.99893 | 0.99905 | 0.99921 | 0.99940 |

On the same tensors with and without RoPE (the model sweep above, fused
pass), RoPE costs 0.0001 cosine on random data, 0.0005 on sharp attention
and 0.0017 on outlier-heavy data. The loss comes from rounding the rotated
elements back to INT8, and it is largest where a few large elements dominate
a row. A wider rotated-row format, or per-row re-normalization after
rotation, would recover it; the adaptive-precision feature is the natural
place for that.

### 10. Paged KV cache

Until now, logical token j of the decode cache had to sit in physical row j.
That works for one conversation, but a real serving stack shares cache
memory between sequences, reuses common prompt prefixes, and evicts or
reorders tokens. With a flat cache, each of those means copying 16-byte K and
V rows over the slow host bus. vLLM's PagedAttention solved this for GPUs:
the cache is split into fixed-size pages, and each sequence keeps a page
table.

**Hardware.** Pages hold 4 tokens, so the 2N = 32-token cache has 8 pages.
PTAB has 8 entries of 3 bits, packed two per byte at 0x501C..F. With PAGED
bit 0 set, a decode step maps every logical key index j to a physical token:

`phys(j) = PTAB[j >> 2] · 4 + (j & 3)`

The physical token is used for:
- the K read address (`AK + phys`, or `phys >> 1` for an MXINT4 K region) and
  the K nibble half;
- the V read address and nibble half, and the V row's dv shift;
- the e_k lookup for the score requant;
- the e_vmax tree. It now spans the physical tokens that hold logical tokens
  0..LEN−1 (a 32 × 8 match against the table). So a sequence's output scale
  ignores pages that belong to other sequences.

Masks and the causal, multi-query and speculative logic all keep using
logical indices, so they are unchanged. The mapping is one 3-bit mux in front
of each address, and a pass takes exactly the same cycles. The cost is 24
flops for the table and the e_vmax valid mask (*estimate*: a few hundred
cells).

**What the host can now do without moving a row:**
- **Several sequences in one cache.** Each sequence has its own page list,
  and the host writes PTAB (four bytes) before each step.
- **Prefix sharing.** Pages holding a common system prompt appear in every
  sequence's table. They are written once.
- **Eviction and reordering.** Retiring a page or splicing in a different one
  is a table write. Combined with attention sinks (feature 7), a sink page can
  stay pinned while ring pages are reassigned.
- Works with the MXINT4 cache (feature 6) and with RoPE positions set through
  ROPEPOS (feature 9).

**Verification** (`test_paged_kv_cache`, `results/paged_N16_D16.txt`):
- Two INT8 sequences share prompt page 0. Sequence A owns physical pages 2 and
  3, and sequence B owns page 1. The pages are allocated interleaved, as
  tokens arrive. Generation alternates between A (12 tokens) and B (8 tokens).
- A 32-token MXINT4 cache is written through the shuffled page order
  [3, 6, 7, 0, 5, 4, 1, 2].
- Every decode step matches the model run on the gathered logical sequence,
  outputs and exponents bit for bit. The mean cosine against FP32 causal
  attention is 0.99986 (A), 0.99980 (B) and 0.97689 (INT4, as in feature 6).
- Measured step times at P = 4 are 46 cycles at LEN = 8 and 66 cycles at
  LEN = 32. The mapping is combinational on the address, so the controller
  schedule, and therefore the cycle count, is the same as for a flat cache.

Paging only applies to decode steps. Prefill and full passes read the
regions in order, and the host can lay those out with the identity table
(the reset value).

### 11. Chip top: SPI host port and idle clock gating

The core's parallel bus (16 address bits, 8 data bits, we and re) is fine
inside an FPGA, but a packaged edge chip needs a narrow host port that any
MCU can drive. `raijin_chip` is the new tapeout top level:

| Pin | Direction | Use |
| --- | --- | --- |
| clk, rst_n | in | core clock (50 MHz target), asynchronous reset |
| sck, cs_n, mosi | in | SPI mode 0 |
| miso (+ miso_oe) | out | SPI read data, driven only while selected |
| done | out | pass finished; use it as the host interrupt |
| busy | out | pass running |

**SPI protocol** (`raijin_spi`). The commands follow SPI SRAM conventions, so
stock MCU drivers work. The address auto-increments, and every data byte is
one core bus access:

- `0x02 AH AL d0 d1 ...` writes a burst. One burst loads a whole Q, K or V
  matrix (256 bytes) or an exponent vector.
- `0x0B AH AL dummy d0 d1 ...` reads a burst (fast-read style, with one dummy
  byte). Each byte's core read is issued one byte ahead, so the core's 3-cycle
  read latency is hidden.
- SCK, CS_N and MOSI pass through two-flop synchronizers into the core clock
  domain. That means SCK ≤ f_clk / 8, which is 6.25 MHz at 50 MHz. In return,
  there is no second clock domain to close timing on.

**Clock gating** (`raijin_chip`). The SPI target runs on the free-running
clock, and the core runs on a gated clock. The gate opens:

- while a pass runs;
- for each bus access the SPI target makes, plus HOLD = 7 cycles. That
  covers the 2-cycle write latency, the 3-cycle read latency and the done
  flag settling.

Under `RAIJIN_SKY130_CELLS` the gate is a sky130 integrated clock gate
(`sky130_fd_sc_hd__dlclkp_1`); otherwise it is a behavioural latch-and-AND.
An SPI byte is 64 clocks at f_clk / 8, so even while the host is streaming
data the core clock runs only about 8 of every 64 cycles. Between tokens,
while the MCU runs the rest of the model, it stops completely. The datapath
flops and their clock tree (most of the 37 mW measured on `sram_p2_v2`) see
no edges in that time. `gated_cycles` is a debug counter of cycles with the
core clock off.

**Verification** (`tb/test_chip.py`, `python3 tb/run_tests.py chip`). A
cocotb SPI master at 6.25 MHz drives the chip pins only.
- `test_spi_registers` covers the ID registers, a register round trip, and a
  burst write and read of an exponent vector.
- `test_spi_passes` loads Q, K, V and the exponents in six bursts, then runs
  fused, causal and unfused passes and reads O back in bursts. It then runs a
  16-token KV-cache decode loop, with one row per tensor written per token and
  a 200-cycle idle gap after each token.
- Everything is bit-exact with the model, at P = 2 and P = 4.

**Measured** (P = 4, `results/chip_spi_N16_D16.txt`):

| Quantity | Value |
| --- | --- |
| Full Q/K/V + exponent load | 834 bytes, 1.07 ms at 6.25 MHz SCK |
| Fused pass | 111 cycles, 2.2 µs at 50 MHz |
| Decode step, SPI traffic per token | 103 bytes (one Q, K and V row, three exponents, LEN/QROW, CTRL, one O row and its exponent, with command overhead) |
| Decode loop, 16 tokens | core clock gated for 90.8% of all cycles |

**What this shows, honestly.** With a single-lane SPI host, the chip
computes a full pass in about 2 µs but needs about 1 ms to receive its
operands. So end-to-end throughput for full passes is host-bound by a factor
of about 500. In decode, about 103 bytes per token take about 130 µs, which
caps the rate at about 7,500 tokens per second per head. That's still far
above what a small MCU can feed through the rest of the model. This is why
the on-chip KV cache (features 6, 7 and 10), on-chip RoPE (feature 9) and
multi-query decode (feature 4) matter. Each one cuts the bytes the host must
send per token. The next steps that follow from it:

- **QSPI** (4 data lines, same protocol with quad phases) gives 4× the host
  bandwidth for 3 more pins.
- **A faster SCK** with a real SPI clock domain and CDC FIFOs would remove the
  f_clk / 8 limit.
- **Output-only readback**, which reads just e_o and the rows that changed, is
  already possible with the burst reads.

**Layout.** `openlane/config_chip_p2.json` hardens `raijin_chip` at P = 2 with
the same floorplan as `sram_p2_v2`. The macro instances are renamed to
`u_core.sram*`. It has not been run yet. The gated clock goes through the
ICG, so CTS has to treat `u_icg/GCLK` as the core's clock root; that run will
show whether it does.

