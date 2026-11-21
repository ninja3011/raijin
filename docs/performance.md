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
