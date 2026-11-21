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
