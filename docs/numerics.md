# Raijin numerics and microarchitecture

This page records the exact integer behaviour that the RTL implements and the
golden model (`model/raijin_model.py`) copies. Where the implementation differs
from the original spec, the difference is listed at the end.

## Formats

| Signal | Format | Bits | Notes |
| --- | --- | --- | --- |
| Q, K, V elements | signed INT8, 6 fraction bits | 8 | value = x · 2⁻⁶ · 2^(E−127) |
| Row exponents e_q, e_k, e_v | E8M0 (bias 127), one per 16-element row | 8 | host folds log2(e)/√d into Q before quantising |
| Dot product q·k | signed integer | 20 | 16 products, max magnitude 2¹⁸ |
| Score s | signed Q7.4, saturating | 12 | s = sat(dot · 2^(e_q+e_k−262+offset)), floor on right shifts |
| Running max m | signed integer | 9 | m = max ceil(s/16); reset value −256 |
| Exponent argument t = 16m − s | unsigned, k = t>>4, f = t&15 | 15 | t ≥ 0 by construction |
| p, exact mode | unsigned Q1.8 | 9 | LUT[f] >> k, zero when k ≥ 9 |
| p, MF1 / MF2 | shift codes sa, sb (15 = no term) | 4+4 | p = 2^(8−sa) + 2^(8−sb) |
| Denominator l | unsigned Q5.8 | 13 | l ≥ 128 always (the max key contributes ≥ 0.5) |
| V row shift dv | e_vmax − e_v[j], clamped to 31 | 5 | applied to each p·v term |
| Accumulator | signed integer, Σ (p·v) >> dv | 24 | max magnitude 2¹⁹ |
| Reciprocal r | floor(2¹⁹ / l) | 13 | 13-cycle restoring divider per lane |
| Output | MXINT8 row | 8 + 8 | y = acc·r; shift by max(0, bitlen−7) with round-half-up; e_o = e_vmax + shift − 19 |

LUT[f] = round(256 · 2^(−f/16)) = 256, 245, 235, 225, 215, 206, 197, 189, 181, 173, 166, 159, 152, 146, 140, 134.

Multiplier-free tables (best fit of 2^(−f/16)):

| f | 0–3 | 4–8 | 9–12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- |
| MF1 | 2⁰ for f ≤ 6, 2⁻¹ for f ≥ 7 | | | | | |
| MF2 | 2⁰ | 2⁻¹ + 2⁻² | 2⁻¹ + 2⁻³ | 2⁻¹ + 2⁻⁴ | 2⁻¹ + 2⁻⁵ | 2⁻¹ + 2⁻⁶ |

## Recurrences

Fused (online), per query row, for each key j:

```
m'   = max(m, ceil(s_j))         delta = m' - m
p_j  = exp2(s_j - 16 m')         (exact LUT, MF1 or MF2)
l   <- (l >> delta) + p_j
acc <- (acc >>> delta) + ((p_j * v_j) >>> dv_j)      skipped when p_j = 0
```

Unfused (batch), per query row: pass 1 computes m over all keys, pass 2 writes
p codes back over the scores in the score buffer and sums l, then the P·V
phase accumulates with delta = 0.

When p_j = 0 the key can never have raised the max (that needs t < 16), so
delta is 0 and leaving the accumulator untouched is exact. The SKIPS counter
counts these (row, key) pairs. The fused dataflow skips fewer than the unfused
one because early keys are compared with a running max that is still low.

## Pipeline and schedule (N = d = 16, P = 4)

- DPU: 2 register stages. Requant is combinational on the DPU output.
- Softmax lane: stage A (max), stage B (exp2, l update), then the accumulate
  array registers the update. One key per cycle in fused mode.
- Output packer: 3 stages (multiply by r, leading-one detect, shift and round).
- The score buffer is banked by lane: lane r only ever touches query rows
  g·P + r, so bank r holds those 4 rows × 16 entries, with one read port and
  one write port. The entry index is {g, j}, which is also the pipeline tag.
- e_vmax (a pairwise max tree) and the per-row V shifts dv[j] are registered.
  They only change through the bus while the core is idle, so this costs no
  cycles and keeps the max tree off the accumulate path.

| Phase | Unfused cycles | Fused cycles |
| --- | --- | --- |
| QKᵀ (4 groups × 16 keys + drain) | 68 | streamed |
| Softmax (per group: clear 1, 2 × 16 keys, drain 3, reciprocal 15) | 4 × 51 | |
| P·V and output (per group: clear 1, 16 keys, drain 2, 4 rows) | 4 × 23 | |
| Fused group (clear 1, 16 keys, drain 5, reciprocal 15, 4 rows) | | 4 × 41 |
| Final drain and done | 5 | 5 |
| Total, measured | **369** | **169** |

Most of the remaining fused time is the reciprocal and output, which do not
overlap with the next group's stream yet. Overlapping them would bring a fused
pass to about 4 × 17 + 25 ≈ 93 cycles.

## Differences from the spec

- The output requant uses 1/l in both dataflows, so the "constant scale in
  unfused mode" path is not needed: unfused mode also normalises by 1/l.
- One reciprocal per softmax lane (4 in total, 13 cycles each) instead of one
  shared unit, so the four rows of a group normalise in parallel.
- Batch softmax makes two passes over a stored score row (max, then exp and
  sum); normalisation happens in the output packer, which removes the third pass.
- The running max is 9 bits so that a saturated score (ceil = 128) does not wrap.
- Score requant rounds towards minus infinity (arithmetic right shift).
