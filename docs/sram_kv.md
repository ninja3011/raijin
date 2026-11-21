# SRAM buffers and KV-cache decode

## Why the flip-flop buffers had to go

The first sky130 layout kept Q, K, V, O and the score buffer in flip-flops:
14,824 flops at P = 2, with each row buffer read through a 16:1 mux that is
128 bits wide. Those muxes made the design hard to route. The router started
at 113k violations at P = 2, and the 4-DPU version never converged. The muxes
also caused most of the slew, capacitance and fanout violations, and the bus
write data fanned out to every storage flop, which is where the 4k hold
violations and 11.6k hold buffers came from.

## Memory organisation

Four `sky130_sram_1kbyte_1rw1r_32x256_8` macros (OpenRAM, 256 × 32 bits, byte
mask, one read/write port and one read port) hold everything:

| Words | Contents |
| --- | --- |
| 0 … N−1 | Q rows |
| N … 2N−1 | K rows |
| 2N … 3N−1 | V rows |
| 3N … 4N−1 | O rows |
| 4N + {g, j} | score buffer: lanes 2b and 2b+1 in bank b, 12 bits each |

Bank b stores bytes 4b … 4b+3 of every row, so the four banks together return a
full 128-bit row per port per cycle. The dataflow keeps one row per cycle:

| Port | Used for |
| --- | --- |
| Port 1 (read) | K rows while keys stream, Q rows at the start of a group, score reads in the softmax and P·V phases |
| Port 0 (read/write) | V rows for the accumulators, score and p-code writes, O row writes, host bus access while idle |

Only the P query rows of the current group are held in flip-flops (P × 128
bits). They are loaded in P cycles before the keys stream. SRAM read data is
registered once before use, so the macro's falling-edge read gets a full half
cycle, and the controller's drain counts absorb the two cycles of latency. A
simulation assertion flags any port-0 conflict. None occur in any test.

Small per-row state stays in flops: the E8M0 exponents, 1/l and e_o
(5 × 16 bytes).

For timing, the accumulator registers its product before the add, and the
output stage registers the selected accumulator row and its 1/l before the
24 × 13 normalising multiply.

The bus inputs are now registered. A write lands two edges after it is
presented. A read returns after three edges, and register 0x5003 reports this
latency. The UART bridge and the testbenches wait accordingly.

`rtl/raijin_mem.v` is a cycle-accurate model of the macro, used for simulation
and for the FPGA build, where it infers block RAM. `tb/run_tests.py core
--macro` instead runs the core against the PDK's own OpenRAM Verilog model,
with `RAIJIN_SKY130_SRAM` defined exactly as for synthesis.

## Cycle counts (RTL simulation, N = d = 16)

| Pass | P = 4, flops | P = 4, SRAM | P = 2, flops | P = 2, SRAM |
| --- | --- | --- | --- | --- |
| Unfused | 369 | 411 | 713 | 779 |
| Fused | 169 | 197 | 317 | 357 |
| Decode, LEN = 1 / 8 / 16 | — | 35 / 42 / 50 | — | 33 / 40 / 48 |

The SRAM version spends a few more cycles on Q-row loads, the two-cycle read
latency and the extra pipeline registers. Every output stays bit-exact with the
golden model.

## KV-cache decode mode

In autoregressive generation, token t needs its query to attend to the keys and
values of tokens 0 … t. The K and V rows of earlier tokens do not change, so a
KV cache keeps them and appends one row per step. In Raijin the K and V rows
already live in SRAM between passes, so the cache costs no new storage. Decode
mode adds the control:

- `LEN` (0x5010) is the number of cached keys, and `QROW` (0x5011) is the
  query row.
- `CTRL` bit 4 starts a decode step. It runs the fused dataflow for QROW's
  group only, streams keys 0 … LEN−1, and writes only O[QROW].
- e_vmax is taken over the LEN cached V rows, matching `attention_decode` in
  the golden model.
- The skip counter and the accumulator enables are masked to QROW's lane.

Per step, the host writes one Q, K and V row plus three exponents (51 bytes),
sets LEN and QROW, starts the step, and reads 17 bytes back. Without the cache,
producing the new token's causal output means rewriting all of Q, K and V
(816 bytes) and running a full 197-cycle pass at P = 4. With the cache, a step
takes 35 cycles at LEN = 1 and 50 at LEN = 16.

`test_decode_kv_cache` builds the cache one token at a time for three tensor
families and checks every step bit-exact against the model. Cosine similarity
against FP32 causal attention is 0.9998 to 0.99999 (`results/decode_N16_D16.txt`).

### Trade-offs

- **DPU utilisation.** A decode step has one query, so only one of the P DPUs
  and lanes does useful work. Splitting the keys across the P DPUs, with a final
  merge of P partial (m, l, acc) states, would cut a step to about LEN/P cycles.
  It needs a merge unit that rescales and adds P accumulators. That is future
  work.
- **Cache size.** N = 16 rows is the limit of this build. The 1 KB macros have
  room for N = 32 at P = 4 in the same address map, but the 4-bit row fields and
  the tests assume 16.
- **Memory-bound regime.** At this size compute dominates. In a real LLM,
  decode is bound by the bandwidth of streaming the cache, which is why KV-cache
  quantisation matters. Raijin's cache is already MXINT8, a quarter the size of
  FP32.
- **Causal prefill.** Full passes are still non-causal. A prefill that produces
  causal outputs for every row at once would need a per-row key limit in the
  fused stream. That is a small change, but it isn't built.

## sky130 layout (P = 2, `openlane/config_sram_p2.json`)

| Metric | Flip-flop buffers (run 7) | SRAM buffers (sram_p2_v2) |
| --- | --- | --- |
| Die | 2.21 × 2.22 mm (4.90 mm²) | 2.20 × 2.20 mm (4.84 mm²) |
| Standard-cell area | 1.94 mm² | 1.04 mm² + 0.76 mm² of SRAM |
| Flip-flops | 14,824 | 5,158 |
| Power, typical corner | 87 mW | 37 mW |
| Setup slack at 50 MHz, typical | +2.38 ns (about 56 MHz) | −3.75 ns (about 42 MHz) |
| Setup slack, fast / slow corner | +8.9 / −14.1 ns | +5.2 / −28.8 ns |
| Hold violations | 4,151 | 93 |
| Detailed-route DRC | 0 | 0 |
| Antenna-violating nets | 58 | 0 |
| LVS | clean | clean |
| Magic DRC | 0 | 0 in the standard-cell area (see below) |

The SRAM version halves the cell area and cuts power by 58%. It also clears
every antenna violation and almost all of the hold problems. The die does not
shrink: the macros take 0.76 mm², and the multipliers and accumulator lanes
need about 8 m of wire, which sets the routing area. The flop buffers were not
what drove that.

**Timing is worse.** The worst typical-corner paths are broadcast nets (1/l
and the rounding shift amount in the output normaliser) whose buffer trees
span the die. Splitting the multiply into two stages (run v4) did not help:
−4.3 ns. The next step is placement, not more pipeline stages: keep the output
normaliser and accumulators in one region, or raise placement density.

**Floorplan.** The macros sit in a column at the left edge with 120 µm channels
between them. A macro has its read port pins on the top edge and its
read/write port pins on the bottom edge, and the layout depends on both edges
facing open space:

| Floorplan tried | Result |
| --- | --- |
| Row across the middle, 2.1 × 1.2 mm | global routing did not converge |
| 2 × 2 grid, 1.7 mm | 100k overflowed tiles (73% track use) |
| 2 × 2 grid, 2.2 mm, 30% density | 38k overflow, mostly nets crossing macros |
| Rotated 90° column | power grid failed (the macro's met4 ring runs parallel to met5) |
| **Left-edge column, 2.2 mm, 30% density** | 6k initial overflow, routed to 0 |

**Magic DRC.** The full GDS check reports 11,325,463 errors. A script that
bins every error by location puts all of them inside the four OpenRAM macros
and none in the standard-cell area. Two runs with different placements gave
the identical count. The OpenRAM bitcells use SRAM-specific rules that Magic's
full deck does not waive. A check with the macros as abstract views reports
only `nwell.4` tap errors, an artefact of abstract standard cells.

**Library edit.** OpenRAM's analytic `.lib` gives every SRAM input pin a
0.04 ns maximum slew. No buffer can meet that, and OpenROAD aborts in
post-route repair. `openlane/macros/` holds a copy with that limit at 0.75 ns,
the same as the design's own slew constraint. The macro's only timing view is
TT, so the slow-corner numbers treat the SRAM as typical.

**P = 4.** Global placement diverged at 2.6 and 3.0 mm with routability-driven
placement. Without it, global routing was still congested after 25 iterations,
so the 4-DPU build is still not hardened.
