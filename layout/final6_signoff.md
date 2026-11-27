# chip_p2_final6 sign-off (raijin_chip, sky130A, 50 MHz)

Commit 8b9ce37, config `openlane/config_chip_p2.json` (RSZ corners nom/max/min tt and ff, setup slack margin 0.3 / 0.2 ns, hold slack margin 0.6 / 0.6 ns), clock period 20 ns, clock uncertainty 0.25 ns, 5% timing derate. LibreLane (OpenLane 2) in the iic-osic-tools container. Signoff STA is post-route with extracted parasitics (step 57).

## Timing per corner

| Corner | Setup WNS (ns) | Setup TNS (ns) | Setup viol. | Hold WNS (ns) | Hold viol. |
|---|---|---|---|---|---|
| nom_tt | +0.140 | 0.00 | 0 | +0.395 | 0 |
| max_tt | -0.670 | -3.77 | 10 | +0.241 | 0 |
| min_tt | +0.954 | 0.00 | 0 | +0.492 | 0 |
| nom_ss | -18.665 | -23155.00 | 4131 | +1.320 | 0 |
| max_ss | -20.127 | -27309.96 | 4365 | +1.254 | 0 |
| min_ss | -17.099 | -19277.42 | 3824 | +1.297 | 0 |
| nom_ff | +6.131 | 0.00 | 0 | +0.071 | 0 |
| max_ff | +5.989 | 0.00 | 0 | -0.049 | 2 |
| min_ff | +6.340 | 0.00 | 0 | +0.150 | 0 |

## Physical verification and area

- Detailed-route DRC: 0
- Antenna-violating nets / pins: 1 / 1
- LVS errors: 0 (device diff 0, net diff 0)
- Magic DRC: 11325463 (all inside the OpenRAM macros, as in every run)
- Die bbox: 0.0 0.0 2200.0 2200.0 um; die area 4840000 um2; core area 4762570 um2
- Std-cell area: 1419220 um2; macro area 762850 um2; utilisation 0.458171
- Std cells: 303496; sequential cells: 8768
- Max slew / cap / fanout violations: 129991 / 3419 / 6984
- Power (vectorless, tt): 0.0009505847119726241 W
- Wirelength: 9421313 um

Magic DRC by rule (top 5 of 30 rules; the rules are the OpenRAM bitcell rules Magic's full deck does not waive, see docs/sram_kv.md):

```
  1232280  N-Diffusion spacing to N-well < 68 (diff/tap.9)
   930679  Local interconnect width < 34 (li.1)
   755692  poly overlap of poly contact < 10 (licon.8)
   749064  poly overhang of SRAM core transistor < 22 (poly.8)
   743732  Diffusion width < 30 (diff/tap.1)
```

## Worst setup paths

```
== nom_tt_025C_1v80 worst 5
  +0.140  u_core.ptab[1][0] -> u_core.u_rope.c_r[8]
  +0.152  u_core.decode -> u_core.g_dpu[1].u_dpu.part_r[1][16]
  +0.267  u_core.ptab[1][0] -> u_core.u_rope.s_r[8]
  +0.280  u_core.ptab[1][0] -> u_core.u_rope.c_r[6]
  +0.378  u_core.decode -> u_core.g_dpu[1].u_dpu.part_r[1][15]
== max_tt_025C_1v80 worst 5
  -0.670  u_core.decode -> u_core.g_dpu[1].u_dpu.part_r[1][16]
  -0.645  u_core.ptab[1][0] -> u_core.u_rope.c_r[8]
  -0.527  u_core.ptab[1][0] -> u_core.u_rope.c_r[6]
  -0.508  u_core.ptab[1][0] -> u_core.u_rope.s_r[8]
  -0.473  u_core.decode -> u_core.g_dpu[1].u_dpu.part_r[1][15]
```

Hold misses at max_ff (both on the registered bus inputs): addr[5] -> u_core.addr_r[5] -0.049 ns, addr[4] -> u_core.addr_r[4] -0.003 ns.

Antenna: one net over the ratio, u_core.u_acc.g_row[0].g_lane[2].t_r[15] at hold13991/A on met1 (422.9 against 400, 1.06x).

## Reading

- Meets 50 MHz at the nominal and fast-leakage typical corners (nom_tt +0.140 ns, min_tt +0.954 ns) and at every fast corner; max_tt (slow-leakage typical) misses by 0.67 ns on ten paths (decode -> DPU1 partial sum, page table -> RoPE angle).
- The slow corner (100 C, 1.60 V) misses by about 18-20 ns. The SRAM macro has only a TT timing view, so the ss numbers treat the SRAM as typical; about 26 MHz is realistic there.
- Detailed-route DRC 0, LVS clean (netgen: circuits match uniquely).
- The reported power is vectorless at the idle clock gate and is not an operating figure; see results/power_gl_sram_p2_speed2.md for the gate-level number.

Renders: layout/raijin_chip_final6_full.png (met1-met5) and layout/raijin_chip_final6_zoom.png (80 x 60 um of core logic), made with scripts/render_layout.py (KLayout batch mode).
