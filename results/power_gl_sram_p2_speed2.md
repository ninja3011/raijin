# Gate-level power: sram_p2_speed2 (typical corner, nom SPEF)

Activity comes from a gate-level simulation of the routed netlist
(`final/nl/raijin_core.nl.v`, sky130_fd_sc_hd functional models, OpenRAM SRAM
model) running `tb/test_core.py::test_gl_short` through `tb/run_gl.py`. The
test loads one random case, runs one fused exact pass and one decode step,
and checks both against the golden model bit for bit. OpenSTA `read_vcd`
annotated 440,708 pins, 100% of them.

| Group (mW) | Whole test, 2,270 cycles (mostly host bus I/O) | Fused pass only, 173 cycles |
| --- | --- | --- |
| Sequential | 15.2 | 18.5 |
| Combinational | 4.4 | 28.2 |
| Clock | 15.9 | 15.9 |
| SRAM macros | 7.7 | 7.7 |
| **Total** | **43.2** | **70.3** |

- The fused-pass window is the first interval with `busy` high: 16.46 to 19.92 µs,
  or 3.46 µs at 50 MHz. `tb/vcd_window.py` cuts that window out of the dump.
- Energy per fused attention pass (N = d = 16, P = 2): 70.3 mW × 3.46 µs ≈ 243 nJ.
- Clock-tree power is 15.9 mW whether the core is busy or not. raijin_core has
  no clock gating; the chip top's idle gating removes it while idle.
- The SRAM figure is the OpenRAM lib's constant internal power, which doesn't
  depend on activity.
- Earlier vectorless figures (32 to 127 mW) assumed SRAM outputs never toggle,
  or propagated 0.1 per cycle through the XOR trees, so neither end was right.
