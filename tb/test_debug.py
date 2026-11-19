"""Debug helper (not part of the regression)."""
import os
import sys

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, ReadOnly

sys.path.insert(0, os.path.dirname(__file__))
from test_core import setup, load, N, D  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import raijin_model as rm  # noqa: E402


@cocotb.test()
async def debug_first_fused(dut):
    bus = await setup(dut)
    rng = np.random.default_rng(99)
    qf, kf, vf = rm.random_case(rng, n=N, d=D, scale=0.7)
    args = rm.prepare(qf, kf, vf)
    await load(bus, *args)
    await bus.write(0x5001, 2)
    await bus.write(0x5000, 1 | 2 | 8)
    for cyc in range(200):
        await RisingEdge(dut.clk)
        await ReadOnly()
        st = str(dut.state.value)
        if st in ("00111", "01000", "01100") or cyc % 20 == 0:
            dut._log.info(f"c={cyc} st={st} g={dut.g.value} rv={dut.sm_rvalid.value} "
                          f"r={dut.sm_r.value} ov={dut.o_valid.value} otag={dut.o_tag.value} "
                          f"acc0={str(dut.acc.value)[-48:]} rinv0={dut.rinv[0].value} "
                          f"r_in={dut.u_out.r.value} em={dut.emax.value}")
        if str(dut.done.value) == "1":
            break
