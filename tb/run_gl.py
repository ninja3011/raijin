#!/usr/bin/env python3
"""Gate-level simulation of a hardened raijin_core netlist, for power annotation.

Builds the final netlist of a LibreLane run with the sky130_fd_sc_hd cell
models and the PDK's OpenRAM SRAM model, runs tb/test_core.py::test_gl_short
(one load, one fused pass, one decode step) and dumps a VCD whose net names
match the netlist, so OpenSTA's read_vcd can annotate it.

Usage (inside the tools container):
  python3 tb/run_gl.py openlane/runs/<tag>
"""
import os
import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tb"))
from run_tests import macro_model  # noqa: E402

CELLS = Path(os.environ.get("PDK_ROOT", "/foss/pdks")) / "sky130A" / "libs.ref" / "sky130_fd_sc_hd" / "verilog"


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    netlist = run_dir / "final" / "nl" / "raijin_core.nl.v"
    build_dir = ROOT / "sim_build" / f"gl_{run_dir.name}"
    build_dir.mkdir(parents=True, exist_ok=True)
    # physical-only cells (fill, decap, tap) have no logic; give any the cell
    # library lacks an empty module so the netlist elaborates
    stubs = build_dir / "physical_stubs.v"
    known = (CELLS / "sky130_fd_sc_hd.v").read_text()
    used = sorted({ln.split()[0] for ln in netlist.read_text().splitlines()
                   if ln.strip().startswith("sky130_")})
    missing = [m for m in used if f"module {m}" not in known and not m.startswith("sky130_sram")]
    stubs.write_text("".join(f"module {m}(); endmodule\n" for m in missing))
    print(f"{len(used)} cell types, {len(missing)} stubbed: {missing}")
    sources = [CELLS / "primitives.v", CELLS / "sky130_fd_sc_hd.v", stubs, macro_model(build_dir), netlist]
    runner = get_runner("icarus")
    runner.build(sources=sources, hdl_toplevel="raijin_core", build_dir=build_dir,
                 defines={"FUNCTIONAL": 1, "UNIT_DELAY": "#0"}, always=True,
                 build_args=["-gno-specify"], waves=True, timescale=("1ns", "1ps"))
    env = {"COCOTB_TOPLEVEL": "raijin_core", "COCOTB_TEST_FILTER": "test_gl_short",
           "RAIJIN_N": "16", "RAIJIN_D": "16", "RAIJIN_P": "2"}
    os.environ.update(env)
    runner.test(hdl_toplevel="raijin_core", test_module="test_core", test_dir=ROOT / "tb",
                build_dir=build_dir, extra_env=env, waves=True)


if __name__ == "__main__":
    main(sys.argv[1])
