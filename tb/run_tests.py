#!/usr/bin/env python3
"""Build and run every cocotb test with Icarus Verilog.

Usage: python3 tb/run_tests.py [blocks|core|all] [--sim icarus|verilator]
Run it inside the tools container: scripts/dock.sh python3 tb/run_tests.py
"""
import argparse
import os
import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl"
SOURCES = [RTL / f for f in (
    "raijin_dpu.v", "raijin_requant.v", "raijin_exp2.v", "raijin_softmax_lane.v",
    "raijin_acc_array.v", "raijin_dupreg.v", "raijin_rope.v", "raijin_out.v", "raijin_mem.v", "raijin_core.v")]

BLOCKS = ["raijin_exp2", "raijin_requant", "raijin_dpu", "raijin_out", "raijin_softmax_lane", "raijin_rope"]


FPGA_SOURCES = [ROOT / "fpga" / "rtl" / f for f in ("uart_rx.v", "uart_tx.v", "raijin_bridge.v")] + \
    [ROOT / "tb" / "raijin_uart_tb_top.v"]


PDK_SRAM = Path(os.environ.get("PDK_ROOT", "/foss/pdks")) / "sky130A" / "libs.ref" /     "sky130_sram_macros" / "verilog" / "sky130_sram_1kbyte_1rw1r_32x256_8.v"


def macro_model(build_dir):
    """Copy of the OpenRAM behavioural model without its per-access log, which
    Icarus cannot elaborate (it indexes a memory inside $display)."""
    build_dir.mkdir(parents=True, exist_ok=True)
    dst = build_dir / "sky130_sram_1kbyte_1rw1r_32x256_8.v"
    lines = PDK_SRAM.read_text().splitlines()
    keep = [";" if ("$display" in ln and "mem[" in ln) else ln for ln in lines]
    dst.write_text(chr(10).join(keep) + chr(10))
    return dst


def run(top, module, sim, params=None, extra_env=None, macro=False):
    runner = get_runner(sim)
    sources = SOURCES + (FPGA_SOURCES if top == "raijin_uart_tb_top" else [])
    build_dir = ROOT / "sim_build" / f"{top}_{sim}{'_macro' if macro else ''}"
    build_args = ["-Wall"] if sim == "icarus" else ["-Wno-fatal", "-Wno-WIDTH"]
    defines = {"RAIJIN_TRACE": 1} if os.environ.get("RAIJIN_TRACE") else {}
    if macro:
        # simulate against the real OpenRAM macro model instead of rtl/raijin_mem.v
        sources = [s for s in sources if s.name != "raijin_mem.v"] + [macro_model(build_dir)]
        defines = dict(defines, RAIJIN_SKY130_SRAM=1)
    runner.build(sources=sources, hdl_toplevel=top, build_dir=build_dir, defines=defines,
                 parameters=params or {}, always=True, build_args=build_args, waves=False,
                 timescale=("1ns", "1ps"))
    env = dict(extra_env or {})
    env["COCOTB_TOPLEVEL"] = top
    os.environ.update(env)
    res = runner.test(hdl_toplevel=top, test_module=module, test_dir=ROOT / "tb",
                      build_dir=build_dir, extra_env=env)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["blocks", "core", "uart", "all"])
    ap.add_argument("--sim", default="icarus")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--p", type=int, default=4, help="DPUs / softmax lanes")
    ap.add_argument("--macro", action="store_true",
                    help="use the PDK's OpenRAM SRAM model (as synthesised for sky130)")
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    if args.what in ("blocks", "all"):
        for top in BLOCKS:
            run(top, "test_blocks", args.sim)
    if args.what in ("core", "all"):
        run("raijin_core", "test_core", args.sim, params={"N": args.n, "D": 16, "P": args.p},
            extra_env={"RAIJIN_N": str(args.n), "RAIJIN_D": "16", "RAIJIN_P": str(args.p)}, macro=args.macro)
    if args.what in ("uart", "all"):
        run("raijin_uart_tb_top", "test_uart", args.sim)


if __name__ == "__main__":
    sys.exit(main())
