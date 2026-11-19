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
    "raijin_acc_array.v", "raijin_out.v", "raijin_core.v")]

BLOCKS = ["raijin_exp2", "raijin_requant", "raijin_dpu", "raijin_out", "raijin_softmax_lane"]


def run(top, module, sim, params=None, extra_env=None):
    runner = get_runner(sim)
    build_dir = ROOT / "sim_build" / f"{top}_{sim}"
    build_args = ["-Wall"] if sim == "icarus" else ["-Wno-fatal", "-Wno-WIDTH"]
    runner.build(sources=SOURCES, hdl_toplevel=top, build_dir=build_dir,
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
    ap.add_argument("what", nargs="?", default="all", choices=["blocks", "core", "all"])
    ap.add_argument("--sim", default="icarus")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    if args.what in ("blocks", "all"):
        for top in BLOCKS:
            run(top, "test_blocks", args.sim)
    if args.what in ("core", "all"):
        run("raijin_core", "test_core", args.sim, params={"N": args.n, "D": 16},
            extra_env={"RAIJIN_N": str(args.n), "RAIJIN_D": "16"})


if __name__ == "__main__":
    sys.exit(main())
