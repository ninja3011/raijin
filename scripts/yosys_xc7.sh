#!/bin/bash
# Open-source sanity synthesis of the Arty top for Artix-7 with Yosys (no Vivado
# needed). Gives a LUT/FF/DSP estimate; Vivado numbers are the ones to report.
# Usage: scripts/yosys_xc7.sh   -> results/yosys_xc7_stat.txt
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/results"
bash "$REPO/scripts/dock.sh" 'yosys -q -l results/yosys_xc7.log -p "read_verilog rtl/*.v fpga/rtl/*.v; synth_xilinx -family xc7 -top raijin_arty_top -flatten; tee -o results/yosys_xc7_stat.txt stat"'
tail -40 "$REPO/results/yosys_xc7_stat.txt"
