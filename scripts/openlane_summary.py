#!/usr/bin/env python3
"""Summarise a LibreLane run into results/openlane_summary.md.

Usage: python3 scripts/openlane_summary.py openlane/runs/<tag> [output.md]
"""
import json
import os
import sys

KEYS = [
    ("design__die__bbox", "Die bounding box (µm)"),
    ("design__core__area", "Core area (µm²)"),
    ("design__instance__area", "Instance area, cells + macros (µm²)"),
    ("design__instance__area__stdcell", "Standard-cell area (µm²)"),
    ("design__instance__area__macros", "SRAM macro area (µm²)"),
    ("design__instance__count", "Cell instances"),
    ("design__instance__count__stdcell", "Standard cells"),
    ("design__instance__utilization", "Core utilization"),
    ("design__instance__count__class:sequential_cell", "Flip-flops"),
    ("design__instance__count__class:multi_input_combinational_cell", "Combinational cells"),
    ("timing__setup__ws__corner:nom_tt_025C_1v80", "Setup WS, typical (ns)"),
    ("timing__setup__ws__corner:nom_ss_100C_1v60", "Setup WS, slow (ns)"),
    ("timing__setup__ws__corner:nom_ff_n40C_1v95", "Setup WS, fast (ns)"),
    ("timing__setup__ws", "Setup WS, all corners (ns)"),
    ("timing__hold__ws", "Hold WS, all corners (ns)"),
    ("timing__setup_vio__count", "Setup violations"),
    ("timing__hold_vio__count", "Hold violations"),
    ("power__total", "Total power, typical (W)"),
    ("power__internal__total", "Internal power (W)"),
    ("power__switching__total", "Switching power (W)"),
    ("power__leakage__total", "Leakage power (W)"),
    ("route__wirelength", "Routed wirelength (µm)"),
    ("route__drc_errors", "Detailed-route DRC errors"),
    ("magic__drc_error__count", "Magic DRC errors"),
    ("klayout__drc_error__count", "KLayout DRC errors"),
    ("design__lvs_error__count", "LVS errors"),
    ("antenna__violating__nets", "Antenna-violating nets"),
]


def main(run, dst="results/openlane_summary.md"):
    path = os.path.join(run, "final", "metrics.json")
    if not os.path.exists(path):
        # flow still running: use the latest step's state
        steps = sorted(d for d in os.listdir(run) if d[:2].isdigit())
        for d in reversed(steps):
            p = os.path.join(run, d, "state_out.json")
            if os.path.exists(p):
                path = p
                break
    with open(path) as f:
        m = json.load(f)
    m = m.get("metrics", m)
    print(f"metrics from {path}", file=sys.stderr)
    lines = [f"# OpenLane 2 (LibreLane) results: `{os.path.basename(run.rstrip('/'))}`", "",
             "sky130A, sky130_fd_sc_hd, 50 MHz clock (20 ns).", "",
             "| Metric | Value |", "| --- | --- |"]
    for k, label in KEYS:
        if k in m:
            v = m[k]
            if isinstance(v, float):
                v = f"{v:.4g}"
            lines.append(f"| {label} | {v} |")
    ws = m.get("timing__setup__ws__corner:nom_tt_025C_1v80")
    if isinstance(ws, (int, float)):
        fmax = 1000.0 / (20.0 - ws)
        lines += ["", f"Estimated fmax at the typical corner: {fmax:.1f} MHz (1 / (20 ns - WS))."]
    out = "\n".join(lines) + "\n"
    os.makedirs("results", exist_ok=True)
    with open(dst, "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
