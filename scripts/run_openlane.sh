#!/bin/bash
# Harden raijin_core on sky130A with LibreLane (OpenLane 2) inside the tools container.
# Usage: scripts/run_openlane.sh [extra librelane args, e.g. --last-run --from STEP]
# Results land in openlane/runs/<tag>/ ; the final GDS, netlist and reports are
# under openlane/runs/<tag>/final/.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${RAIJIN_RUN_TAG:-run_$(date +%Y%m%d_%H%M)}"
bash "$REPO/scripts/dock.sh" \
  "export STD_CELL_LIBRARY=sky130_fd_sc_hd; librelane --pdk-root /foss/pdks --manual-pdk -p sky130A --run-tag $TAG $* openlane/config.json"
