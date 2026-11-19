#!/bin/bash
# Run a command inside the iic-osic-tools container with the repo mounted at /work.
# Usage: scripts/dock.sh <command...>
# The image bundles Icarus, Verilator, cocotb, Yosys, OpenLane 2 and the sky130 PDK.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${RAIJIN_IMAGE:-hpretl/iic-osic-tools:latest}"
exec docker run --rm -v "$REPO":/work -w /work --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/home --entrypoint /bin/bash "$IMAGE" \
  -c 'mkdir -p /tmp/home; for f in /etc/profile.d/*.sh; do . "$f" >/dev/null 2>&1; done; export PDK_ROOT=/foss/pdks PDK=sky130A; '"$*"
