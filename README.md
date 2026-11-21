# Raijin

Raijin is a single-head MXINT8 attention accelerator built from three reusable
blocks: a dot-product unit (DPU) array, a GEMM accumulate array, and a softmax
unit. One controller runs attention two ways on the same hardware:

- **Unfused**: S = QKᵀ into a score buffer, batch softmax per row, then O = P·V.
- **Fused**: keys stream past four query rows with base-2 online softmax, so the
  N×N score matrix is never stored.

A causal flag (CTRL bit 5) makes the fused pass causal, for prompt prefill in
decoder models. Fused query groups are pipelined back to back
(`docs/performance.md`).

A third mode, decode, is one KV-cache step of autoregressive generation: one
query row attends to the LEN keys and values already held in SRAM.

Q, K, V, O and the score buffer live in four 1 KB sky130 SRAM macros
(`docs/sram_kv.md`).

The softmax-to-V path has an exact mode (16-entry LUT, 9×8 multipliers) and two
multiplier-free modes where each probability is one or two powers of two, so
P·V becomes shifts and adds. Every RTL block is bit-exact with the Python golden
model in `model/`.

The same Verilog targets sky130 through OpenLane 2 (LibreLane) and the Arty
A7-100T through Vivado.

## Layout

| Path | Contents |
| --- | --- |
| `model/raijin_model.py` | Bit-exact golden model, quantisation helpers, FP32 reference |
| `model/accuracy.py` | Accuracy study of every numeric variant against FP32 |
| `rtl/` | Core RTL (Verilog-2005); `raijin_mem.v` models the SRAM macro |
| `tb/` | cocotb tests (block level and full pass) |
| `openlane/` | OpenLane 2 configs; `config_sram_p2.json` is the SRAM build, `macros/` its macro views |
| `fpga/` | Arty A7-100T top, UART bridge, constraints, Vivado scripts, host script |
| `docs/` | Numerics and architecture notes |
| `scripts/dock.sh` | Runs any command inside the `hpretl/iic-osic-tools` container |

## Quick start (WSL or Linux with Docker)

```bash
scripts/dock.sh python3 model/raijin_model.py        # golden model self-check
scripts/dock.sh python3 tb/run_tests.py all          # every cocotb test (Icarus)
scripts/dock.sh python3 tb/run_tests.py core --macro # core against the PDK OpenRAM model
scripts/dock.sh python3 model/accuracy.py            # accuracy table
RAIJIN_CONFIG=openlane/config_sram_p2.json scripts/run_openlane.sh   # sky130 RTL-to-GDS
```

The container image `hpretl/iic-osic-tools` bundles Icarus Verilog, Verilator,
cocotb, Yosys, LibreLane (OpenLane 2) and the sky130 PDK, so nothing needs to be
installed on the host besides Docker.
