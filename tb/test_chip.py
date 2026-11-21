"""Chip-level cocotb tests: the host talks to raijin_chip over SPI only, with
idle clock gating on. Results must match the golden model bit for bit."""
import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import raijin_model as rm  # noqa: E402

N = int(os.environ.get("RAIJIN_N", 16))
P = int(os.environ.get("RAIJIN_P", 4))
D = int(os.environ.get("RAIJIN_D", 16))
CLK_NS = 20                    # 50 MHz
SCK_HALF_NS = 4 * CLK_NS       # SCK = f_clk / 8 = 6.25 MHz


class Spi:
    def __init__(self, dut):
        self.dut = dut
        self.bytes = 0

    async def _xfer(self, out_bytes, n_in=0):
        d = self.dut
        got = []
        d.cs_n.value = 0
        await Timer(SCK_HALF_NS, unit="ns")
        for i in range(len(out_bytes) + n_in):
            b = out_bytes[i] if i < len(out_bytes) else 0
            v = 0
            for k in range(7, -1, -1):
                d.mosi.value = (b >> k) & 1
                await Timer(SCK_HALF_NS, unit="ns")
                d.sck.value = 1
                v = (v << 1) | int(d.miso.value)
                await Timer(SCK_HALF_NS, unit="ns")
                d.sck.value = 0
            if i >= len(out_bytes):
                got.append(v)
        await Timer(SCK_HALF_NS, unit="ns")
        d.cs_n.value = 1
        await Timer(2 * SCK_HALF_NS, unit="ns")
        self.bytes += len(out_bytes) + n_in
        return got

    async def write(self, addr, data):
        data = [int(x) & 0xFF for x in (data if hasattr(data, "__len__") else [data])]
        await self._xfer([0x02, addr >> 8, addr & 0xFF] + data)

    async def read(self, addr, n=1):
        return await self._xfer([0x0B, addr >> 8, addr & 0xFF, 0x00], n)

    async def read32(self, addr):
        b = await self.read(addr, 4)
        return b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    dut.rst_n.value = 0
    dut.cs_n.value = 1
    dut.sck.value = 0
    dut.mosi.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    return Spi(dut)


async def load(spi, q, eq, k, ek, v, ev):
    for base, mat in ((0x0000, q), (0x1000, k), (0x2000, v)):
        await spi.write(base, np.asarray(mat).reshape(-1))      # one burst per tensor
    for base, ex in ((0x4000, eq), (0x4100, ek), (0x4200, ev)):
        await spi.write(base, ex)


async def wait_done(dut):
    for _ in range(20000):
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            return
    raise AssertionError("pass did not finish")


async def read_out(spi, rows):
    o = np.zeros((N, D), dtype=np.int64)
    eo = np.zeros(N, dtype=np.int64)
    if rows == list(range(rows[0], rows[0] + len(rows))):
        flat = await spi.read(0x3000 + rows[0] * D, len(rows) * D)
        ex = await spi.read(0x4300 + rows[0], len(rows))
        for i, r in enumerate(rows):
            o[r] = [b - 256 if b & 0x80 else b for b in flat[i * D:(i + 1) * D]]
            eo[r] = ex[i]
    return o, eo


@cocotb.test()
async def test_spi_registers(dut):
    """ID registers and a register write/read round trip over SPI."""
    spi = await setup(dut)
    ids = await spi.read(0x500C, 4)
    assert ids[0] == N and ids[1] == D and ids[2] == P, ids
    await spi.write(0x5001, 0xFD)
    assert (await spi.read(0x5001))[0] == 0xFD
    await spi.write(0x5001, 0)
    # exponent memory: a burst write then a burst read back
    await spi.write(0x4100, list(range(100, 100 + N)))
    assert await spi.read(0x4100, N) == list(range(100, 100 + N))


@cocotb.test()
async def test_spi_passes(dut):
    """Fused, causal and unfused passes, and KV-cache decode, all over SPI with
    the clock gate active. Bit-exact; records the SPI bytes per pass, the
    gated-clock share of an idle-heavy decode loop, and host-limited rates."""
    spi = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 61)
    qf, kf, vf = rm.random_case(rng, n=N, d=D, scale=1.5)
    args = rm.prepare(qf, kf, vf)
    spi.bytes = 0
    await load(spi, *args)
    load_bytes = spi.bytes
    summary = []
    for fused, causal in ((True, False), (True, True), (False, False)):
        exp = rm.attention(*args, fused=fused, mode=rm.MODE_EXACT, causal=causal)
        await spi.write(0x5000, 1 | (int(fused) << 1) | (int(causal) << 5))
        await wait_done(dut)
        o, eo = await read_out(spi, list(range(N)))
        cycles = await spi.read32(0x5004)
        tag = f"spi fused={int(fused)} causal={int(causal)}"
        assert np.array_equal(o, exp.o), f"{tag}: output mismatch\nrtl=\n{o}\nmodel=\n{exp.o}"
        assert np.array_equal(eo, exp.eo), f"{tag}: exponent mismatch"
        summary.append(f"{tag:28s} cycles={cycles:4d}")
        dut._log.info(summary[-1])
    # decode loop: per token, write one K/V/Q row + exponents, start, read one row
    q, eq, k, ek, v, ev = args
    g0, t0 = int(dut.gated_cycles.value), cocotb.utils.get_sim_time("ns")
    spi.bytes = 0
    for t in range(N):
        for base, mat in ((0x0000, q), (0x1000, k), (0x2000, v)):
            await spi.write(base + t * D, mat[t])
        for base, ex in ((0x4000, eq), (0x4100, ek), (0x4200, ev)):
            await spi.write(base + t, int(ex[t]))
        await spi.write(0x5010, [t + 1, t])
        await spi.write(0x5000, 1 | (1 << 4))
        await wait_done(dut)
        o, eo = await read_out(spi, [t])
        exp = rm.attention_decode(q, eq, k, ek, v, ev, qrow=t, length=t + 1)
        assert np.array_equal(o[t], exp.o[0]) and eo[t] == exp.eo[0], f"spi decode t={t}"
        # an idle gap between tokens, as when the host runs the rest of the model
        await ClockCycles(dut.clk, 200)
    total = (cocotb.utils.get_sim_time("ns") - t0) / CLK_NS
    gated = int(dut.gated_cycles.value) - g0
    summary.append(f"spi load: {load_bytes} bytes for Q, K, V and exponents "
                   f"({load_bytes * 8 * 2 * SCK_HALF_NS / 1000:.1f} us at 6.25 MHz SCK)")
    summary.append(f"spi decode loop: {N} tokens, {spi.bytes} bytes, {total:.0f} cycles, "
                   f"core clock gated {gated} cycles ({100 * gated / total:.1f}%)")
    for line in summary[-2:]:
        dut._log.info(line)
    assert gated > 0
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"chip_spi_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))
