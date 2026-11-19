"""End-to-end test of the FPGA datapath (UART -> bridge -> core -> UART) using
the same command protocol as fpga/host/raijin_host.py."""
import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.queue import Queue
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import raijin_model as rm  # noqa: E402

CPB = 8
N = D = 16


class UartHost:
    def __init__(self, dut):
        self.dut = dut
        self.rxq = Queue()
        cocotb.start_soon(self._monitor())

    async def _monitor(self):
        d = self.dut
        while True:
            await FallingEdge(d.tx)
            await ClockCycles(d.clk, CPB // 2)
            if int(d.tx.value):
                continue
            val = 0
            for i in range(8):
                await ClockCycles(d.clk, CPB)
                val |= int(d.tx.value) << i
            await ClockCycles(d.clk, CPB)
            await self.rxq.put(val)

    async def send(self, data):
        d = self.dut
        for b in data:
            for bit in [0] + [(b >> i) & 1 for i in range(8)] + [1]:
                d.rx.value = bit
                await ClockCycles(d.clk, CPB)

    async def recv(self, n):
        return bytes([await self.rxq.get() for _ in range(n)])

    async def write(self, addr, val):
        await self.send([ord("W"), addr >> 8, addr & 0xFF, val & 0xFF])

    async def read(self, addr):
        await self.send([ord("R"), addr >> 8, addr & 0xFF])
        return (await self.recv(1))[0]

    async def burst_write(self, addr, data):
        data = [int(x) & 0xFF for x in data]
        await self.send([ord("B"), addr >> 8, addr & 0xFF, len(data) - 1] + data)

    async def burst_read(self, addr, n):
        await self.send([ord("X"), addr >> 8, addr & 0xFF, n - 1])
        return await self.recv(n)


@cocotb.test()
async def test_uart_end_to_end(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rx.value = 1
    dut.rst.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 5)
    host = UartHost(dut)

    await host.send([ord("P")])
    assert await host.recv(1) == b"K"
    assert await host.read(0x500C) == N

    rng = np.random.default_rng(3)
    qf, kf, vf = rm.random_case(rng, n=N, d=D, scale=2.0, outliers=True)
    q, eq, k, ek, v, ev = rm.prepare(qf, kf, vf)
    await host.burst_write(0x0000, q.reshape(-1))
    await host.burst_write(0x1000, k.reshape(-1))
    await host.burst_write(0x2000, v.reshape(-1))
    await host.burst_write(0x4000, eq)
    await host.burst_write(0x4100, ek)
    await host.burst_write(0x4200, ev)
    assert await host.read(0x4205) == ev[5]

    for fused, mode in ((False, 0), (True, 2)):
        await host.write(0x5000, 1 | (int(fused) << 1) | (mode << 2))
        while not (await host.read(0x5002)) & 2:
            pass
        raw = await host.burst_read(0x3000, N * D)
        o = np.frombuffer(raw, dtype=np.int8).astype(np.int64).reshape(N, D)
        eo = np.frombuffer(await host.burst_read(0x4300, N), dtype=np.uint8).astype(np.int64)
        skips = int.from_bytes(await host.burst_read(0x5008, 4), "little")
        exp = rm.attention(q, eq, k, ek, v, ev, fused=fused, mode=mode)
        assert np.array_equal(o, exp.o), f"fused={fused} mode={mode}"
        assert np.array_equal(eo, exp.eo)
        assert skips == exp.skips
        dut._log.info(f"UART pass fused={fused} mode={mode}: match, skips={skips}")
