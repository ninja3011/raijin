"""Block-level cocotb tests. The toplevel is chosen by the runner (TOPLEVEL env);
each test skips itself unless it matches the current toplevel."""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import raijin_model as rm  # noqa: E402

TOP = os.environ.get("COCOTB_TOPLEVEL", os.environ.get("TOPLEVEL", ""))


def s8(x):
    return x & 0xFF


def pack(vals, w):
    out = 0
    for i, v in enumerate(vals):
        out |= (v & ((1 << w) - 1)) << (i * w)
    return out


def signed(v, w):
    v = int(v) & ((1 << w) - 1)
    return v - (1 << w) if v >> (w - 1) else v


@cocotb.test(skip=TOP != "raijin_exp2")
async def test_exp2_sweep(dut):
    count = 0
    for mode in (0, 1, 2):
        for m in range(-128, 129, 3):
            for s in range(max(-2048, m * 16 - 2600), min(2047, m * 16) + 1, 7):
                dut.s.value = s & 0xFFF
                dut.m.value = m & 0x1FF
                dut.mode.value = mode
                await Timer(1, unit="ns")
                p, sa, sb = rm.exp2(s, m, mode)
                assert int(dut.p.value) == p, (mode, s, m)
                if mode:
                    assert int(dut.sa.value) == sa and int(dut.sb.value) == sb, (mode, s, m)
                count += 1
    dut._log.info(f"exp2: {count} points match")


@cocotb.test(skip=TOP != "raijin_requant")
async def test_requant_sweep(dut):
    rnd = random.Random(3)
    for _ in range(4000):
        dot = rnd.randint(-(1 << 19), (1 << 19) - 1) >> rnd.randint(0, 19)
        es = rnd.randint(200, 320)
        off = rnd.randint(-128, 127) if rnd.random() < 0.3 else rnd.randint(-4, 4)
        dut.dot.value = dot & 0xFFFFF
        dut.esum.value = es
        dut.offset.value = off & 0xFF
        await Timer(1, unit="ns")
        assert signed(dut.s.value, 12) == rm.requant(dot, es, off), (dot, es, off)


@cocotb.test(skip=TOP != "raijin_dpu")
async def test_dpu_random(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    rnd = random.Random(5)
    vecs = []
    for i in range(200):
        a = [rnd.randint(-128, 127) for _ in range(16)]
        b = [rnd.randint(-128, 127) for _ in range(16)]
        if i < 2:
            a = [-128] * 16
            b = [-128 if i == 0 else 127] * 16
        vecs.append((a, b, rnd.randint(0, 255), rnd.randint(0, 255)))
    got = []
    for idx in range(len(vecs) + 3):
        if idx < len(vecs):
            a, b, ea, eb = vecs[idx]
            dut.a.value = pack(a, 8)
            dut.b.value = pack(b, 8)
            dut.ea.value = ea
            dut.eb.value = eb
            dut.in_valid.value = 1
            dut.in_tag.value = idx & 0xFF
        else:
            dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.out_valid.value):
            got.append((int(dut.out_tag.value), signed(dut.dot.value, 20), int(dut.esum.value)))
    assert len(got) == len(vecs)
    for i, (tag, dot, es) in enumerate(got):
        a, b, ea, eb = vecs[i]
        assert tag == i & 0xFF
        assert dot == rm.dpu(a, b), i
        assert es == ea + eb


@cocotb.test(skip=TOP != "raijin_out")
async def test_out_random(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    rnd = random.Random(11)
    vecs = []
    for _ in range(150):
        mag = rnd.randint(1, 23)
        acc = [rnd.randint(-(1 << mag), (1 << mag) - 1) for _ in range(16)]
        if rnd.random() < 0.1:
            acc = [0] * 16
        r = rnd.randint(128, 4096)
        emax = rnd.randint(0, 255)
        vecs.append((acc, r, emax))
    got = []
    for idx in range(len(vecs) + 4):
        if idx < len(vecs):
            acc, r, emax = vecs[idx]
            dut.acc.value = pack(acc, 24)
            dut.r.value = pack([r] * 4, 13)   # RG = 4 identical copies of 1/l
            dut.emax.value = emax
            dut.in_tag.value = idx & 0xFF
            dut.in_valid.value = 1
        else:
            dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.out_valid.value):
            o = int(dut.o.value)
            got.append(([signed(o >> (8 * c), 8) for c in range(16)], int(dut.eo.value)))
    assert len(got) == len(vecs)
    for i, (o, eo) in enumerate(got):
        exp_o, exp_e = rm.out_row(*vecs[i])
        assert o == exp_o and eo == exp_e, (i, o, exp_o, eo, exp_e)


@cocotb.test(skip=TOP != "raijin_softmax_lane")
async def test_lane_online_and_recip(dut):
    """Drive one lane in online mode and compare m, l and 1/l with the model."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst_n.value = 0
    for sig in (dut.clear, dut.s_valid, dut.m_update, dut.emit, dut.recip_start, dut.first, dut.last, dut.mask, dut.thr):
        sig.value = 0
    dut.mode.value = 0
    dut.tag.value = 0
    dut.s.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    rnd = random.Random(21)
    for trial in range(40):
        mode = trial % 3
        n = 16
        scores = [rnd.randint(-2048, 2047) >> rnd.randint(0, 6) for _ in range(n)]
        dut.mode.value = mode
        dut.clear.value = 1
        await RisingEdge(dut.clk)
        dut.clear.value = 0
        dut.m_update.value = 1
        dut.emit.value = 1
        for s in scores:
            dut.s.value = s & 0xFFF
            dut.s_valid.value = 1
            await RisingEdge(dut.clk)
        dut.s_valid.value = 0
        await ClockCycles(dut.clk, 3)
        # model
        m, lsum = rm.M_INIT, 0
        for s in scores:
            mn = max(m, rm.ceil_int(s))
            dl = mn - m
            m = mn
            p, _, _ = rm.exp2(s, m, mode)
            lsum = (lsum >> min(dl, 31)) + p
        assert signed(dut.m.value, 9) == m
        assert int(dut.l.value) == lsum, (trial, int(dut.l.value), lsum)
        dut.recip_start.value = 1
        await RisingEdge(dut.clk)
        dut.recip_start.value = 0
        for _ in range(20):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if int(dut.r_valid.value):
                break
        assert int(dut.r.value) == rm.recip(lsum)


@cocotb.test(skip=TOP != "raijin_rope")
async def test_rope_rotator(dut):
    """The RoPE rotator against rm.rope_row: every position 0..255, every pair,
    random element values plus the extremes."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.we_even.value = 0
    await ClockCycles(dut.clk, 2)
    rng = random.Random(5)
    n = 0
    for pos in range(256):
        for pair in range(8):
            xs = [(-128, -128), (127, -128), (127, 127)] if pos % 64 == 0 else []
            xs += [(rng.randint(-128, 127), rng.randint(-128, 127)) for _ in range(2)]
            for x, y in xs:
                dut.pos.value = pos
                dut.pair.value = pair
                dut.we_even.value = 1
                dut.wdata.value = s8(x)
                await RisingEdge(dut.clk)
                dut.we_even.value = 0
                dut.wdata.value = s8(y)
                await Timer(1, unit="ns")
                c, s = rm.rope_cs(pos, pair)
                ex = rm._sat((x * c - y * s + 128) >> 8, 8)
                ey = rm._sat((x * s + y * c + 128) >> 8, 8)
                assert (signed(dut.xo.value, 8), signed(dut.yo.value, 8)) == (ex, ey), (pos, pair, x, y)
                n += 1
    dut._log.info(f"rope rotator: {n} pairs match")
