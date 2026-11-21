"""Full-pass cocotb tests: the RTL core must match the golden model bit for bit."""
import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, ReadOnly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import raijin_model as rm  # noqa: E402

N = int(os.environ.get("RAIJIN_N", 16))
P = int(os.environ.get("RAIJIN_P", 4))
D = int(os.environ.get("RAIJIN_D", 16))
RD_LATENCY = 3   # rdata is valid after the third rising edge following re


class Bus:
    def __init__(self, dut):
        self.dut = dut

    async def write(self, addr, data):
        d = self.dut
        d.addr.value = addr
        d.wdata.value = data & 0xFF
        d.we.value = 1
        await RisingEdge(d.clk)
        d.we.value = 0

    async def read(self, addr):
        d = self.dut
        d.addr.value = addr
        d.re.value = 1
        await RisingEdge(d.clk)
        d.re.value = 0
        await ClockCycles(d.clk, RD_LATENCY - 1)
        await ReadOnly()
        val = int(d.rdata.value)
        await RisingEdge(d.clk)
        return val

    async def read32(self, addr):
        b = [await self.read(addr + i) for i in range(4)]
        return b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst_n.value = 0
    dut.we.value = 0
    dut.re.value = 0
    dut.addr.value = 0
    dut.wdata.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    return Bus(dut)


async def load(bus, q, eq, k, ek, v, ev):
    for base, mat in ((0x0000, q), (0x1000, k), (0x2000, v)):
        for r in range(N):
            for c in range(D):
                await bus.write(base + r * D + c, int(mat[r][c]))
    for base, ex in ((0x4000, eq), (0x4100, ek), (0x4200, ev)):
        for r in range(N):
            await bus.write(base + r, int(ex[r]))


async def run_pass(bus, fused, mode, offset=0, causal=False):
    dut = bus.dut
    await bus.write(0x5001, offset)
    await bus.write(0x5000, 1 | (int(fused) << 1) | (mode << 2) | (int(causal) << 5))
    await wait_done(dut)
    return await read_out(bus)


async def wait_done(dut):
    # the start write lands two edges after it is presented; let done drop first
    await ClockCycles(dut.clk, 3)
    for _ in range(5000):
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            break
    else:
        raise AssertionError("pass did not finish")


async def read_out(bus, rows=None):
    rows = range(N) if rows is None else rows
    o = np.zeros((N, D), dtype=np.int64)
    eo = np.zeros(N, dtype=np.int64)
    for r in rows:
        for c in range(D):
            b = await bus.read(0x3000 + r * D + c)
            o[r][c] = b - 256 if b & 0x80 else b
        eo[r] = await bus.read(0x4300 + r)
    cycles = await bus.read32(0x5004)
    skips = await bus.read32(0x5008)
    return o, eo, cycles, skips


CASES = [
    ("random", dict(scale=1.0)),
    ("sharp", dict(scale=2.5)),
    ("outliers", dict(scale=1.5, outliers=True)),
]


@cocotb.test()
async def test_ids(dut):
    bus = await setup(dut)
    assert await bus.read(0x500C) == N
    assert await bus.read(0x500D) == D
    assert await bus.read(0x500F) == 2
    assert await bus.read(0x5003) == RD_LATENCY


@cocotb.test()
async def test_all_modes(dut):
    """Every case, both dataflows, all three exponent modes, bit-exact."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)))
    summary = []
    for name, kw in CASES:
        qf, kf, vf = rm.random_case(rng, n=N, d=D, **kw)
        args = rm.prepare(qf, kf, vf)
        await load(bus, *args)
        ref = rm.reference(qf, kf, vf)
        for fused in (False, True):
            for mode in (rm.MODE_EXACT, rm.MODE_MF1, rm.MODE_MF2):
                exp = rm.attention(*args, fused=fused, mode=mode)
                o, eo, cycles, skips = await run_pass(bus, fused, mode)
                tag = f"{name} fused={int(fused)} mode={mode}"
                assert np.array_equal(o, exp.o), f"{tag}: output mismatch\nrtl=\n{o}\nmodel=\n{exp.o}"
                assert np.array_equal(eo, exp.eo), f"{tag}: exponent mismatch rtl={eo} model={exp.eo}"
                assert skips == exp.skips, f"{tag}: skips rtl={skips} model={exp.skips}"
                got = rm.mx_dequantize(o, eo)
                cos = float(np.sum(got * ref) / np.linalg.norm(got) / np.linalg.norm(ref))
                summary.append(f"{tag:34s} cycles={cycles:4d} skips={skips:3d} cos={cos:.5f}")
                dut._log.info(summary[-1])
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"core_N{N}_D{D}.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")


@cocotb.test()
async def test_offset_and_reuse(dut):
    """A non-zero requant offset, and back-to-back passes without reloading."""
    bus = await setup(dut)
    rng = np.random.default_rng(99)
    qf, kf, vf = rm.random_case(rng, n=N, d=D, scale=0.7)
    args = rm.prepare(qf, kf, vf)
    await load(bus, *args)
    for off in (2, -3):
        for fused in (True, False):
            exp = rm.attention(*args, fused=fused, mode=rm.MODE_MF2, offset=off)
            o, eo, _, skips = await run_pass(bus, fused, rm.MODE_MF2, offset=off)
            assert np.array_equal(o, exp.o), f"off={off} fused={fused}\n{o}\n{exp.o}"
            assert np.array_equal(eo, exp.eo), f"off={off} fused={fused} {eo} {exp.eo}"
            assert skips == exp.skips


@cocotb.test()
async def test_random_regression(dut):
    """Many random tensors with random exponents spread, fused and unfused, exact and MF2."""
    bus = await setup(dut)
    rng = np.random.default_rng(1234)
    iters = int(os.environ.get("RAIJIN_ITERS", 6))
    for it in range(iters):
        scale = float(rng.uniform(0.3, 4.0))
        qf, kf, vf = rm.random_case(rng, n=N, d=D, scale=scale, outliers=bool(it % 2))
        args = rm.prepare(qf, kf, vf)
        await load(bus, *args)
        for fused in (False, True):
            mode = int(rng.integers(0, 3))
            exp = rm.attention(*args, fused=fused, mode=mode)
            o, eo, _, skips = await run_pass(bus, fused, mode)
            assert np.array_equal(o, exp.o), f"iter {it} fused={fused} mode={mode}"
            assert np.array_equal(eo, exp.eo)
            assert skips == exp.skips


@cocotb.test()
async def test_decode_kv_cache(dut):
    """Autoregressive decode with a KV cache: each step writes only the new
    token's Q, K and V rows, then one query attends to every cached key."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 5)
    summary = []
    for name, kw in CASES:
        qf, kf, vf = rm.random_case(rng, n=N, d=D, **kw)
        q, eq, k, ek, v, ev = rm.prepare(qf, kf, vf)
        ref = rm.reference_causal(qf, kf, vf)
        mode = rm.MODE_EXACT if name != "sharp" else rm.MODE_MF2
        for t in range(N):
            for base, mat in ((0x0000, q), (0x1000, k), (0x2000, v)):
                for c in range(D):
                    await bus.write(base + t * D + c, int(mat[t][c]))
            for base, ex in ((0x4000, eq), (0x4100, ek), (0x4200, ev)):
                await bus.write(base + t, int(ex[t]))
            await bus.write(0x5010, t + 1)
            await bus.write(0x5011, t)
            await bus.write(0x5000, 1 | (mode << 2) | (1 << 4))
            await wait_done(dut)
            o, eo, cycles, skips = await read_out(bus, rows=[t])
            exp = rm.attention_decode(q, eq, k, ek, v, ev, qrow=t, length=t + 1, mode=mode)
            assert np.array_equal(o[t], exp.o[0]), f"{name} t={t}: rtl={o[t]} model={exp.o[0]}"
            assert eo[t] == exp.eo[0], f"{name} t={t}: e rtl={eo[t]} model={exp.eo[0]}"
            assert skips == exp.skips, f"{name} t={t}: skips rtl={skips} model={exp.skips}"
            got = rm.mx_dequantize(o[t:t + 1], eo[t:t + 1])[0]
            cos = float(np.dot(got, ref[t]) / np.linalg.norm(got) / np.linalg.norm(ref[t]))
            if t in (0, N // 2 - 1, N - 1):
                summary.append(f"decode {name:8s} mode={mode} len={t + 1:2d} cycles={cycles:4d} cos={cos:.5f}")
                dut._log.info(summary[-1])
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"decode_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))


@cocotb.test()
async def test_causal_prefill(dut):
    """Causal prefill in one fused pass: row i attends to keys 0..i. Bit-exact
    with the model, compared with FP32 causal attention, and timed against
    building the same outputs with N decode steps."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 11)
    summary = []
    for name, kw in CASES:
        qf, kf, vf = rm.random_case(rng, n=N, d=D, **kw)
        args = rm.prepare(qf, kf, vf)
        await load(bus, *args)
        ref = rm.reference_causal(qf, kf, vf)
        for mode in (rm.MODE_EXACT, rm.MODE_MF1, rm.MODE_MF2):
            for off in (0, 2):
                exp = rm.attention(*args, fused=True, mode=mode, offset=off, causal=True)
                o, eo, cycles, skips = await run_pass(bus, True, mode, offset=off, causal=True)
                tag = f"causal {name} mode={mode} off={off}"
                assert np.array_equal(o, exp.o), f"{tag}: output mismatch\nrtl=\n{o}\nmodel=\n{exp.o}"
                assert np.array_equal(eo, exp.eo), f"{tag}: exponent mismatch rtl={eo} model={exp.eo}"
                assert skips == exp.skips, f"{tag}: skips rtl={skips} model={exp.skips}"
                if off == 0:
                    got = rm.mx_dequantize(o, eo)
                    cos = float(np.mean([np.dot(got[i], ref[i]) / np.linalg.norm(got[i]) /
                                         np.linalg.norm(ref[i]) for i in range(N)]))
                    summary.append(f"{tag:30s} cycles={cycles:4d} skips={skips:3d} mean_row_cos={cos:.5f}")
                    dut._log.info(summary[-1])
        # a plain fused pass right after a causal one must be unaffected
        exp = rm.attention(*args, fused=True, mode=rm.MODE_EXACT)
        o, eo, _, _ = await run_pass(bus, True, rm.MODE_EXACT)
        assert np.array_equal(o, exp.o) and np.array_equal(eo, exp.eo), f"{name}: non-causal after causal"
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"causal_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))


@cocotb.test()
async def test_multi_query_decode(dut):
    """Multi-query decode: P query rows share one K/V stream in a single step.
    GQA (every row sees all LEN keys) and speculative verify (row r sees keys
    0..LEN-P+r), each bit-exact with the model, timed against P single-row steps."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 23)
    summary = []
    for name, kw in CASES:
        qf, kf, vf = rm.random_case(rng, n=N, d=D, **kw)
        args = rm.prepare(qf, kf, vf)
        await load(bus, *args)
        q, eq, k, ek, v, ev = args
        mode = rm.MODE_EXACT if name != "sharp" else rm.MODE_MF2
        for spec in (False, True):
            for length in sorted({P, N // 2, N}):
                for group in (0, N // P - 1):
                    await bus.write(0x5010, length)
                    await bus.write(0x5011, group * P)
                    ctrl = 1 | (mode << 2) | (1 << 4) | (1 << 6) | (int(spec) << 7)
                    await bus.write(0x5000, ctrl)
                    await wait_done(dut)
                    rows = list(range(group * P, group * P + P))
                    o, eo, cycles, skips = await read_out(bus, rows=rows)
                    exp = rm.attention_decode_multi(q, eq, k, ek, v, ev, group=group, p=P,
                                                    length=length, spec=spec, mode=mode)
                    tag = f"{name} spec={int(spec)} len={length} group={group}"
                    assert np.array_equal(o[rows], exp.o), f"{tag}: rtl=\n{o[rows]}\nmodel=\n{exp.o}"
                    assert np.array_equal(eo[rows], exp.eo), f"{tag}: e rtl={eo[rows]} model={exp.eo}"
                    assert skips == exp.skips, f"{tag}: skips rtl={skips} model={exp.skips}"
                    if group == 0:
                        # the same P outputs one row per step, for the speed comparison
                        single = 0
                        for r in rows:
                            await bus.write(0x5010, length - P + 1 + (r - rows[0]) if spec else length)
                            await bus.write(0x5011, r)
                            await bus.write(0x5000, 1 | (mode << 2) | (1 << 4))
                            await wait_done(dut)
                            single += await bus.read32(0x5004)
                        summary.append(f"{tag:36s} multi={cycles:3d} cycles  {P} single steps={single:4d} cycles")
                        dut._log.info(summary[-1])
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"multi_decode_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))


@cocotb.test()
async def test_streaming_kv_cache(dut):
    """StreamingLLM-style generation past the cache size: 4 attention-sink rows
    stay pinned and the other N - 4 rows are a ring of the most recent tokens.
    Attention is order-free over keys, so the decode step needs no change; the
    host only picks the row to overwrite. 3N tokens, each step bit-exact."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 31)
    sinks, total = 4, 3 * N
    qf, kf, vf = rm.random_case(rng, n=total, d=D, scale=1.5)
    q, eq, k, ek, v, ev = rm.prepare(qf, kf, vf)
    ref = rm.reference_streaming(qf, kf, vf, N, sinks)
    # mirror of the on-chip cache contents, for the bit-exact model
    mq, mk, mv = (np.zeros((N, D), dtype=np.int64) for _ in range(3))
    meq, mek, mev = (np.zeros(N, dtype=np.int64) for _ in range(3))
    coss = []
    for t in range(total):
        slot = rm.streaming_slot(t, N, sinks)
        for base, mat, mir in ((0x0000, q, mq), (0x1000, k, mk), (0x2000, v, mv)):
            mir[slot] = mat[t]
            for c in range(D):
                await bus.write(base + slot * D + c, int(mat[t][c]))
        for base, ex, mir in ((0x4000, eq, meq), (0x4100, ek, mek), (0x4200, ev, mev)):
            mir[slot] = ex[t]
            await bus.write(base + slot, int(ex[t]))
        length = min(t + 1, N)
        await bus.write(0x5010, length)
        await bus.write(0x5011, slot)
        await bus.write(0x5000, 1 | (1 << 4))
        await wait_done(dut)
        o, eo, cycles, skips = await read_out(bus, rows=[slot])
        exp = rm.attention_decode(mq, meq, mk, mek, mv, mev, qrow=slot, length=length)
        assert np.array_equal(o[slot], exp.o[0]), f"t={t} slot={slot}: rtl={o[slot]} model={exp.o[0]}"
        assert eo[slot] == exp.eo[0], f"t={t}: e rtl={eo[slot]} model={exp.eo[0]}"
        got = rm.mx_dequantize(o[slot:slot + 1], eo[slot:slot + 1])[0]
        coss.append(float(np.dot(got, ref[t]) / np.linalg.norm(got) / np.linalg.norm(ref[t])))
    line = (f"streaming N={N} sinks={sinks} tokens={total}: mean cos={np.mean(coss):.5f} "
            f"min cos={np.min(coss):.5f} (tokens past the cache: mean {np.mean(coss[N:]):.5f})")
    dut._log.info(line)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"streaming_N{N}_D{D}.txt"), "w") as f:
        f.write(line + chr(10))


async def load_kv4(bus, q, eq, k, ek, v, ev):
    """Q as MXINT8 rows; K and V packed MXINT4, token t at byte 8t of its region."""
    for r in range(N):
        for c in range(D):
            await bus.write(0x0000 + r * D + c, int(q[r][c]))
        await bus.write(0x4000 + r, int(eq[r]))
    for base, mat in ((0x1000, k), (0x2000, v)):
        for t in range(mat.shape[0]):
            for b, byte in enumerate(rm.pack4(mat[t])):
                await bus.write(base + 8 * t + b, byte)
    for base, ex in ((0x4100, ek), (0x4200, ev)):
        for t in range(len(ex)):
            await bus.write(base + t, int(ex[t]))


@cocotb.test()
async def test_int4_kv_cache(dut):
    """MXINT4 K/V (KVFMT = 1): full passes on N tokens, then a 2N-token KV cache
    in the same SRAM, every step bit-exact. Accuracy is compared with the same
    steps run with an MXINT8 cache (model only, it cannot hold 2N tokens)."""
    bus = await setup(dut)
    await bus.write(0x5012, 3)
    assert await bus.read(0x5012) == 3
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 41)
    summary = []
    # full passes with an INT4 K/V
    qf, kf, vf = rm.random_case(rng, n=N, d=D, scale=1.5)
    args = rm.prepare_kv4(qf, kf, vf)
    await load_kv4(bus, *args)
    ref = rm.reference(qf, kf, vf)
    for fused, causal in ((False, False), (True, False), (True, True)):
        exp = rm.attention(*args, fused=fused, mode=rm.MODE_EXACT, causal=causal)
        o, eo, cycles, skips = await run_pass(bus, fused, rm.MODE_EXACT, causal=causal)
        tag = f"kv4 full fused={int(fused)} causal={int(causal)}"
        assert np.array_equal(o, exp.o) and np.array_equal(eo, exp.eo), f"{tag}: mismatch"
        assert skips == exp.skips
        if not causal:
            got = rm.mx_dequantize(o, eo)
            cos = float(np.sum(got * ref) / np.linalg.norm(got) / np.linalg.norm(ref))
            summary.append(f"{tag:34s} cycles={cycles:4d} cos={cos:.5f}")
            dut._log.info(summary[-1])
    # a 2N-token KV cache
    T = 2 * N
    for name, kw in CASES:
        qf, kf, vf = rm.random_case(rng, n=T, d=D, **kw)
        q, eq, k, ek, v, ev = rm.prepare_kv4(qf, kf, vf)
        q8, eq8, k8, ek8, v8, ev8 = rm.prepare(qf, kf, vf)
        ref = rm.reference_causal(qf, kf, vf)
        cos4, cos8 = [], []
        mq = np.zeros((N, D), dtype=np.int64)
        meq = np.zeros(N, dtype=np.int64)
        for t in range(T):
            row = t % N
            mq[row], meq[row] = q[t], eq[t]
            for c in range(D):
                await bus.write(0x0000 + row * D + c, int(q[t][c]))
            await bus.write(0x4000 + row, int(eq[t]))
            for base, mat in ((0x1000, k), (0x2000, v)):
                for b, byte in enumerate(rm.pack4(mat[t])):
                    await bus.write(base + 8 * t + b, byte)
            await bus.write(0x4100 + t, int(ek[t]))
            await bus.write(0x4200 + t, int(ev[t]))
            await bus.write(0x5010, t + 1)
            await bus.write(0x5011, row)
            await bus.write(0x5000, 1 | (1 << 4))
            await wait_done(dut)
            o, eo, cycles, _ = await read_out(bus, rows=[row])
            exp = rm.attention_decode(mq, meq, k, ek, v, ev, qrow=row, length=t + 1)
            assert np.array_equal(o[row], exp.o[0]), f"{name} t={t}: rtl={o[row]} model={exp.o[0]}"
            assert eo[row] == exp.eo[0]
            got = rm.mx_dequantize(o[row:row + 1], eo[row:row + 1])[0]
            cos4.append(float(np.dot(got, ref[t]) / np.linalg.norm(got) / np.linalg.norm(ref[t])))
            e8 = rm.attention_decode(q8, eq8, k8, ek8, v8, ev8, qrow=t, length=t + 1)
            g8 = rm.mx_dequantize(e8.o, e8.eo)[0]
            cos8.append(float(np.dot(g8, ref[t]) / np.linalg.norm(g8) / np.linalg.norm(ref[t])))
            if t == T - 1:
                last_cycles = cycles
        summary.append(f"kv4 decode {name:8s} tokens={T} cycles@len{T}={last_cycles:3d} "
                       f"mean cos INT4={np.mean(cos4):.5f} (INT8 cache, model={np.mean(cos8):.5f})")
        dut._log.info(summary[-1])
    # K in MXINT8, V in MXINT4 (KVFMT = 2): the accuracy-first setting
    qf, kf, vf = rm.random_case(rng, n=N, d=D, scale=2.5)
    q, eq, k, ek, _, _ = rm.prepare(qf, kf, vf)
    _, _, _, _, v, ev = rm.prepare_kv4(qf, kf, vf)
    await bus.write(0x5012, 2)
    for r in range(N):
        for c in range(D):
            await bus.write(0x0000 + r * D + c, int(q[r][c]))
            await bus.write(0x1000 + r * D + c, int(k[r][c]))
        for b, byte in enumerate(rm.pack4(v[r])):
            await bus.write(0x2000 + 8 * r + b, byte)
        for base, ex in ((0x4000, eq), (0x4100, ek), (0x4200, ev)):
            await bus.write(base + r, int(ex[r]))
    exp = rm.attention(q, eq, k, ek, v, ev, fused=True, mode=rm.MODE_EXACT, causal=True)
    o, eo, _, _ = await run_pass(bus, True, rm.MODE_EXACT, causal=True)
    assert np.array_equal(o, exp.o) and np.array_equal(eo, exp.eo), "K8/V4 causal pass mismatch"
    await bus.write(0x5012, 0)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"kv4_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))
