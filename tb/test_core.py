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


@cocotb.test()
async def test_key_skip(dut):
    """Hardware key skip (SKIPTHR, 0x5013): keys scoring SKIPTHR or more powers
    of two below the running max get p = 0. When no row of a P-row group uses a
    key, its V row is never read (VSKIPS, 0x5014). Bit-exact with the model for
    fused, unfused, causal and decode passes; accuracy vs FP32 is recorded."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 23)
    summary = []
    for name, kw in CASES:
        qf, kf, vf = rm.random_case(rng, n=N, d=D, **kw)
        args = rm.prepare(qf, kf, vf)
        await load(bus, *args)
        ref = rm.reference(qf, kf, vf)
        refc = rm.reference_causal(qf, kf, vf)
        for thr in (0, 6, 4, 2):
            await bus.write(0x5013, thr)
            assert await bus.read(0x5013) == thr
            for fused, causal in ((False, False), (True, False), (True, True)):
                for mode in (rm.MODE_EXACT, rm.MODE_MF2):
                    exp = rm.attention(*args, fused=fused, mode=mode, causal=causal, thr=thr)
                    o, eo, cycles, skips = await run_pass(bus, fused, mode, causal=causal)
                    vsk = await bus.read32(0x5014)
                    tag = f"{name} thr={thr} fused={int(fused)} causal={int(causal)} mode={mode}"
                    assert np.array_equal(o, exp.o), f"{tag}: output mismatch\nrtl=\n{o}\nmodel=\n{exp.o}"
                    assert np.array_equal(eo, exp.eo), f"{tag}: exponent mismatch"
                    assert skips == exp.skips, f"{tag}: skips rtl={skips} model={exp.skips}"
                    exp_v = rm.v_reads_saved(exp.used, P, causal)
                    assert vsk == exp_v, f"{tag}: vskips rtl={vsk} model={exp_v}"
                    r = refc if causal else ref
                    got = rm.mx_dequantize(o, eo)
                    cos = float(np.mean([np.dot(got[i], r[i]) / np.linalg.norm(got[i]) /
                                         np.linalg.norm(r[i]) for i in range(N)]))
                    total = N * N if not causal else N * (N + 1) // 2
                    beats = (N // P) * N if not causal else sum((g + 1) * P for g in range(N // P))
                    summary.append(f"{tag:46s} cycles={cycles:4d} p0_keys={skips:3d}/{total:3d} "
                                   f"v_saved={vsk:3d}/{beats:3d} mean_row_cos={cos:.5f}")
                    dut._log.info(summary[-1])
            # decode: one row, every zero-p key saves its V read
            for t in (N // 2 - 1, N - 1):
                q, eq, k, ek, v, ev = args
                await bus.write(0x5010, t + 1)
                await bus.write(0x5011, t)
                await bus.write(0x5000, 1 | (1 << 4))
                await wait_done(dut)
                o, eo, cycles, skips = await read_out(bus, rows=[t])
                vsk = await bus.read32(0x5014)
                exp = rm.attention_decode(*args, qrow=t, length=t + 1, thr=thr)
                tag = f"{name} thr={thr} decode len={t + 1}"
                assert np.array_equal(o[t], exp.o[0]) and eo[t] == exp.eo[0], f"{tag}: mismatch"
                assert skips == exp.skips and vsk == exp.skips, f"{tag}: skips={skips} vsk={vsk} model={exp.skips}"
    await bus.write(0x5013, 0)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"keyskip_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))


@cocotb.test()
async def test_rope(dut):
    """RoPE on write (ROPE 0x5018, ROPEPOS 0x5019): the host writes raw Q and K
    rows and the chip rotates each pair on its way into SRAM. Passes are
    bit-exact with the model of the rotator, and are compared with FP32 RoPE
    attention. Decode uses absolute positions from ROPEPOS."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 31)
    summary = []
    for name, kw in CASES:
        qf, kf, vf = rm.random_case(rng, n=N, d=D, **kw)
        q, eq, k, ek, v, ev = rm.prepare(qf, kf, vf)
        pos = list(range(N))
        qr, eqr = rm.rope_quantized(q, eq, pos)
        kr, ekr = rm.rope_quantized(k, ek, pos)
        await bus.write(0x5018, 1)
        assert await bus.read(0x5018) == 1
        await load(bus, q, eqr, k, ekr, v, ev)          # raw rows; the chip rotates
        await bus.write(0x5018, 0)
        qrf, krf = rm.rope_float(qf, pos), rm.rope_float(kf, pos)
        for fused, causal in ((False, False), (True, False), (True, True)):
            for mode in (rm.MODE_EXACT, rm.MODE_MF2):
                exp = rm.attention(qr, eqr, kr, ekr, v, ev, fused=fused, mode=mode, causal=causal)
                o, eo, cycles, skips = await run_pass(bus, fused, mode, causal=causal)
                tag = f"rope {name} fused={int(fused)} causal={int(causal)} mode={mode}"
                assert np.array_equal(o, exp.o), f"{tag}: output mismatch\nrtl=\n{o}\nmodel=\n{exp.o}"
                assert np.array_equal(eo, exp.eo), f"{tag}: exponent mismatch"
                assert skips == exp.skips, f"{tag}: skips"
                ref = (rm.reference_causal if causal else rm.reference)(qrf, krf, vf)
                got = rm.mx_dequantize(o, eo)
                cos = float(np.mean([np.dot(got[i], ref[i]) / np.linalg.norm(got[i]) /
                                     np.linalg.norm(ref[i]) for i in range(N)]))
                if mode == rm.MODE_EXACT:
                    summary.append(f"{tag:42s} cycles={cycles:4d} mean_row_cos={cos:.5f}")
                    dut._log.info(summary[-1])
        # decode at absolute positions 200 + t (ROPEPOS), one token at a time
        base = 200
        apos = [base + t for t in range(N)]
        qa, eqa = rm.rope_quantized(q, eq, apos)
        ka, eka = rm.rope_quantized(k, ek, apos)
        refa = rm.reference_causal(rm.rope_float(qf, apos), rm.rope_float(kf, apos), vf)
        coss = []
        for t in range(N):
            await bus.write(0x5018, 3)
            await bus.write(0x5019, base + t)
            for b, mat in ((0x0000, q), (0x1000, k)):
                for c in range(D):
                    await bus.write(b + t * D + c, int(mat[t][c]))
            await bus.write(0x5018, 0)
            for c in range(D):
                await bus.write(0x2000 + t * D + c, int(v[t][c]))
            for b, ex in ((0x4000, eqa), (0x4100, eka), (0x4200, ev)):
                await bus.write(b + t, int(ex[t]))
            await bus.write(0x5010, t + 1)
            await bus.write(0x5011, t)
            await bus.write(0x5000, 1 | (1 << 4))
            await wait_done(dut)
            o, eo, cycles, skips = await read_out(bus, rows=[t])
            exp = rm.attention_decode(qa, eqa, ka, eka, v, ev, qrow=t, length=t + 1)
            assert np.array_equal(o[t], exp.o[0]) and eo[t] == exp.eo[0], f"rope decode {name} t={t}"
            got = rm.mx_dequantize(o[t:t + 1], eo[t:t + 1])[0]
            coss.append(float(np.dot(got, refa[t]) / np.linalg.norm(got) / np.linalg.norm(refa[t])))
        summary.append(f"rope {name} decode pos={base}..{base + N - 1}{'':14s} cycles={cycles:4d} mean_cos={np.mean(coss):.5f}")
        dut._log.info(summary[-1])
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"rope_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))


async def write_ptab(bus, ptab):
    """PTAB (0x501C..F): 2N/4 entries, entry 2b in the low nibble of byte b."""
    ent = list(ptab) + [0] * (2 * N // 4 - len(ptab))
    for b in range(len(ent) // 2):
        await bus.write(0x501C + b, ent[2 * b] | (ent[2 * b + 1] << 4))


@cocotb.test()
async def test_paged_kv_cache(dut):
    """Paged KV cache (PAGED 0x501A, PTAB 0x501C): the decode step reads logical
    token j from physical token PTAB[j/4]*4 + j%4. Two sequences share one
    cache and one 4-token prompt page (prefix sharing), their pages are
    interleaved as a page allocator would hand them out, and an MXINT4 cache of
    2N tokens is scattered over a random page order. Every step is bit-exact
    with decode on the logical (gathered) sequence."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 53)
    summary = []
    await bus.write(0x501A, 1)
    assert await bus.read(0x501A) == 1
    # ---- two sequences, shared prompt page, INT8 ----
    pf = 4
    qf, kf, vf = rm.random_case(rng, n=pf + 8 + 4, d=D, scale=1.5)
    q, eq, k, ek, v, ev = rm.prepare(qf, kf, vf)
    seqs = {"A": list(range(pf)) + list(range(pf, pf + 8)),          # prompt + 8 own tokens
            "B": list(range(pf)) + list(range(pf + 8, pf + 12))}     # prompt + 4 own tokens
    ptabs = {"A": [0, 2, 3], "B": [0, 1]}        # logical page -> physical page
    qrow_of = {"A": 12, "B": 13}

    async def put_token(seq, t):
        src = seqs[seq][t]
        phys = ptabs[seq][t // 4] * 4 + t % 4
        for base, mat in ((0x1000, k), (0x2000, v)):
            for c in range(D):
                await bus.write(base + phys * D + c, int(mat[src][c]))
        await bus.write(0x4100 + phys, int(ek[src]))
        await bus.write(0x4200 + phys, int(ev[src]))

    async def step(seq, t, fmt="int8", arrays=None):
        qa, eqa, ka, eka, va, eva = arrays
        qr = qrow_of[seq] if fmt == "int8" else t % N
        src = seqs[seq][t]
        for c in range(D):
            await bus.write(qr * D + c, int(qa[src][c]))
        await bus.write(0x4000 + qr, int(eqa[src]))
        await write_ptab(bus, ptabs[seq])
        await bus.write(0x5010, t + 1)
        await bus.write(0x5011, qr)
        await bus.write(0x5000, 1 | (1 << 4))
        await wait_done(dut)
        o, eo, cycles, _ = await read_out(bus, rows=[qr])
        idx = seqs[seq][:t + 1]
        exp = rm.attention_decode(qa[idx], eqa[idx], ka[idx], eka[idx], va[idx], eva[idx],
                                  qrow=t, length=t + 1)
        assert np.array_equal(o[qr], exp.o[0]) and eo[qr] == exp.eo[0], \
            f"paged {fmt} {seq} t={t}: rtl={o[qr]} e={eo[qr]} model={exp.o[0]} e={exp.eo[0]}"
        return rm.mx_dequantize(o[qr:qr + 1], eo[qr:qr + 1])[0], cycles

    for t in range(pf):                      # the shared prompt, written once
        await put_token("A", t)
    coss = {"A": [], "B": []}
    refs = {s: rm.reference_causal(qf[seqs[s]], kf[seqs[s]], vf[seqs[s]]) for s in seqs}
    order = []
    for i in range(8):                       # interleaved generation
        order.append(("A", pf + i))
        if i < 4:
            order.append(("B", pf + i))
    for seq, t in order:
        await put_token(seq, t)
        got, cycles = await step(seq, t, arrays=(q, eq, k, ek, v, ev))
        r = refs[seq][t]
        coss[seq].append(float(np.dot(got, r) / np.linalg.norm(got) / np.linalg.norm(r)))
    for s in seqs:
        summary.append(f"paged int8 seq {s}: {len(seqs[s])} tokens, pages {ptabs[s]}, "
                       f"shared prompt page 0, last step {cycles} cycles, mean cos={np.mean(coss[s]):.5f}")
        dut._log.info(summary[-1])
    # ---- MXINT4 cache of 2N tokens over a shuffled page order ----
    await bus.write(0x5012, 3)
    T = 2 * N
    qf, kf, vf = rm.random_case(rng, n=T, d=D, scale=1.5)
    arr4 = rm.prepare_kv4(qf, kf, vf)
    q4, eq4, k4, ek4, v4, ev4 = arr4
    perm = [int(x) for x in rng.permutation(T // 4)]
    seqs["C"], ptabs["C"] = list(range(T)), perm
    ref = rm.reference_causal(qf, kf, vf)
    cos4 = []
    for t in range(T):
        phys = perm[t // 4] * 4 + t % 4
        for base, mat in ((0x1000, k4), (0x2000, v4)):
            for b, byte in enumerate(rm.pack4(mat[t])):
                await bus.write(base + 8 * phys + b, byte)
        await bus.write(0x4100 + phys, int(ek4[t]))
        await bus.write(0x4200 + phys, int(ev4[t]))
        got, cycles = await step("C", t, fmt="int4", arrays=arr4)
        cos4.append(float(np.dot(got, ref[t]) / np.linalg.norm(got) / np.linalg.norm(ref[t])))
    summary.append(f"paged int4: {T} tokens, page order {perm}, LEN={T} step {cycles} cycles, "
                   f"mean cos={np.mean(cos4):.5f}")
    dut._log.info(summary[-1])
    await bus.write(0x5012, 0)
    await bus.write(0x501A, 0)
    await write_ptab(bus, list(range(2 * N // 4)))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"paged_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))


@cocotb.test()
async def test_token_window(dut):
    """Token window (region 6): each decode token is one 52-byte write burst
    (Q, K, V rows, three exponents, CTRL with auto LEN + 1) and one 17-byte
    read burst (O row and e_o). Plain, then paged with a shuffled page order
    and RoPE at absolute positions, every step bit-exact."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 71)
    summary = []
    for variant in ("plain", "paged+rope"):
        qf, kf, vf = rm.random_case(rng, n=N, d=D, scale=1.5)
        q, eq, k, ek, v, ev = rm.prepare(qf, kf, vf)
        paged = variant != "plain"
        perm = [int(x) for x in rng.permutation(N // 4)] if paged else list(range(2 * N // 4))
        base = 50
        if paged:
            await write_ptab(bus, perm)
            await bus.write(0x501A, 1)
            qm, eqm = rm.rope_quantized(q, eq, [base + t for t in range(N)])
            km, ekm = rm.rope_quantized(k, ek, [base + t for t in range(N)])
        else:
            qm, eqm, km, ekm = q, eq, k, ek
        await bus.write(0x5010, 0)
        coss = []
        for t in range(N):
            qr = t
            await bus.write(0x5011, qr)
            if paged:
                await bus.write(0x5018, 3)
                await bus.write(0x5019, base + t)
            burst = list(q[t]) + list(k[t]) + list(v[t]) + [eq[t], ek[t], ev[t], 1 | (1 << 4)]
            for b, val in enumerate(burst):
                await bus.write(0x6000 + b, int(val))
            await wait_done(dut)
            out = [await bus.read(0x6000 + b) for b in range(D + 1)]
            o = np.array([x - 256 if x & 0x80 else x for x in out[:D]])
            assert await bus.read(0x5010) == t + 1
            exp = rm.attention_decode(qm, eqm, km, ekm, v, ev, qrow=t, length=t + 1)
            assert np.array_equal(o, exp.o[0]) and out[D] == exp.eo[0], \
                f"{variant} t={t}: rtl={o} e={out[D]} model={exp.o[0]} e={exp.eo[0]}"
            ref = (rm.reference_causal(rm.rope_float(qf, [base + i for i in range(N)]),
                                       rm.rope_float(kf, [base + i for i in range(N)]), vf)
                   if paged else rm.reference_causal(qf, kf, vf))[t]
            got = rm.mx_dequantize(o[None, :], np.array([out[D]]))[0]
            coss.append(float(np.dot(got, ref) / np.linalg.norm(got) / np.linalg.norm(ref)))
        summary.append(f"token window {variant}: {N} tokens, 52 + 17 bytes per token, mean cos={np.mean(coss):.5f}")
        dut._log.info(summary[-1])
        await bus.write(0x5018, 0)
        await bus.write(0x501A, 0)
        await write_ptab(bus, list(range(2 * N // 4)))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"token_window_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))


@cocotb.test()
async def test_mixed_precision_kv(dut):
    """Per-page KV precision (PAGED bit 1, PFMT 0x501B): each 4-token page is
    MXINT8 or MXINT4. In the 16-word cache, 2 INT8 pages + 4 INT4 pages hold
    24 tokens. Policies: all INT4, INT8 for the sink page and the most recent
    page, and INT8 for the sink page and the page with the largest INT4 K error.
    Every decode step is bit-exact; accuracy is compared with an all-INT8
    cache (model only, it cannot hold 24 tokens)."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 83)
    T, NP = 24, 6
    # stale cache contents from an earlier sequence (never read for LEN keys)
    for base in (0x1000, 0x2000):
        for a in range(N * D):
            await bus.write(base + a, int(rng.integers(0, 256)))
    summary = []
    for name, kw in CASES:
        qf, kf, vf = rm.random_case(rng, n=T, d=D, **kw)
        q, eq, k8, ek8, v8, ev8 = rm.prepare(qf, kf, vf)
        _, _, k4, ek4, v4, ev4 = rm.prepare_kv4(qf, kf, vf)
        ref = rm.reference_causal(qf, kf, vf)
        kerr = [float(np.abs(rm.mx_dequantize(k4[p*4:p*4+4], ek4[p*4:p*4+4]) - kf[p*4:p*4+4]).max())
                for p in range(NP)]
        worst = max(range(1, NP), key=lambda p: kerr[p])
        policies = {
            "all INT4": set(),
            "INT8 sink + recent": {0, NP - 1},
            "INT8 sink + outlier": {0, worst},
        }
        # model-only all-INT8 reference
        c8 = []
        for t in range(T):
            e = rm.attention_decode(q, eq, k8, ek8, v8, ev8, qrow=t, length=t + 1)
            g = rm.mx_dequantize(e.o, e.eo)[0]
            c8.append(float(np.dot(g, ref[t]) / np.linalg.norm(g) / np.linalg.norm(ref[t])))
        summary.append(f"{name:8s} all INT8 (model, 1.5x the SRAM)     mean cos={np.mean(c8):.5f}")
        for pol, int8 in policies.items():
            # slot pages: INT8 pages take an even pair first, INT4 pages one each
            ptab, nxt = [0] * (2 * N // 4), 0
            for p in sorted(int8):
                ptab[p], nxt = nxt, nxt + 2
            for p in range(NP):
                if p not in int8:
                    ptab[p], nxt = nxt, nxt + 1
            assert nxt <= 2 * N // 4
            pfmt = sum(1 << p for p in range(NP) if p not in int8)
            await write_ptab(bus, ptab)
            await bus.write(0x501B, pfmt)
            await bus.write(0x501A, 3)
            r1a = await bus.read(0x501A)
            r1b = await bus.read(0x501B)
            assert r1a == 3 and r1b == pfmt, (r1a, r1b)
            km = np.array([k8[t] if t // 4 in int8 else k4[t] for t in range(T)])
            ekm = np.array([ek8[t] if t // 4 in int8 else ek4[t] for t in range(T)])
            vm = np.array([v8[t] if t // 4 in int8 else v4[t] for t in range(T)])
            evm = np.array([ev8[t] if t // 4 in int8 else ev4[t] for t in range(T)])
            coss = []
            for t in range(T):
                p, i = t // 4, t % 4
                slot = ptab[p] * 4 + (i if p not in int8 else 2 * i)
                for base, mat in ((0x1000, km), (0x2000, vm)):
                    data = rm.pack4(mat[t]) if p not in int8 else [int(x) for x in mat[t]]
                    for b, byte in enumerate(data):
                        await bus.write(base + 8 * slot + b, int(byte) & 0xFF)
                await bus.write(0x4100 + slot, int(ekm[t]))
                await bus.write(0x4200 + slot, int(evm[t]))
                qr = t % N
                for c in range(D):
                    await bus.write(qr * D + c, int(q[t][c]))
                await bus.write(0x4000 + qr, int(eq[t]))
                await bus.write(0x5010, t + 1)
                await bus.write(0x5011, qr)
                await bus.write(0x5000, 1 | (1 << 4))
                await wait_done(dut)
                o, eo, cycles, _ = await read_out(bus, rows=[qr])
                qa = q[:t + 1].copy()
                eqa = eq[:t + 1].copy()
                exp = rm.attention_decode(qa, eqa, km[:t + 1], ekm[:t + 1], vm[:t + 1], evm[:t + 1],
                                          qrow=t, length=t + 1)
                assert np.array_equal(o[qr], exp.o[0]) and eo[qr] == exp.eo[0], \
                    f"{name} {pol} t={t}: rtl={o[qr]} e={eo[qr]} model={exp.o[0]} e={exp.eo[0]}"
                g = rm.mx_dequantize(o[qr:qr + 1], eo[qr:qr + 1])[0]
                coss.append(float(np.dot(g, ref[t]) / np.linalg.norm(g) / np.linalg.norm(ref[t])))
            words = 4 * len(int8) + 2 * (NP - len(int8))
            summary.append(f"{name:8s} {pol:22s} {words:2d} words  mean cos={np.mean(coss):.5f} "
                           f"min cos={np.min(coss):.5f}")
            dut._log.info(summary[-1])
    await bus.write(0x501A, 0)
    await bus.write(0x501B, 0)
    await write_ptab(bus, list(range(2 * N // 4)))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"mixed_kv_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))


LM_VECTORS = os.path.join(os.path.dirname(__file__), "..", "results", "lm_head_vectors.npz")


@cocotb.test(skip=not os.path.exists(LM_VECTORS))
async def test_lm_head(dut):
    """Replay one real attention head through the RTL: layer 0, head 0 of
    TinyStories-8M on the prompt line from model/lm_eval.py, as one causal
    prefill pass. The RTL must match the golden model's causal pass bit for bit."""
    vec = np.load(LM_VECTORS)
    t_len = len(vec["tokens"])
    assert t_len <= N
    pad = lambda a: np.concatenate([a, np.zeros((N - len(a),) + a.shape[1:], a.dtype)])  # noqa: E731
    q, k, v = pad(vec["q"]), pad(vec["k"]), pad(vec["v"])
    # padded rows get a neutral exponent; causal rows < t_len never see them
    eq, ek, ev = (np.concatenate([vec[e], np.full(N - t_len, 127)]) for e in ("eq", "ek", "ev"))
    bus = await setup(dut)
    await load(bus, q, eq, k, ek, v, ev)
    o, eo, cycles, skips = await run_pass(bus, True, rm.MODE_EXACT, causal=True)
    exp = rm.attention(q, eq, k, ek, v, ev, fused=True, mode=rm.MODE_EXACT, causal=True)
    # the benchmark ran each row as a decode step; compare the dequantised rows too
    got = rm.mx_dequantize(o[:t_len], eo[:t_len])
    dec = rm.mx_dequantize(vec["o"], vec["eo"])
    cos = float(np.mean([np.dot(got[i], dec[i]) / (np.linalg.norm(got[i]) * np.linalg.norm(dec[i]) + 1e-30)
                         for i in range(t_len)]))
    lines = [f"layer 0 head 0, {t_len} prompt tokens, causal prefill pass: {cycles} cycles, {skips} skips",
             f"mean row cosine vs the benchmark's decode-step outputs: {cos:.6f}"]
    for t in range(t_len):
        ok = np.array_equal(o[t], exp.o[t]) and eo[t] == exp.eo[t]
        lines.append(f"row {t:2d} token {int(vec['tokens'][t]):5d} e_o={int(eo[t]):3d} "
                     f"o={' '.join(f'{int(x):4d}' for x in o[t])} {'match' if ok else 'MISMATCH'}")
        assert ok, f"row {t}: rtl {o[t]} e={eo[t]} model {exp.o[t]} e={exp.eo[t]}"
    dut._log.info(lines[0])
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "lm_head_rtl.txt"), "w") as f:
        f.write(chr(10).join(lines) + chr(10))


@cocotb.test(skip=os.environ.get("COCOTB_TEST_FILTER") != "test_gl_short")
async def test_gl_short(dut):
    """Short workload for gate-level power: one load, one fused pass, one
    decode step. Logs the simulated window of each phase in ns and cycles."""
    from cocotb.utils import get_sim_time
    bus = await setup(dut)
    rng = np.random.default_rng(5)
    qf, kf, vf = rm.random_case(rng, n=N, d=D, scale=1.5)
    args = rm.prepare(qf, kf, vf)
    t0 = get_sim_time("ns")
    await load(bus, *args)
    t1 = get_sim_time("ns")
    exp = rm.attention(*args, fused=True, mode=rm.MODE_EXACT)
    o, eo, cycles, _ = await run_pass(bus, True, rm.MODE_EXACT)
    t2 = get_sim_time("ns")
    assert np.array_equal(o, exp.o) and np.array_equal(eo, exp.eo), "fused pass mismatch"
    await bus.write(0x5010, N)
    await bus.write(0x5011, N - 1)
    await bus.write(0x5000, 1 | (1 << 4))
    await wait_done(dut)
    o2, eo2, dcycles, _ = await read_out(bus, rows=[N - 1])
    t3 = get_sim_time("ns")
    dec = rm.attention_decode(*args, qrow=N - 1, length=N, mode=rm.MODE_EXACT)
    assert np.array_equal(o2[N - 1], dec.o[0]) and eo2[N - 1] == dec.eo[0], "decode mismatch"
    msg = (f"GL window: load {t0:.0f}-{t1:.0f} ns, fused pass + readback {t1:.0f}-{t2:.0f} ns "
           f"({cycles} busy cycles), decode + readback {t2:.0f}-{t3:.0f} ns ({dcycles} busy cycles); "
           f"total {t3:.0f} ns = {t3 / 20:.0f} clock cycles")
    dut._log.info(msg)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "gl_window.txt"), "w") as f:
        f.write(msg + chr(10))


@cocotb.test()
async def test_heavy_hitter(dut):
    """Heavy-hitter tracker (feature 15, region 7): every single-row decode step
    adds each cached key's max-normalised attention weight to its slot's
    accumulator, and HHMIN names the lowest-scoring slot outside the sinks and
    the recent window. All
    accumulators and HHMIN are bit-exact with rm.hh_step; clear, halve, the
    per-slot clear and the token-window readback (byte 17) are checked too."""
    bus = await setup(dut)
    rng = np.random.default_rng(int(os.environ.get("RAIJIN_SEED", 7)) + 71)
    skip, recent = 2, 4
    summary = []

    async def read_acc():
        a = []
        for sl in range(rm.HH_NK):
            lo = await bus.read(0x7000 + 2 * sl)
            hi = await bus.read(0x7001 + 2 * sl)
            a.append(lo | hi << 8)
        return a

    await bus.write(0x7080, 0b011)          # enable, clear all
    await bus.write(0x7082, skip)
    await bus.write(0x7083, recent)
    assert await bus.read(0x7080) == 1 and await bus.read(0x7082) == skip and await bus.read(0x7083) == recent
    for name, kw in CASES[:2]:
        qf, kf, vf = rm.random_case(rng, n=N, d=D, **kw)
        q, eq, k, ek, v, ev = rm.prepare(qf, kf, vf)
        await load(bus, q, eq, k, ek, v, ev)
        await bus.write(0x7080, 0b011)
        acc = [0] * rm.HH_NK
        for t in range(N):
            await bus.write(0x5010, t + 1)
            await bus.write(0x5011, t)
            await bus.write(0x5000, 1 | (1 << 4))
            await wait_done(dut)
            o, eo, cycles, _ = await read_out(bus, rows=[t])
            exp = rm.attention_decode(q, eq, k, ek, v, ev, qrow=t, length=t + 1)
            assert np.array_equal(o[t], exp.o[0]) and eo[t] == exp.eo[0], f"{name} t={t}: decode output"
            sc = rm.decode_scores(q, eq, k, ek, qrow=t, length=t + 1)
            mexp = rm.hh_step(acc, sc, list(range(t + 1)), skip, recent)
            if t >= skip + recent:
                assert await bus.read(0x7081) == mexp, f"{name} t={t}: HHMIN"
        got = await read_acc()
        assert got == acc, f"{name}: accumulators\nrtl={got}\nmodel={acc}"
        summary.append(f"hh {name:8s} after {N} steps: acc={acc[:N]} HHMIN={mexp} decode cycles at LEN=N: {cycles}")
        dut._log.info(summary[-1])
    # decay, per-slot clear, token-window byte 17
    await bus.write(0x7080, 0b101)
    acc = [a >> 1 for a in acc]
    await bus.write(0x7000 + 2 * 5, 0)
    acc[5] = 0
    assert await read_acc() == acc, "halve / per-slot clear"
    assert await bus.read(0x6011) == mexp, "token window HHMIN"
    # disabled: accumulators stay put
    await bus.write(0x7080, 0)
    await bus.write(0x5000, 1 | (1 << 4))
    await wait_done(dut)
    assert await read_acc() == acc, "tracker off"
    with open(os.path.join(os.path.dirname(__file__), "..", "results", f"heavy_hitter_N{N}_D{D}.txt"), "w") as f:
        f.write(chr(10).join(summary) + chr(10))
