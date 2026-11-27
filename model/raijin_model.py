"""Bit-exact golden model of the Raijin attention accelerator.

Every function here mirrors one RTL block. The integer formats are described in
docs/numerics.md. Run this file directly for a quick self-check.

Formats (summary):
  Q, K, V elements : signed INT8, 6 fraction bits; one E8M0 exponent per row
                     value = x * 2^-6 * 2^(E - 127)
  dot q.k          : signed 20-bit integer
  score s          : signed 12-bit, Q7.4 (log2 domain; host folds log2(e)/sqrt(d) into Q)
  running max m    : signed 9-bit integer, initial value -256
  p                : unsigned 9-bit, Q1.8
  l                : unsigned 14-bit, Q6.8 (sums up to 63.99, 32 keys of 1.0)
  acc              : signed 24-bit, sum of (p * v) >> dv
  1/l              : unsigned 13-bit, floor(2^19 / l)
  output           : MXINT8 row (INT8 elements + E8M0 exponent)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants shared with the RTL (rtl/raijin_exp2.v)
# ---------------------------------------------------------------------------
EXP_LUT = [256, 245, 235, 225, 215, 206, 197, 189, 181, 173, 166, 159, 152, 146, 140, 134]
# Multiplier-free, one power of two: 2^-f/16 ~= 2^-MF1_S1[f]
MF1_S1 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Multiplier-free, two powers of two: 2^-f/16 ~= 2^-s1 + 2^-s2  (s2 = 15 means no second term)
MF2_S1 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
MF2_S2 = [15, 15, 15, 15, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6]

MODE_EXACT, MODE_MF1, MODE_MF2 = 0, 1, 2
M_INIT = -256          # running-max reset value (below any reachable ceil(s))
NONE_CODE = 15         # shift code meaning "no term"
EXP_BIAS = 127
FRAC = 6               # fraction bits of MXINT8 elements


def _sat(x: int, bits: int) -> int:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return lo if x < lo else hi if x > hi else x


def _sra(x: int, n: int) -> int:
    """Arithmetic shift right, Python >> is already floor for negatives."""
    return x >> n


# ---------------------------------------------------------------------------
# Block-level functions
# ---------------------------------------------------------------------------
def dpu(q: Sequence[int], k: Sequence[int]) -> int:
    """Dot product of two INT8 rows, signed 20-bit result."""
    d = int(sum(int(a) * int(b) for a, b in zip(q, k)))
    assert -(1 << 19) <= d < (1 << 19)
    return d


def requant(dot: int, esum: int, offset: int) -> int:
    """Move a raw dot product into Q7.4 using the row exponents.

    sh = e_q + e_k - 262 + offset.  s = sat12(dot << sh) or sat12(dot >> -sh).
    The shift amount is clamped to [-31, 15] in hardware.
    """
    sh = esum - 262 + offset
    if sh >= 0:
        sh = min(sh, 15)
        return _sat(dot << sh, 12)
    sh = min(-sh, 31)
    return _sat(_sra(dot, sh), 12)


def ceil_int(s: int) -> int:
    """ceil(s / 16) for Q7.4 s, a 9-bit signed integer."""
    return (s + 15) >> 4


def term(sh: int) -> int:
    """256 >> sh, or 0 when the shift code means 'no term' (sh > 8)."""
    return 0 if sh > 8 else 256 >> sh


def exp2(s: int, m: int, mode: int, thr: int = 0) -> Tuple[int, int, int]:
    """p = 2^(s/16 - m) in Q1.8 plus the two shift codes used by the AV path.

    Returns (p, sa, sb). In exact mode sa/sb are unused (returned as NONE_CODE).
    thr (key-skip threshold, 0 = off): a key whose score is thr or more
    powers of two below the running max gets p = 0 and is skipped.
    """
    t = m * 16 - s
    assert t >= 0
    k, f = t >> 4, t & 15
    if thr and k >= thr:
        return 0, NONE_CODE, NONE_CODE
    if mode == MODE_EXACT:
        p = 0 if k >= 9 else EXP_LUT[f] >> k
        return p, NONE_CODE, NONE_CODE
    if mode == MODE_MF1:
        s1, s2 = MF1_S1[f], NONE_CODE
    else:
        s1, s2 = MF2_S1[f], MF2_S2[f]
    sa = min(k + s1, NONE_CODE)
    sb = NONE_CODE if s2 == NONE_CODE else min(k + s2, NONE_CODE)
    return term(sa) + term(sb), sa, sb


def pv_term(p: int, v: int, dv: int) -> int:
    """(p * v) >> dv. In multiplier-free mode p has at most two bits set, so the
    RTL computes the same value with shifts and one add."""
    return _sra(p * int(v), min(dv, 31))


def recip(l: int) -> int:
    """floor(2^19 / l), 13 bits. l >= 128 is guaranteed by construction."""
    assert l >= 128, l
    return (1 << 19) // l


def _bitlen_signed(y: int) -> int:
    """Number of magnitude bits of a signed value (excluding sign)."""
    return (y ^ (y >> 63)).bit_length() if y < 0 else y.bit_length()


def out_row(acc: Sequence[int], r: int, emax_v: int) -> Tuple[List[int], int]:
    """Normalise one accumulator row by 1/l and pack it as MXINT8."""
    y = [int(a) * r for a in acc]
    nb = max(_bitlen_signed(v) for v in y)
    sh = max(0, nb - 7)
    out = []
    for v in y:
        if sh:
            v = (v + (1 << (sh - 1))) >> sh
        out.append(_sat(v, 8))
    eo = min(max(emax_v + sh - 19, 0), 255)
    return out, eo


# ---------------------------------------------------------------------------
# Full attention pass
# ---------------------------------------------------------------------------
@dataclass
class PassResult:
    o: np.ndarray                 # (N, D) int8 elements
    eo: np.ndarray                # (N,) output row exponents
    skips: int = 0
    l: List[int] = field(default_factory=list)
    scores: np.ndarray | None = None
    used: np.ndarray | None = None   # (rows, keys) bool: key j added to row i


def v_reads_saved(used: np.ndarray, p: int, causal: bool = False) -> int:
    """V-row reads the RTL saves: one per (group, key) beat in which no row of
    the P-row group uses the key. A causal pass streams group g only up to
    key g*P+P-1."""
    n_rows, n_keys = used.shape
    saved = 0
    for g in range(n_rows // p):
        last = g * p + p - 1 if causal else n_keys - 1
        for j in range(last + 1):
            if not used[g * p:(g + 1) * p, j].any():
                saved += 1
    return saved


def attention(q, eq, k, ek, v, ev, *, fused: bool, mode: int = MODE_EXACT,
              offset: int = 0, causal: bool = False, thr: int = 0) -> PassResult:
    """Full attention pass. causal (fused only): row i sees keys 0..i; masked
    keys are not processed at all (and not counted as skips). e_vmax is still
    taken over all N V rows, as in the RTL."""
    assert fused or not causal
    q, k, v = (np.asarray(a, dtype=np.int64) for a in (q, k, v))
    eq, ek, ev = (np.asarray(a, dtype=np.int64) for a in (eq, ek, ev))
    n, d = q.shape
    emax = int(ev.max())
    dvs = [emax - int(e) for e in ev]
    o = np.zeros((n, d), dtype=np.int64)
    eo = np.zeros(n, dtype=np.int64)
    skips = 0
    ls = []
    scores = np.zeros((n, n), dtype=np.int64)
    used = np.zeros((n, n), dtype=bool)
    for i in range(n):
        s = [requant(dpu(q[i], k[j]), int(eq[i]) + int(ek[j]), offset) for j in range(n)]
        scores[i] = s
        acc = [0] * d
        lsum = 0
        if fused:
            m = M_INIT
            for j in range(i + 1 if causal else n):
                mn = max(m, ceil_int(s[j]))
                delta = mn - m
                m = mn
                p, _, _ = exp2(s[j], m, mode, thr)
                if p == 0:
                    skips += 1          # delta is always 0 here
                    continue
                used[i, j] = True
                dsh = min(delta, 31)
                lsum = (lsum >> dsh) + p
                for c in range(d):
                    acc[c] = _sra(acc[c], dsh) + pv_term(p, v[j][c], dvs[j])
        else:
            m = M_INIT
            for j in range(n):
                m = max(m, ceil_int(s[j]))
            for j in range(n):
                p, _, _ = exp2(s[j], m, mode, thr)
                if p == 0:
                    skips += 1
                    continue
                used[i, j] = True
                lsum += p
                for c in range(d):
                    acc[c] += pv_term(p, v[j][c], dvs[j])
        for c in range(d):
            assert -(1 << 23) <= acc[c] < (1 << 23)
        assert lsum < (1 << 14)
        ls.append(lsum)
        row, e = out_row(acc, recip(lsum), emax)
        o[i] = row
        eo[i] = e
    return PassResult(o=o, eo=eo, skips=skips, l=ls, scores=scores, used=used)


def attention_decode(q, eq, k, ek, v, ev, *, qrow: int, length: int,
                     mode: int = MODE_EXACT, offset: int = 0,
                     emax_len: int | None = None, thr: int = 0,
                     final_max: bool = False) -> PassResult:
    """One KV-cache decode step: query row `qrow` attends to keys 0..length-1.

    final_max (max-first decode, feature 17): the running maximum starts at the
    row's final maximum, as when a score pass finds it before any weight is
    formed. No rescale then happens, and every key whose weight underflows
    against the true maximum is skipped.

    This is the fused (online softmax) pass for one query over a key prefix.
    e_vmax is taken over the cached V rows only, as in the RTL. The result
    holds a single output row.
    """
    q, k, v = (np.asarray(a, dtype=np.int64) for a in (q, k, v))
    eq, ek, ev = (np.asarray(a, dtype=np.int64) for a in (eq, ek, ev))
    d = q.shape[1]
    emax = int(ev[:length if emax_len is None else emax_len].max())
    s = [requant(dpu(q[qrow], k[j]), int(eq[qrow]) + int(ek[j]), offset) for j in range(length)]
    acc = [0] * d
    lsum, m, skips = 0, M_INIT, 0
    if final_max:
        m = max(M_INIT, max(ceil_int(x) for x in s))
    for j in range(length):
        mn = max(m, ceil_int(s[j]))
        delta = mn - m
        m = mn
        p, _, _ = exp2(s[j], m, mode, thr)
        if p == 0:
            skips += 1
            continue
        dsh = min(delta, 31)
        lsum = (lsum >> dsh) + p
        dv = emax - int(ev[j])
        for c in range(d):
            acc[c] = _sra(acc[c], dsh) + pv_term(p, v[j][c], dv)
    assert lsum < (1 << 14), lsum          # the 14-bit l register
    row, e = out_row(acc, recip(lsum), emax)
    return PassResult(o=np.array([row], dtype=np.int64), eo=np.array([e], dtype=np.int64),
                      skips=skips, l=[lsum])


def decode_scores(q, eq, k, ek, *, qrow: int, length: int, offset: int = 0):
    """The Q7.4 scores of one decode step (what the softmax lane sees)."""
    q, k = np.asarray(q, dtype=np.int64), np.asarray(k, dtype=np.int64)
    return [requant(dpu(q[qrow], k[j]), int(eq[qrow]) + int(ek[j]), offset) for j in range(length)]


HH_NK = 32


def hh_step(acc, scores, slots, skip=0, recent=0):
    """Heavy-hitter tracker (rtl/raijin_hh.v) after one single-row decode step.

    acc: list of HH_NK 16-bit accumulators (updated in place); scores[i] is the
    score of the key held in physical slot slots[i]. Each key adds its
    max-normalised weight 2^((s - 16 m) / 16) in Q0.8, as
    (frac >= 1/2 ? 181 : 256) >> floor((16 m - s) / 16), 0 below 2^-8, where m
    is the row's final running maximum. Returns the eviction candidate: the
    slot >= skip with the smallest accumulator among this step's slots (lowest
    index on ties). Keys among the step's newest `recent` positions are never
    chosen (H2O keeps a recent window)."""
    m = M_INIT
    for sj in scores:
        m = max(m, ceil_int(sj))
    used, rec = {}, set()
    n = len(scores)
    for j, (sj, sl) in enumerate(zip(scores, slots)):
        used[sl] = sj
        if j + recent >= n:
            rec.add(sl)
    best, best_k = 1 << 17, 0
    for sl in range(HH_NK):
        if sl not in used:
            continue
        d = 16 * m - used[sl]
        if d < 0:
            w = 256
        else:
            di = d >> 4
            w = 0 if di > 8 else ((181 if d & 8 else 256) >> di)
        acc[sl] = min(acc[sl] + w, 0xFFFF)
        if sl >= skip and sl not in rec and acc[sl] < best:
            best, best_k = acc[sl], sl
    return best_k


def attention_decode_multi(q, eq, k, ek, v, ev, *, group: int, p: int, length: int,
                           spec: bool = False, mode: int = MODE_EXACT,
                           offset: int = 0, thr: int = 0) -> PassResult:
    """Multi-query decode step: the p query rows of `group` (rows group*p ..
    group*p+p-1) attend to the cached keys in one step, as the RTL does.

    GQA / MQA (spec=False): every row sees keys 0..length-1.
    Speculative verify (spec=True): the rows are p draft tokens whose K/V are
    the last p cached rows; row r sees keys 0..length-p+r.
    e_vmax is over the length cached V rows for every row."""
    rows, eos, ls, skips = [], [], [], 0
    for r in range(p):
        n_keys = length - p + r + 1 if spec else length
        res = attention_decode(q, eq, k, ek, v, ev, qrow=group * p + r, length=n_keys,
                               mode=mode, offset=offset, emax_len=length, thr=thr)
        rows.append(res.o[0])
        eos.append(res.eo[0])
        ls += res.l
        skips += res.skips
    return PassResult(o=np.array(rows, dtype=np.int64), eo=np.array(eos, dtype=np.int64),
                      skips=skips, l=ls)


def _check_decode_matches_fused(rng) -> None:
    """With length = N, decode of row i must equal row i of the fused pass."""
    qf, kf, vf = random_case(rng, scale=1.5)
    args = prepare(qf, kf, vf)
    full = attention(*args, fused=True)
    for i in range(qf.shape[0]):
        r = attention_decode(*args, qrow=i, length=qf.shape[0])
        assert np.array_equal(r.o[0], full.o[i]) and r.eo[0] == full.eo[i], i


def streaming_slot(t: int, n: int, sinks: int) -> int:
    """KV-cache row for token t in a StreamingLLM-style cache of n rows: the
    first `sinks` tokens stay pinned (attention sinks), and the other n - sinks
    rows form a ring that always holds the most recent tokens."""
    if t < n:
        return t
    return sinks + (t - sinks) % (n - sinks)


def reference_streaming(qf, kf, vf, n: int, sinks: int):
    """FP32 attention for every token t over the keys a streaming cache of n
    rows holds at step t: tokens 0..sinks-1 plus the last n - sinks tokens."""
    out = []
    for t in range(qf.shape[0]):
        keys = list(range(t + 1)) if t < n else \
            list(range(sinks)) + list(range(t - (n - sinks) + 1, t + 1))
        s = qf[t] @ kf[keys].T / np.sqrt(qf.shape[1])
        p = np.exp(s - s.max())
        out.append((p / p.sum()) @ vf[keys])
    return np.array(out)


# ---------------------------------------------------------------------------
# Quantisation helpers (host side)
# ---------------------------------------------------------------------------
def mx_quantize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantise each row of a float matrix to MXINT8 (INT8 with 6 fraction bits
    and one shared E8M0 exponent per row)."""
    x = np.asarray(x, dtype=np.float64)
    q = np.zeros(x.shape, dtype=np.int64)
    e = np.zeros(x.shape[0], dtype=np.int64)
    for i, row in enumerate(x):
        amax = float(np.max(np.abs(row)))
        if amax == 0:
            ex = 0
        else:
            # want |row| / 2^ex * 64 <= 127  ->  ex >= log2(amax * 64 / 127)
            ex = int(np.ceil(np.log2(amax * 64 / 127)))
        ex = min(max(ex, -127), 128)
        q[i] = np.clip(np.round(row / 2.0 ** ex * 64), -128, 127)
        e[i] = ex + EXP_BIAS
    return q, e


def mx_quantize4(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantise each row to MXINT4: INT4 elements in [-8, 7] with 2 fraction
    bits and one shared E8M0 exponent per row. Returns the elements as the
    INT8 datapath sees them (x4 << 4), so the INT8 golden model applies as is,
    together with the exponents."""
    x = np.asarray(x, dtype=np.float64)
    q4 = np.zeros(x.shape, dtype=np.int64)
    e = np.zeros(x.shape[0], dtype=np.int64)
    for i, row in enumerate(x):
        amax = float(np.max(np.abs(row)))
        ex = 0 if amax == 0 else int(np.ceil(np.log2(amax * 4 / 7)))
        ex = min(max(ex, -127), 128)
        q4[i] = np.clip(np.round(row / 2.0 ** ex * 4), -8, 7)
        e[i] = ex + EXP_BIAS
    return q4 * 16, e


def pack4(row8: Sequence[int]) -> List[int]:
    """Pack one MXINT4 row (given as x4 << 4) into 8 bytes: element 2b in the
    low nibble of byte b, element 2b+1 in the high nibble."""
    n = [(int(x) >> 4) & 0xF for x in row8]
    return [n[2 * b] | (n[2 * b + 1] << 4) for b in range(len(n) // 2)]


# MXFP4 (OCP MX, E2M1 elements): nibble s e1 e0 m. Twice the value is an
# integer, 2|x| in {0, 1, 2, 3, 4, 6, 8, 12}, so the chip widens a nibble to
# INT8 as +-(2|x|) << 3 = 16 x, the same scale as an MXINT4 element x4 << 4.
FP4_MAG2 = (0, 1, 2, 3, 4, 6, 8, 12)          # 2|x| for codes 0..7
FP4_GRID = np.array(sorted({s * m / 2 for m in FP4_MAG2 for s in (1, -1)}))


def fp4_code8(nib: int) -> int:
    """INT8 element the datapath sees for one E2M1 nibble (16 x)."""
    mag = FP4_MAG2[nib & 7] << 3
    return -mag if nib & 8 else mag


def mx_quantize_fp4(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantise each row to MXFP4: E2M1 elements (|x| <= 6) and one shared E8M0
    exponent per row. The host tries the exponent at which the row maximum just
    fits (|x| <= 6) and the one below it (the maximum then clips to 6, the
    other elements get a finer grid) and keeps the one with the lower squared
    error; the chip only sees the exponent, so this choice is free.
    Elements are returned as the INT8 datapath sees them (16 x), with the
    exponents; a value is then 16 x / 64 * 2^(e - bias), as for MXINT4."""
    x = np.asarray(x, dtype=np.float64)
    q = np.zeros(x.shape, dtype=np.int64)
    e = np.zeros(x.shape[0], dtype=np.int64)
    for i, row in enumerate(x):
        amax = float(np.max(np.abs(row)))
        ex0 = 0 if amax == 0 else int(np.ceil(np.log2(amax * 4 / 6)))
        best = None
        for ex in (ex0, ex0 - 1):            # host picks the lower-error exponent
            ex = min(max(ex, -127), 128)
            y = row / 2.0 ** ex * 4
            g = FP4_GRID[np.abs(y[:, None] - FP4_GRID[None, :]).argmin(axis=1)]
            err = float(np.sum((g / 4 * 2.0 ** ex - row) ** 2))
            if best is None or err < best[0]:
                best = (err, g, ex)
        q[i] = np.round(best[1] * 16).astype(np.int64)
        e[i] = best[2] + EXP_BIAS
    return q, e


def fp4_nibble(c8: int) -> int:
    """E2M1 nibble of an INT8 element 16 x (inverse of fp4_code8)."""
    mag = FP4_MAG2.index(abs(int(c8)) >> 3)
    return mag | (8 if c8 < 0 else 0)


def pack_fp4(row8: Sequence[int]) -> List[int]:
    """Pack one MXFP4 row (given as 16 x) into 8 bytes, as pack4."""
    n = [fp4_nibble(x) for x in row8]
    return [n[2 * b] | (n[2 * b + 1] << 4) for b in range(len(n) // 2)]


def prepare_kvfp4(qf, kf, vf):
    """Like prepare(), but K and V in MXFP4 (returned as 16 x)."""
    d = qf.shape[1]
    q, eq = mx_quantize(qf * (np.log2(np.e) / np.sqrt(d)))
    k, ek = mx_quantize_fp4(kf)
    v, ev = mx_quantize_fp4(vf)
    return q, eq, k, ek, v, ev


def prepare_kv4(qf, kf, vf):
    """Like prepare(), but K and V in MXINT4 (returned as x4 << 4)."""
    d = qf.shape[1]
    q, eq = mx_quantize(qf * (np.log2(np.e) / np.sqrt(d)))
    k, ek = mx_quantize4(kf)
    v, ev = mx_quantize4(vf)
    return q, eq, k, ek, v, ev


def mx_dequantize(q: np.ndarray, e: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    return q / 64.0 * 2.0 ** (e[:, None] - EXP_BIAS)


def prepare(qf, kf, vf):
    """Fold log2(e)/sqrt(d) into Q and quantise all three to MXINT8."""
    d = qf.shape[1]
    q, eq = mx_quantize(qf * (np.log2(np.e) / np.sqrt(d)))
    k, ek = mx_quantize(kf)
    v, ev = mx_quantize(vf)
    return q, eq, k, ek, v, ev


def reference(qf, kf, vf):
    s = qf @ kf.T / np.sqrt(qf.shape[1])
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return p @ vf


def reference_causal(qf, kf, vf):
    """FP32 causal attention: row i sees keys 0..i (what decode computes)."""
    n = qf.shape[0]
    s = qf @ kf.T / np.sqrt(qf.shape[1])
    s = np.where(np.tril(np.ones((n, n), dtype=bool)), s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return p @ vf


# ---------------------------------------------------------------------------
# RoPE on write (feature 9)
# ---------------------------------------------------------------------------
# Pair i = elements (2i, 2i+1) turns by pos * theta_i, theta_i = 10000^(-2i/d).
# Phase is in turns, Q0.20: ROPE_W[i] = round(2^20 * theta_i / (2 pi)).
ROPE_W = [166886, 52774, 16689, 5277, 1669, 528, 167, 53]
# quarter-wave sine, Q1.8: ROPE_SIN[k] = round(256 * sin(2 pi k / 1024)), k = 0..256
ROPE_SIN = [int(round(256 * np.sin(2 * np.pi * k / 1024))) for k in range(257)]


def rope_cs(pos: int, i: int) -> Tuple[int, int]:
    """(cos, sin) in Q1.8 for pair i at position pos, as the RTL computes it."""
    ph = (pos * ROPE_W[i]) & 0xFFFFF
    a = ((ph + (1 << 9)) >> 10) & 1023          # 10-bit phase, rounded
    q, k = a >> 8, a & 255
    s0, c0 = ROPE_SIN[k], ROPE_SIN[256 - k]
    return [(c0, s0), (-s0, c0), (-c0, -s0), (s0, -c0)][q]


def rope_row(row: Sequence[int], pos: int) -> List[int]:
    """Rotate one INT8 row as the write-path rotator does: round, then
    saturate to INT8. The row keeps its exponent."""
    out = []
    for i in range(len(row) // 2):
        x, y = int(row[2 * i]), int(row[2 * i + 1])
        c, s = rope_cs(pos, i)
        out += [_sat((x * c - y * s + 128) >> 8, 8), _sat((x * s + y * c + 128) >> 8, 8)]
    return out


def rope_float(x: np.ndarray, pos: Sequence[int]) -> np.ndarray:
    """Exact (float) RoPE, interleaved pairs, for the FP32 reference."""
    x = np.array(x, dtype=np.float64)
    d = x.shape[1]
    th = 10000.0 ** (-2 * np.arange(d // 2) / d)
    for r, p in enumerate(pos):
        a = p * th
        c, s = np.cos(a), np.sin(a)
        x0, x1 = x[r, 0::2].copy(), x[r, 1::2].copy()
        x[r, 0::2] = x0 * c - x1 * s
        x[r, 1::2] = x0 * s + x1 * c
    return x


def rope_quantized(m: np.ndarray, e: np.ndarray, pos: Sequence[int]):
    """What the chip stores for rows written with RoPE on: rotated elements,
    same exponents."""
    out = np.array([rope_row(m[r], pos[r]) for r in range(m.shape[0])], dtype=np.int64)
    return out, np.asarray(e, dtype=np.int64)


def random_case(rng: np.random.Generator, n=16, d=16, scale=1.0, outliers=False):
    qf = rng.standard_normal((n, d)) * scale
    kf = rng.standard_normal((n, d)) * scale
    vf = rng.standard_normal((n, d))
    if outliers:
        kf[rng.integers(0, n, 2)] *= 8
        vf[rng.integers(0, n, 2)] *= 16
    return qf, kf, vf


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for fused in (False, True):
        for mode in (MODE_EXACT, MODE_MF1, MODE_MF2):
            cos = []
            for _ in range(20):
                qf, kf, vf = random_case(rng, scale=1.5)
                args = prepare(qf, kf, vf)
                r = attention(*args, fused=fused, mode=mode)
                got = mx_dequantize(r.o, r.eo)
                ref = reference(qf, kf, vf)
                cos.append(np.sum(got * ref) / np.linalg.norm(got) / np.linalg.norm(ref))
            print(f"fused={fused!s:5} mode={mode} mean cos={np.mean(cos):.5f} min={np.min(cos):.5f}")
    _check_decode_matches_fused(rng)
    print("decode with length N matches the fused pass row by row")
