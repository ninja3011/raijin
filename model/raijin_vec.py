"""Vectorised, bit-exact twin of the decode path of raijin_model.

`decode_causal` computes, for every query i of a sequence, the decode step
attention_decode(qrow=i, length=i+1): the online-softmax pass over keys 0..i
that Raijin runs per generated token. It is vectorised over any leading
dimensions (batch, head) and over queries, looping only over keys, so it can
stand in for the attention of a small language model. `check()` compares it
with raijin_model.attention_decode on random tensors, bit for bit.
"""
from __future__ import annotations

import numpy as np

import raijin_model as rm

_LUT = np.array(rm.EXP_LUT, dtype=np.int64)
_S1 = {rm.MODE_MF1: np.array(rm.MF1_S1), rm.MODE_MF2: np.array(rm.MF2_S1)}
_S2 = {rm.MODE_MF1: np.full(16, rm.NONE_CODE), rm.MODE_MF2: np.array(rm.MF2_S2)}


def _sat(x, bits):
    return np.clip(x, -(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def _term(sh):
    return np.where(sh > 8, 0, 256 >> np.minimum(sh, 8))


def scores(q, eq, k, ek, offset=0):
    """Requantised Q7.4 scores, shape (..., Tq, Tk)."""
    dot = np.einsum("...id,...jd->...ij", q.astype(np.int64), k.astype(np.int64))
    sh = eq[..., :, None].astype(np.int64) + ek[..., None, :] - 262 + offset
    left = _sat(dot << np.clip(sh, 0, 15), 12)
    right = _sat(dot >> np.clip(-sh, 0, 31), 12)
    return np.where(sh >= 0, left, right)


def exp2(s, m, mode, thr=0):
    t = m * 16 - s
    kk, f = t >> 4, t & 15
    if mode == rm.MODE_EXACT:
        p = np.where(kk >= 9, 0, _LUT[f] >> np.minimum(kk, 9))
    else:
        s1, s2 = _S1[mode][f], _S2[mode][f]
        sa = np.minimum(kk + s1, rm.NONE_CODE)
        sb = np.where(s2 == rm.NONE_CODE, rm.NONE_CODE, np.minimum(kk + s2, rm.NONE_CODE))
        p = _term(sa) + _term(sb)
    if thr:
        p = np.where(kk >= thr, 0, p)
    return p


def _bitlen(y):
    mag = np.where(y < 0, ~y, y)
    nb = np.zeros(y.shape, dtype=np.int64)
    for b in range(48):
        nb = np.where((mag >> b) != 0, b + 1, nb)
    return nb


def decode_causal(q, eq, k, ek, v, ev, mode=rm.MODE_EXACT, thr=0, offset=0, stats=None):
    """q, k, v: int arrays (..., T, D); eq, ek, ev: (..., T). Returns (o, eo)."""
    q, k, v = (np.asarray(a, dtype=np.int64) for a in (q, k, v))
    eq, ek, ev = (np.asarray(a, dtype=np.int64) for a in (eq, ek, ev))
    T, D = q.shape[-2], q.shape[-1]
    S = scores(q, eq, k, ek, offset)                       # (..., T, T)
    emax = np.maximum.accumulate(ev, axis=-1)              # e_vmax over keys 0..i
    lead = q.shape[:-2]
    m = np.full(lead + (T,), rm.M_INIT, dtype=np.int64)
    l = np.zeros(lead + (T,), dtype=np.int64)
    acc = np.zeros(lead + (T, D), dtype=np.int64)
    qi = np.arange(T)
    zeros = 0
    for j in range(T):
        live = qi >= j                                     # queries that see key j
        s = S[..., :, j]
        mn = np.maximum(m, (s + 15) >> 4)
        delta = mn - m
        m = np.where(live, mn, m)
        p = exp2(s, m, mode, thr)
        use = live & (p != 0)
        zeros += int(np.sum(live & (p == 0)))
        dsh = np.minimum(delta, 31)
        l = np.where(use, ((l >> dsh) + p) & 0x1FFF, l)
        dv = np.minimum(emax - ev[..., j:j + 1], 31)       # (..., T)
        term = (p[..., None] * v[..., None, j, :]) >> dv[..., None]
        acc = np.where(use[..., None], (acc >> dsh[..., None]) + term, acc)
    if stats is not None:
        stats["zero_p"] = stats.get("zero_p", 0) + zeros
        stats["pairs"] = stats.get("pairs", 0) + int(np.prod(lead)) * T * (T + 1) // 2
        stats["acc_overflow"] = stats.get("acc_overflow", 0) + int(np.sum((acc < -(1 << 23)) | (acc >= (1 << 23))))
    r = (1 << 19) // np.maximum(l, 1)
    y = acc * r[..., None]
    sh = np.maximum(0, _bitlen(y).max(axis=-1) - 7)        # (..., T)
    rnd = np.where(sh[..., None] > 0, (y + (1 << np.maximum(sh - 1, 0))[..., None]) >> sh[..., None], y)
    o = _sat(rnd, 8)
    eo = np.clip(emax + sh - 19, 0, 255)
    return o, eo


def check(trials=20, seed=0):
    rng = np.random.default_rng(seed)
    for t in range(trials):
        T = int(rng.integers(1, 33))
        qf, kf, vf = rm.random_case(rng, n=T, d=16, scale=float(rng.uniform(0.3, 3)), outliers=bool(t % 2))
        prep = rm.prepare_kv4 if t % 3 == 2 else rm.prepare
        q, eq, k, ek, v, ev = prep(qf, kf, vf)
        for mode in (rm.MODE_EXACT, rm.MODE_MF1, rm.MODE_MF2):
            thr = [0, 4, 6][t % 3]
            o, eo = decode_causal(q, eq, k, ek, v, ev, mode=mode, thr=thr)
            for i in range(T):
                r = rm.attention_decode(q, eq, k, ek, v, ev, qrow=i, length=i + 1, mode=mode, thr=thr)
                assert np.array_equal(o[i], r.o[0]) and eo[i] == r.eo[0], (t, mode, i)
    return True


if __name__ == "__main__":
    check()
    print("raijin_vec matches raijin_model.attention_decode")
