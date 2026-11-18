#!/usr/bin/env python3
"""Accuracy study: every numeric variant against FP32 softmax attention.

Variants (spec, evaluation plan):
  int8_tensor  plain INT8 with one scale per tensor, float softmax
  mx_float     MXINT8 inputs, float softmax
  hw_exact     Raijin, base-2 softmax with the 16-entry LUT     (bit-exact model)
  hw_mf1       Raijin, multiplier-free, one power of two         (bit-exact model)
  hw_mf2       Raijin, multiplier-free, two powers of two        (bit-exact model)
each hw_* in the unfused and the fused dataflow.

Workloads are synthetic: Gaussian Q/K/V at three attention sharpness levels,
and an "LLM-like" set where a few channels carry large outliers, which is the
case MX block scaling is meant for. Real ViT/LLM tensors can be dropped in with
--npz (arrays q, k, v of shape [heads, N, d]).

Writes results/accuracy.md.
"""
import argparse
import os

import numpy as np

import raijin_model as rm


def cos(a, b):
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def int8_tensor(x):
    s = np.max(np.abs(x)) / 127 if np.max(np.abs(x)) > 0 else 1.0
    return np.clip(np.round(x / s), -128, 127) * s


def mx(x):
    return rm.mx_dequantize(*rm.mx_quantize(x))


def workloads(rng, n, d, count):
    for name, scale, outl in (("gauss_soft", 0.7, False), ("gauss", 1.0, False),
                              ("gauss_sharp", 2.0, False), ("llm_outliers", 1.0, True)):
        cases = []
        for _ in range(count):
            q = rng.standard_normal((n, d)) * scale
            k = rng.standard_normal((n, d)) * scale
            v = rng.standard_normal((n, d))
            if outl:
                ch = rng.choice(d, 2, replace=False)
                q[:, ch] *= 6
                k[:, ch] *= 6
                v[rng.integers(0, n, 2)] *= 20
                q /= 2.5
                k /= 2.5
            cases.append((q, k, v))
        yield name, cases


def evaluate(cases):
    rows = {}
    def add(key, got, ref):
        c = cos(got, ref)
        e = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))
        rows.setdefault(key, []).append((c, e))
    for qf, kf, vf in cases:
        ref = rm.reference(qf, kf, vf)
        add("int8_tensor", rm.reference(int8_tensor(qf), int8_tensor(kf), int8_tensor(vf)), ref)
        add("mx_float", rm.reference(mx(qf), mx(kf), mx(vf)), ref)
        t = rm.prepare(qf, kf, vf)
        for fused in (False, True):
            for mode, nm in ((0, "hw_exact"), (1, "hw_mf1"), (2, "hw_mf2")):
                r = rm.attention(*t, fused=fused, mode=mode)
                add(f"{nm}_{'fused' if fused else 'unfused'}", rm.mx_dequantize(r.o, r.eo), ref)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--npz", help="optional real tensors: q, k, v arrays [heads, N, d]")
    args = ap.parse_args()
    rng = np.random.default_rng(2026)
    sets = list(workloads(rng, args.n, args.d, args.count))
    if args.npz:
        z = np.load(args.npz)
        sets.append((os.path.basename(args.npz), list(zip(z["q"], z["k"], z["v"]))))
    order = ["int8_tensor", "mx_float", "hw_exact_unfused", "hw_exact_fused",
             "hw_mf1_unfused", "hw_mf1_fused", "hw_mf2_unfused", "hw_mf2_fused"]
    lines = ["# Accuracy against FP32 attention", "",
             f"N = {args.n}, d = {args.d}, {args.count} random heads per workload. "
             "Cells show mean cosine similarity / worst-case max abs error "
             "(relative to max |reference|).", ""]
    hdr = "| Variant | " + " | ".join(name for name, _ in sets) + " |"
    lines += [hdr, "|" + " --- |" * (len(sets) + 1)]
    results = {name: evaluate(cases) for name, cases in sets}
    for key in order:
        cells = []
        for name, _ in sets:
            v = np.array(results[name][key])
            cells.append(f"{v[:, 0].mean():.5f} / {v[:, 1].max():.3f}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    out = "\n".join(lines) + "\n"
    print(out)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "accuracy.md"), "w") as f:
        f.write(out)


if __name__ == "__main__":
    main()
