"""KV-cache element formats on the golden model: MXINT8, MXINT4 and MXFP4 for K
and V, in every combination the chip supports (KVFMT bits 0-3).

32-token causal decode, mean cosine against FP32 attention, averaged over
TRIALS random cases per score family (the families of tb/test_core.py).
  python3 eval/kv_formats.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import raijin_model as rm  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CASES = [("random", dict(scale=1.0)), ("sharp", dict(scale=2.5)), ("outliers", dict(scale=1.5, outliers=True))]
FMT = {"INT8": rm.mx_quantize, "INT4": rm.mx_quantize4, "FP4": rm.mx_quantize_fp4}
T, D, TRIALS = 32, 16, 16


def main():
    rng = np.random.default_rng(5)
    combos = [(kf, vf) for kf in FMT for vf in FMT]
    res = {(c, kv): [] for c, _ in CASES for kv in combos}
    rms = {f: [] for f in FMT}
    for cname, kw in CASES:
        for _ in range(TRIALS):
            qf, kf, vf = rm.random_case(rng, n=T, d=D, **kw)
            ref = rm.reference_causal(qf, kf, vf)
            q, eq = rm.mx_quantize(qf * (np.log2(np.e) / np.sqrt(D)))
            for f, fn in FMT.items():
                for x in (kf, vf):
                    m, e = fn(x)
                    rms[f].append(np.sqrt(np.mean((rm.mx_dequantize(m, e) - x) ** 2) / np.mean(x ** 2)))
            for kn, vn in combos:
                k, ek = FMT[kn](kf)
                v, ev = FMT[vn](vf)
                c = []
                for t in range(T):
                    o = rm.attention_decode(q, eq, k, ek, v, ev, qrow=t, length=t + 1)
                    g = rm.mx_dequantize(o.o, o.eo)[0]
                    c.append(np.dot(g, ref[t]) / np.linalg.norm(g) / np.linalg.norm(ref[t]))
                res[(cname, (kn, vn))].append(np.mean(c))
    lines = [f"KV element formats: {T}-token causal decode, d={D}, mean cosine vs FP32 over {TRIALS} cases per family",
             "relative RMS quantisation error of K and V rows: " +
             ", ".join(f"{f} {np.mean(v):.4f}" for f, v in rms.items()),
             f"{'K':5s} {'V':5s} bits/token " + " ".join(f"{c:>9s}" for c, _ in CASES)]
    for kn, vn in combos:
        bits = sum(16 * (8 if f == "INT8" else 4) + 8 for f in (kn, vn))
        lines.append(f"{kn:5s} {vn:5s} {bits:10d} " + " ".join(f"{np.mean(res[(c, (kn, vn))]):9.5f}" for c, _ in CASES))
    text = "\n".join(lines) + "\n"
    open(os.path.join(ROOT, "results", "kv_formats.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
