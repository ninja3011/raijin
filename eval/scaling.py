"""Analytical cycle and throughput model of Raijin, fitted to and checked
against the cycle counts the RTL reports (results/*.txt), then extrapolated to
LLM-sized heads.

Parameters: N keys/queries, d head dim, P query lanes (DPUs and accumulator
rows), W = 16 elements per DPU beat (a d-wide row takes d/W beats).

  fused, pipelined   C = (N/P) * N * d/W + (43 + P)
  causal prefill     C = sum_g max((g+1) P d/W, 14 + P) + (23 + P)
  decode, 1 row      C = LEN * d/W + 34
  decode, P rows     C = LEN * d/W + 37        (GQA / MQA / speculative)
  unfused            C = 2 (N/P) N d/W + ...   (measured 411 at P = 4, N = 16)

Work: QK^T and PV are 2 N^2 d MACs = 4 N^2 d ops per head.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = 16


def fused(n, d, p):
    return (n // p) * n * (d // W) + 43 + p


def causal(n, d, p):
    return sum(max((g + 1) * p * (d // W), 14 + p) for g in range(n // p)) + 23 + p


def decode(length, d, rows=1):
    return length * (d // W) + (34 if rows == 1 else 37)


MEASURED = [  # (kind, N or LEN, P, measured cycles, source)
    ("fused", 16, 2, 173, "test_all_modes, P=2"),
    ("fused", 16, 4, 111, "results/core_N16_D16.txt"),
    ("causal", 16, 2, 153, "test_causal_prefill, P=2"),
    ("causal", 16, 4, 97, "results/causal_N16_D16.txt"),
    ("decode", 1, 4, 35, "results/decode_N16_D16.txt"),
    ("decode", 8, 4, 42, "results/decode_N16_D16.txt"),
    ("decode", 16, 4, 50, "results/decode_N16_D16.txt"),
    ("decode", 32, 4, 66, "results/paged_N16_D16.txt (INT4, LEN=32)"),
    ("multi", 4, 4, 41, "results/multi_decode_N16_D16.txt"),
    ("multi", 8, 4, 45, "results/multi_decode_N16_D16.txt"),
    ("multi", 16, 4, 53, "results/multi_decode_N16_D16.txt"),
]


def model(kind, n, p, d=16):
    return {"fused": lambda: fused(n, d, p), "causal": lambda: causal(n, d, p),
            "decode": lambda: decode(n, d), "multi": lambda: decode(n, d, rows=p)}[kind]()


def main(f_mhz=45.0):
    out = ["Cycle model against the RTL (d = 16):",
           f"{'kind':8s} {'N/LEN':>5s} {'P':>2s} {'RTL':>5s} {'model':>5s} {'error':>6s}  source"]
    for kind, n, p, cyc, src in MEASURED:
        m = model(kind, n, p)
        out.append(f"{kind:8s} {n:5d} {p:2d} {cyc:5d} {m:5d} {100 * (m - cyc) / cyc:+5.1f}%  {src}")
    out += ["", f"Extrapolation at {f_mhz:.0f} MHz (one head; fused pass = prefill without mask, "
            "decode = one token over LEN cached keys):",
            f"{'N':>5s} {'d':>3s} {'P':>2s} {'fused cycles':>12s} {'util':>5s} {'GOPS':>6s} "
            f"{'causal cycles':>13s} {'decode LEN=N':>12s} {'tokens/s':>9s}"]
    for n, d in ((16, 16), (64, 64), (128, 64), (512, 64), (2048, 128)):
        for p in (2, 4, 8):
            c = fused(n, d, p)
            ideal = 2 * n * n * d / (32 * p)            # MACs / (32 P MACs per cycle)
            gops = 4 * n * n * d / (c / (f_mhz * 1e6)) / 1e9
            dc = decode(n, d)
            out.append(f"{n:5d} {d:3d} {p:2d} {c:12d} {100 * ideal / c:4.0f}% {gops:6.2f} "
                       f"{causal(n, d, p):13d} {dc:12d} {f_mhz * 1e6 / dc:9,.0f}")
    out += ["", "Peak: 32 P MACs per cycle = 64 P ops per cycle; at 45 MHz that is "
            "5.76 GOPS (P = 2), 11.5 GOPS (P = 4), 23.0 GOPS (P = 8). Utilisation of the "
            "pipelined fused pass tends to 100% as N grows, because the fixed 43 + P cycles "
            "(drain, reciprocal, last output rows) are paid once per pass.",
            "Decode is memory-bound by design: one token reads LEN K rows and LEN V rows, "
            "one each per cycle, so its time is LEN d/W + 34 cycles whatever P is; multi-query "
            "decode serves P query heads (GQA) in the same time."]
    text = chr(10).join(out) + chr(10)
    with open(os.path.join(ROOT, "results", "scaling.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
