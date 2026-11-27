"""Ablation of Raijin's features on one attention layer of a small edge LLM.

Workload: 8 query heads, 2 KV heads (GQA 4:1), head dim d = 64, a 512-token
prompt (causal prefill) followed by 512 generated tokens. P = 4, 45 MHz.
Cycle counts come from the validated model in eval/scaling.py; accuracy from
results/lm_perplexity.json. Each row adds one feature to the row above.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaling import causal, decode, fused, W, ROOT  # noqa: E402

HQ, HKV, D, PROMPT, GEN, P, F = 8, 2, 64, 512, 512, 4, 45e6
G = HQ // HKV
B = D // W


def fused_unpipelined(n, d, p):
    # sram-kv baseline: each group of P rows runs clear, Q load, stream, drain,
    # reciprocal and output strictly in sequence (357 cycles measured at N = 16, P = 2)
    return (n // p) * (n * (d // W) + 2 * p + 24)


def prefill_by_decode(n, d):
    return sum(decode(t, d) for t in range(1, n + 1))


def kv_bytes(k4, v4):
    exps = 2 * (D // 16)
    return (D // 2 if k4 else D) + (D // 2 if v4 else D) + exps


lens = range(PROMPT, PROMPT + GEN)
ppl = {r["variant"]: r["ppl"] for r in json.load(open(os.path.join(ROOT, "results", "lm_perplexity.json")))} \
    if os.path.exists(os.path.join(ROOT, "results", "lm_perplexity.json")) else {}


def find(key):
    for k, v in ppl.items():
        if key in k:
            return v
    return None


rows = []
# 0 baseline: no causal mode, so a causal prompt is built from decode steps;
# decode serves one query head per pass; INT8 K and V; no skipping
pre0 = HQ * prefill_by_decode(PROMPT, D)
dec0 = sum(HQ * decode(L, D) for L in lens) / GEN
vr0 = sum(HQ * L for L in lens) / GEN
rows.append(("baseline (sram-kv)", pre0, dec0, kv_bytes(0, 0), vr0, find("exact")))
# 2 group pipelining: speeds up full (non-causal) fused passes; recorded separately below
rows.append(("+ group pipelining (2)", pre0, dec0, kv_bytes(0, 0), vr0, find("exact")))
# 3 causal prefill
pre3 = HQ * causal(PROMPT, D, P)
rows.append(("+ causal prefill (3)", pre3, dec0, kv_bytes(0, 0), vr0, find("exact")))
# 4 multi-query decode: the G query heads of a KV group share one pass
dec4 = sum(HKV * decode(L, D, rows=G) for L in lens) / GEN
vr4 = sum(HKV * L for L in lens) / GEN
rows.append(("+ GQA decode (4)", pre3, dec4, kv_bytes(0, 0), vr4, find("exact")))
# 6/14 INT4 V
rows.append(("+ INT4 V cache (6)", pre3, dec4, kv_bytes(0, 1), vr4, find("K INT8, V INT4")))
# 8 key skip at SKIPTHR = 6: 24.7% of keys get zero weight in the LM
skip = 0.247
rows.append(("+ key skip, SKIPTHR=6 (8)", pre3, dec4, kv_bytes(0, 1), vr4 * (1 - skip), find("V INT4, SKIPTHR=6")))
# all of the above with the cheaper two-shift exponent
rows.append(("+ MF2 exponent (operating pt)", pre3, dec4, kv_bytes(0, 1), vr4 * (1 - skip), find("operating point")))

out = []
out.append(__doc__.strip())
out.append("")
out.append(f"{'configuration':28s} {'prefill cyc':>12s} {'prefill ms':>10s} {'decode cyc/tok':>15s} {'tok/s':>8s} "
           f"{'KV B/tok/head':>13s} {'V reads/tok':>11s} {'ppl':>7s}")
for name, pre, dec, kvb, vr, pp in rows:
    out.append(f"{name:28s} {pre:12,.0f} {pre / F * 1e3:10.2f} {dec:15,.0f} {F / dec:8,.0f} "
               f"{kvb:13d} {vr:11,.0f} {('%.4f' % pp) if pp else '  —':>7s}")
first, last = rows[0], rows[-1]
out.append("")
out.append(f"Prefill {first[1] / last[1]:.1f}x faster, decode {first[2] / last[2]:.1f}x faster, "
           f"KV storage {first[3] / last[3]:.2f}x smaller, V reads {first[4] / last[4]:.1f}x fewer.")
out.append(f"Perplexity: FP32 attention is {ppl.get('FP32 attention', float('nan')):.4f}. The skip fraction (24.7% of keys) is the one measured in the LM.")
out.append("")
out.append("Group pipelining on a full fused (non-causal, encoder-style) pass, one head:")
out.append(f"{'N':>5s} {'d':>3s} {'unpipelined':>12s} {'pipelined':>10s} {'speed-up':>9s}")
for n, d in ((16, 16), (64, 64), (128, 64), (512, 64)):
    a, b = fused_unpipelined(n, d, P), fused(n, d, P)
    out.append(f"{n:5d} {d:3d} {a:12,d} {b:10,d} {a / b:8.2f}x")
out.append("")
out.append("Host link per decode token at N = 16 (measured, features 11-13): SPI 135 us, QSPI 66.9 us, "
           "QSPI + token window 36.2 us (7,400 -> 27,700 tokens/s).")
text = "\n".join(out) + "\n"
open(os.path.join(ROOT, "results", "ablation.txt"), "w").write(text)
print(text)
