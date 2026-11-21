#!/usr/bin/env python3
"""Where does Raijin lose accuracy on a pretrained model, and how to win it back.

TinyStories-8M loses 4.3% perplexity with Raijin attention (results/lm_eval.md).
This study splits that loss and tests two host-side fixes that need no RTL change.

1. Attribution (fake quantisation, FP32 softmax): quantise-dequantise only Q, only
   K, only V, or all three to MXINT8, then run exact FP32 attention. The rest of
   the gap, from the full chip model, is the chip's softmax/exp2/reciprocal/output
   path.

2. Fixes. Both are exact rewrites of attention, so FP32 output is unchanged, and
   both can be folded offline into the model's W_q, W_k, W_v, W_o and biases:
   - K mean removal ("K smoothing", as in SageAttention): k' = k - mu_k. Every
     score of a query shifts by the same q.mu_k, so softmax is unchanged.
   - Channel balancing (SmoothQuant-style): q' = q * s, k' = k / s, with
     s_c = sqrt(max|k_c| / max|q_c|), so q'.k' = q.k. V gets v' = v / t,
     o = o' * t, with t_c = max|v_c| (attention is linear in V).
   Statistics (mu_k, max|q|, max|k|, max|v| per layer, head and channel) are
   calibrated on windows that do not overlap the evaluation windows.

Usage (same environment as model/lm_eval.py):
  python3 model/lm_study.py --data ~/lmdata/TinyStories-valid.txt --windows 300
  python3 model/lm_study.py --selftest      # random-weight GPT-Neo, no download
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import raijin_model as rm  # noqa: E402
import raijin_vec as rv    # noqa: E402
from transformers.models.gpt_neo import modeling_gpt_neo as neo  # noqa: E402

LOG2E = math.log2(math.e)
WIN = 16
_orig_attn = neo.GPTNeoSelfAttention._attn
_cfg = {}            # current variant
_stats = {}          # (layer, head) -> dict of per-channel statistics
_collect = False


def mxq(x, bits=8):
    """Row-wise MX quantisation of (..., d) arrays, equal to rm.mx_quantize."""
    amax = np.abs(x).max(axis=-1)
    qmax, frac = (127, 64) if bits == 8 else (7, 4)
    with np.errstate(divide="ignore"):
        ex = np.where(amax == 0, 0, np.ceil(np.log2(np.where(amax == 0, 1, amax) * frac / qmax)))
    ex = np.clip(ex, -127, 128)
    q = np.clip(np.round(x / 2.0 ** ex[..., None] * frac), -qmax - 1, qmax).astype(np.int64)
    if bits == 4:
        q = q * 16
    return q, (ex + rm.EXP_BIAS).astype(np.int64)


def fq(x, bits=8):
    q, e = mxq(x, bits)
    return q / 64.0 * 2.0 ** (e[..., None] - rm.EXP_BIAS)


def fp_attn(q, k, v):
    """Causal FP32 attention, GPT-Neo style (no 1/sqrt(d)). q, k, v: (h, t, d)."""
    s = q @ np.swapaxes(k, -1, -2)
    t = s.shape[-1]
    s = np.where(np.tril(np.ones((t, t), bool)), s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v


def chip_attn(q, k, v, mode, kbits=8, vbits=8):
    qi, eq = mxq(q * LOG2E, 8)
    ki, ek = mxq(k, kbits)
    vi, ev = mxq(v, vbits)
    o, eo = rv.decode_causal(qi, eq, ki, ek, vi, ev, mode=mode)
    return o / 64.0 * 2.0 ** (eo[..., None] - rm.EXP_BIAS)


def study_attn(self, query, key, value, attention_mask=None):
    q = query.detach().double().numpy()[0]       # (h, t, d), batch 1
    k = key.detach().double().numpy()[0]
    v = value.detach().double().numpy()[0]
    lid = self.layer_id
    if _collect:
        for h in range(q.shape[0]):
            st = _stats.setdefault((lid, h), dict(n=0, ksum=0, qmax=0, kmax=0, kcmax=0, vmax=0))
            st["n"] += k.shape[1]
            st["ksum"] = st["ksum"] + k[h].sum(axis=0)
            st["qmax"] = np.maximum(st["qmax"], np.abs(q[h]).max(axis=0))
            st["kmax"] = np.maximum(st["kmax"], np.abs(k[h]).max(axis=0))
            st["vmax"] = np.maximum(st["vmax"], np.abs(v[h]).max(axis=0))
        o = fp_attn(q, k, v)
    else:
        c = _cfg
        tv = np.ones((q.shape[0], 1, q.shape[2]))
        if c.get("kmean") or c.get("balance") or c.get("vbal"):
            mu = np.stack([_stats[(lid, h)]["mu"] for h in range(q.shape[0])])[:, None, :]
            if c.get("kmean"):
                k = k - mu
            if c.get("balance"):
                s = np.stack([_stats[(lid, h)]["s"] for h in range(q.shape[0])])[:, None, :]
                q, k = q * s, k / s
            if c.get("vbal"):
                tv = np.stack([_stats[(lid, h)]["t"] for h in range(q.shape[0])])[:, None, :]
                v = v / tv
        if c.get("chip"):
            o = chip_attn(q, k, v, c.get("mode", rm.MODE_EXACT), c.get("kbits", 8), c.get("vbits", 8))
        else:
            fqs = c.get("fq", "")
            o = fp_attn(fq(q * LOG2E) / LOG2E if "q" in fqs else q,
                        fq(k) if "k" in fqs else k,
                        fq(v) if "v" in fqs else v)
        o = o * tv
    return torch.from_numpy(o[None]).to(query.dtype), None


def finish_stats():
    for st in _stats.values():
        mu = st["ksum"] / st["n"]
        st["mu"] = mu
        st["s"] = np.sqrt(np.maximum(st["kmax"], 1e-6) / np.maximum(st["qmax"], 1e-6))
        st["t"] = np.maximum(st["vmax"], 1e-6)


@torch.no_grad()
def evaluate(model, wins):
    nll, count, preds = 0.0, 0, []
    for w in wins:
        logits = model(w[None]).logits[0]
        lp = torch.log_softmax(logits[:-1].float(), dim=-1)
        tgt = w[1:]
        nll -= lp[torch.arange(len(tgt)), tgt].sum().item()
        count += len(tgt)
        preds.append(logits[:-1].argmax(-1))
    return math.exp(nll / count), torch.cat(preds)


VARIANTS = [
    ("FP32", None),
    ("fake-quant Q only (FP softmax)", dict(fq="q")),
    ("fake-quant K only (FP softmax)", dict(fq="k")),
    ("fake-quant V only (FP softmax)", dict(fq="v")),
    ("fake-quant Q, K, V (FP softmax)", dict(fq="qkv")),
    ("chip, exact exp2", dict(chip=1, mode=rm.MODE_EXACT)),
    ("chip, MF2", dict(chip=1, mode=rm.MODE_MF2)),
    ("fix: K mean removal, fake-quant Q K V", dict(fq="qkv", kmean=1)),
    ("fix: K mean + Q/K balance, fake-quant Q K V", dict(fq="qkv", kmean=1, balance=1)),
    ("fix: K mean + Q/K balance + V balance, fake-quant", dict(fq="qkv", kmean=1, balance=1, vbal=1)),
    ("chip exact + K mean", dict(chip=1, kmean=1)),
    ("chip exact + K mean + Q/K balance", dict(chip=1, kmean=1, balance=1)),
    ("chip exact + K mean + Q/K + V balance", dict(chip=1, kmean=1, balance=1, vbal=1)),
    ("chip MF2 + K mean + Q/K + V balance", dict(chip=1, mode=rm.MODE_MF2, kmean=1, balance=1, vbal=1)),
    ("chip MF2 + all fixes, V INT4", dict(chip=1, mode=rm.MODE_MF2, kmean=1, balance=1, vbal=1, vbits=4)),
]


def run(model, wins, calib, out=None, title=""):
    global _collect, _cfg
    neo.GPTNeoSelfAttention._attn = study_attn
    _collect = True
    evaluate(model, calib)
    _collect = False
    finish_stats()
    rows, pred_fp = [], None
    for name, cfg in VARIANTS:
        t0 = time.time()
        if cfg is None:
            neo.GPTNeoSelfAttention._attn = _orig_attn
        else:
            neo.GPTNeoSelfAttention._attn = study_attn
            _cfg = cfg
        ppl, pred = evaluate(model, wins)
        pred_fp = pred if pred_fp is None else pred_fp
        agree = (pred == pred_fp).float().mean().item()
        rows.append((name, ppl, agree))
        print(f"{name:52s} ppl={ppl:9.3f} agree={100 * agree:6.2f}% ({time.time() - t0:.0f} s)", flush=True)
    neo.GPTNeoSelfAttention._attn = _orig_attn
    base = rows[0][1]
    lines = [f"# {title}", "",
             f"{wins.shape[0]} evaluation windows of {WIN} tokens, {calib.shape[0]} disjoint calibration windows.", "",
             "| Variant | Perplexity | vs FP32 | Top-1 agreement with FP32 |", "| --- | --- | --- | --- |"]
    for name, ppl, agree in rows:
        lines.append(f"| {name} | {ppl:.3f} | {100 * (ppl / base - 1):+.2f}% | {100 * agree:.2f}% |")
    text = "\n".join(lines) + "\n"
    if out:
        with open(out, "w") as f:
            f.write(text)
    print(text)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.selftest:
        torch.manual_seed(0)
        cfg = neo.GPTNeoConfig(vocab_size=500, hidden_size=256, num_layers=2, num_heads=16,
                               attention_types=[[["global", "local"], 1]], max_position_embeddings=64,
                               window_size=256)
        model = neo.GPTNeoForCausalLM(cfg).eval()
        model.config._attn_implementation = "eager"
        g = torch.Generator().manual_seed(1)
        wins = torch.randint(0, 500, (6, WIN), generator=g)
        calib = torch.randint(0, 500, (4, WIN), generator=g)
        rows = run(model, wins, calib, title="selftest (random weights)")
        # exactness of the rewrites: with no quantisation they must leave FP32 unchanged
        global _cfg
        neo.GPTNeoSelfAttention._attn = study_attn
        _cfg = dict(kmean=1, balance=1, vbal=1)
        p1, _ = evaluate(model, wins)
        neo.GPTNeoSelfAttention._attn = _orig_attn
        assert abs(p1 / rows[0][1] - 1) < 1e-6, (p1, rows[0][1])
        print("selftest ok: rewrites are exact in FP32")
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval import MODEL, TOKENIZER
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").eval()
    text = open(args.data, encoding="utf-8").read(4_000_000)
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = len(ids) // WIN
    allw = ids[: n * WIN].view(n, WIN)
    wins = allw[: args.windows]                       # same windows as lm_eval.py
    calib = allw[-args.calib:]                        # from the far end: no overlap
    assert args.windows + args.calib <= n
    out = os.path.join(os.path.dirname(__file__), "..", "results", "lm_study.md")
    run(model, wins, calib, out=out, title=f"Accuracy study: Raijin attention in {MODEL}")


if __name__ == "__main__":
    main()
