#!/usr/bin/env python3
"""Language-model benchmark: TinyStories-8M with Raijin attention in every head.

TinyStories-8M (GPT-Neo, 8 layers x 16 heads, head dimension 16) has exactly
Raijin's head size, so every attention head of the model can be replaced by
the bit-exact golden model (model/raijin_model.py). Each causal row runs as
one KV-cache decode step, the same operation the RTL performs in decode mode.

Benchmark: next-token perplexity and top-1 agreement with FP32 on the
TinyStories validation set, in non-overlapping 16-token windows (Raijin's
N = 16), for FP32 attention and the exact, MF1 and MF2 Raijin modes.

Example: greedy continuation of one prompt line with FP32 and with Raijin
attention, plus the quantised Q/K/V of layer 0, head 0 for the RTL check in
tb/test_core.py (test_lm_head).

Usage (outside the tools container, in a venv with torch + transformers):
  python3 model/lm_eval.py --data ~/lmdata/TinyStories-valid.txt --windows 300
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_neo import modeling_gpt_neo as neo

sys.path.insert(0, os.path.dirname(__file__))
import raijin_model as rm  # noqa: E402

MODEL = "roneneldan/TinyStories-8M"
TOKENIZER = "EleutherAI/gpt-neo-125M"
WIN = 16                      # Raijin sequence length N
LOG2E = math.log2(math.e)     # GPT-Neo does not scale scores by 1/sqrt(d)
MODE_NAMES = {0: "exact", 1: "mf1", 2: "mf2"}

_mode = 0
_capture = None               # set to a dict to capture layer-0 head-0 vectors
_orig_attn = neo.GPTNeoSelfAttention._attn


def quantise_head(q, k, v):
    """FP32 head tensors (T, 16) -> MXINT8 operands in Raijin's formats."""
    qq, eq = rm.mx_quantize(q * LOG2E)
    kq, ek = rm.mx_quantize(k)
    vq, ev = rm.mx_quantize(v)
    return qq, eq, kq, ek, vq, ev


def raijin_head(args):
    """Causal attention of one head: row t is one decode step over keys 0..t."""
    q, k, v, mode = args
    ops = quantise_head(q, k, v)
    t_len = q.shape[0]
    out = np.zeros_like(q)
    for t in range(t_len):
        r = rm.attention_decode(*ops, qrow=t, length=t + 1, mode=mode)
        out[t] = rm.mx_dequantize(r.o, r.eo)[0]
    return out


def raijin_attn(self, query, key, value, attention_mask=None):
    b, h, t, d = query.shape
    assert d == 16 and t <= WIN, (d, t)
    q = query.detach().float().numpy()
    k = key.detach().float().numpy()
    v = value.detach().float().numpy()
    jobs = [(q[i, j], k[i, j], v[i, j], _mode) for i in range(b) for j in range(h)]
    outs = _pool.map(raijin_head, jobs)
    if _capture is not None and self.layer_id == 0 and "q" not in _capture:
        ops = quantise_head(q[0, 0], k[0, 0], v[0, 0])
        _capture.update(dict(zip(("q", "eq", "k", "ek", "v", "ev"), ops)), mode=_mode)
    o = torch.from_numpy(np.stack(outs).reshape(b, h, t, d)).to(query.dtype)
    return o, None


def use_raijin(enable: bool):
    neo.GPTNeoSelfAttention._attn = raijin_attn if enable else _orig_attn


def windows(tok, path, n_win):
    text = open(path, encoding="utf-8").read(2_000_000)
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = min(n_win, len(ids) // WIN)
    return ids[: n * WIN].view(n, WIN)


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


@torch.no_grad()
def continue_line(model, tok, prompt, n_new=12):
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    for _ in range(n_new):
        ctx = ids[-WIN:]
        nxt = model(ctx[None]).logits[0, -1].argmax()
        ids = torch.cat([ids, nxt[None]])
    return tok.decode(ids)


def main():
    global _mode, _capture, _pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="TinyStories-valid.txt")
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--prompt", default="Once upon a time, there was a little girl named Lily. She")
    args = ap.parse_args()
    torch.set_num_threads(4)
    _pool = Pool(args.procs)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").eval()
    wins = windows(tok, args.data, args.windows)
    n_pred = wins.shape[0] * (WIN - 1)
    print(f"{MODEL}: {wins.shape[0]} windows of {WIN} tokens, {n_pred} predictions")

    rows = []
    use_raijin(False)
    t0 = time.time()
    ppl_fp, pred_fp = evaluate(model, wins)
    rows.append(("FP32 (reference)", ppl_fp, 1.0, time.time() - t0))
    print(rows[-1])
    use_raijin(True)
    for mode in (0, 1, 2):
        _mode = mode
        t0 = time.time()
        ppl, pred = evaluate(model, wins)
        agree = (pred == pred_fp).float().mean().item()
        rows.append((f"Raijin {MODE_NAMES[mode]}", ppl, agree, time.time() - t0))
        print(rows[-1])

    # worked example: one prompt line
    use_raijin(False)
    fp_text = continue_line(model, tok, args.prompt)
    use_raijin(True)
    _mode = 0
    rj_text = continue_line(model, tok, args.prompt)
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids[0][:WIN]
    _capture = {}
    with torch.no_grad():
        model(prompt_ids[None])
    cap = _capture
    exp = [rm.attention_decode(cap["q"], cap["eq"], cap["k"], cap["ek"], cap["v"], cap["ev"],
                               qrow=t, length=t + 1, mode=0) for t in range(len(prompt_ids))]
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "lm_head_vectors.npz"),
             q=cap["q"], eq=cap["eq"], k=cap["k"], ek=cap["ek"], v=cap["v"], ev=cap["ev"],
             o=np.stack([e.o[0] for e in exp]), eo=np.array([e.eo[0] for e in exp]),
             tokens=prompt_ids.numpy())
    _pool.close()

    lines = [
        "# TinyStories benchmark: Raijin attention in every head of TinyStories-8M", "",
        f"Model `{MODEL}` (GPT-Neo, 8 layers x 16 heads, head dimension 16 = Raijin's d).",
        "Every attention head is replaced by the bit-exact Raijin golden model; each causal row",
        "is one KV-cache decode step (the RTL's decode mode). Q/K/V are quantised to MXINT8 per row;",
        "log2(e) is folded into Q (GPT-Neo has no 1/sqrt(d) scaling).", "",
        f"Data: TinyStories validation set, first {wins.shape[0]} non-overlapping {WIN}-token windows "
        f"({n_pred} next-token predictions, context <= {WIN} tokens).", "",
        "| Attention | Perplexity | Top-1 agreement with FP32 | Eval time (s) |",
        "| --- | --- | --- | --- |",
    ]
    for name, ppl, agree, sec in rows:
        lines.append(f"| {name} | {ppl:.3f} | {100 * agree:.2f}% | {sec:.0f} |")
    lines += ["", "## Worked example (greedy decoding, 16-token context window)", "",
              f"Prompt: `{args.prompt}`", "",
              f"- FP32 attention: {fp_text!r}",
              f"- Raijin exact attention: {rj_text!r}", "",
              f"Layer 0, head 0 of the prompt ({len(prompt_ids)} tokens) is saved in `results/lm_head_vectors.npz`",
              "and replayed through the RTL by `tb/test_core.py::test_lm_head` (see `results/lm_head_rtl.txt`)."]
    with open(os.path.join(args.out, "lm_eval.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
