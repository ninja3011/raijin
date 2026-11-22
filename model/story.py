#!/usr/bin/env python3
"""Long generation on Raijin: a 1000-word story from TinyStories-8M with every
attention head on the chip model and the chip's own KV-cache budget.

TinyStories-8M (GPT-Neo, 8 layers x 16 heads, d = 16) matches Raijin's head
size. The chip's KV SRAM holds 16 tokens in MXINT8 or 32 in MXINT4/MXFP4, so
a 1,300-token story only fits by eviction, as the host driver does it:

  window : StreamingLLM. Keep SINKS sink tokens and the newest tokens (feature 7).
  h2o    : keep SINKS sinks and a recent window of budget/2; evict the token with
           the smallest accumulated weight, using the heavy-hitter tracker's
           divider-free statistic (feature 15). With decay, the accumulators are
           halved every 16 steps (the tracker's HHCTL halve bit), so old heavy
           hitters fade instead of holding their slots forever.

GPT-Neo alternates global layers with local layers that only see the last
window_size (256) positions. Every policy honours that: a local layer never
keeps a position more than 255 tokens back, sinks included, exactly as the
model's own attention mask does.

Each decode step of each head runs through the bit-exact vector model of the
decode path (model/raijin_vec.py, MF2 exponent, max-first decode, SKIPTHR = 6),
with the key mean folded out of K (the fix measured in results/lm_study.md) and
K/V rows quantised as the cache stores them. Position embeddings are GPT-Neo's
learned absolute ones (host side), so no RoPE is involved.

Outputs (results/):
  story_<config>.md        the generated story (seeded sampling, same seed for all)
  story_long_ppl.md        teacher-forced perplexity on long TinyStories text,
                           positions >= 32, chip configurations vs FP32 full context

Usage (laptop venv with torch + transformers, like model/lm_eval.py):
  python3 model/story.py --data ~/lmdata/TinyStories-valid.txt
  python3 model/story.py --selftest          # random-weight GPT-Neo, no download
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from transformers.models.gpt_neo import modeling_gpt_neo as neo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raijin_model as rm  # noqa: E402
import raijin_vec as rv    # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOG2E = math.log2(math.e)      # GPT-Neo does not scale scores by 1/sqrt(d)
SINKS = 4
_orig_attn = neo.GPTNeoSelfAttention._attn

# configurations: name -> (policy, budget, chip?, K format, V format)
CONFIGS = {
    "fp32-full":         dict(policy="full",   budget=None, chip=False),
    "fp32-window32":     dict(policy="window", budget=32,   chip=False),
    "chip-int8-window16": dict(policy="window", budget=16,  chip=True, kf=8, vf=8),
    "chip-fp4-window32": dict(policy="window", budget=32,   chip=True, kf="fp4", vf="fp4"),
    "fp32-h2o32":        dict(policy="h2o",    budget=32,   chip=False, decay=True),
    "fp32-h2o32-nodecay": dict(policy="h2o",   budget=32,   chip=False, decay=False),
    "chip-fp4-h2o32":    dict(policy="h2o",    budget=32,   chip=True, kf="fp4", vf="fp4", decay=True),
}
STORY_CONFIGS = ("fp32-full", "chip-fp4-window32", "chip-fp4-h2o32")
DECAY_EVERY = 16


def mxq(x, bits=8):
    """Row-wise MX quantisation as the chip's cache stores rows (equals
    rm.mx_quantize, rm.mx_quantize4 and rm.mx_quantize_fp4)."""
    amax = np.abs(x).max(axis=-1)
    if bits == "fp4":
        with np.errstate(divide="ignore"):
            ex = np.where(amax == 0, 0, np.ceil(np.log2(np.where(amax == 0, 1, amax) * 4 / 6)))
        cand = []
        for exc in (np.clip(ex, -127, 128), np.clip(ex - 1, -127, 128)):
            y = x / 2.0 ** exc[..., None] * 4
            g = rm.FP4_GRID[np.abs(y[..., None] - rm.FP4_GRID).argmin(axis=-1)]
            cand.append((g, exc, ((g / 4 * 2.0 ** exc[..., None] - x) ** 2).sum(-1)))
        lo = cand[1][2] < cand[0][2]
        g = np.where(lo[..., None], cand[1][0], cand[0][0])
        ex = np.where(lo, cand[1][1], cand[0][1])
        return np.round(g * 16).astype(np.int64), (ex + rm.EXP_BIAS).astype(np.int64)
    qmax, frac = (127, 64) if bits == 8 else (7, 4)
    with np.errstate(divide="ignore"):
        ex = np.where(amax == 0, 0, np.ceil(np.log2(np.where(amax == 0, 1, amax) * frac / qmax)))
    ex = np.clip(ex, -127, 128)
    q = np.clip(np.round(x / 2.0 ** ex[..., None] * frac), -qmax - 1, qmax).astype(np.int64)
    if bits == 4:
        q = q * 16
    return q, (ex + rm.EXP_BIAS).astype(np.int64)


def chip_weight(s):
    """The tracker's max-normalised weight (Q0.8) from base-2 scores, per head."""
    d = np.floor((s.max(axis=-1, keepdims=True) - s) * 16.0)
    di = np.floor(d / 16)
    w0 = np.where(np.mod(d, 16) >= 8, 181, 256)
    return np.where(di > 8, 0, np.floor(w0 / 2.0 ** np.minimum(di, 8)))


class State:
    """Host-side cache bookkeeping, per layer: the kept positions of each head
    and, for H2O, their accumulated weights."""

    def __init__(self, cfg, mu):
        self.cfg = cfg
        self.mu = mu                 # layer -> (h, d) key means, or None
        self.keep = {}               # layer -> list (per head) of kept positions
        self.acc = {}                # layer -> list (per head) of dict pos -> weight
        self.collect = None          # calibration: layer -> [sum (h, d), count]
        self.v_reads = [0, 0]        # V rows read, keys scored (max-first)

    def select(self, lid, h, p, win=None):
        """Positions head-by-head that the cache holds when position p decodes.
        win: the layer's local attention window (GPT-Neo local layers), or None."""
        c = self.cfg
        lo = 0 if win is None else max(0, p - win + 1)    # oldest position the layer may see
        if c["policy"] == "full" or p + 1 <= c["budget"]:
            return [list(range(lo, p + 1))] * h
        b = c["budget"]
        if c["policy"] == "window":
            sinks = [x for x in range(SINKS) if x >= lo]
            return [sinks + list(range(p - (b - len(sinks)) + 1, p + 1))] * h
        keep = self.keep.setdefault(lid, [list(range(p)) for _ in range(h)])
        acc = self.acc.setdefault(lid, [dict.fromkeys(range(p), 0.0) for _ in range(h)])
        if c.get("decay") and p % DECAY_EVERY == 0:      # HHCTL halve
            for a in acc:
                for x in a:
                    a[x] *= 0.5
        recent = b // 2
        for hh in range(h):
            keep[hh].append(p)
            acc[hh][p] = 0.0
            for x in [x for x in keep[hh] if x < lo]:     # out of a local layer's reach
                keep[hh].remove(x)
                del acc[hh][x]
            while len(keep[hh]) > b:
                cand = [x for x in keep[hh] if x >= SINKS and x <= p - recent]
                victim = min(cand, key=lambda x: (acc[hh][x], x))
                keep[hh].remove(victim)
                del acc[hh][victim]
        return [list(x) for x in keep]

    def credit(self, lid, sel, s):
        """H2O: add the tracker weight of this step to every kept key."""
        if self.cfg["policy"] != "h2o":
            return
        acc = self.acc.setdefault(lid, [dict() for _ in range(len(sel))])
        for hh, pos in enumerate(sel):
            w = chip_weight(s[hh][: len(pos)])
            for x, wx in zip(pos, w):
                acc[hh][x] = acc[hh].get(x, 0.0) + float(wx)


_state: State | None = None


def head_step(q, k, v, lid, p, win=None):
    """One decode step of every head of a layer for the query at position p.
    q: (h, d); k, v: (h, p + 1, d) FP32 projections of all positions so far."""
    st = _state
    c = st.cfg
    h, d = q.shape
    sel = st.select(lid, h, p, win)
    n = len(sel[0])
    ks = np.stack([k[hh, sel[hh]] for hh in range(h)])         # (h, n, d)
    vs = np.stack([v[hh, sel[hh]] for hh in range(h)])
    if not c["chip"]:
        s = np.einsum("hd,hnd->hn", q, ks)
        w = np.exp(s - s.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        if c["policy"] == "h2o":
            st.credit(lid, sel, s * LOG2E)
        return np.einsum("hn,hnd->hd", w, vs)
    if st.mu is not None:
        ks = ks - st.mu[lid][:, None, :]
    qi, eq = mxq(np.repeat((q * LOG2E)[:, None, :], n, axis=1), 8)   # every row = the query
    ki, ek = mxq(ks, c["kf"])
    vi, ev = mxq(vs, c["vf"])
    stats = {}
    o, eo = rv.decode_causal(qi, eq, ki, ek, vi, ev, mode=rm.MODE_MF2, thr=6,
                             final_max=True, stats=stats)
    if c["policy"] == "h2o":
        s = rv.scores(qi[:, -1:], eq[:, -1:], ki, ek)[:, 0] / 16.0   # chip's Q.4 scores
        st.credit(lid, sel, s)
    # V reads of the last row only (the step the chip runs)
    srow = rv.scores(qi[:, -1:], eq[:, -1:], ki, ek)[:, 0]
    m = (srow.max(-1, keepdims=True) + 15) >> 4
    kk = (m * 16 - srow) >> 4
    st.v_reads[0] += int(np.sum(kk < 6))
    st.v_reads[1] += srow.size
    return o[:, -1] / 64.0 * 2.0 ** (eo[:, -1, None] - rm.EXP_BIAS)


def story_attn(self, query, key, value, attention_mask=None, *args, **kwargs):
    q = query.detach().double().numpy()[0]       # (h, tq, d), batch 1
    k = key.detach().double().numpy()[0]         # (h, tk, d), every past position
    v = value.detach().double().numpy()[0]
    lid = self.layer_id
    win = self.config.window_size if self.attention_type == "local" else None
    st = _state
    if st.collect is not None:                   # calibration of the key means
        acc = st.collect.setdefault(lid, [0.0, 0])
        acc[0] = acc[0] + k.sum(axis=1)
        acc[1] += k.shape[1]
    tq, tk = q.shape[1], k.shape[1]
    out = np.zeros_like(q)
    for i in range(tq):
        p = tk - tq + i
        out[:, i] = head_step(q[:, i], k[:, : p + 1], v[:, : p + 1], lid, p, win)
    return torch.from_numpy(out[None]).to(query.dtype), None


@torch.no_grad()
def calibrate(model, texts):
    """Per layer and head key means, on text disjoint from the evaluation."""
    global _state
    _state = State(CONFIGS["fp32-full"], None)
    _state.collect = {}
    neo.GPTNeoSelfAttention._attn = story_attn
    for ids in texts:
        model(ids[None])
    neo.GPTNeoSelfAttention._attn = _orig_attn
    return {lid: s / n for lid, (s, n) in _state.collect.items()}


@torch.no_grad()
def run_tokens(model, ids, cfg, mu, generate=0, seed=0, ban=None, temp=0.8, topk=40, stop=None):
    """Feed ids one token at a time (the decode loop); optionally keep sampling
    up to `generate` more tokens, ending early once stop(ids) is true (checked
    every 16 new tokens). Never passes the model's position limit.
    Returns (all ids, per-position log-probs of ids[1:], V-read counts)."""
    global _state
    _state = State(cfg, mu)
    neo.GPTNeoSelfAttention._attn = story_attn
    g = torch.Generator().manual_seed(seed)
    past, lps, out = None, [], list(ids.tolist())
    total = min(len(out) + generate, model.config.max_position_embeddings)
    t = 0
    while t < total - 1 or t < len(out) - 1:
        x = torch.tensor([[out[t]]])
        r = model(input_ids=x, past_key_values=past, use_cache=True,
                  position_ids=torch.tensor([[t]]))
        past = r.past_key_values
        logits = r.logits[0, -1].float()
        if t + 1 < len(out):
            lps.append(torch.log_softmax(logits, -1)[out[t + 1]].item())
        elif len(out) < total:
            if ban is not None:
                logits[ban] = -float("inf")
            top = torch.topk(logits / temp, topk)
            nxt = top.indices[torch.multinomial(torch.softmax(top.values, -1), 1, generator=g)]
            out.append(int(nxt))
            if stop is not None and (len(out) - len(ids)) % 16 == 0 and stop(out):
                total = len(out)
        t += 1
    neo.GPTNeoSelfAttention._attn = _orig_attn
    return out, np.array(lps), _state.v_reads


def selftest():
    torch.manual_seed(0)
    cfg = neo.GPTNeoConfig(vocab_size=300, hidden_size=256, num_layers=2, num_heads=16,
                           attention_types=[[["global", "local"], 1]], max_position_embeddings=256,
                           window_size=24)          # < 80 tokens: the local mask matters
    model = neo.GPTNeoForCausalLM(cfg).eval()
    model.config._attn_implementation = "eager"
    ids = torch.randint(0, 300, (80,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = torch.log_softmax(model(ids[None]).logits[0, :-1].float(), -1)
    ref = ref[torch.arange(79), ids[1:]].numpy()
    _, lp, _ = run_tokens(model, ids, CONFIGS["fp32-full"], None)
    assert np.allclose(lp, ref, atol=1e-4), np.abs(lp - ref).max()
    out, _, _ = run_tokens(model, ids[:10], CONFIGS["fp32-full"], None, generate=1000, seed=1,
                           stop=lambda o: len(o) >= 50)
    assert len(out) == 58, len(out)                      # stops at the first check past 50
    out, _, _ = run_tokens(model, ids[:10], CONFIGS["fp32-full"], None, generate=1000, seed=1)
    assert len(out) == cfg.max_position_embeddings, len(out)
    mu = calibrate(model, [ids[:40]])
    for name, c in CONFIGS.items():
        t0 = time.time()
        out, lp, vr = run_tokens(model, ids[:60], c, mu, generate=20, seed=3)
        print(f"{name:20s} ppl={math.exp(-lp.mean()):8.3f} tokens={len(out)} "
              f"V reads {vr[0]}/{vr[1]} ({time.time() - t0:.1f} s)")
    print("selftest ok: the decode loop matches a full forward pass in FP32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--words", type=int, default=1000)
    ap.add_argument("--ppl-tokens", type=int, default=1024)
    ap.add_argument("--ppl-seqs", type=int, default=2)
    ap.add_argument("--prompt", default="Once upon a time, in a small village by the sea, there lived a girl named Lily.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.selftest:
        return selftest()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval import MODEL, TOKENIZER
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").eval()
    text = open(args.data, encoding="utf-8").read(4_000_000)
    ids = tok(text, return_tensors="pt").input_ids[0]
    calib = [ids[-(i + 1) * 64: -i * 64 or None] for i in range(16)]       # far end of the file
    mu = calibrate(model, calib)
    eos = tok.eos_token_id
    # 1) long-context perplexity, teacher forced
    seqs = [ids[i * args.ppl_tokens:(i + 1) * args.ppl_tokens] for i in range(args.ppl_seqs)]
    rows = []
    for name, c in CONFIGS.items():
        t0 = time.time()
        lps, vr = [], [0, 0]
        for s in seqs:
            _, lp, v = run_tokens(model, s, c, mu)
            lps.append(lp[31:])                                  # positions >= 32
            vr = [vr[0] + v[0], vr[1] + v[1]]
        lp = np.concatenate(lps)
        rows.append((name, math.exp(-lp.mean()), vr))
        print(f"{name:20s} ppl={rows[-1][1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    base = rows[0][1]
    lines = [f"# Long-context perplexity: {MODEL} with Raijin's KV-cache budget", "",
             f"{args.ppl_seqs} sequences of {args.ppl_tokens} tokens of TinyStories validation text, decoded one token "
             f"at a time; perplexity over positions 32 and later. {SINKS} sink tokens; H2O keeps a recent window of "
             "half the budget; with decay the accumulators halve every 16 steps (HHCTL halve). GPT-Neo's local "
             "layers keep their 256-token reach under every policy. "
             "Chip rows: MF2, SKIPTHR = 6, max-first decode, key mean folded out of K.", "",
             "| Configuration | Cache | Perplexity | vs FP32 full context | Keys needing a V read |",
             "| --- | --- | --- | --- | --- |"]
    for name, ppl, vr in rows:
        c = CONFIGS[name]
        cache = "all tokens" if c["budget"] is None else f"{c['budget']} tokens, {c['policy']}" + (
            "" if c["policy"] != "h2o" else (" + decay" if c.get("decay") else ", no decay"))
        vfrac = f"{100 * vr[0] / vr[1]:.1f}%" if vr[1] else "n/a"
        lines.append(f"| {name} | {cache} | {ppl:.3f} | {100 * (ppl / base - 1):+.2f}% | {vfrac} |")
    open(os.path.join(ROOT, "results", "story_long_ppl.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    # 2) the story: same prompt and seed for every configuration, EOS banned so it runs long
    prompt = tok(args.prompt, return_tensors="pt").input_ids[0]
    enough = lambda out: len(tok.decode(out).split()) >= args.words   # noqa: E731
    for name in STORY_CONFIGS:
        t0 = time.time()
        out, _, vr = run_tokens(model, prompt, CONFIGS[name], mu, generate=10_000, seed=7, ban=eos,
                                stop=enough)
        story = tok.decode(out)
        words = len(story.split())
        c = CONFIGS[name]
        head = (f"# Story: {name}\n\n{MODEL}, {len(out)} tokens, {words} words, "
                f"{'every attention head on the Raijin chip model, ' if c['chip'] else 'FP32 attention, '}"
                f"cache: {'all tokens' if c['budget'] is None else str(c['budget']) + ' tokens, ' + c['policy']}. "
                f"Sampling: top-k 40, temperature 0.8, seed 7, end-of-text banned. "
                f"Generated in {time.time() - t0:.0f} s.\n\n")
        open(os.path.join(ROOT, "results", f"story_{name}.md"), "w").write(head + story + "\n")
        print(f"{name}: {words} words, {len(out)} tokens", flush=True)


if __name__ == "__main__":
    main()
