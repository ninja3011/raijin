"""KV-cache eviction policies under a fixed cache budget (feature 15).

The char LMs of eval/lm.py are run with a per-head KV cache that holds at most
B tokens of the 32-token context. When the cache is full, one token is
evicted per step:

  window : StreamingLLM - keep SINKS sink tokens and the most recent ones
  h2o    : the chip's heavy-hitter tracker - keep SINKS sinks and RECENT recent
           tokens, evict the slot with the smallest accumulated max-normalised
           weight (rm.hh_step's weight: (frac >= 1/2 ? 181 : 256) >> floor(d))
  h2o-fp : the same policy with exact softmax probabilities (H2O as published)

Attention itself is FP32 here, so the comparison isolates the policy.
  python3 eval/eviction.py                       # Tiny Shakespeare model
  RAIJIN_LM_DATA=wikitext2 python3 eval/eviction.py
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm  # noqa: E402

SINKS = 2
RECENT = None      # None: half the budget, as in the H2O paper
LOG2E = math.log2(math.e)


def chip_weight(s):
    """Max-normalised weight in Q0.8, as the tracker: s are base-2 scores."""
    d = (s.max() - s) * 16.0                  # Q.4 distance below the max
    d = np.floor(d)
    di = np.floor(d / 16)
    w0 = np.where(np.mod(d, 16) >= 8, 181, 256)
    return np.where(di > 8, 0, np.floor(w0 / 2.0 ** np.minimum(di, 8)))


def attend(q, k, v, budget, policy):
    """q, k, v: (H, T, D) for one layer. Returns (H, T, D)."""
    H, T, D = q.shape
    out = np.zeros_like(q)
    for h in range(H):
        cache = []                             # token positions held
        acc = {}
        for t in range(T):
            cache.append(t)
            acc[t] = 0.0
            if len(cache) > budget:
                if policy == "window":
                    victim = cache[SINKS]
                else:
                    r = budget // 2 if RECENT is None else RECENT
                    cand = [c for c in cache[SINKS:] if c < t - r + 1]
                    victim = min(cand, key=lambda c: (acc[c], c))
                cache.remove(victim)
            idx = np.array(cache)
            s = q[h, t] @ k[h, idx].T / math.sqrt(D)
            p = np.exp(s - s.max())
            p /= p.sum()
            out[h, t] = p @ v[h, idx]
            if policy == "h2o":
                w = chip_weight(s * LOG2E)
            else:
                w = p
            for c, wc in zip(cache, w):
                acc[c] += wc
    return out


@torch.no_grad()
def forward(model, x, budget, policy):
    h = model.emb(x)
    for blk in model.blocks:
        b, t, _ = h.shape
        q, k, v = blk.qkv(blk.ln1(h)).view(b, t, 3, lm.NH, lm.DH).permute(2, 0, 3, 1, 4)
        q, k = lm.rope(q), lm.rope(k)
        if budget >= t:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = torch.stack([torch.from_numpy(attend(q[i].double().numpy(), k[i].double().numpy(),
                                                     v[i].double().numpy(), budget, policy)).float()
                             for i in range(b)])
        h = h + blk.proj(y.transpose(1, 2).reshape(b, t, lm.DM))
        h = h + blk.mlp(blk.ln2(h))
    return model.head(model.ln(h))


def main(n_seq=128):
    torch.set_num_threads(os.cpu_count())
    model = lm.LM()
    model.load_state_dict(torch.load(os.path.join(lm.ROOT, "eval", lm.CKPT)))
    model.eval()
    x, y = lm.batch("val", n_seq, np.random.default_rng(123))
    rows = []
    full = forward(model, x, lm.T, "full")
    for budget in (16, 12, 8):
        # score only positions t >= budget, where the cache is full and policies differ
        sel = slice(budget, None)
        for policy in ("full", "window", "h2o", "h2o-fp"):
            logits = full if policy == "full" else forward(model, x, budget, policy)
            loss = F.cross_entropy(logits[:, sel].reshape(-1, len(lm.CHARS)), y[:, sel].reshape(-1)).item()
            rows.append(dict(budget=budget, policy=policy, ppl=math.exp(loss)))
            print(rows[-1], flush=True)
    name = "eviction_wikitext2" if lm.DATASET == "wikitext2" else "eviction_shakespeare"
    lines = [f"KV eviction under a cache budget: {lm.DATASET} char LM, context {lm.T}, {n_seq} sequences, "
             f"{SINKS} sinks, recent window {RECENT or 'budget/2'} for h2o; perplexity over positions t >= budget; full = no eviction on the same positions"]
    for r in rows:
        lines.append(f"budget {r['budget']:2d}  {r['policy']:7s} ppl={r['ppl']:.4f}")
    open(os.path.join(lm.ROOT, "results", name + ".txt"), "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(lm.ROOT, "results", name + ".json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
