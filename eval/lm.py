"""A small character-level transformer whose attention heads match Raijin
(head dimension 16, interleaved RoPE with theta_i = 10000^(-2i/16)), and an
evaluation that swaps its attention for the bit-exact chip model.

  python3 eval/lm.py train      # trains on Tiny Shakespeare, saves eval/lm.pt
  python3 eval/lm.py eval       # perplexity for FP32 and every chip variant
  RAIJIN_LM_DATA=wikitext2 python3 eval/lm.py train|eval
                                # the same on WikiText-2 (raw characters, standard
                                # train/test split), context 32, eval/lm_wt2.pt
  WikiText-2 is not committed; fetch it with
    for f in train valid test; do curl -fL --create-dirs -o eval/data/wikitext-2/$f.txt \
      https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/$f.txt; done

The chip variant runs exactly what Raijin does per generated token: Q (with
log2(e)/sqrt(d) folded in), K and V are quantised to MXINT8 (or MXINT4), Q and
K are rotated by the on-write RoPE unit, and every query i attends to keys
0..i through the decode-step online softmax (model/raijin_vec.py, verified bit
for bit against model/raijin_model.py, which the RTL matches). The output is
dequantised and the rest of the network stays in FP32.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import raijin_model as rm  # noqa: E402
import raijin_vec as rv    # noqa: E402

DATASET = os.environ.get("RAIJIN_LM_DATA", "shakespeare")
DH, NH, DM, NL = 16, 8, 128, 4
torch.manual_seed(0)
if DATASET == "wikitext2":
    # WikiText-2 (Merity et al. 2016), tokenised release as mirrored in pytorch/examples;
    # modelled as characters, trained on train.txt and scored on test.txt; context 32, since the 13-bit l register holds sums up to about 31 keys
    T, CKPT, TAG = 32, "lm_wt2.pt", "lm_wikitext2"
    _d = os.path.join(ROOT, "eval", "data", "wikitext-2")
    _tr, _te = (open(os.path.join(_d, f + ".txt"), encoding="utf-8").read() for f in ("train", "test"))
    TEXT = _tr + _te
    SPLIT = len(_tr)
else:
    T, CKPT, TAG = 32, "lm.pt", "lm_perplexity"
    TEXT = open(os.path.join(ROOT, "eval", "data", "tinyshakespeare.txt")).read()
    SPLIT = None
CHARS = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(CHARS)}
DATA = np.array([STOI[c] for c in TEXT], dtype=np.int64)
SPLIT = SPLIT if SPLIT is not None else int(0.9 * len(DATA))


def rope(x):
    """Interleaved RoPE over the last dim (DH), positions 0..T-1, as rm.rope_float."""
    t = x.shape[-2]
    th = 10000.0 ** (-2 * torch.arange(DH // 2) / DH)
    a = torch.arange(t)[:, None] * th[None, :]
    c, s = torch.cos(a), torch.sin(a)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(DM), nn.LayerNorm(DM)
        self.qkv = nn.Linear(DM, 3 * DM)
        self.proj = nn.Linear(DM, DM)
        self.mlp = nn.Sequential(nn.Linear(DM, 4 * DM), nn.GELU(), nn.Linear(4 * DM, DM))
        self.attn_fn = None          # None: FP32; else a callable on (q, k, v) numpy

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, NH, DH).permute(2, 0, 3, 1, 4)
        if self.attn_fn is None:
            y = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        else:
            y = torch.from_numpy(self.attn_fn(q.detach().numpy(), k.detach().numpy(),
                                              v.detach().numpy())).float()
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, DM))
        return x + self.mlp(self.ln2(x))


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(CHARS), DM)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.ln = nn.LayerNorm(DM)
        self.head = nn.Linear(DM, len(CHARS))

    def forward(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def batch(split, n, rng):
    lo, hi = (0, SPLIT - T - 1) if split == "train" else (SPLIT, len(DATA) - T - 1)
    ix = rng.integers(lo, hi, n) if split == "train" else np.linspace(lo, hi, n).astype(np.int64)
    x = np.stack([DATA[i:i + T] for i in ix])
    y = np.stack([DATA[i + 1:i + T + 1] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


def train(steps=4000):
    torch.set_num_threads(os.cpu_count())
    model = LM()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(steps):
        x, y = batch("train", 64, rng)
        loss = F.cross_entropy(model(x).view(-1, len(CHARS)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 500 == 0 or it == steps - 1:
            print(f"step {it} loss {loss.item():.3f} ({time.time() - t0:.0f} s)", flush=True)
    torch.save(model.state_dict(), os.path.join(ROOT, "eval", CKPT))
    n = sum(p.numel() for p in model.parameters())
    print(f"saved, {n} parameters")


# ---------------------------------------------------------------------------
# Chip attention
# ---------------------------------------------------------------------------
def mxq(x, bits=8):
    """Row-wise MX quantisation, vectorised; equals rm.mx_quantize / mx_quantize4,
    and rm.mx_quantize_fp4 for bits="fp4"."""
    amax = np.abs(x).max(axis=-1)
    if bits == "fp4":
        with np.errstate(divide="ignore"):
            ex = np.where(amax == 0, 0, np.ceil(np.log2(np.where(amax == 0, 1, amax) * 4 / 6)))
        cand = []
        for exc in (np.clip(ex, -127, 128), np.clip(ex - 1, -127, 128)):
            y = x / 2.0 ** exc[..., None] * 4
            g = rm.FP4_GRID[np.abs(y[..., None] - rm.FP4_GRID).argmin(axis=-1)]
            cand.append((g, exc, ((g / 4 * 2.0 ** exc[..., None] - x) ** 2).sum(-1)))
        lo = cand[1][2] < cand[0][2]                       # ties keep the larger exponent
        g = np.where(lo[..., None], cand[1][0], cand[0][0])
        ex = np.where(lo, cand[1][1], cand[0][1])
        return np.round(g * 16).astype(np.int64), (ex + rm.EXP_BIAS).astype(np.int64)
    qmax, frac = (127, 64) if bits == 8 else (7, 4)
    with np.errstate(divide="ignore"):
        ex = np.where(amax == 0, 0, np.ceil(np.log2(amax * frac / qmax)))
    ex = np.clip(ex, -127, 128)
    q = np.clip(np.round(x / 2.0 ** ex[..., None] * frac), -qmax - 1, qmax).astype(np.int64)
    if bits == 4:
        q = q * 16
    return q, (ex + rm.EXP_BIAS).astype(np.int64)


def rope_int(m):
    """The on-write rotator on every row (positions = row index 0..T-1)."""
    out = np.empty_like(m)
    cs = np.array([[rm.rope_cs(p, i) for i in range(DH // 2)] for p in range(m.shape[-2])])  # (T, 8, 2)
    c, s = cs[..., 0], cs[..., 1]
    x, y = m[..., 0::2], m[..., 1::2]
    out[..., 0::2] = np.clip((x * c - y * s + 128) >> 8, -128, 127)
    out[..., 1::2] = np.clip((x * s + y * c + 128) >> 8, -128, 127)
    return out


def chip_attn(mode=rm.MODE_EXACT, thr=0, kbits=8, vbits=8, int8_pages=None, stats=None):
    def fn(q, k, v):
        qs = q * (math.log2(math.e) / math.sqrt(DH))
        qi, eq = mxq(qs, 8)
        # the chip rotates INT8 K rows on write; a 4-bit K is rotated by the host
        # before it is quantised (RoPE is not applied to 4-bit K on chip)
        k4rot = kbits != 8
        ki, ek = mxq(rope(torch.from_numpy(k)).numpy() if k4rot else k, kbits)
        vi, ev = mxq(v, vbits)
        if int8_pages is not None:               # per-page mixed precision
            k8, ek8 = mxq(k, 8)
            v8, ev8 = mxq(v, 8)
            k8 = rope_int(k8) if k4rot else k8
            sel = int8_pages(k, ki, ek)          # (..., T) bool: token kept in INT8
            ki = np.where(sel[..., None], k8, ki)
            ek = np.where(sel, ek8, ek)
            vi = np.where(sel[..., None], v8, vi)
            ev = np.where(sel, ev8, ev)
        qi = rope_int(qi)
        ki = ki if k4rot else rope_int(ki)
        o, eo = rv.decode_causal(qi, eq, ki, ek, vi, ev, mode=mode, thr=thr, stats=stats)
        return o / 64.0 * 2.0 ** (eo[..., None] - rm.EXP_BIAS)
    return fn


def sink_and_worst(k, ki, ek):
    """Pages of 4 tokens: INT8 for page 0 (attention sink) and, per head, the
    page with the largest INT4 dequantisation error of K."""
    t = k.shape[-2]
    err = np.abs(ki / 64.0 * 2.0 ** (ek[..., None] - rm.EXP_BIAS) - k).max(axis=-1)
    perr = err.reshape(err.shape[:-1] + (t // 4, 4)).max(axis=-1)
    perr[..., 0] = -1
    worst = perr.argmax(axis=-1)
    page = np.arange(t) // 4
    return (page == 0) | (page == worst[..., None])


def evaluate(n_seq=256):
    torch.set_num_threads(os.cpu_count())
    model = LM()
    model.load_state_dict(torch.load(os.path.join(ROOT, "eval", CKPT)))
    model.eval()
    rng = np.random.default_rng(123)
    x, y = batch("val", n_seq, rng)
    variants = [
        ("FP32 attention", None, {}),
        ("Raijin MXINT8, exact exp", rm.MODE_EXACT, {}),
        ("Raijin MXINT8, MF1 (one shift)", rm.MODE_MF1, {}),
        ("Raijin MXINT8, MF2 (two shifts)", rm.MODE_MF2, {}),
        ("Raijin MXINT8, key skip SKIPTHR=6", rm.MODE_EXACT, dict(thr=6)),
        ("Raijin MXINT8, key skip SKIPTHR=4", rm.MODE_EXACT, dict(thr=4)),
        ("Raijin MXINT8, key skip SKIPTHR=2", rm.MODE_EXACT, dict(thr=2)),
        ("Raijin K INT8, V INT4", rm.MODE_EXACT, dict(vbits=4)),
        ("Raijin K INT4, V INT4", rm.MODE_EXACT, dict(kbits=4, vbits=4)),
        ("Raijin K INT8, V MXFP4", rm.MODE_EXACT, dict(vbits="fp4")),
        ("Raijin K MXFP4, V MXFP4", rm.MODE_EXACT, dict(kbits="fp4", vbits="fp4")),
        ("Raijin K INT4, V MXFP4", rm.MODE_EXACT, dict(kbits=4, vbits="fp4")),
        ("Raijin INT4 + INT8 sink & worst page", rm.MODE_EXACT, dict(kbits=4, vbits=4, int8_pages=sink_and_worst)),
        ("Raijin K INT8, V INT4, SKIPTHR=6", rm.MODE_EXACT, dict(vbits=4, thr=6)),
        ("Raijin operating point: MF2, K8 V4, SKIPTHR=6", rm.MODE_MF2, dict(vbits=4, thr=6)),
        ("Raijin MF2, K8 V-MXFP4, SKIPTHR=6", rm.MODE_MF2, dict(vbits="fp4", thr=6)),
    ]
    res = []
    with torch.no_grad():
        for name, mode, kw in variants:
            stats = {}
            for blk in model.blocks:
                blk.attn_fn = None if mode is None else chip_attn(mode=mode, stats=stats, **kw)
            t0 = time.time()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, len(CHARS)), y.view(-1)).item()
            acc = (logits.argmax(-1) == y).float().mean().item()
            row = dict(variant=name, loss=loss, ppl=math.exp(loss), acc=acc,
                       zero_p=stats.get("zero_p", 0) / max(stats.get("pairs", 1), 1),
                       acc_overflow=stats.get("acc_overflow", 0),
                       l_overflow=stats.get("l_overflow", 0))
            res.append(row)
            print(f"{name:40s} ppl={row['ppl']:.4f} loss={loss:.4f} top1={acc:.4f} "
                  f"zero-weight keys={100 * row['zero_p']:.1f}% ({time.time() - t0:.0f} s)", flush=True)
    base = res[0]["ppl"]
    where = ("evenly spaced windows of the WikiText-2 test set" if DATASET == "wikitext2"
             else "validation sequences of Tiny Shakespeare")
    with open(os.path.join(ROOT, "results", TAG + ".txt"), "w") as f:
        f.write(f"char LM: {NL} layers, d_model {DM}, {NH} heads x {DH}, RoPE, context {T}; "
                f"{n_seq} {where} ({n_seq * T} predicted characters)\n")
        for r in res:
            f.write(f"{r['variant']:46s} ppl={r['ppl']:.4f} ({100 * (r['ppl'] / base - 1):+.2f}%) "
                    f"bpc={r['loss'] / math.log(2):.4f} top1={r['acc']:.4f} "
                    f"zero-weight keys={100 * r['zero_p']:.1f}% l-wraps={r['l_overflow']}\n")
    with open(os.path.join(ROOT, "results", TAG + ".json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    train() if sys.argv[1:] == ["train"] else evaluate()
