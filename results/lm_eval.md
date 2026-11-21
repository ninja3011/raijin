# TinyStories benchmark: Raijin attention in every head of TinyStories-8M

Model `roneneldan/TinyStories-8M` (GPT-Neo, 8 layers x 16 heads, head dimension 16 = Raijin's d).
Every attention head is replaced by the bit-exact Raijin golden model; each causal row
is one KV-cache decode step (the RTL's decode mode). Q/K/V are quantised to MXINT8 per row;
log2(e) is folded into Q (GPT-Neo has no 1/sqrt(d) scaling).

Data: TinyStories validation set, first 2000 non-overlapping 16-token windows (30000 next-token predictions, context <= 16 tokens).

| Attention | Perplexity | Top-1 agreement with FP32 | Eval time (s) |
| --- | --- | --- | --- |
| FP32 (reference) | 8.571 | 100.00% | 35 |
| Raijin exact | 8.942 | 93.18% | 185 |
| Raijin mf1 | 9.146 | 89.62% | 183 |
| Raijin mf2 | 8.961 | 92.52% | 184 |

## Worked example (greedy decoding, 16-token context window)

Prompt: `Once upon a time, there was a little girl named Lily. She`

- FP32 attention: 'Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she'
- Raijin exact attention: 'Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she'

Layer 0, head 0 of the prompt (14 tokens) is saved in `results/lm_head_vectors.npz`
and replayed through the RTL by `tb/test_core.py::test_lm_head` (see `results/lm_head_rtl.txt`).

## RTL check of one real head

`tb/test_core.py::test_lm_head` loads layer 0, head 0 of the prompt (14 tokens, quantised
Q/K/V and exponents from `results/lm_head_vectors.npz`) into the RTL and runs one causal
prefill pass: 97 cycles. All 14 output rows match the golden model bit for bit, and their
mean cosine against the benchmark's decode-step outputs is 0.999993 (`results/lm_head_rtl.txt`).

## Reading the numbers

- Raijin exact costs +4.3% perplexity (8.571 -> 8.942) with a 16-token context; that is the
  MXINT8 quantisation of Q, K and V plus the 16-entry exp LUT and the integer softmax, all at once.
- MF2 (two powers of two, no P.V multipliers) costs only 0.2% more than exact (8.961);
  MF1 (one power of two) costs 2.3% more than exact (9.146).
- Top-1 agreement is the fraction of the 30,000 positions where the model's most likely next
  token is the same as with FP32 attention.
