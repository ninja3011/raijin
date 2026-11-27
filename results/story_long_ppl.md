# Long-context perplexity: roneneldan/TinyStories-8M with Raijin's KV-cache budget

The 8 longest TinyStories validation stories (746-1011 tokens, one story per sequence, starting at position 0), decoded one token at a time; perplexity over positions 32 and later (6682 predictions). 4 sink tokens; H2O keeps a recent window of half the budget (r24: 24 tokens); with decay the accumulators halve every 16 steps (HHCTL halve). GPT-Neo's local layers keep their 256-token reach under every policy. Chip rows: MF2, SKIPTHR = 6, max-first decode, key mean folded out of K.

| Configuration | Cache | Perplexity | vs FP32 full context | Keys needing a V read |
| --- | --- | --- | --- | --- |
| fp32-full | all tokens | 18.360 | +0.00% | n/a |
| fp32-window32 | 32 tokens, window | 23.073 | +25.67% | n/a |
| chip-int8-window16 | 16 tokens, window | 36.992 | +101.48% | 63.0% |
| chip-fp4-window32 | 32 tokens, window | 57.422 | +212.75% | 55.5% |
| fp32-h2o32 | 32 tokens, h2o + decay | 40.786 | +122.15% | n/a |
| fp32-h2o32-nodecay | 32 tokens, h2o, no decay | 53.200 | +189.76% | n/a |
| fp32-h2o32-r24 | 32 tokens, h2o + decay | 41.848 | +127.93% | n/a |
| chip-fp4-h2o32 | 32 tokens, h2o + decay | 63.397 | +245.30% | 55.4% |
| chip-fp4-h2o32-r24 | 32 tokens, h2o + decay | 66.201 | +260.57% | 50.8% |
