# Long-context perplexity: roneneldan/TinyStories-8M with Raijin's KV-cache budget

2 sequences of 1024 tokens of TinyStories validation text, decoded one token at a time; perplexity over positions 32 and later. 4 sink tokens; H2O keeps a recent window of half the budget; with decay the accumulators halve every 16 steps (HHCTL halve). GPT-Neo's local layers keep their 256-token reach under every policy. Chip rows: MF2, SKIPTHR = 6, max-first decode, key mean folded out of K.

| Configuration | Cache | Perplexity | vs FP32 full context | Keys needing a V read |
| --- | --- | --- | --- | --- |
| fp32-full | all tokens | 25.093 | +0.00% | n/a |
| fp32-window32 | 32 tokens, window | 14.446 | -42.43% | n/a |
| chip-int8-window16 | 16 tokens, window | 29.831 | +18.88% | 61.1% |
| chip-fp4-window32 | 32 tokens, window | 39.132 | +55.95% | 51.1% |
| fp32-h2o32 | 32 tokens, h2o + decay | 60.742 | +142.07% | n/a |
| fp32-h2o32-nodecay | 32 tokens, h2o, no decay | 71.004 | +182.96% | n/a |
| chip-fp4-h2o32 | 32 tokens, h2o + decay | 103.271 | +311.55% | 46.0% |
