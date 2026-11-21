# Accuracy study: Raijin attention in roneneldan/TinyStories-8M

300 evaluation windows of 16 tokens, 64 disjoint calibration windows.

| Variant | Perplexity | vs FP32 | Top-1 agreement with FP32 |
| --- | --- | --- | --- |
| FP32 | 9.951 | +0.00% | 100.00% |
| fake-quant Q only (FP softmax) | 9.953 | +0.03% | 99.60% |
| fake-quant K only (FP softmax) | 9.962 | +0.11% | 98.47% |
| fake-quant V only (FP softmax) | 9.955 | +0.04% | 99.60% |
| fake-quant Q, K, V (FP softmax) | 9.958 | +0.08% | 98.38% |
| chip, exact exp2 | 10.304 | +3.55% | 93.58% |
| chip, MF2 | 10.332 | +3.83% | 92.56% |
| fix: K mean removal, fake-quant Q K V | 9.955 | +0.04% | 99.47% |
| fix: K mean + Q/K balance, fake-quant Q K V | 9.958 | +0.08% | 99.24% |
| fix: K mean + Q/K balance + V balance, fake-quant | 9.956 | +0.06% | 99.24% |
| chip exact + K mean | 9.949 | -0.01% | 98.47% |
| chip exact + K mean + Q/K balance | 9.950 | -0.01% | 98.51% |
| chip exact + K mean + Q/K + V balance | 9.946 | -0.05% | 98.98% |
| chip MF2 + K mean + Q/K + V balance | 9.973 | +0.22% | 97.18% |
| chip MF2 + all fixes, V INT4 | 10.223 | +2.74% | 90.78% |
