# Accuracy against FP32 attention

N = 16, d = 16, 50 random heads per workload. Cells show mean cosine similarity / worst-case max abs error (relative to max |reference|).

| Variant | gauss_soft | gauss | gauss_sharp | llm_outliers |
| --- | --- | --- | --- | --- |
| int8_tensor | 0.99997 / 0.016 | 0.99994 / 0.020 | 0.99985 / 0.053 | 0.99958 / 0.051 |
| mx_float | 0.99996 / 0.014 | 0.99993 / 0.031 | 0.99984 / 0.054 | 0.99994 / 0.068 |
| hw_exact_unfused | 0.99985 / 0.028 | 0.99980 / 0.068 | 0.99976 / 0.052 | 0.99974 / 0.086 |
| hw_exact_fused | 0.99985 / 0.028 | 0.99983 / 0.060 | 0.99978 / 0.050 | 0.99981 / 0.078 |
| hw_mf1_unfused | 0.98142 / 0.365 | 0.98639 / 0.338 | 0.99639 / 0.250 | 0.99049 / 0.359 |
| hw_mf1_fused | 0.98142 / 0.365 | 0.98639 / 0.338 | 0.99639 / 0.250 | 0.99051 / 0.359 |
| hw_mf2_unfused | 0.99811 / 0.129 | 0.99854 / 0.112 | 0.99946 / 0.096 | 0.99869 / 0.156 |
| hw_mf2_fused | 0.99811 / 0.129 | 0.99854 / 0.112 | 0.99947 / 0.096 | 0.99872 / 0.156 |
