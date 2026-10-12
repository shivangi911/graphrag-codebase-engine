# Pooled retrieval evaluation

Instances per repo: pallets/flask: 11, psf/requests: 30, pytest-dev/pytest: 40
Total: 81

Mean (95% bootstrap CI):

| method | hit@1 | hit@3 | hit@5 | hit@10 | recall@1 | recall@10 | mrr | symbol_hit@10 |
|---|---|---|---|---|---|---|---|---|
| vector-only | 0.235 (0.15-0.33) | 0.481 (0.37-0.59) | 0.617 (0.51-0.72) | 0.704 (0.60-0.80) | 0.195 (0.12-0.28) | 0.645 (0.55-0.74) | 0.382 (0.30-0.47) | 0.198 (0.11-0.28) |
| graphrag (BFS+RRF) | 0.321 (0.22-0.42) | 0.519 (0.41-0.63) | 0.617 (0.51-0.73) | 0.728 (0.63-0.83) | 0.260 (0.17-0.35) | 0.674 (0.58-0.77) | 0.452 (0.36-0.54) | 0.272 (0.17-0.37) |

Paired difference (graphrag (BFS+RRF) minus vector-only), n=81, 95% bootstrap CI over instances:

| metric | baseline | graphrag | delta | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|
| hit@1 | 0.235 | 0.321 | +0.086 | [-0.025, +0.198] | no |
| hit@3 | 0.481 | 0.519 | +0.037 | [-0.074, +0.148] | no |
| hit@5 | 0.617 | 0.617 | +0.000 | [-0.099, +0.099] | no |
| hit@10 | 0.704 | 0.728 | +0.025 | [-0.062, +0.111] | no |
| recall@1 | 0.195 | 0.260 | +0.065 | [-0.029, +0.162] | no |
| recall@10 | 0.645 | 0.674 | +0.029 | [-0.058, +0.118] | no |
| mrr | 0.382 | 0.452 | +0.070 | [-0.010, +0.150] | no |
| symbol_hit@10 | 0.198 | 0.272 | +0.074 | [-0.012, +0.160] | no |
