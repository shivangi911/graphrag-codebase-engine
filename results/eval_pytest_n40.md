# Retrieval evaluation (pytest-dev/pytest, embedder: sbert-BAAI/bge-small-en-v1.5)

Instances evaluated: 40  (skipped: 0)

| method | recall@1 | recall@3 | recall@5 | recall@10 | hit@1 | hit@3 | hit@5 | hit@10 | mrr | symbol_hit@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| vector-only | 0.138 | 0.338 | 0.463 | 0.525 | 0.150 | 0.350 | 0.475 | 0.550 | 0.277 | 0.075 |
| graphrag (BFS+RRF) | 0.150 | 0.388 | 0.463 | 0.512 | 0.150 | 0.400 | 0.475 | 0.525 | 0.288 | 0.075 |

Paired difference (graphrag (BFS+RRF) minus vector-only), n=40, 95% bootstrap CI over instances:

| metric | baseline | graphrag | delta | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|
| hit@1 | 0.150 | 0.150 | +0.000 | [-0.150, +0.125] | no |
| hit@3 | 0.350 | 0.400 | +0.050 | [-0.075, +0.175] | no |
| hit@5 | 0.475 | 0.475 | +0.000 | [-0.100, +0.100] | no |
| hit@10 | 0.550 | 0.525 | -0.025 | [-0.150, +0.100] | no |
| recall@1 | 0.138 | 0.150 | +0.013 | [-0.113, +0.150] | no |
| recall@10 | 0.525 | 0.512 | -0.013 | [-0.138, +0.113] | no |
| mrr | 0.277 | 0.288 | +0.011 | [-0.082, +0.102] | no |
| symbol_hit@10 | 0.075 | 0.075 | +0.000 | [-0.075, +0.075] | no |
