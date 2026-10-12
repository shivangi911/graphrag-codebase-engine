# Retrieval evaluation (pallets/flask, embedder: sbert-BAAI/bge-small-en-v1.5)

Instances evaluated: 11  (skipped: 0)

| method | recall@1 | recall@3 | recall@5 | recall@10 | hit@1 | hit@3 | hit@5 | hit@10 | mrr | symbol_hit@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| vector-only | 0.212 | 0.439 | 0.652 | 0.788 | 0.273 | 0.545 | 0.727 | 0.818 | 0.435 | 0.273 |
| graphrag (BFS+RRF) | 0.356 | 0.523 | 0.553 | 0.848 | 0.545 | 0.636 | 0.636 | 0.909 | 0.630 | 0.455 |

Paired difference (graphrag (BFS+RRF) minus vector-only), n=11, 95% bootstrap CI over instances:

| metric | baseline | graphrag | delta | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|
| hit@1 | 0.273 | 0.545 | +0.273 | [-0.091, +0.636] | no |
| hit@3 | 0.545 | 0.636 | +0.091 | [-0.273, +0.455] | no |
| hit@5 | 0.727 | 0.636 | -0.091 | [-0.455, +0.273] | no |
| hit@10 | 0.818 | 0.909 | +0.091 | [-0.182, +0.364] | no |
| recall@1 | 0.212 | 0.356 | +0.144 | [-0.159, +0.455] | no |
| recall@10 | 0.788 | 0.848 | +0.061 | [-0.212, +0.333] | no |
| mrr | 0.435 | 0.630 | +0.196 | [-0.105, +0.482] | no |
| symbol_hit@10 | 0.273 | 0.455 | +0.182 | [-0.182, +0.545] | no |
