# Retrieval evaluation (pallets/flask, embedder: sbert-BAAI/bge-small-en-v1.5)

Instances evaluated: 11  (skipped: 0)

| method | recall@1 | recall@3 | recall@5 | recall@10 | hit@1 | hit@3 | hit@5 | hit@10 | mrr | symbol_hit@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| vector-only | 0.212 | 0.439 | 0.652 | 0.788 | 0.273 | 0.545 | 0.727 | 0.818 | 0.435 | 0.273 |
| graphrag (BFS+RRF) | 0.356 | 0.523 | 0.553 | 0.848 | 0.545 | 0.636 | 0.636 | 0.909 | 0.630 | 0.455 |
