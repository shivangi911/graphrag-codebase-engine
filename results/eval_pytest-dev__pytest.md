# Retrieval evaluation (pytest-dev/pytest, embedder: sbert-BAAI/bge-small-en-v1.5)

Instances evaluated: 10  (skipped: 0)

| method | recall@1 | recall@3 | recall@5 | recall@10 | hit@1 | hit@3 | hit@5 | hit@10 | mrr | symbol_hit@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| vector-only | 0.200 | 0.600 | 0.600 | 0.600 | 0.200 | 0.600 | 0.600 | 0.600 | 0.374 | 0.100 |
| graphrag (BFS+RRF) | 0.300 | 0.600 | 0.700 | 0.700 | 0.300 | 0.600 | 0.700 | 0.700 | 0.475 | 0.100 |
