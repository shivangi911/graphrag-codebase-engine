# Retrieval evaluation (psf/requests, embedder: sbert-BAAI/bge-small-en-v1.5)

Instances evaluated: 30  (skipped: 0)

| method | recall@1 | recall@3 | recall@5 | recall@10 | hit@1 | hit@3 | hit@5 | hit@10 | mrr | symbol_hit@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| vector-only | 0.267 | 0.502 | 0.634 | 0.753 | 0.333 | 0.633 | 0.767 | 0.867 | 0.503 | 0.333 |
| graphrag (BFS+RRF) | 0.372 | 0.530 | 0.693 | 0.826 | 0.467 | 0.633 | 0.800 | 0.933 | 0.606 | 0.467 |
