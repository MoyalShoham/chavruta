# Retrieval, dev split

Recall@k: share of questions with at least one gold source in the top k distinct references.

Latency is measured separately (README): reranker scores are cached across configurations.

| Configuration | n | Recall@5 | Recall@10 | All gold@10 | MRR@10 | Hebrew R@10 | English R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bge-m3 | 66 | 65% | 73% | 62% | 0.50 | 75% | 71% |
| hybrid:bge-m3 | 66 | 65% | 73% | 63% | 0.52 | 79% | 69% |
| hybrid:bge-m3+rerank | 66 | 74% | 80% | 70% | 0.60 | 88% | 76% |
| hybrid:bge-m3+rerank20 | 66 | 71% | 79% | 69% | 0.60 | 88% | 74% |

Best-source score of hybrid:bge-m3+rerank: share of questions at or above each threshold

| Threshold | Answerable (kept) | Not in library (passed) | Out of scope (passed) |
|---:|---:|---:|---:|
| 0.01 | 100% | 40% | 41% |
| 0.02 | 100% | 40% | 35% |
| 0.05 | 97% | 40% | 24% |
| 0.1 | 94% | 20% | 6% |
| 0.2 | 92% | 20% | 0% |
| 0.3 | 89% | 0% | 0% |
| 0.5 | 70% | 0% | 0% |

Best-source score of hybrid:bge-m3+rerank20: share of questions at or above each threshold

| Threshold | Answerable (kept) | Not in library (passed) | Out of scope (passed) |
|---:|---:|---:|---:|
| 0.01 | 100% | 40% | 35% |
| 0.02 | 100% | 40% | 29% |
| 0.05 | 97% | 40% | 12% |
| 0.1 | 94% | 20% | 0% |
| 0.2 | 91% | 20% | 0% |
| 0.3 | 88% | 0% | 0% |
| 0.5 | 68% | 0% | 0% |
