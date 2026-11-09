# Retrieval, dev split

Recall@k: share of questions with at least one gold source in the top k distinct references.

Latency is measured separately (README): reranker scores are cached across configurations.

| Configuration | n | Recall@5 | Recall@10 | All gold@10 | MRR@10 | Hebrew R@10 | English R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 66 | 52% | 55% | 48% | 0.44 | 58% | 52% |
| e5-small | 66 | 52% | 58% | 48% | 0.39 | 54% | 60% |
| hybrid:e5-small | 66 | 59% | 67% | 59% | 0.47 | 71% | 64% |
| e5-base | 66 | 47% | 53% | 44% | 0.39 | 50% | 55% |
| hybrid:e5-base | 66 | 59% | 68% | 58% | 0.47 | 71% | 67% |
| minilm | 66 | 36% | 42% | 35% | 0.29 | 29% | 50% |
| hybrid:minilm | 66 | 53% | 59% | 51% | 0.42 | 58% | 60% |
| hybrid:e5-small+rerank | 66 | 65% | 74% | 64% | 0.57 | 79% | 71% |
| hybrid:e5-small+rerank20 | 66 | 67% | 71% | 64% | 0.56 | 75% | 69% |

Best-source score of hybrid:e5-small+rerank: share of questions at or above each threshold

| Threshold | Answerable (kept) | Not in library (passed) | Out of scope (passed) |
|---:|---:|---:|---:|
| 0.01 | 100% | 40% | 24% |
| 0.02 | 100% | 40% | 12% |
| 0.05 | 95% | 20% | 6% |
| 0.1 | 94% | 20% | 0% |
| 0.2 | 91% | 20% | 0% |
| 0.3 | 88% | 0% | 0% |
| 0.5 | 68% | 0% | 0% |

Best-source score of hybrid:e5-small+rerank20: share of questions at or above each threshold

| Threshold | Answerable (kept) | Not in library (passed) | Out of scope (passed) |
|---:|---:|---:|---:|
| 0.01 | 100% | 40% | 24% |
| 0.02 | 97% | 40% | 12% |
| 0.05 | 94% | 20% | 0% |
| 0.1 | 91% | 0% | 0% |
| 0.2 | 88% | 0% | 0% |
| 0.3 | 83% | 0% | 0% |
| 0.5 | 68% | 0% | 0% |
