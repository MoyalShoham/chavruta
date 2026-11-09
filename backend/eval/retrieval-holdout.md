# Retrieval, holdout split

Recall@k: share of questions with at least one gold source in the top k distinct references.

Latency is measured separately (README): reranker scores are cached across configurations.

| Configuration | n | Recall@5 | Recall@10 | All gold@10 | MRR@10 | Hebrew R@10 | English R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 97 | 53% | 55% | 45% | 0.38 | 58% | 52% |
| e5-small | 97 | 52% | 58% | 42% | 0.41 | 61% | 56% |
| hybrid:e5-small | 97 | 53% | 70% | 56% | 0.41 | 72% | 69% |
| e5-base | 97 | 56% | 68% | 52% | 0.44 | 67% | 69% |
| hybrid:e5-base | 97 | 56% | 66% | 54% | 0.42 | 67% | 66% |
| minilm | 97 | 33% | 39% | 28% | 0.24 | 22% | 49% |
| hybrid:minilm | 97 | 42% | 58% | 46% | 0.33 | 56% | 59% |
| hybrid:e5-small+rerank | 97 | 71% | 79% | 64% | 0.55 | 86% | 75% |
| hybrid:e5-small+rerank20 | 97 | 69% | 75% | 61% | 0.55 | 86% | 69% |

Best-source score of hybrid:e5-small+rerank: share of questions at or above each threshold

| Threshold | Answerable (kept) | Not in library (passed) | Out of scope (passed) |
|---:|---:|---:|---:|
| 0.01 | 98% | 50% | 30% |
| 0.02 | 96% | 50% | 9% |
| 0.05 | 94% | 0% | 9% |
| 0.1 | 91% | 0% | 9% |
| 0.2 | 88% | 0% | 0% |
| 0.3 | 80% | 0% | 0% |
| 0.5 | 74% | 0% | 0% |

Best-source score of hybrid:e5-small+rerank20: share of questions at or above each threshold

| Threshold | Answerable (kept) | Not in library (passed) | Out of scope (passed) |
|---:|---:|---:|---:|
| 0.01 | 97% | 50% | 22% |
| 0.02 | 95% | 50% | 4% |
| 0.05 | 94% | 0% | 4% |
| 0.1 | 91% | 0% | 4% |
| 0.2 | 86% | 0% | 0% |
| 0.3 | 78% | 0% | 0% |
| 0.5 | 72% | 0% | 0% |
