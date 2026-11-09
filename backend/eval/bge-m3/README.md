# bge-m3 on dev: added after the holdout run; not used for the choice

`BAAI/bge-m3` could not be embedded on the 8 GB development machine, so the embedding model
(e5-small) and the reranking depth were chosen without it, and holdout was then run once. These
results were recorded later, on a GPU, on the **dev split only**. Holdout was not touched.

They are kept apart from `../retrieval-dev.json` so the committed comparison stays exactly what the
choice was made on. CI replays them from `../recordings/retrieval/dev/*bge-m3*.json`:

```bash
uv run python -m chavruta.evaluation retrieval --split dev --models bge-m3 --rerank-models bge-m3 \
  --only bge-m3 hybrid:bge-m3 hybrid:bge-m3+rerank hybrid:bge-m3+rerank20 --replay --check eval/bge-m3
```

To record them again you need the model weights and the bge-m3 vectors: `fetch_models bge-m3`,
then `evaluation embed --model bge-m3`, then the command above without `--replay` and with
`--out eval/bge-m3`. The bge-m3 vectors are not shipped.
