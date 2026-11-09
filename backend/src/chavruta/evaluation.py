"""Evaluation: the question set, retrieval metrics, and (later) answers.

python -m chavruta.evaluation verify                  # every gold reference exists and says what we claim
python -m chavruta.evaluation embed --model e5-small  # vectors for the whole corpus
python -m chavruta.evaluation retrieval --split dev   # compare configurations, record rankings
python -m chavruta.evaluation retrieval --split dev --replay   # metrics from the recordings only

Questions live in eval/questions.jsonl. Dev is for every choice (model, fusion, reranker, prompt);
holdout is run once with the chosen configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chavruta.corpus import Chunk
from chavruta.retrieval import matches
from chavruta.text import normalize

BACKEND = Path(__file__).resolve().parents[2]
DATA = BACKEND / "data"
EVAL = BACKEND / "eval"
QUESTIONS = EVAL / "questions.jsonl"
RECORDINGS = EVAL / "recordings"
KS = (5, 10)


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    lang: str
    type: str
    question: str
    gold: tuple[str, ...]
    verify: dict[str, list[str]] = field(default_factory=dict)
    #: For a machloket: groups of name variants; a good answer names one from every group.
    expect_sides: tuple[tuple[str, ...], ...] = ()

    @property
    def in_scope(self) -> bool:
        return self.type != "out_of_scope"


def load_questions(path: Path = QUESTIONS, split: str | None = None) -> list[Question]:
    out: list[Question] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sides = tuple(tuple(g) for g in r.get("expect_sides", []))
        q = Question(
            r["id"], r["split"], r["lang"], r["type"], r["question"], tuple(r["gold"]), r["verify"], sides
        )
        if split is None or q.split == split:
            out.append(q)
    return out


def load_chunks(path: Path = DATA / "corpus" / "chunks.jsonl") -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


# ---- verifying the question set -------------------------------------------------------------------


def verify_questions(questions: Sequence[Question], chunks: Sequence[Chunk]) -> list[str]:
    """Problems with the question set: gold references missing from the corpus, or keywords
    claimed for a gold reference that its text does not contain."""
    refs = sorted({c.ref for c in chunks})
    hebrew: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        if c.language == "he":
            hebrew[c.ref].append(normalize(c.text))
    problems: list[str] = []
    ids: set[str] = set()
    for q in questions:
        if q.id in ids:
            problems.append(f"{q.id}: duplicate id")
        ids.add(q.id)
        if q.split not in ("dev", "holdout"):
            problems.append(f"{q.id}: unknown split {q.split}")
        for gold in q.gold:
            if not any(matches(r, gold) for r in _prefixed(refs, gold)):
                problems.append(f"{q.id}: gold {gold!r} is not in the corpus")
        for gold, words in q.verify.items():
            if gold not in q.gold:
                problems.append(f"{q.id}: verified ref {gold!r} is not a gold ref")
            text = " ".join(t for r in _prefixed(refs, gold) if matches(r, gold) for t in hebrew[r])
            for w in words:
                if normalize(w) not in text:
                    problems.append(f"{q.id}: {gold!r} does not contain {w!r}")
    return problems


def _prefixed(sorted_refs: list[str], gold: str) -> Iterable[str]:
    import bisect

    i = bisect.bisect_left(sorted_refs, gold)
    while i < len(sorted_refs) and sorted_refs[i].startswith(gold):
        yield sorted_refs[i]
        i += 1


# ---- retrieval metrics ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Ranking:
    question_id: str
    refs: tuple[str, ...]
    seconds: float
    scores: tuple[float, ...] = ()


def score_rankings(
    questions: Sequence[Question], rankings: dict[str, Ranking], corpus_refs: set[str] | None = None
) -> dict[str, Any]:
    """Recall@k = share of questions with at least one gold source in the top k (several gold
    references are alternatives: the Mishnah or the Gemara on it both answer the question).
    Also reported: the mean share of all gold references found, and MRR@10."""
    rows = [q for q in questions if q.in_scope and q.gold]
    by: dict[str, list[dict[str, float]]] = defaultdict(list)
    for q in rows:
        r = rankings[q.id]
        gold = [g for g in q.gold if corpus_refs is None or any(matches(c, g) for c in corpus_refs)]
        first = next((i for i, ref in enumerate(r.refs) if any(matches(ref, g) for g in gold)), None)
        m: dict[str, float] = {"mrr@10": 0.0 if first is None or first >= 10 else 1.0 / (first + 1)}
        for k in KS:
            top = r.refs[:k]
            m[f"recall@{k}"] = float(first is not None and first < k)
            m[f"all_gold@{k}"] = sum(any(matches(t, g) for t in top) for g in gold) / len(gold)
        m["seconds"] = r.seconds
        for key in ("all", f"lang:{q.lang}", f"type:{q.type}"):
            by[key].append(m)
    return {
        key: {"n": len(ms), **{name: _mean(m[name] for m in ms) for name in ms[0]}}
        for key, ms in sorted(by.items())
    }


def _mean(xs: Iterable[float]) -> float:
    values = list(xs)
    return sum(values) / len(values) if values else 0.0


# ---- commands -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    name: str
    mode: str  # keyword | semantic | hybrid
    model: str | None
    rerank: bool
    rerank_top: int = 50


def configs(models: Sequence[str], rerank_models: Sequence[str] = ()) -> list[Config]:
    """BM25 alone; each model alone and fused with BM25; reranking only for the models named in
    `rerank_models`. The reranker is the expensive part (a 568M-parameter cross-encoder), so it is
    evaluated for the chosen model, at two depths."""
    out = [Config("bm25", "keyword", None, False)]
    for m in models:
        out += [Config(f"{m}", "semantic", m, False), Config(f"hybrid:{m}", "hybrid", m, False)]
    for m in rerank_models:
        out += [
            Config(f"hybrid:{m}+rerank", "hybrid", m, True),
            Config(f"hybrid:{m}+rerank20", "hybrid", m, True, 20),
        ]
    return out


def cmd_verify(_: argparse.Namespace) -> int:
    questions = load_questions()
    problems = verify_questions(questions, load_chunks())
    for p in problems:
        print(p)
    counts: dict[str, int] = defaultdict(int)
    for q in questions:
        counts[f"{q.split}/{q.type}/{q.lang}"] += 1
    print(json.dumps(counts, indent=1, sort_keys=True))
    print(f"{len(questions)} questions, {len(problems)} problems")
    return 1 if problems else 0


def cmd_embed(args: argparse.Namespace) -> int:
    from chavruta.embeddings import MODELS, SentenceEmbedder, save_vectors

    chunks = load_chunks()
    embedder = SentenceEmbedder(MODELS[args.model])
    embedder.model.max_seq_length = 512
    start = time.perf_counter()
    vectors = embedder.passages([c.text for c in chunks])
    elapsed = time.perf_counter() - start
    save_vectors(DATA / "vectors" / args.model, [_chunk_id(c) for c in chunks], vectors)
    print(f"{args.model}: {len(chunks)} chunks in {elapsed:.0f}s ({len(chunks) / elapsed:.0f}/s)")
    return 0


def cmd_latency(args: argparse.Namespace) -> int:
    """Per-question retrieval time of the chosen configuration, with a fresh (uncached) reranker,
    on the device given: what a person waits for before the model starts answering."""
    import statistics

    from chavruta.embeddings import CHOSEN_MODEL, MODELS, CrossEncoderReranker, SentenceEmbedder, load_vectors
    from chavruta.retrieval import Retriever

    chunks = load_chunks()
    _, vectors = load_vectors(DATA / "vectors" / CHOSEN_MODEL)
    embedder = SentenceEmbedder(MODELS[CHOSEN_MODEL], device=args.device)
    retriever = Retriever(chunks, embedder, vectors, CrossEncoderReranker(device=args.device))
    questions = [q for q in load_questions(split="dev") if q.gold][: args.n]
    retriever.search(questions[0].question, k=10)  # warm-up
    for depth in args.depths:
        retriever.rerank_top = depth
        times = []
        for q in questions:
            start = time.perf_counter()
            retriever.search(q.question, k=10)
            times.append(time.perf_counter() - start)
        print(
            f"{args.device} depth {depth}: median {statistics.median(times):.2f}s, "
            f"max {max(times):.2f}s, n={len(times)}"
        )
    return 0


def cmd_export_vectors(args: argparse.Namespace) -> int:
    """Ships the chosen model's vectors with the corpus snapshot (float16, compressed)."""
    import numpy as np

    from chavruta.embeddings import CHOSEN_MODEL, load_vectors

    ids, vectors = load_vectors(DATA / "vectors" / CHOSEN_MODEL)
    out = BACKEND / "corpus" / f"vectors-{CHOSEN_MODEL}.npz"
    np.savez_compressed(out, ids=np.array(ids), vectors=vectors.astype(np.float16))
    print(f"{len(ids)} vectors -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def _chunk_id(c: Chunk) -> str:
    return f"{c.ref}|{c.language}|{c.version}"


def cmd_retrieval(args: argparse.Namespace) -> int:
    # Every question is retrieved (out-of-scope ones too: their best scores set the relevance
    # threshold); only questions with gold references are scored for recall.
    questions = load_questions(split=args.split)
    out_dir = RECORDINGS / "retrieval" / args.split
    wanted = configs(args.models, args.rerank_models)
    if args.only:
        wanted = [c for c in wanted if c.name in args.only]
    results: dict[str, Any] = {}
    if not args.replay:
        _record(questions, wanted, out_dir)
    for c in wanted:
        rankings = _load_rankings(out_dir / f"{_slug(c.name)}.json")
        results[c.name] = score_rankings(questions, rankings)
        if c.rerank:
            results[c.name]["relevance"] = relevance_profile(questions, rankings)
    _write_results(args.split, results, args.out)
    if args.check:
        committed = json.loads((args.check / f"retrieval-{args.split}.json").read_text(encoding="utf-8"))
        if _without_timing(committed) != _without_timing(json.loads(json.dumps(results))):
            print(f"retrieval-{args.split}.json does not match the recorded rankings")
            return 1
        print(f"retrieval-{args.split}.json matches the recorded rankings")
    return 0


def _without_timing(x: Any) -> Any:
    """Latency is measured, not replayed; everything else must match exactly."""
    if isinstance(x, dict):
        return {k: _without_timing(v) for k, v in x.items() if k != "seconds"}
    return x


def _record(questions: Sequence[Question], wanted: Sequence[Config], out_dir: Path) -> None:
    from chavruta.embeddings import MODELS, CrossEncoderReranker, SentenceEmbedder, load_vectors
    from chavruta.retrieval import Retriever

    chunks = load_chunks()
    reranker = MemoReranker(CrossEncoderReranker()) if any(c.rerank for c in wanted) else None
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in dict.fromkeys(c.model for c in wanted):
        embedder = vectors = None
        if model is not None:
            ids, vectors = load_vectors(DATA / "vectors" / model)
            if ids != [_chunk_id(c) for c in chunks]:
                raise SystemExit(f"vectors for {model} are stale; run embed --model {model}")
            embedder = SentenceEmbedder(MODELS[model])
        retriever = Retriever(chunks, embedder, vectors, reranker)
        for c in (c for c in wanted if c.model == model):
            rankings = []
            retriever.rerank_top = c.rerank_top
            for q in questions:
                start = time.perf_counter()
                hits = retriever.search(q.question, k=10, mode=c.mode, rerank=c.rerank)
                rankings.append(
                    {
                        "id": q.id,
                        "refs": [h.chunk.ref for h in hits],
                        "scores": [round(h.score, 6) for h in hits],
                        "seconds": time.perf_counter() - start,
                    }
                )
            path = out_dir / f"{_slug(c.name)}.json"
            path.write_text(
                json.dumps({"config": c.__dict__, "rankings": rankings}, ensure_ascii=False, indent=1),
                encoding="utf-8",
                newline="\n",
            )
            print(f"recorded {c.name}", flush=True)


class MemoReranker:
    """Configurations share most of their candidates; each (question, passage) is scored once."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.cache: dict[tuple[str, str], float] = {}

    def scores(self, query: str, passages: list[str]) -> list[float]:
        missing = list(dict.fromkeys(p for p in passages if (query, p) not in self.cache))
        for p, score in zip(missing, self.inner.scores(query, missing), strict=True):
            self.cache[(query, p)] = score
        return [self.cache[(query, p)] for p in passages]


def _load_rankings(path: Path) -> dict[str, Ranking]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        r["id"]: Ranking(r["id"], tuple(r["refs"]), r["seconds"], tuple(r.get("scores", ())))
        for r in data["rankings"]
    }


THRESHOLDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def relevance_profile(questions: Sequence[Question], rankings: dict[str, Ranking]) -> dict[str, Any]:
    """How the best reranker score separates answerable questions from the rest, per threshold:
    the share of answerable questions kept, and of out-of-scope / not-in-library ones stopped."""
    groups: dict[str, list[float]] = defaultdict(list)
    for q in questions:
        r = rankings.get(q.id)
        if r is None or not r.scores:
            continue
        group = "answerable" if q.gold else q.type
        groups[group].append(r.scores[0])
    return {
        str(t): {
            g: {"n": len(v), "above": sum(x >= t for x in v) / len(v)} for g, v in sorted(groups.items())
        }
        for t in THRESHOLDS
    }


def _slug(name: str) -> str:
    return name.replace(":", "_").replace("+", "_")


def _write_results(split: str, results: dict[str, Any], out: Path | None) -> None:
    lines = [
        f"# Retrieval, {split} split",
        "",
        "Recall@k: share of questions with at least one gold source in the top k distinct references.",
        "",
        "Latency is measured separately (README): reranker scores are cached across configurations.",
        "",
        "| Configuration | n | Recall@5 | Recall@10 | All gold@10 | MRR@10 | Hebrew R@10 | English R@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        a, he, en = r["all"], r.get("lang:he", {}), r.get("lang:en", {})
        lines.append(
            f"| {name} | {a['n']} | {a['recall@5']:.0%} | {a['recall@10']:.0%} | {a['all_gold@10']:.0%} "
            f"| {a['mrr@10']:.2f} | {he.get('recall@10', 0):.0%} | {en.get('recall@10', 0):.0%} |"
        )
    for name, r in results.items():
        if "relevance" not in r:
            continue
        lines += [
            "",
            f"Best-source score of {name}: share of questions at or above each threshold",
            "",
            "| Threshold | Answerable (kept) | Not in library (passed) | Out of scope (passed) |",
            "|---:|---:|---:|---:|",
        ]
        for t, g in r["relevance"].items():
            row = [g.get(k, {}).get("above", 0.0) for k in ("answerable", "not_in_corpus", "out_of_scope")]
            lines.append(f"| {t} | {row[0]:.0%} | {row[1]:.0%} | {row[2]:.0%} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"retrieval-{split}.md").write_text(text, encoding="utf-8", newline="\n")
        (out / f"retrieval-{split}.json").write_text(
            json.dumps(results, indent=1), encoding="utf-8", newline="\n"
        )


def main(argv: list[str] | None = None) -> int:
    from chavruta.embeddings import CHOSEN_MODEL, MODELS

    parser = argparse.ArgumentParser(prog="python -m chavruta.evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("export-vectors").set_defaults(fn=cmd_export_vectors)
    lat = sub.add_parser("latency")
    lat.add_argument("--device", default="cpu")
    lat.add_argument("--n", type=int, default=10)
    lat.add_argument("--depths", type=int, nargs="+", default=[20, 50])
    lat.set_defaults(fn=cmd_latency)
    e = sub.add_parser("embed")
    e.add_argument("--model", choices=sorted(MODELS), required=True)
    e.set_defaults(fn=cmd_embed)
    r = sub.add_parser("retrieval")
    r.add_argument("--split", choices=("dev", "holdout"), required=True)
    r.add_argument("--models", nargs="+", default=["e5-small", "e5-base", "minilm"])
    r.add_argument("--rerank-models", nargs="*", default=[CHOSEN_MODEL])
    r.add_argument("--only", nargs="*", help="configuration names to run")
    r.add_argument("--replay", action="store_true", help="score the recorded rankings only")
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--check", type=Path, default=None, help="compare with the results committed here")
    r.set_defaults(fn=cmd_retrieval)
    args = parser.parse_args(argv)
    from chavruta.net import install_socket_guard

    install_socket_guard()
    code: int = args.fn(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
