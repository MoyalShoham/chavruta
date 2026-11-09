"""Evaluation of the Sefaria lookup, on questions about texts outside the library.

    python -m chavruta.eval_lookup verify                        # the gold places say what is claimed
    python -m chavruta.eval_lookup run --split dev --llm record --sefaria record
    python -m chavruta.eval_lookup judge --split dev --llm record
    python -m chavruta.eval_lookup library --split dev --llm record --sefaria record   # main questions
    python -m chavruta.eval_lookup score --split dev --out eval            # committed files only (CI)

The questions (eval/lookup-questions.jsonl) name a place outside the library ("named"), ask about a
well-known passage without naming it ("implicit"), describe a topic ("topic"), or name a place that
does not exist ("missing", where the right outcome is no answer). Every Sefaria response the run
used is recorded in eval/recordings/sefaria/ and every model response in eval/recordings/llm/, so a
run replays offline. `score` needs neither: it re-checks the stored answers against the recorded
model responses, as eval_answers does, and computes the metrics from the stored rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from chavruta.eval_answers import (
    LLM_RECORDINGS,
    _grounding_metrics,
    _pct,
    _recordings_by_key,
    _sent_sources,
    answer_row,
    check_against_recordings,
    judge_request,
)
from chavruta.evaluation import EVAL, Question, load_questions
from chavruta.llm import LLM, Mode, RecordingLLM
from chavruta.retrieval import matches
from chavruta.text import normalize

QUESTIONS = EVAL / "lookup-questions.jsonl"
ROWS = EVAL / "answers"
SEFARIA_RECORDINGS = EVAL / "recordings" / "sefaria"
TYPES = ("named", "implicit", "topic", "missing")


def questions(split: str | None = None) -> list[Question]:
    return load_questions(QUESTIONS, split=split)


def _sefaria(mode: str) -> Any:
    from chavruta.sefaria import SefariaClient

    return SefariaClient(cache_dir=SEFARIA_RECORDINGS, offline=mode == "replay", min_interval=0.1)


def _covers(refs: list[str], gold: tuple[str, ...]) -> bool:
    """A gold place is found if a source is that place, a part of it, or the section around it."""
    return any(matches(r, g) or matches(g, r) for r in refs for g in gold)


# ---- verify ----------------------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    from chavruta.lookup import SefariaLookup

    lookup = SefariaLookup(_sefaria(args.sefaria), llm=None)  # type: ignore[arg-type]
    problems: list[str] = []
    qs = questions()
    seen: set[str] = set()
    for q in qs:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.type not in TYPES:
            problems.append(f"{q.id}: unknown type {q.type}")
        if (q.type == "missing") != (not q.gold):
            problems.append(f"{q.id}: only 'missing' questions have no gold place")
        for gold, words in q.verify.items():
            part = lookup._fetch(gold, False, False, lambda r: False, lambda w: False, float("inf"))
            text = normalize(" ".join(c.text for c in part.chunks if c.language == "he"))
            if not text:
                problems.append(f"{q.id}: {gold!r} has no openly licensed Hebrew text ({part.errors})")
            for w in words:
                if text and normalize(w) not in text:
                    problems.append(f"{q.id}: {gold!r} does not contain {w!r}")
    groups: dict[str, set[str]] = {}
    for q in qs:
        for g in q.gold:
            groups.setdefault(g, set()).add(q.split)
    problems += [f"gold {g!r} is in both splits" for g, splits in groups.items() if len(splits) > 1]
    for p in problems:
        print(p)
    print(f"{len(qs)} questions, {len(problems)} problems")
    return 1 if problems else 0


# ---- run -------------------------------------------------------------------------------------------


def _bot(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """The library as the main evaluation searched it, with the lookup on: (bot, llm, sefaria)."""
    from chavruta.answer import Chavruta, CorpusIndex, Settings
    from chavruta.embeddings import MODELS, CrossEncoderReranker, SentenceEmbedder, load_vectors
    from chavruta.evaluation import DATA, load_chunks
    from chavruta.llm import AnthropicLLM
    from chavruta.lookup import SefariaLookup
    from chavruta.retrieval import Retriever

    chunks = load_chunks()
    _, vectors = load_vectors(DATA / "vectors" / args.model)
    retriever = Retriever(chunks, SentenceEmbedder(MODELS[args.model]), vectors, CrossEncoderReranker())
    mode = Mode(args.llm)
    llm = RecordingLLM(LLM_RECORDINGS, mode, AnthropicLLM() if mode != Mode.REPLAY else None)
    sefaria = _sefaria(args.sefaria)
    bot = Chavruta(retriever, CorpusIndex(chunks), llm, Settings(), lookup=SefariaLookup(sefaria, llm))
    return bot, llm, sefaria


def cmd_run(args: argparse.Namespace) -> int:
    bot, llm, sefaria = _bot(args)
    rows = []
    for q in questions(args.split):
        before = sefaria.requests
        a = bot.ask(q.question)
        row = answer_row(q, a)
        lookup = a.lookup or {}
        sent = [s.ref for s in a.sources]  # only what reached the answering model
        row.update(
            type=q.type,
            lookup=lookup,
            sefaria_requests=sefaria.requests - before,  # 0 when every response came from the cache
            sources=[
                {
                    "ref": s.ref,
                    "work": s.work,
                    "fetched": s.fetched,
                    "licenses": sorted({t.license for t in s.texts}),
                }
                for s in a.sources
            ],
            gold_fetched=_covers(lookup.get("passages", []), q.gold) if q.gold else None,
            gold_sent=_covers(sent, q.gold) if q.gold else None,
        )
        rows.append(row)
        print(
            f"{q.id} {q.type:8} {a.status.value:9} fetched={lookup.get('fetched', [])[:3]} "
            f"gold_sent={row['gold_sent']}",
            flush=True,
        )
    path = ROWS / f"lookup-{args.split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    print(f"{len(rows)} answers -> {path}; {llm.calls} API calls; {sefaria.requests} Sefaria requests")
    return 0


# ---- the library's own questions, with the lookup on --------------------------------------------------


def cmd_library(args: argparse.Namespace) -> int:
    """Runs the main evaluation's questions with the lookup on, up to the answering model, and
    compares with what was recorded with it off: which questions the lookup changes, and whether
    every other answered question sends a byte-identical request (so its recorded answer stands)."""
    from chavruta.answer import answer_request
    from chavruta.llm import request_key

    bot, llm, sefaria = _bot(args)
    recorded = {r["id"]: r for r in _main_rows(args.split)}
    rows = []
    for q in load_questions(split=args.split):
        _, decision, sources, found = bot.prepare(q.question)
        before = recorded[q.id]
        fetched = [s.ref for s in sources if s.fetched]
        key = request_key(answer_request(q.question, sources, (), "ask")) if decision.answer else None
        rows.append(
            {
                "id": q.id,
                "type": q.type,
                "status_before": before["status"],
                "answers_now": decision.answer,
                "planned": found.record()["refs"] if found else None,
                "fetched_sources": fetched,
                "same_request": key == before["answer_key"] if before["answer_key"] and key else None,
            }
        )
        print(
            f"{q.id} {q.type:13} {before['status']:9} -> {decision.reason:12} fetched={fetched[:3]}",
            flush=True,
        )
    path = ROWS / f"lookup-library-{args.split}.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    print(f"{len(rows)} questions -> {path}; {llm.calls} API calls; {sefaria.requests} Sefaria requests")
    return 0


def _main_rows(split: str) -> list[dict[str, Any]]:
    path = ROWS / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def library_check(split: str) -> dict[str, Any] | None:
    path = ROWS / f"lookup-library-{split}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    answered_before = [r for r in rows if r["status_before"] == "answered"]
    changed = [r for r in answered_before if r["fetched_sources"]]
    unchanged = [r for r in answered_before if not r["fetched_sources"] and r["answers_now"]]
    outside = [r for r in rows if r["type"] == "not_in_corpus"]
    return {
        "n": len(rows),
        "answered_before": len(answered_before),
        "with_fetched_sources": [r["id"] for r in changed],
        "unchanged": len(unchanged),
        "unchanged_with_identical_request": sum(bool(r["same_request"]) for r in unchanged),
        "not_in_library": len(outside),
        "not_in_library_answered_now": [r["id"] for r in outside if r["answers_now"]],
        "declined_before_answered_now": [
            r["id"] for r in rows if r["status_before"] != "answered" and r["answers_now"]
        ],
    }


def _rows(split: str) -> list[dict[str, Any]]:
    path = ROWS / f"lookup-{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---- judge -----------------------------------------------------------------------------------------


def cmd_judge(args: argparse.Namespace) -> int:
    from chavruta.llm import AnthropicLLM

    mode = Mode(args.llm)
    llm: LLM = RecordingLLM(LLM_RECORDINGS, mode, AnthropicLLM() if mode != Mode.REPLAY else None)
    qs = {q.id: q for q in questions(args.split)}
    recordings = _recordings_by_key()
    out = []
    for row in _rows(args.split):
        if row["status"] != "answered":
            continue
        rec = recordings[row["answer_key"]]
        response = llm.create(judge_request(qs[row["id"]].question, row, _sent_sources(rec)))
        text = next(b["text"] for b in response["content"] if b["type"] == "text")
        out.append({"id": row["id"], "verdicts": json.loads(text)["segments"]})
        print(row["id"], Counter(v["verdict"] for v in out[-1]["verdicts"]), flush=True)
    (ROWS / f"lookup-{args.split}-judge.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8", newline="\n"
    )
    return 0


# ---- score -----------------------------------------------------------------------------------------


def score(split: str) -> dict[str, Any]:
    qs = {q.id: q for q in questions(split)}
    rows = _rows(split)
    problems = check_against_recordings(rows, _recordings_by_key())
    judge_path = ROWS / f"lookup-{split}-judge.jsonl"
    judged = (
        {r["id"]: r["verdicts"] for r in map(json.loads, judge_path.read_text(encoding="utf-8").splitlines())}
        if judge_path.exists()
        else {}
    )
    by_type: dict[str, dict[str, Any]] = {}
    for t in TYPES:
        rs = [r for r in rows if r["type"] == t]
        if not rs:
            continue
        m: dict[str, Any] = {"n": len(rs), "answered": sum(r["status"] == "answered" for r in rs)}
        if t != "missing":
            m["gold_fetched"] = sum(bool(r["gold_fetched"]) for r in rs)
            m["gold_sent"] = sum(bool(r["gold_sent"]) for r in rs)
            cited = [
                any(_covers([c[1]], qs[r["id"]].gold) for seg in r["segments"] for c in seg["citations"])
                for r in rs
                if r["status"] == "answered"
            ]
            m["answers_citing_gold"] = [sum(cited), len(cited)]
        by_type[t] = m
    findable = [r for r in rows if r["type"] != "missing"]
    # Timing only where Sefaria was really asked (not served from the cache) in the run that wrote
    # these rows; the planner runs while the library is searched, so it is reported apart.
    live = [r for r in rows if r.get("sefaria_requests")]
    fetch_seconds = sorted(r["lookup"]["fetch_seconds"] for r in live)
    plan_seconds = sorted(r["lookup"]["plan_seconds"] for r in rows if r.get("lookup"))
    licenses = {lic for r in rows for s in r["sources"] for lic in s["licenses"]}
    return {
        "split": split,
        "n": len(rows),
        "integrity_problems": problems,
        "by_type": by_type,
        "findable": {
            "n": len(findable),
            "answered": sum(r["status"] == "answered" for r in findable),
            "gold_fetched": sum(bool(r["gold_fetched"]) for r in findable),
            "gold_sent": sum(bool(r["gold_sent"]) for r in findable),
        },
        "missing_declined": [
            sum(r["status"] != "answered" for r in rows if r["type"] == "missing"),
            sum(r["type"] == "missing" for r in rows),
        ],
        "grounding": _grounding_metrics(rows, qs, judged),
        "licenses_sent": sorted(licenses),
        "sefaria_requests_median": _median([r.get("sefaria_requests", 0) for r in rows if r.get("lookup")]),
        "fetch_seconds": {
            "n": len(fetch_seconds),
            "median": _median(fetch_seconds),
            "max": max(fetch_seconds, default=None),
        },
        "plan_seconds": {"median": _median(plan_seconds), "max": max(plan_seconds, default=None)},
        "library_check": library_check(split),
    }


def _median(values: list[Any]) -> Any:
    values = sorted(values)
    return values[len(values) // 2] if values else None


def _of(k: int, n: int) -> str:
    return f"{k} of {n}" if n else "–"


def cmd_score(args: argparse.Namespace) -> int:
    r = score(args.split)
    g, f = r["grounding"], r["findable"]
    lines = [
        f"# Sefaria lookup, {r['split']} split ({r['n']} questions about texts outside the library)",
        "",
        f"Integrity check: {len(r['integrity_problems'])} problems (stored answers re-parsed from the "
        "recorded responses; every cited text matched against the sources sent).",
        "",
        "| Questions | Gold passage fetched | Gold passage sent to the model | Answered "
        "| Answers citing gold |",
        "|---|---:|---:|---:|---:|",
    ]
    for t, m in r["by_type"].items():
        if t == "missing":
            continue
        n = m["n"]
        cited, answered = m["answers_citing_gold"]
        lines.append(
            f"| {t} ({n}) | {_of(m['gold_fetched'], n)} | {_of(m['gold_sent'], n)} | {_of(m['answered'], n)} "
            f"| {_of(cited, answered)} |"
        )
    n = f["n"]
    lines.append(
        f"| **all ({n})** | {_of(f['gold_fetched'], n)} | {_of(f['gold_sent'], n)} "
        f"| {_of(f['answered'], n)} | |"
    )
    declined, missing = r["missing_declined"]
    fs, ps = r["fetch_seconds"], r["plan_seconds"]
    lines += [
        "",
        f"- Places that do not exist, declined rather than answered: {_of(declined, missing)}",
        f"- Answers naming a source that was not sent: {_pct(g['names_an_unretrieved_source'])}",
        "- Judge: claims fully supported by the sources sent: "
        f"{_pct(g['judge']['supported_share_of_claims'])} ({g['judge']['answers']} answers)",
        f"- Licenses of every source sent: {', '.join(r['licenses_sent'])}",
        f"- Sefaria requests per question: median {r['sefaria_requests_median']}",
        f"- Fetching (questions where Sefaria was really asked, {fs['n']}): median {fs['median']} s, "
        f"max {fs['max']} s. The planner (median {ps['median']} s) runs while the library is searched.",
        "",
    ]
    lib = r["library_check"]
    if lib:
        lines += [
            f"## The library's {r['split']} questions with the lookup on ({lib['n']})",
            "",
            f"- Answered with the lookup off: {lib['answered_before']}. Of these, "
            f"{len(lib['with_fetched_sources'])} now also get a fetched source "
            f"({', '.join(lib['with_fetched_sources']) or 'none'}); "
            f"{lib['unchanged_with_identical_request']} of the other {lib['unchanged']} send a "
            "byte-identical request, so their recorded answers stand.",
            f"- Not in the library ({lib['not_in_library']}): now answered from Sefaria: "
            f"{', '.join(lib['not_in_library_answered_now']) or 'none'}.",
            f"- Declined before, answered now: {', '.join(lib['declined_before_answered_now']) or 'none'}.",
            "",
        ]
    text = "\n".join(lines)
    print(text)
    for p in r["integrity_problems"]:
        print("PROBLEM:", p)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"lookup-{args.split}.md").write_text(text, encoding="utf-8", newline="\n")
        (args.out / f"lookup-{args.split}.json").write_text(
            json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
    if args.check:
        committed = json.loads((args.check / f"lookup-{args.split}.json").read_text(encoding="utf-8"))
        if committed != json.loads(json.dumps(r)):
            print(f"lookup-{args.split}.json does not match the recorded answers")
            return 1
        print(f"lookup-{args.split}.json matches the recorded answers")
    return 1 if r["integrity_problems"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chavruta.eval_lookup")
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--sefaria", choices=("record", "replay"), default="replay")
    for name in ("run", "judge", "score", "library"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=("dev", "holdout"), required=True)
        if name in ("run", "judge", "library"):
            p.add_argument("--llm", choices=[m.value for m in Mode], default="replay")
        if name in ("run", "library"):
            p.add_argument("--sefaria", choices=("record", "replay"), default="replay")
            p.add_argument("--model", default="e5-small")
        if name == "score":
            p.add_argument("--out", type=Path, default=None)
            p.add_argument("--check", type=Path, default=None, help="compare with the results committed here")
    args = parser.parse_args(argv)
    commands = {
        "verify": cmd_verify,
        "run": cmd_run,
        "judge": cmd_judge,
        "score": cmd_score,
        "library": cmd_library,
    }
    return commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
