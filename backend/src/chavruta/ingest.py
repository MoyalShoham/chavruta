"""Loads the corpus into PostgreSQL: works, every version considered (with its license and whether it
was kept), chunks, and embeddings.

    python -m chavruta.ingest --preset demo      # a smaller library for a quick first run
    python -m chavruta.ingest --preset full      # everything in corpus.CORPUS
    python -m chavruta.ingest --from-file data/corpus/chunks.jsonl   # no fetching

Idempotent: works, versions and chunks are upserted by natural key; only chunks without an
embedding are embedded.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from chavruta.corpus import CORPUS, Chunk, Skipped, WorkSpec, fetch_work
from chavruta.db import ChunkRow, Version, Work, make_engine, make_sessions, now
from chavruta.embeddings import CHOSEN_MODEL, MODELS, Embedder
from chavruta.sefaria import SefariaClient, SefariaError, is_open

# The demo library: the first chapter of Berakhot (2a-13a) with its Mishnah, Rashi and Tosafot,
# Deuteronomy, and the laws of Shema; enough for every feature, quick to fetch and embed.
DEMO_LIMITS = {
    "Deuteronomy": None,
    "Mishnah Berakhot": "Mishnah Berakhot 2",
    "Berakhot": "Berakhot 13a",
    "Rashi on Berakhot": "Rashi on Berakhot 13a",
    "Tosafot on Berakhot": "Tosafot on Berakhot 13a",
    "Mishneh Torah, Reading the Shema": None,
    "Shulchan Arukh, Orach Chayim": "Shulchan Arukh, Orach Chayim 89",
}


def preset(name: str) -> list[WorkSpec]:
    if name == "full":
        return list(CORPUS)
    if name == "demo":
        return [
            replace(w, last_ref=DEMO_LIMITS[w.title] or w.last_ref) for w in CORPUS if w.title in DEMO_LIMITS
        ]
    raise ValueError(f"unknown preset {name}")


def upsert_work(s: Session, spec: WorkSpec) -> Work:
    work = s.scalars(select(Work).where(Work.title == spec.title)).first()
    if work is None:
        work = Work(
            title=spec.title, he_title=spec.he_title, category=spec.category, base_title=spec.base_title
        )
        s.add(work)
        s.flush()
    return work


def upsert_version(
    s: Session, work: Work, language: str, title: str, license_name: str, included: bool, reason: str | None
) -> Version:
    v = s.scalars(
        select(Version).where(
            Version.work_id == work.id, Version.language == language, Version.title == title
        )
    ).first()
    if v is None:
        v = Version(work_id=work.id, language=language, title=title)
        s.add(v)
    v.license, v.included, v.reason = license_name, included, reason
    s.flush()
    return v


def survey_versions(s: Session, client: SefariaClient, spec: WorkSpec) -> None:
    """Records every version Sefaria has for the work, so the dashboard shows what was left out
    and why (license, or not chosen)."""
    wanted = {(lang[:2], title) for lang, title in spec.versions}
    work = upsert_work(s, spec)
    for v in client.versions(spec.title):
        language = str(v.get("language", ""))[:2]
        title = str(v.get("versionTitle", ""))
        lic = str(v.get("license") or "unknown")
        if not is_open(lic):
            upsert_version(s, work, language, title, lic, False, f"license {lic} is not open")
        elif (language, title) not in wanted:
            upsert_version(s, work, language, title, lic, False, "open, not selected")
        else:
            upsert_version(s, work, language, title, lic, True, None)


def store_chunks(s: Session, spec: WorkSpec, chunks: Sequence[Chunk], skipped: Iterable[Skipped]) -> int:
    work = upsert_work(s, spec)
    versions: dict[tuple[str, str], Version] = {}
    for c in chunks:
        key = (c.language, c.version)
        if key not in versions:
            versions[key] = upsert_version(s, work, c.language, c.version, c.license, True, None)
    for sk in {(k.language, k.version): k for k in skipped if k.reason.startswith("license")}.values():
        upsert_version(s, work, sk.language, sk.version, sk.license, False, sk.reason)
    rows = [
        {
            "work_id": work.id,
            "version_id": versions[(c.language, c.version)].id,
            "ref": c.ref,
            "section_ref": c.section_ref,
            "language": c.language,
            "position": c.position,
            "text": c.text,
        }
        for c in chunks
    ]
    inserted = 0
    for i in range(0, len(rows), 1000):
        stmt = (
            pg_insert(ChunkRow)
            .values(rows[i : i + 1000])
            .on_conflict_do_nothing(index_elements=["version_id", "ref"])
        )
        result: Any = s.execute(stmt)
        inserted += result.rowcount or 0
    work.synced_at = now()
    return inserted


def embed_missing(s: Session, embedder: Embedder, batch: int = 256) -> int:
    done = 0
    while True:
        rows = list(
            s.scalars(select(ChunkRow).where(ChunkRow.embedding.is_(None)).order_by(ChunkRow.id).limit(batch))
        )
        if not rows:
            return done
        vectors = embedder.passages([r.text for r in rows])
        for r, v in zip(rows, vectors, strict=True):
            r.embedding = v.tolist()
            r.embedding_model = embedder.key
        s.commit()
        done += len(rows)
        print(f"embedded {done}", flush=True)


def until(chunks: Sequence[Chunk], last_section: str | None) -> list[Chunk]:
    """The chunks of a work up to and including a section (positions are in reading order)."""
    order = list(dict.fromkeys(c.section_ref for c in sorted(chunks, key=lambda c: c.position)))
    if last_section is None or last_section not in order:
        return list(chunks)
    keep = set(order[: order.index(last_section) + 1])
    return [c for c in chunks if c.section_ref in keep]


def load_file(path: Path) -> dict[str, list[Chunk]]:
    by_work: dict[str, list[Chunk]] = {}
    opener: Any = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            c = Chunk(**json.loads(line))
            by_work.setdefault(c.work, []).append(c)
    return by_work


def main(argv: list[str] | None = None) -> None:
    from chavruta.config import Settings
    from chavruta.net import install_socket_guard

    parser = argparse.ArgumentParser(prog="python -m chavruta.ingest")
    parser.add_argument(
        "--preset", choices=("demo", "full"), default=os.environ.get("CHAVRUTA_CORPUS", "demo")
    )
    parser.add_argument("--from-file", type=Path, default=None)
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings()
    install_socket_guard(extra_hosts=[settings.db_host])
    sessions = make_sessions(make_engine(settings.dsn()))
    specs = preset(args.preset)
    client = SefariaClient(cache_dir=settings.data_dir / "sefaria-cache")
    with sessions() as s:
        local = load_file(args.from_file) if args.from_file else None
        for spec in specs:
            try:
                survey_versions(s, client, spec)  # every version Sefaria has, with its license
            except (SefariaError, OSError) as e:
                print(f"{spec.title}: version survey unavailable ({e}); continuing", flush=True)
            if local is not None:
                chunks = until(local.get(spec.title, []), spec.last_ref)
                skipped: list[Skipped] = []
            else:
                chunks, skipped = fetch_work(client, spec)
            n = store_chunks(s, spec, chunks, skipped)
            s.commit()
            print(f"{spec.title}: {len(chunks)} chunks ({n} new)", flush=True)
        total = s.scalar(select(func.count()).select_from(ChunkRow))
        print(f"{total} chunks in the library", flush=True)
        if not args.no_embed:
            from chavruta.embeddings import SentenceEmbedder

            embed_missing(s, SentenceEmbedder(MODELS[CHOSEN_MODEL]))


if __name__ == "__main__":
    main()
