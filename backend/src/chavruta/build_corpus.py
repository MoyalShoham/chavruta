"""Fetches the whole corpus from Sefaria into data/corpus/ (chunks.jsonl, skipped.jsonl).

python -m chavruta.build_corpus [--only "Berakhot"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chavruta.corpus import CORPUS, fetch_work
from chavruta.net import install_socket_guard
from chavruta.sefaria import SefariaClient

DATA = Path(__file__).resolve().parents[2] / "data"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m chavruta.build_corpus")
    parser.add_argument("--only", action="append", default=None)
    args = parser.parse_args(argv)
    install_socket_guard()
    client = SefariaClient(cache_dir=DATA / "sefaria-cache")
    out = DATA / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    works = [w for w in CORPUS if not args.only or w.title in args.only]
    with (
        (out / "chunks.jsonl").open("w", encoding="utf-8") as cf,
        (out / "skipped.jsonl").open("w", encoding="utf-8") as sf,
    ):
        for w in works:
            chunks, skipped = fetch_work(client, w)
            for c in chunks:
                cf.write(json.dumps(c.as_dict(), ensure_ascii=False) + "\n")
            for s in skipped:
                sf.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")
            print(f"{w.title}: {len(chunks)} chunks, {len(skipped)} skipped", flush=True)


if __name__ == "__main__":
    main()
