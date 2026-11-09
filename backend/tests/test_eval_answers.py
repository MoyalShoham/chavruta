"""The answer evaluation, end to end with a scripted model: run, check against recordings, judge,
score and sample. Also: tampering with a stored answer or a cited text is detected."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chavruta import eval_answers as ev
from chavruta.evaluation import load_questions
from chavruta.llm import Mode, RecordingLLM
from tests.test_answer import ScriptedLLM, bot, cite, classification


class EvalLLM(ScriptedLLM):
    """Classifies by question id prefix, answers with one cited and one uncited segment, and
    judges every segment as supported."""

    def __init__(self, out_of_scope: set[str]) -> None:
        super().__init__(classification())
        self.out_of_scope = out_of_scope

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        if request["model"] == "claude-haiku-5-5":
            q = request["messages"][0]["content"]
            c = (
                classification(scope="out_of_scope", kind="other")
                if q in self.out_of_scope
                else classification()
            )
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(c)}]}
        if request.get("output_config", {}).get("format"):  # the judge
            n = request["messages"][0]["content"].count("\n(")
            verdicts = {"segments": [{"i": i, "verdict": "supported", "note": ""} for i in range(n)]}
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(verdicts)}]}
        first = request["messages"][-1]["content"][0]["content"][0]["text"]
        return {
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "content": [
                {"type": "text", "text": "The source says so.", "citations": [cite(0, first)]},
                {"type": "text", "text": " Shall we go on?"},
            ],
        }


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ev, "ANSWERS", tmp_path / "answers")
    monkeypatch.setattr(ev, "LLM_RECORDINGS", tmp_path / "llm")
    (tmp_path / "answers").mkdir()
    return tmp_path


def picked() -> list[Any]:
    """One dev question of each kind: a source question, a sensitive one, an out-of-scope one."""
    dev = load_questions(split="dev")
    return [next(q for q in dev if q.type == t) for t in ("source", "sensitive", "out_of_scope")]


def run(dirs: Path) -> list[dict[str, Any]]:
    questions = picked()
    scripted = EvalLLM(out_of_scope={q.question for q in questions if q.type == "out_of_scope"})
    llm = RecordingLLM(dirs / "llm", Mode.RECORD, scripted)
    b = bot(llm)
    rows = [ev.answer_row(q, b.ask(q.question)) for q in questions]
    (dirs / "answers" / "dev.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    judged = ev.judge_rows(RecordingLLM(dirs / "llm", Mode.RECORD, scripted), "dev")
    (dirs / "answers" / "dev-judge.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in judged), encoding="utf-8"
    )
    return rows


def test_scores_come_from_the_recordings(dirs: Path) -> None:
    rows = run(dirs)
    assert [r["status"] for r in rows] == ["answered", "answered", "refused"]
    r = ev.score("dev")
    assert r["integrity_problems"] == []
    g = r["grounding"]
    assert g["answered"] == 2
    assert g["with_citations"] == 1.0
    assert 0 < g["cited_share_of_text"] < 1  # " Shall we go on?" is uncited
    assert g["judge"]["supported_share_of_claims"] == 1.0
    assert r["scope"]["balanced"]["out_of_scope_refused"] == 1.0
    assert r["sensitive"]["ask_rabbi_first"] == 1.0  # first-person, keyword backstop


def test_a_doctored_answer_or_citation_is_caught(dirs: Path) -> None:
    rows = run(dirs)
    rows[0]["segments"][0]["text"] = "Something the model never said."
    (dirs / "answers" / "dev.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    assert any("differs from the recorded response" in p for p in ev.score("dev")["integrity_problems"])

    path = next((dirs / "llm").glob(f"*/{rows[1]['answer_key']}.json"))
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["response"]["content"][0]["citations"][0]["cited_text"] = "a quotation that was never in the source"
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert any("not verbatim" in p for p in ev.score("dev")["integrity_problems"])


def test_manual_review_sheet(
    dirs: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run(dirs)
    monkeypatch.setattr(ev, "EVAL", dirs)
    assert ev.main(["sample", "--split", "dev", "--n", "5"]) == 0
    sheet = (dirs / "manual-review-dev.md").read_text(encoding="utf-8")
    assert sheet.count("Verdict: ") == 2
    (dirs / "manual-review-dev.md").write_text(
        sheet.replace("Verdict: ", "Verdict: supported", 1), encoding="utf-8"
    )
    ev.main(["manual", "--split", "dev"])
    assert "1 of 2 reviewed" in capsys.readouterr().out
