"""Test du harnais d'évaluation : exécute eval_rag.main() de bout en bout avec
des services simulés et un « LLM parfait » (qui répond la réponse attendue).
Vérifie que le rapport JSON est généré et que le scoring renvoie ~1.0.
"""

import json
import sys
from pathlib import Path

import pytest

import eval_rag
from app import ollama_client as ollama

QA_PATH = Path(__file__).resolve().parents[2] / "tests" / "eval" / "qa_france2030.json"


def load_qa() -> list[dict]:
    with open(QA_PATH, encoding="utf-8") as fh:
        return json.load(fh)["questions"]


@pytest.fixture
def simulated_stack(fake_stack, monkeypatch):
    """Pile simulée + LLM parfait : renvoie la réponse attendue de la question."""

    def canned_chat(messages, on_delta):
        user = messages[-1]["content"]
        question = user.split("Question : ", 1)[1]
        expected = next(q["expected"] for q in load_qa() if q["question"] == question)
        on_delta(expected)

    monkeypatch.setattr(ollama, "chat_stream", canned_chat)
    return fake_stack


def test_eval_harness_generates_report(tmp_path, monkeypatch, simulated_stack):
    out = tmp_path / "eval_results.json"
    monkeypatch.setattr(sys, "argv", ["eval_rag.py", "--out", str(out)])
    eval_rag.main()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["questions"] == 10
    assert data["summary"]["ok_count"] == 10
    assert data["summary"]["mean_score"] >= 0.99
    assert data["summary"]["threshold"] == eval_rag.PASS_THRESHOLD
    assert all(r["score"] >= 0.99 for r in data["results"])
    first = data["results"][0]
    # chaque question doit exposer la réponse, le score et les extraits récupérés
    assert {"answer", "score", "top_hits", "expected"} <= set(first)


def test_absence_questions_are_special_cased(tmp_path, monkeypatch, simulated_stack):
    """Une question expected_absence réussit si le modèle REFUSE (score 1.0)."""
    out = tmp_path / "eval_results.json"
    monkeypatch.setattr(sys, "argv", ["eval_rag.py", "--out", str(out)])
    eval_rag.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    absence = [r for r in data["results"] if r["expected_absence"]]
    assert absence, "le jeu doit contenir des questions expected_absence"
    # Depuis l'ajout de la source La Tribune, seule Q10 (hallucination)
    # exige une information absente de la base.
    assert [r["id"] for r in absence] == [10]
    assert absence[0]["absence_clean"] is True
