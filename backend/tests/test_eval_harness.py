"""Test du harnais d'évaluation : exécute eval_rag.main() de bout en bout avec
des services simulés et un « LLM parfait » (qui répond la réponse attendue).
Vérifie que le rapport JSON est généré et que le scoring renvoie ~1.0."""

import json
import sys
from pathlib import Path

import pytest

import eval_rag
from app import config, fluree, ollama_client as ollama
from test_pipeline_smoke import FakeFluree, _fake_embed, _fake_embed_batch

QA_PATH = Path(__file__).resolve().parents[2] / "tests" / "eval" / "qa_france2030.json"


@pytest.fixture
def simulated_stack(monkeypatch):
    store = FakeFluree()
    monkeypatch.setattr(fluree, "ensure_ledger", store.ensure_ledger)
    monkeypatch.setattr(fluree, "insert_chunks", store.insert_chunks)
    monkeypatch.setattr(fluree, "all_chunks", store.all_chunks)
    monkeypatch.setattr(ollama, "embed_batch", _fake_embed_batch)
    monkeypatch.setattr(ollama, "embed_one", lambda t: _fake_embed(t))

    def canned_chat(messages, on_delta):
        # Simule un LLM parfait : renvoie la réponse attendue de la question posée.
        user = messages[-1]["content"]
        question = user.split("Question : ", 1)[1]
        expected = next(q["expected"] for q in load_qa() if q["question"] == question)
        on_delta(expected)

    monkeypatch.setattr(ollama, "chat_stream", canned_chat)
    # Voir test_pipeline_smoke : le seuil est désactivé pour les embeddings simulés.
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    return store


def load_qa() -> list[dict]:
    with open(QA_PATH, encoding="utf-8") as fh:
        return json.load(fh)["questions"]


def test_eval_harness_generates_report(tmp_path, monkeypatch, simulated_stack):
    out = tmp_path / "eval_results.json"
    monkeypatch.setattr(sys, "argv", ["eval_rag.py", "--out", str(out)])
    eval_rag.main()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["questions"] == 10
    assert data["summary"]["ok_count"] == 10
    assert data["summary"]["mean_score"] >= 0.99
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
