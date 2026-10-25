"""Test du pipeline RAG complet (ingestion → retrieval) SANS les services réels.

Fluree est remplacé par un stockage mémoire et Ollama par des embeddings
déterministes (fixtures `conftest.py`). Vérifie que les questions topiques
retrouvent bien les bons passages.
"""

import json
from pathlib import Path

from app import fluree, ingester, retriever

ARTICLE = Path(__file__).resolve().parents[2] / "docs" / "article-tech-francaise.txt"


def _ingest_article(fake_stack) -> None:
    text = ARTICLE.read_text(encoding="utf-8")
    chunks = ingester.ingest_text(text, ARTICLE.name)
    assert len(chunks) >= 6
    assert len(fake_stack.rows) == len(chunks)


def test_article_ingests_into_chunks(fake_stack):
    _ingest_article(fake_stack)


def test_ingested_chunks_stay_active_and_searchable(fake_stack):
    """Non-régression : l'ingestion ne doit pas marquer ses propres chunks obsolètes."""
    _ingest_article(fake_stack)
    assert all(not row["rag:obsolete"] for row in fake_stack.rows)
    assert len(fake_stack.all_chunks()) == len(fake_stack.rows)
    hits = retriever.retrieve(
        "Quelle entreprise a développé l'ordinateur quantique Lucy ?", top_k=3
    )
    assert hits


def test_chunk_local_id_normalizes_id_forms():
    assert fluree.chunk_local_id("rag:abc123") == "abc123"
    assert fluree.chunk_local_id("https://ns.flurag.dev/abc123") == "abc123"
    assert fluree.chunk_local_id("abc123") == "abc123"
    assert fluree.chunk_local_id(None) is None


def test_history_uses_fluree_log_endpoint(monkeypatch):
    """L'historique interroge l'API /log de Fluree et en extrait les commits."""
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ledger_id": "kb/main", "commits": [{"t": 1}], "count": 1}

    class FakeClient:
        def get(self, url, params=None, timeout=30):
            calls["url"] = url
            calls["params"] = params
            return FakeResponse()

    monkeypatch.setattr(fluree, "_client", lambda: FakeClient())
    assert fluree.history(limit=10) == [{"t": 1}]
    assert calls["url"].endswith("/log/kb/main")
    assert calls["params"] == {"limit": 10}


def test_mark_obsolete_keeps_new_chunk_ids(monkeypatch):
    """Les nouveaux chunk_id (nus) doivent correspondre aux @id renvoyés par Fluree."""
    rows = [
        {"@id": "rag:keep-compact", "rag:source": "doc.md"},
        {"@id": "https://ns.flurag.dev/keep-iri", "rag:source": "doc.md"},
        {"@id": "rag:ancien", "rag:source": "doc.md"},
    ]
    monkeypatch.setattr(fluree, "all_chunks", lambda source=None: rows)
    sent = {}

    def fake_post(path, payload, timeout=60):
        sent["payload"] = payload
        return {"ok": True}

    monkeypatch.setattr(fluree, "_post", fake_post)
    fluree.mark_obsolete("doc.md", keep_ids={"keep-compact", "keep-iri"})

    assert [item["@id"] for item in sent["payload"]["@graph"]] == ["rag:ancien"]


def test_retrieval_finds_quantum_entities(fake_stack):
    _ingest_article(fake_stack)
    hits = retriever.retrieve(
        "Quelle entreprise a développé l'ordinateur quantique Lucy ?", top_k=3
    )
    merged = " ".join(h["content"] for h in hits)
    assert "Quandela" in merged
    hits = retriever.retrieve(
        "Quel organisme a conclu un contrat historique avec Alice & Bob ?", top_k=3
    )
    merged = " ".join(h["content"] for h in hits)
    assert "Alice" in merged and "GENCI" in merged


def test_retrieval_finds_palantir_switch(fake_stack):
    _ingest_article(fake_stack)
    hits = retriever.retrieve(
        "La solution de Palantir a-t-elle été remplacée par ChapsVision ?", top_k=3
    )
    merged = " ".join(h["content"] for h in hits)
    assert "Palantir" in merged and "ChapsVision" in merged


def test_retrieval_finds_agent_count(fake_stack):
    _ingest_article(fake_stack)
    hits = retriever.retrieve(
        "Combien d'agents de l'État vont bénéficier de l'assistant IA et à quel coût ?",
        top_k=3,
    )
    merged = " ".join(h["content"] for h in hits)
    assert "million" in merged and "fonction publique" in merged


# ---------------------------------------------------------------------------
# Tests d'intégration des endpoints
# ---------------------------------------------------------------------------


def test_search_endpoint_empty_question():
    """L'endpoint /api/search rejette une question vide."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/search", json={"question": ""})
    assert resp.status_code == 400


def test_chat_without_hits_asks_the_llm(monkeypatch):
    """Sans extrait pertinent, la réponse vient du LLM, pas d'une phrase codée en dur."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr("app.main.retriever.retrieve", lambda *a, **k: [])
    captured = {}

    def fake_chat(messages, on_delta):
        captured["content"] = messages[-1]["content"]
        on_delta(
            "Je ne dispose pas de cette information dans ma base de connaissances."
        )

    monkeypatch.setattr("app.main.ollama.chat_stream", fake_chat)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/chat", json={"question": "Qui est le président ?"})

    assert resp.status_code == 200
    assert "Je ne dispose pas de cette information" in resp.text
    assert "aucun extrait pertinent" in captured["content"]
    assert "Ingérez d'abord des documents" not in resp.text


def test_chat_sources_include_short_snippet(monkeypatch):
    """Les sources SSE exposent un extrait court, jamais le chunk complet."""
    from fastapi.testclient import TestClient

    from app.main import app

    content = "Le contenu pertinent. " + "x" * 300
    monkeypatch.setattr(
        "app.main.retriever.retrieve",
        lambda *a, **k: [
            {"content": content, "source": "doc.md", "index": 2, "score": 0.91}
        ],
    )
    monkeypatch.setattr(
        "app.main.ollama.chat_stream",
        lambda messages, on_delta: on_delta("Réponse."),
    )

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/chat", json={"question": "Une question ?"})

    assert resp.status_code == 200
    events = [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    source = next(event for event in events if event["type"] == "sources")["sources"][0]
    assert source["source"] == "doc.md"
    assert source["index"] == 2
    assert source["score"] == 0.91
    assert source["snippet"].startswith("Le contenu pertinent.")
    assert len(source["snippet"]) <= 241
    assert "content" not in source


def test_search_endpoint_returns_hits(monkeypatch):
    """L'endpoint /api/search retourne les chunks récupérés par le retriever."""
    from fastapi.testclient import TestClient

    from app.main import app

    fake_hits = [
        {
            "content": "Le contenu pertinent.",
            "source": "doc.md",
            "index": 0,
            "score": 0.95,
        }
    ]

    def fake_retrieve(question, top_k=None, matrix=None, source=None, entity=None):
        return fake_hits

    monkeypatch.setattr("app.main.retriever.retrieve", fake_retrieve)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search", json={"question": "Une question de test", "top_k": 5}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == "Une question de test"
    assert data["top_k"] == 5
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "Le contenu pertinent."
    assert data["results"][0]["score"] == 0.95
