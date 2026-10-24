"""Test du pipeline RAG complet (ingestion → retrieval) SANS les services réels :
Fluree est remplacé par un stockage mémoire, Ollama par des embeddings
déterministes (n-grammes de caractères + tokens + paires de tokens, pour un
contrôle thématique proche d'un vrai modèle). Vérifie que les questions
topiques retrouvent bien les bons passages."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from app import config, fluree, ingester, ollama_client as ollama, retriever

ARTICLE = Path(__file__).resolve().parents[2] / "docs" / "article-tech-francaise.txt"

_DIM = 512
_STOP = frozenset(
    "le la les de des du et en un une pour avec est sont au aux qui que se sa son ses \
ce cette il elle ils elles dans par sur plus comme".split()
)


def _hash_idx(prefix: str, token: str) -> int:
    return int(hashlib.md5((prefix + token).encode()).hexdigest()[:8], 16) % _DIM


def _fake_embed(text: str) -> list[float]:
    vec = [0.0] * _DIM
    t = text.lower()
    # n-grammes de caractères : thème approximatif
    for i in range(len(t) - 2):
        vec[_hash_idx("", t[i : i + 3])] += 1.0
    # tokens (entités, vocabulaire thématique)
    tokens = [
        tok
        for tok in re.findall(r"[a-zà-ÿ0-9]+", t)
        if tok not in _STOP and len(tok) >= 3
    ]
    for tok in tokens:
        vec[_hash_idx("t", tok)] += 4.0
    # paires de tokens consécutifs : expressions-clés ("calcul quantique", …)
    for a, b in zip(tokens, tokens[1:]):
        vec[_hash_idx("p", f"{a} {b}")] += 3.0
    return vec


def _fake_embed_batch(texts: list[str]) -> list[list[float]]:
    return [_fake_embed(t) for t in texts]


class FakeFluree:
    """Remplacement de app.fluree : garde les chunks en mémoire."""

    def __init__(self):
        self.rows: list[dict] = []

    def ensure_ledger(self) -> str:
        return "rag/smoke"

    def insert_chunks(self, chunks: list[dict]) -> dict:
        for c in chunks:
            self.rows.append(
                {
                    "@id": f"rag:{c['chunk_id']}",
                    "@type": "rag:Chunk",
                    "rag:content": c["content"],
                    "rag:embedding": c["embedding"],
                    "rag:source": c["source"],
                    "rag:chunkIndex": c["index"],
                    "rag:obsolete": False,
                }
            )
        return {"ok": True}

    def mark_obsolete(self, source: str, keep_ids: set | None = None) -> dict:
        # Même normalisation que fluree.mark_obsolete : les keep_ids sont des
        # chunk_id nus alors que les lignes portent un @id préfixé.
        keep = {fluree.chunk_local_id(value) for value in (keep_ids or set())}
        for r in self.rows:
            if (
                r.get("rag:source") == source
                and fluree.chunk_local_id(r.get("@id")) not in keep
            ):
                r["rag:obsolete"] = True
        return {"ok": True}

    def all_chunks(
        self, source: str | None = None, entity: str | None = None
    ) -> list[dict]:
        rows = [r for r in self.rows if not r.get("rag:obsolete", False)]
        if source:
            rows = [r for r in rows if r.get("rag:source") == source]
        if entity:
            rows = [
                r
                for r in rows
                if r.get("rag:mentions")
                and any(m.get("ent:name") == entity for m in r.get("rag:mentions", []))
            ]
        return rows


@pytest.fixture
def fake_stack(monkeypatch):
    store = FakeFluree()
    monkeypatch.setattr(fluree, "ensure_ledger", store.ensure_ledger)
    monkeypatch.setattr(fluree, "insert_chunks", store.insert_chunks)
    monkeypatch.setattr(fluree, "all_chunks", store.all_chunks)
    monkeypatch.setattr(fluree, "mark_obsolete", store.mark_obsolete)
    monkeypatch.setattr(ollama, "embed_batch", _fake_embed_batch)
    monkeypatch.setattr(ollama, "embed_one", lambda t: _fake_embed(t))
    # Les embeddings simulés ont une échelle de similarité inférieure aux vrais
    # (nomic-embed-text) : le seuil RETRIEVAL_MIN_SCORE est désactivé ici, sa
    # logique étant testée à part (test_min_score_filters_low_similarity).
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    return store


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

    def fake_get(url, params=None, timeout=30):
        calls["url"] = url
        calls["params"] = params
        return FakeResponse()

    monkeypatch.setattr(fluree.httpx, "get", fake_get)
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
    monkeypatch.setattr(fluree, "all_chunks", lambda source=None, entity=None: rows)
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
# Test d'intégration : endpoint /api/search
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
