"""Test du pipeline RAG complet (ingestion → retrieval) SANS les services réels :
Fluree est remplacé par un stockage mémoire, Ollama par des embeddings
déterministes (n-grammes de caractères + tokens + paires de tokens, pour un
contrôle thématique proche d'un vrai modèle). Vérifie que les questions
topiques retrouvent bien les bons passages."""

import hashlib
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
    tokens = [tok for tok in re.findall(r"[a-zà-ÿ0-9]+", t) if tok not in _STOP and len(tok) >= 3]
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

    def all_chunks(self, source: str | None = None, entity: str | None = None) -> list[dict]:
        rows = list(self.rows)
        if source:
            rows = [r for r in rows if r.get("rag:source") == source]
        if entity:
            rows = [r for r in rows if r.get("rag:mentions") and any(
                m.get("ent:name") == entity for m in r.get("rag:mentions", [])
            )]
        return rows


@pytest.fixture
def fake_stack(monkeypatch):
    store = FakeFluree()
    monkeypatch.setattr(fluree, "ensure_ledger", store.ensure_ledger)
    monkeypatch.setattr(fluree, "insert_chunks", store.insert_chunks)
    monkeypatch.setattr(fluree, "all_chunks", store.all_chunks)
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


def test_retrieval_finds_quantum_entities(fake_stack):
    _ingest_article(fake_stack)
    hits = retriever.retrieve("Quelle entreprise a développé l'ordinateur quantique Lucy ?", top_k=3)
    merged = " ".join(h["content"] for h in hits)
    assert "Quandela" in merged
    hits = retriever.retrieve("Quel organisme a conclu un contrat historique avec Alice & Bob ?", top_k=3)
    merged = " ".join(h["content"] for h in hits)
    assert "Alice" in merged and "GENCI" in merged


def test_retrieval_finds_palantir_switch(fake_stack):
    _ingest_article(fake_stack)
    hits = retriever.retrieve("La solution de Palantir a-t-elle été remplacée par ChapsVision ?", top_k=3)
    merged = " ".join(h["content"] for h in hits)
    assert "Palantir" in merged and "ChapsVision" in merged


def test_retrieval_finds_agent_count(fake_stack):
    _ingest_article(fake_stack)
    hits = retriever.retrieve("Combien d'agents de l'État vont bénéficier de l'assistant IA et à quel coût ?", top_k=3)
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
    resp = client.post("/api/search", json={"question": "Une question de test", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == "Une question de test"
    assert data["top_k"] == 5
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "Le contenu pertinent."
    assert data["results"][0]["score"] == 0.95