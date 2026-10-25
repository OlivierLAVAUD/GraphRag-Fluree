"""Tests de la désindexation : source unique et réinitialisation complète.

La désindexation ne supprime rien du ledger (immuable) : elle marque les chunks
obsolètes, exactement comme une ré-ingestion, ce qui les retire de la recherche.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app import config, fluree, ingester, retriever
from app.main import app

ARTICLE = Path(__file__).resolve().parents[2] / "docs" / "article-tech-francaise.txt"


def test_deindex_source_batches_requests(monkeypatch):
    """600 chunks -> 2 transactions (500 + 100) au lieu d'une requête géante."""
    ids = [f"rag:{i}" for i in range(600)]
    monkeypatch.setattr(
        fluree, "all_chunks", lambda source=None: [{"@id": cid} for cid in ids]
    )
    payloads = []

    def fake_post(path, payload, timeout=60):
        payloads.append(payload)
        return {"ok": True}

    monkeypatch.setattr(fluree, "_post", fake_post)

    assert fluree.deindex_source("docs/a.md") == 600
    assert [len(p["@graph"]) for p in payloads] == [500, 100]
    assert all(node["rag:obsolete"] is True for p in payloads for node in p["@graph"])


def test_deindex_unknown_source_returns_zero(monkeypatch):
    monkeypatch.setattr(fluree, "all_chunks", lambda source=None: [])
    called = {"post": 0}

    def fake_post(path, payload, timeout=60):
        called["post"] += 1
        return {"ok": True}

    monkeypatch.setattr(fluree, "_post", fake_post)
    assert fluree.deindex_source("inconnu.md") == 0
    assert called["post"] == 0


def test_deindex_source_propagates_failures(monkeypatch):
    """Un échec de transaction doit remonter : sinon l'utilisateur croit à tort
    que la source est désindexée."""
    monkeypatch.setattr(
        fluree, "all_chunks", lambda source=None: [{"@id": "rag:a"}, {"@id": "rag:b"}]
    )

    def fake_post(path, payload, timeout=60):
        raise fluree.FlureeError(path, 500, "boom")

    monkeypatch.setattr(fluree, "_post", fake_post)
    try:
        fluree.deindex_source("docs/a.md")
    except fluree.FlureeError:
        pass
    else:  # pragma: no cover - échec du test explicite
        raise AssertionError("l'erreur Fluree aurait dû être propagée")


def test_api_deindex_source_removes_it_from_search(fake_stack):
    text = ARTICLE.read_text(encoding="utf-8")
    ingester.ingest_text(text, "docs/article.txt")
    ingester.ingest_text("Autre document sur les services Docker.", "docs/autre.md")
    assert len(fake_stack.all_chunks()) >= 3

    client = TestClient(app, raise_server_exceptions=False)
    before = client.post(
        "/api/search", json={"question": "ordinateur quantique Lucy", "top_k": 3}
    ).json()["results"]
    assert any(hit["source"] == "docs/article.txt" for hit in before)

    resp = client.delete("/api/sources", params={"source": "docs/article.txt"})
    assert resp.status_code == 200
    assert resp.json()["deindexed"] >= 2
    assert resp.json()["source"] == "docs/article.txt"

    after = client.post(
        "/api/search", json={"question": "ordinateur quantique Lucy", "top_k": 5}
    ).json()["results"]
    # Les chunks de la source désindexée ne sont plus renvoyés, les autres oui.
    assert all(hit["source"] != "docs/article.txt" for hit in after)
    assert fake_stack.all_chunks()  # la base n'est pas vidée


def test_api_deindex_unknown_source_is_404(fake_stack):
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.delete("/api/sources", params={"source": "absent.md"})
    assert resp.status_code == 404


def test_api_reset_empties_the_knowledge_base(fake_stack):
    ingester.ingest_text(ARTICLE.read_text(encoding="utf-8"), "docs/article.txt")
    ingester.ingest_text("Deuxième document.", "docs/deux.md")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/reset")
    assert resp.status_code == 200
    assert resp.json()["deindexed"] >= 2
    assert fake_stack.all_chunks() == []
    assert (
        client.post("/api/search", json={"question": "quantique"}).json()["results"]
        == []
    )


def test_api_reset_on_empty_base_is_idempotent(fake_stack):
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/reset")
    assert resp.status_code == 200
    assert resp.json()["deindexed"] == 0


def test_destructive_endpoints_require_token(monkeypatch):
    monkeypatch.setattr(config, "API_TOKEN", "s3cret")
    client = TestClient(app, raise_server_exceptions=False)

    assert client.delete("/api/sources", params={"source": "a.md"}).status_code == 401
    assert client.post("/api/reset").status_code == 401


def test_deindex_invalidates_the_search_index(fake_stack, monkeypatch):
    """Après désindexation, l'index en cache ne doit pas resservir les chunks."""
    ingester.ingest_text(ARTICLE.read_text(encoding="utf-8"), "docs/article.txt")
    retriever.retrieve("quantique", top_k=3)
    assert retriever._index_cache  # index en cache

    ingester.deindex_source("docs/article.txt")
    assert retriever._index_cache == {}
    assert retriever.retrieve("quantique", top_k=3) == []
