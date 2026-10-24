"""Tests des correctifs de robustesse : identité des sources, rollback
d'ingestion, entités, validation API et authentification optionnelle."""

import pytest
from fastapi.testclient import TestClient

from app import config, fluree, ingester, ollama_client as ollama
from app.main import app


def _chunk(chunk_id, content="contenu", entities=None):
    return {
        "chunk_id": chunk_id,
        "index": 0,
        "source": "doc.md",
        "content": content,
        "embedding": [0.0],
        "entities": entities or [],
    }


def test_fluree_slugs_and_deduplicates_entities(monkeypatch):
    sent = {}

    def fake_post(path, payload, timeout=60):
        sent["payload"] = payload
        return {"ok": True}

    monkeypatch.setattr(fluree, "_post", fake_post)
    chunks = [
        _chunk("a", entities=[{"name": "Alice & Bob", "type": "Company"}]),
        _chunk("b", entities=[{"name": "Alice & Bob", "type": "Company"}]),
        _chunk("c", entities=[{"name": "État", "type": "Company"}]),
    ]

    fluree.insert_chunks(chunks)

    graph = sent["payload"]["@graph"]
    entity_nodes = [node for node in graph if node["@id"].startswith("ent:")]
    assert [node["@id"] for node in entity_nodes] == ["ent:alice_bob", "ent:etat"]
    chunk_nodes = [node for node in graph if node.get("@type") == "rag:Chunk"]
    assert all(len(node["rag:mentions"]) == 1 for node in chunk_nodes)


def test_ingest_rolls_back_partially_inserted_chunks(monkeypatch):
    calls = {"insert": 0, "obsolete": [], "mark_obsolete": 0}
    chunks = [_chunk(f"c{i}", content=f"chunk {i}") for i in range(150)]

    monkeypatch.setattr(ingester, "prepare_chunks", lambda text, source: chunks)
    monkeypatch.setattr(fluree, "ensure_ledger", lambda: "kb/test")
    monkeypatch.setattr(
        ollama, "embed_batch", lambda texts, batch_size=32: [[0.0] for _ in texts]
    )

    def fake_insert(batch):
        calls["insert"] += 1
        if calls["insert"] == 2:
            raise RuntimeError("Fluree indisponible")
        return {"ok": True}

    monkeypatch.setattr(fluree, "insert_chunks", fake_insert)
    monkeypatch.setattr(
        fluree, "mark_ids_obsolete", lambda ids: calls["obsolete"].extend(ids) or {}
    )
    monkeypatch.setattr(
        fluree,
        "mark_obsolete",
        lambda source, keep_ids=None: calls.update(
            mark_obsolete=calls["mark_obsolete"] + 1
        ),
    )

    with pytest.raises(RuntimeError):
        ingester.ingest_text("texte", "doc.md")

    assert calls["obsolete"] == [f"c{i}" for i in range(100)]
    assert calls["mark_obsolete"] == 0


def test_source_for_path_keeps_subdirectories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "a" / "doc.md"
    second = tmp_path / "b" / "doc.md"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    assert ingester.source_for_path(first) == "a/doc.md"
    assert ingester.source_for_path(second) == "b/doc.md"


def test_search_rejects_invalid_top_k(monkeypatch):
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/search", json={"question": "test", "top_k": 0})
    assert resp.status_code == 422


def test_api_token_is_enforced_when_configured(monkeypatch):
    monkeypatch.setattr(config, "API_TOKEN", "s3cret")
    monkeypatch.setattr("app.main.retriever.retrieve", lambda *a, **k: [])
    client = TestClient(app, raise_server_exceptions=False)

    assert client.post("/api/search", json={"question": "test"}).status_code == 401
    ok = client.post(
        "/api/search",
        json={"question": "test"},
        headers={"X-API-Token": "s3cret"},
    )
    assert ok.status_code == 200


def test_ingest_rejects_unsupported_extension(monkeypatch):
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/ingest",
        files={"files": ("malware.exe", b"data", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_ingest_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE", 5)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/ingest", files={"files": ("doc.md", b"trop long", "text/markdown")}
    )
    assert resp.status_code == 413
