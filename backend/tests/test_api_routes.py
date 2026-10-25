"""Tests des routes sans couverture : /api/status, /api/sources, /api/entities,
/api/gpu, /api/history, réconciliation des sources à l'upload et limiteur de
générations simultanées.

Le serveur Fluree n'est jamais joint : `fluree._post` et les clients Ollama sont
remplacés par des doubles.
"""

import pytest
from fastapi.testclient import TestClient

from app import config, fluree, ingester, ollama_client as ollama
from app.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_status_exposes_models_and_ledger(monkeypatch, client):
    monkeypatch.setattr(ollama, "list_models", lambda: ["llama3.2:3b"])
    data = client.get("/api/status").json()
    assert data["ledger"] == config.LEDGER
    assert data["chat_model"] == config.CHAT_MODEL
    assert data["embed_model"] == config.EMBED_MODEL
    assert data["ollama_models"] == ["llama3.2:3b"]


def test_status_survives_unreachable_ollama(monkeypatch, client):
    def boom():
        raise RuntimeError("ollama down")

    monkeypatch.setattr(ollama, "list_models", boom)
    resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.json()["ollama_models"] == []


def test_sources_are_counted_per_source(monkeypatch, client):
    rows = [
        {"rag:source": "docs/a.md"},
        {"rag:source": "docs/a.md"},
        {"rag:source": "docs/b.md"},
        {"rag:source": None},
    ]
    monkeypatch.setattr(fluree, "_post", lambda path, payload, timeout=60: rows)

    data = client.get("/api/sources").json()
    assert data["sources"] == [
        {"name": "docs/a.md", "chunks": 2},
        {"name": "docs/b.md", "chunks": 1},
    ]


def test_entities_are_aggregated_with_fallback_names(monkeypatch, client):
    rows = [
        {
            "rag:mentions": [
                {"@id": "ent:palantir", "ent:name": "Palantir", "@type": "ent:Company"}
            ]
        },
        {"rag:mentions": [{"@id": "ent:palantir", "ent:name": "Palantir"}]},
        # Mention sans nom ni type : le repli interroge les chunks actifs.
        {"rag:mentions": [{"@id": "ent:quandela"}]},
    ]
    monkeypatch.setattr(fluree, "_post", lambda path, payload, timeout=60: rows)
    monkeypatch.setattr(
        fluree, "active_entity_names", lambda: {"ent:quandela": "Quandela"}
    )

    data = client.get("/api/entities").json()
    entities = {entity["name"]: entity for entity in data["entities"]}
    assert entities["Palantir"] == {"name": "Palantir", "type": "Company", "chunks": 2}
    assert entities["Quandela"] == {"name": "Quandela", "type": "", "chunks": 1}


def test_history_returns_commits(monkeypatch, client):
    monkeypatch.setattr(fluree, "history", lambda: [{"t": 1}])
    assert client.get("/api/history").json() == {"history": [{"t": 1}]}


def test_history_reports_unavailable_ledger(monkeypatch, client):
    def boom():
        raise RuntimeError("Fluree down")

    monkeypatch.setattr(fluree, "history", boom)
    assert client.get("/api/history").status_code == 503


def test_gpu_reports_vram(monkeypatch, client):
    monkeypatch.setattr(ollama, "gpu_status", lambda: {"models": [{"name": "m"}]})
    assert client.get("/api/gpu").json() == {"models": [{"name": "m"}]}


def test_gpu_reports_ollama_down(monkeypatch, client):
    def boom():
        raise RuntimeError("Ollama down")

    monkeypatch.setattr(ollama, "gpu_status", boom)
    assert client.get("/api/gpu").status_code == 503


def _stub_chat(monkeypatch, answer: str = "Réponse.") -> None:
    monkeypatch.setattr("app.main.retriever.retrieve", lambda *a, **k: [])
    monkeypatch.setattr(
        "app.main.ollama.chat_stream", lambda messages, on_delta: on_delta(answer)
    )


def test_chat_releases_its_slot(monkeypatch, client):
    """Le créneau doit être rendu après chaque réponse (pas de 429 fantôme)."""
    monkeypatch.setattr(config, "MAX_CONCURRENT_CHATS", 1)
    _stub_chat(monkeypatch)

    for _ in range(3):
        resp = client.post("/api/chat", json={"question": "question"})
        assert resp.status_code == 200


def test_chat_refuses_when_gpu_is_saturated(monkeypatch, client):
    """Aucune génération possible : 429 immédiat, pas de mise en file d'attente."""
    monkeypatch.setattr(config, "MAX_CONCURRENT_CHATS", 0)
    _stub_chat(monkeypatch)

    resp = client.post("/api/chat", json={"question": "question"})
    assert resp.status_code == 429


def test_upload_reuses_the_cli_source_name(monkeypatch, client):
    """Un upload `faq.md` rejoint `docs/faq.md` au lieu de créer un doublon."""
    monkeypatch.setattr(fluree, "source_names", lambda: ["autre/faq.md", "docs/faq.md"])
    monkeypatch.setattr(ingester, "read_bytes", lambda data, filename: "texte")
    seen = {}

    def fake_ingest_text(text, source):
        seen["source"] = source
        return [{"chunk_id": "abc"}]

    monkeypatch.setattr(ingester, "ingest_text", fake_ingest_text)

    resp = client.post(
        "/api/ingest", files={"files": ("faq.md", b"texte", "text/markdown")}
    )
    assert resp.status_code == 200
    assert seen["source"] == "docs/faq.md"  # le chemin le plus court l'emporte
    assert resp.json()["ingested"][0]["source"] == "docs/faq.md"


def test_upload_prefers_the_path_form_when_both_exist(monkeypatch, client):
    """Si `faq.md` et `docs/faq.md` existent, la forme CLI (avec chemin) gagne."""
    monkeypatch.setattr(fluree, "source_names", lambda: ["faq.md", "docs/faq.md"])
    monkeypatch.setattr(ingester, "read_bytes", lambda data, filename: "texte")
    seen = {}

    def fake_ingest_text(text, source):
        seen["source"] = source
        return []

    monkeypatch.setattr(ingester, "ingest_text", fake_ingest_text)

    client.post("/api/ingest", files={"files": ("faq.md", b"texte", "text/markdown")})
    assert seen["source"] == "docs/faq.md"


def test_upload_keeps_filename_without_existing_source(monkeypatch, client):
    monkeypatch.setattr(fluree, "source_names", lambda: [])
    monkeypatch.setattr(ingester, "read_bytes", lambda data, filename: "texte")
    monkeypatch.setattr(
        ingester, "ingest_text", lambda text, source: [{"chunk_id": "a"}]
    )

    resp = client.post(
        "/api/ingest", files={"files": ("faq.md", b"texte", "text/markdown")}
    )
    assert resp.json()["ingested"][0]["source"] == "faq.md"


def test_upload_of_a_folder_keeps_the_relative_path(monkeypatch, client):
    """Un upload de dossier conserve le chemin : pas de doublon par homonymie."""
    monkeypatch.setattr(fluree, "source_names", lambda: ["docs/notes.md"])
    monkeypatch.setattr(ingester, "read_bytes", lambda data, filename: "texte")
    seen = []

    def fake_ingest_text(text, source):
        seen.append(source)
        return []

    monkeypatch.setattr(ingester, "ingest_text", fake_ingest_text)

    resp = client.post(
        "/api/ingest",
        files={"files": ("mon-dossier/sous/notes.md", b"texte", "text/markdown")},
    )
    assert resp.status_code == 200
    # Le chemin n'est pas réconcilié avec `docs/notes.md` : deux documents
    # homonymes de dossiers différents restent distincts.
    assert seen == ["mon-dossier/sous/notes.md"]
    assert resp.json()["ingested"][0]["source"] == "mon-dossier/sous/notes.md"


def test_upload_source_cannot_escape_its_directory(monkeypatch, client):
    monkeypatch.setattr(ingester, "read_bytes", lambda data, filename: "texte")
    seen = {}

    def fake_ingest_text(text, source):
        seen["source"] = source
        return []

    monkeypatch.setattr(ingester, "ingest_text", fake_ingest_text)

    resp = client.post(
        "/api/ingest",
        files={"files": ("../../etc/passwd.md", b"texte", "text/markdown")},
    )
    assert resp.status_code == 200
    assert seen["source"] == "etc/passwd.md"


def test_resolve_upload_source_skips_fluree_for_relative_paths(monkeypatch):
    """Un chemin relatif est sa propre source : aucune requête de réconciliation."""
    calls = []
    monkeypatch.setattr(fluree, "source_names", lambda: calls.append(1) or [])

    assert ingester.resolve_upload_source("dossier/a.md") == "dossier/a.md"
    assert ingester.resolve_upload_source("dossier\\sous\\a.md") == "dossier/sous/a.md"
    assert ingester.resolve_upload_source("") == "fichier"
    assert calls == []


def test_status_exposes_upload_limits(monkeypatch, client):
    monkeypatch.setattr(ollama, "list_models", lambda: [])
    limits = client.get("/api/status").json()["limits"]
    assert limits["max_upload_files"] == config.MAX_UPLOAD_FILES
    assert limits["max_total_upload_size"] == config.MAX_TOTAL_UPLOAD_SIZE
    assert ".pdf" in limits["supported_extensions"]


def test_ui_assets_are_revalidated(client):
    """`/` et `/static/*` doivent être revalidés pour éviter une UI périmée."""
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
    # Les réponses d'API ne sont pas concernées.
    assert "cache-control" not in client.get("/openapi.json").headers


def test_index_serves_the_current_documents_tab(client):
    """Garde-fou : les contrôles de désindexation sont dans la page servie."""
    html = client.get("/").text
    assert 'id="resetKb"' in html
    assert 'id="sourceList"' in html
    # Sélection de dossier et barre de progression de l'ingestion.
    assert 'id="libraryDir"' in html
    assert "webkitdirectory" in html
    assert 'id="pickDirBtn"' in html
    assert 'id="uploadProgress"' in html
