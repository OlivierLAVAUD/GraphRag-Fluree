"""Tests du client Ollama (module sans couverture jusqu'ici).

Le client est bâti sur des appels httpx module-level : on les remplace par des
doubles pour vérifier les URL, les charges utiles et le découpage en lots, sans
réseau.
"""

import json

import pytest

from app import config, ollama_client as ollama


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self):
        for line in self._payload.get("lines", []):
            yield line


def test_is_ready_reflects_http_status(monkeypatch):
    monkeypatch.setattr(ollama.httpx, "get", lambda url, timeout=5: FakeResponse())
    assert ollama.is_ready() is True

    monkeypatch.setattr(
        ollama.httpx, "get", lambda url, timeout=5: FakeResponse(status_code=500)
    )
    assert ollama.is_ready() is False

    def boom(url, timeout=5):
        raise ollama.httpx.HTTPError("down")

    monkeypatch.setattr(ollama.httpx, "get", boom)
    assert ollama.is_ready() is False


def test_list_models_returns_names(monkeypatch):
    payload = {"models": [{"name": "llama3.2:3b"}, {"name": "nomic-embed-text"}]}
    monkeypatch.setattr(
        ollama.httpx, "get", lambda url, timeout=10: FakeResponse(payload)
    )
    assert ollama.list_models() == ["llama3.2:3b", "nomic-embed-text"]


def test_gpu_status_exposes_vram(monkeypatch):
    payload = {
        "models": [
            {
                "name": "llama3.2:3b",
                "size": 2600000000,
                "size_vram": 2600000000,
                "details": {"gpu": "RTX 3060", "parallel": 1},
            }
        ]
    }
    monkeypatch.setattr(
        ollama.httpx, "get", lambda url, timeout=10: FakeResponse(payload)
    )
    status = ollama.gpu_status()
    assert status["models"][0]["size_vram"] == 2600000000
    assert status["models"][0]["gpus"] == "RTX 3060"


def test_embed_batch_is_chunked_and_ordered(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=300):
        calls.append(json)
        return FakeResponse({"embeddings": [[float(len(t))] for t in json["input"]]})

    monkeypatch.setattr(ollama.httpx, "post", fake_post)
    texts = [f"texte {i}" for i in range(33)]
    vectors = ollama.embed_batch(texts, batch_size=32)

    assert len(calls) == 2
    assert len(calls[0]["input"]) == 32
    assert len(calls[1]["input"]) == 1
    assert calls[0]["model"] == config.EMBED_MODEL
    assert len(vectors) == 33
    # l'ordre d'entrée est conservé entre les lots
    assert vectors[0] == [7.0]  # len("texte 0")
    assert vectors[-1] == [8.0]  # len("texte 32")


def test_embed_batch_without_texts_does_not_call_http(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("aucun appel HTTP attendu")

    monkeypatch.setattr(ollama.httpx, "post", fail)
    assert ollama.embed_batch([]) == []


def test_embed_one_returns_single_vector(monkeypatch):
    monkeypatch.setattr(
        ollama.httpx,
        "post",
        lambda url, json=None, timeout=300: FakeResponse({"embeddings": [[0.1, 0.2]]}),
    )
    assert ollama.embed_one("question") == [0.1, 0.2]


def test_chat_stream_forwards_deltas_and_stops_at_done(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "Bon"}, "done": False}),
        "",
        "pas du json",
        json.dumps({"message": {"content": "jour"}, "done": False}),
        json.dumps({"message": {"content": "ignoré"}, "done": True}),
    ]
    sent = {}

    class FakeStream:
        def __enter__(self):
            return FakeResponse({"lines": lines})

        def __exit__(self, *exc):
            return False

    def fake_stream(method, url, json=None, timeout=None):
        sent["method"] = method
        sent["url"] = url
        sent["payload"] = json
        return FakeStream()

    monkeypatch.setattr(ollama.httpx, "stream", fake_stream)
    deltas = []
    ollama.chat_stream([{"role": "user", "content": "salut"}], on_delta=deltas.append)

    assert "".join(deltas) == "Bonjour"
    assert sent["payload"]["model"] == config.CHAT_MODEL
    assert sent["payload"]["stream"] is True
    assert sent["url"].endswith("/api/chat")


def test_chat_stream_propagates_http_errors(monkeypatch):
    class FakeStream:
        def __enter__(self):
            return FakeResponse(status_code=500)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(ollama.httpx, "stream", lambda *a, **k: FakeStream())
    with pytest.raises(RuntimeError):
        ollama.chat_stream([], on_delta=lambda _: None)
