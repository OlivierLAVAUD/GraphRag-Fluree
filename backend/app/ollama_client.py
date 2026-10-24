"""Client minimal pour l'API HTTP d'Ollama (embeddings + génération)."""

import json
import logging
from typing import Callable

import httpx

from . import config

log = logging.getLogger(__name__)


def is_ready() -> bool:
    try:
        resp = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def list_models() -> list[str]:
    resp = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m.get("name", "") for m in resp.json().get("models", [])]


def gpu_status() -> dict:
    """Infos d'exécution des modèles : vérifie que le GPU est utilisé."""
    resp = httpx.get(f"{config.OLLAMA_URL}/api/ps", timeout=10)
    resp.raise_for_status()
    models = []
    for m in resp.json().get("models", []):
        details = m.get("details", {})
        models.append(
            {
                "name": m.get("name", ""),
                "size_vram": m.get("size_vram", 0),
                "size": m.get("size", 0),
                "gpus": details.get("gpu", ""),
                "parallel": details.get("parallel", 0),
            }
        )
    return {"models": models}


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embedd une liste de textes par lots (modèle : nomic-embed-text).

    Envoie les textes par sous-lots de *batch_size* pour éviter les timeouts
    et la saturation mémoire sur de gros documents. Les résultats sont
    concaténés dans l'ordre d'entrée.
    """
    if not texts:
        return []
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = httpx.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": batch},
            timeout=300,
        )
        resp.raise_for_status()
        results.extend(resp.json()["embeddings"])
    return results


def embed_one(text: str) -> list[float]:
    return embed_batch([text])[0]


def chat_stream(messages: list[dict], on_delta: Callable[[str], None]) -> None:
    """Appelle /api/chat en streaming et transmet chaque morceau à on_delta."""
    payload = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2},
    }
    with httpx.stream(
        "POST", f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=None
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("done"):
                break
            delta = data.get("message", {}).get("content", "")
            if delta:
                on_delta(delta)
