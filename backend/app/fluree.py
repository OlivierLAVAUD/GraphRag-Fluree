"""Client HTTP pour Fluree Server v4.x (API sous /v1/fluree).

Référence : https://labs.flur.ee/docs/db/api/overview
- POST /v1/fluree/create          { "ledger": "kb/main" }
- POST /v1/fluree/upsert?ledger=… corps JSON-LD { "@context", "@graph" }  (idempotent)
- POST /v1/fluree/query/{ledger}  corps JSON-LD { "@context", "select", "where" }
"""

import logging
from urllib.parse import quote

import httpx

from . import config

log = logging.getLogger(__name__)

API_BASE = f"{config.FLUREE_URL}/v1/fluree"


def _post(path: str, payload: dict, timeout: float = 60.0) -> dict:
    url = f"{API_BASE}{path}"
    resp = httpx.post(url, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"Fluree {path} -> HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def is_ready() -> bool:
    try:
        resp = httpx.get(f"{config.FLUREE_URL}/health", timeout=5)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_ledger() -> str:
    """Crée le ledger s'il n'existe pas, puis renvoie son nom."""
    try:
        out = _post("/create", {"ledger": config.LEDGER}, timeout=30)
        log.info("Ledger créé : %s", out.get("ledger_id"))
    except RuntimeError as exc:
        # Conflit si le ledger existe déjà : ce n'est pas une erreur bloquante.
        log.info("Ledger probablement déjà présent : %s", str(exc)[:200])
    return config.LEDGER


def insert_chunks(chunks: list[dict]) -> dict:
    """Insère des chunks (content, embedding, source, index) via /upsert.

    /upsert est idempotent : ré-ingérer un chunk déjà présent (même @id et
    mêmes valeurs) n'ajoute pas de doublons — les chunks gardent un @id stable
    dérivé de leur contenu.

    Si un chunk contient des entités (clé 'entities'), les entités et les liens
    rag:mentions sont insérés dans la même transaction.
    """
    inserts = []
    for c in chunks:
        chunk_obj = {
            "@id": f"rag:{c['chunk_id']}",
            "@type": "rag:Chunk",
            "rag:content": c["content"],
            "rag:embedding": {"@list": c["embedding"]},
            "rag:source": c["source"],
            "rag:chunkIndex": c["index"],
            "rag:obsolete": False,
        }
        # Ajouter les mentions d'entités
        entities = c.get("entities", [])
        if entities:
            chunk_obj["rag:mentions"] = [
                {"@id": f"ent:{e['name'].lower().replace(' ', '_')}"}
                for e in entities
            ]
            for e in entities:
                ent_id = f"ent:{e['name'].lower().replace(' ', '_')}"
                inserts.append({
                    "@id": ent_id,
                    "@type": f"ent:{e['type']}",
                    "ent:name": e["name"],
                })
        inserts.append(chunk_obj)

    payload = {"@context": config.CTX, "@graph": inserts}
    ledger_qs = quote(config.LEDGER, safe="")
    return _post(f"/upsert?ledger={ledger_qs}", payload, timeout=120)


def all_chunks(source: str | None = None, entity: str | None = None) -> list[dict]:
    """Renvoie les chunks du graphe, optionnellement filtrés.

    Filtres :
    - source : nom du fichier source
    - entity : nom d'une entité mentionnée par le chunk
    """
    where = {"@id": "?c", "@type": "rag:Chunk", "rag:obsolete": False}
    if source:
        where["rag:source"] = source
    if entity:
        where["rag:mentions"] = {"ent:name": entity}

    payload = {
        "@context": config.CTX,
        "select": {"?c": ["*"]},
        "where": where,
    }
    out = _post(f"/query/{config.LEDGER}", payload, timeout=120)
    if isinstance(out, list):
        return out
    if isinstance(out, dict):
        for key in ("results", "result", "bindings", "rows"):
            if key in out:
                return out[key]
    raise RuntimeError(f"Réponse Fluree inattendue : {str(out)[:300]}")


def history() -> list[dict]:
    """Renvoie l'historique des transactions du ledger (via /v1/fluree/history)."""
    try:
        out = _post(f"/history/{config.LEDGER}", {}, timeout=30)
        if isinstance(out, list):
            return out
        if isinstance(out, dict):
            for key in ("results", "result", "bindings", "rows", "history"):
                if key in out:
                    return out[key]
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning("Historique Fluree indisponible : %s", exc)
        return []