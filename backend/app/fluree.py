"""Client HTTP pour Fluree Server v4.x (API sous /v1/fluree).

Référence : https://labs.flur.ee/docs/db/api/overview
- POST /v1/fluree/create          { "ledger": "kb/main" }
- POST /v1/fluree/upsert?ledger=… corps JSON-LD { "@context", "@graph" }  (idempotent)
- POST /v1/fluree/query/{ledger}  corps JSON-LD { "@context", "select", "where" }
"""

import logging
import re
import unicodedata
from urllib.parse import quote

import httpx

from . import config

log = logging.getLogger(__name__)

API_BASE = f"{config.FLUREE_URL}/v1/fluree"


def _post(path: str, payload: dict, timeout: float = 60.0) -> dict:
    url = f"{API_BASE}{path}"
    resp = httpx.post(url, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Fluree {path} -> HTTP {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


def is_ready() -> bool:
    try:
        resp = httpx.get(f"{config.FLUREE_URL}/health", timeout=5)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_ledger() -> str:
    """Crée le ledger s'il n'existe pas, puis renvoie son nom.

    Seul le conflit « ledger déjà existant » est toléré : une erreur réseau,
    d'authentification ou de configuration doit remonter à l'appelant au lieu
    d'être masquée.
    """
    try:
        out = _post("/create", {"ledger": config.LEDGER}, timeout=30)
        log.info("Ledger créé : %s", out.get("ledger_id"))
    except RuntimeError as exc:
        message = str(exc).lower()
        conflict_markers = (
            "already exist",
            "already exists",
            "exists",
            "conflict",
            "déjà",
            " 409",
        )
        if any(marker in message for marker in conflict_markers):
            log.info("Ledger déjà présent : %s", config.LEDGER)
        else:
            raise
    return config.LEDGER


_ENTITY_SLUG_RE = re.compile(r"[^a-z0-9]+")


def entity_slug(name: str) -> str:
    """Transforme un nom d'entité en identifiant local RDF sûr.

    « Alice & Bob » -> « alice_bob », « État » -> « etat ». Les accents et la
    ponctuation sont supprimés pour produire une IRI valide.
    """
    ascii_name = (
        unicodedata.normalize("NFKD", name or "")
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = _ENTITY_SLUG_RE.sub("_", ascii_name).strip("_")
    return slug or "entity"


def insert_chunks(chunks: list[dict]) -> dict:
    """Insère des chunks (content, embedding, source, index) via /upsert.

    /upsert est idempotent : ré-ingérer un chunk déjà présent (même @id et
    mêmes valeurs) n'ajoute pas de doublons — les chunks gardent un @id stable
    dérivé de leur contenu.

    Si un chunk contient des entités (clé 'entities'), les entités et les liens
    rag:mentions sont insérés dans la même transaction.
    """
    chunk_inserts = []
    # Les entités sont globales : on les déduplique sur l'ensemble du lot pour
    # ne pas envoyer plusieurs fois le même sujet dans un même @graph (Fluree
    # peut rejeter ou écraser les doublons).
    entity_nodes: dict[str, dict] = {}
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
        mentions = []
        seen_entities: set[str] = set()
        for e in c.get("entities", []):
            ent_id = f"ent:{entity_slug(e['name'])}"
            if ent_id in seen_entities:
                continue
            seen_entities.add(ent_id)
            mentions.append({"@id": ent_id, "ent:name": e["name"]})
            entity_nodes.setdefault(
                ent_id,
                {
                    "@id": ent_id,
                    "@type": f"ent:{e['type']}",
                    "ent:name": e["name"],
                },
            )
        if mentions:
            chunk_obj["rag:mentions"] = mentions
        chunk_inserts.append(chunk_obj)

    payload = {
        "@context": config.CTX,
        "@graph": list(entity_nodes.values()) + chunk_inserts,
    }
    ledger_qs = quote(config.LEDGER, safe="")
    return _post(f"/upsert?ledger={ledger_qs}", payload, timeout=120)


def chunk_local_id(value: object) -> str | None:
    """Renvoie l'identifiant local d'un chunk, quelle que soit sa forme.

    Les chunks sont insérés avec un @id compact (`rag:<hash>`), mais Fluree peut
    renvoyer l'IRI complète (`https://ns.flurag.dev/<hash>`) : on normalise les
    deux formes (ainsi qu'un identifiant nu) pour pouvoir les comparer.
    """
    if not isinstance(value, str):
        return None
    for prefix in (config.RAG_NS, "rag:"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def mark_obsolete(source: str, keep_ids: set | None = None) -> dict:
    """Marque tous les chunks existants d'une source comme obsolètes, en excluant les IDs spécifiés.

    Args:
        source: La source dont les chunks doivent être marqués obsolètes.
        keep_ids: Ensemble d'IDs de chunks à conserver (ne pas marquer comme obsolètes).
                  Utile pour exclure les nouveaux chunks récemment insérés. Les IDs
                  peuvent être nus (`<hash>`), compacts (`rag:<hash>`) ou des IRI.
    """
    chunks = all_chunks(source=source)
    # Les appelants passent des chunk_id nus alors que le graphe renvoie des @id
    # compacts ou des IRI : sans normalisation, *tous* les chunks de la source
    # (y compris ceux qui viennent d'être insérés) étaient marqués obsolètes et
    # le document devenait introuvable.
    keep = {chunk_local_id(value) for value in (keep_ids or set())}
    updates = [
        {"@id": chunk["@id"], "rag:obsolete": True}
        for chunk in chunks
        if chunk.get("@id") and chunk_local_id(chunk["@id"]) not in keep
    ]
    if not updates:
        return {}
    payload = {"@context": config.CTX, "@graph": updates}
    ledger_qs = quote(config.LEDGER, safe="")
    try:
        return _post(f"/upsert?ledger={ledger_qs}", payload, timeout=60)
    except Exception as exc:  # noqa: BLE001
        # Ne pas masquer l'échec : sans ce marquage, anciens et nouveaux chunks
        # resteraient actifs (doublons). L'appelant annule l'insertion partielle.
        log.error("Impossible de marquer obsolètes : %s", exc)
        raise


def mark_ids_obsolete(chunk_ids: list[str] | set[str]) -> dict:
    """Marque uniquement les ids fournis comme obsolètes.

    Utilisé pour annuler une ingestion partielle : les chunks déjà insérés sont
    désactivés afin que l'ancienne version du document reste la seule active.
    """
    updates = [
        {"@id": f"rag:{chunk_local_id(chunk_id)}", "rag:obsolete": True}
        for chunk_id in chunk_ids
        if chunk_local_id(chunk_id)
    ]
    if not updates:
        return {}
    payload = {"@context": config.CTX, "@graph": updates}
    ledger_qs = quote(config.LEDGER, safe="")
    try:
        return _post(f"/upsert?ledger={ledger_qs}", payload, timeout=60)
    except Exception as exc:  # noqa: BLE001
        log.warning("Impossible d'annuler les chunks insérés : %s", exc)
        return {}


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


def history(limit: int = 100) -> list[dict]:
    """Renvoie l'historique des transactions du ledger (API `/log` de Fluree).

    Fluree renvoie `{"ledger_id": ..., "commits": [...], "count": ...}` ; on
    n'expose que la liste des commits. Une liste vide est renvoyée si l'API
    d'historique est indisponible (le reste de l'application continue).
    """
    try:
        resp = httpx.get(
            f"{API_BASE}/log/{config.LEDGER}", params={"limit": limit}, timeout=30
        )
        resp.raise_for_status()
        out = resp.json()
        if isinstance(out, list):
            return out
        if isinstance(out, dict):
            for key in ("commits", "results", "history"):
                if key in out:
                    return out[key]
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning("Historique Fluree indisponible : %s", exc)
        return []
