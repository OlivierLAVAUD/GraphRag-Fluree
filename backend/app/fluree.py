"""Client HTTP pour Fluree Server v4.x (API sous /v1/fluree).

Référence : https://labs.flur.ee/docs/db/api/overview
- POST /v1/fluree/create          { "ledger": "kb/main" }
- POST /v1/fluree/upsert?ledger=… corps JSON-LD { "@context", "@graph" }  (idempotent)
- POST /v1/fluree/query/{ledger}  corps JSON-LD { "@context", "select", "where" }
"""

import logging
import re
import threading
import unicodedata
from urllib.parse import quote

import httpx

from . import config

log = logging.getLogger(__name__)

API_BASE = f"{config.FLUREE_URL}/v1/fluree"


class FlureeError(RuntimeError):
    """Erreur HTTP renvoyée par Fluree (code de statut conservé).

    Le statut est exposé structuré : les appelants n'ont pas à deviner la
    nature d'un échec en cherchant des mots dans le corps de la réponse.
    """

    def __init__(self, path: str, status_code: int, message: str):
        super().__init__(f"Fluree {path} -> HTTP {status_code}: {message}")
        self.path = path
        self.status_code = status_code


_client_lock = threading.Lock()
_client_instance: httpx.Client | None = None


def _client() -> httpx.Client:
    """Client HTTP partagé vers Fluree (une connexion réutilisée par appel).

    Les tests peuvent remplacer cet objet pour simuler le serveur.
    """
    global _client_instance
    with _client_lock:
        if _client_instance is None or _client_instance.is_closed:
            _client_instance = httpx.Client(timeout=httpx.Timeout(60.0))
        return _client_instance


def close_client() -> None:
    """Ferme le client partagé (arrêt propre, tests)."""
    global _client_instance
    with _client_lock:
        if _client_instance is not None:
            _client_instance.close()
            _client_instance = None


def _post(path: str, payload: dict, timeout: float = 60.0) -> dict:
    resp = _client().post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise FlureeError(path, resp.status_code, resp.text[:500])
    return resp.json()


def is_ready() -> bool:
    try:
        resp = _client().get(f"{config.FLUREE_URL}/health", timeout=5)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_ledger() -> str:
    """Crée le ledger s'il n'existe pas, puis renvoie son nom.

    Seul le conflit « ledger déjà existant » est toléré (409, ou 400 accompagné
    d'un message explicite) : une erreur réseau, d'authentification ou de
    configuration doit remonter à l'appelant au lieu d'être masquée.
    """
    try:
        out = _post("/create", {"ledger": config.LEDGER}, timeout=30)
        log.info("Ledger créé : %s", out.get("ledger_id"))
    except FlureeError as exc:
        already_exists = exc.status_code in (409, 400) and "exist" in str(exc).lower()
        if not already_exists:
            raise
        log.info("Ledger déjà présent : %s", config.LEDGER)
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


def _mark_obsolete_ids(ids: list[str], batch_size: int = 500) -> int:
    """Marque des @id de chunks obsolètes par lots, en propageant les erreurs.

    Le découpage en lots évite d'envoyer une transaction gigantesque lors d'une
    désindexation de corpus entier.
    """
    ledger_qs = quote(config.LEDGER, safe="")
    marked = 0
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        payload = {
            "@context": config.CTX,
            "@graph": [{"@id": cid, "rag:obsolete": True} for cid in batch],
        }
        _post(f"/upsert?ledger={ledger_qs}", payload, timeout=60)
        marked += len(batch)
    return marked


def mark_ids_obsolete(chunk_ids: list[str] | set[str]) -> dict:
    """Marque uniquement les ids fournis comme obsolètes.

    Utilisé pour annuler une ingestion partielle : les chunks déjà insérés sont
    désactivés afin que l'ancienne version du document reste la seule active.
    Best effort : un échec est journalisé, pas propagé (l'appelant est déjà en
    train de remonter une erreur d'ingestion).
    """
    ids = [
        f"rag:{chunk_local_id(chunk_id)}"
        for chunk_id in chunk_ids
        if chunk_local_id(chunk_id)
    ]
    if not ids:
        return {}
    try:
        return {"marked": _mark_obsolete_ids(ids)}
    except Exception as exc:  # noqa: BLE001
        log.warning("Impossible d'annuler les chunks insérés : %s", exc)
        return {}


def deindex_source(source: str) -> int:
    """Désindexe une source : marque tous ses chunks actifs obsolètes.

    Le ledger reste immuable (aucune donnée effacée, l'historique est conservé) :
    les chunks deviennent invisibles pour la recherche, exactement comme lors
    d'une ré-ingestion. Renvoie le nombre de chunks désindexés.
    """
    ids = [c["@id"] for c in all_chunks(source=source) if c.get("@id")]
    if not ids:
        return 0
    return _mark_obsolete_ids(ids)


def deindex_all() -> int:
    """Désindexe tout le contenu du ledger (réinitialisation de la base).

    Les chunks sont marqués obsolètes mais le ledger n'est **pas** supprimé :
    Fluree conserve un « tombeau » après un `drop` (toute transaction ultérieure
    échoue avec « Ledger has been retracted »), et l'historique d'audit reste
    consultable. Renvoie le nombre de chunks désindexés.
    """
    ids = [c["@id"] for c in all_chunks() if c.get("@id")]
    if not ids:
        return 0
    return _mark_obsolete_ids(ids)


def _rows(out, path: str) -> list[dict]:
    """Normalise les différentes formes de réponse d'une requête Fluree."""
    if isinstance(out, list):
        return out
    if isinstance(out, dict):
        for key in ("results", "result", "bindings", "rows"):
            if key in out:
                return out[key]
    raise RuntimeError(f"Réponse Fluree inattendue ({path}) : {str(out)[:300]}")


def all_chunks(source: str | None = None) -> list[dict]:
    """Renvoie les chunks actifs du graphe (contenu + mentions).

    Le filtre *source* reste appliqué côté serveur (utilisé par l'ingestion
    pour retrouver les chunks d'un document). Le filtrage par source/entité de
    la *recherche* est fait localement par le retriever sur un index unique du
    ledger : indexer le corpus une fois par combinaison de filtres coûtait une
    matrice complète par filtre.
    """
    where = {"@id": "?c", "@type": "rag:Chunk", "rag:obsolete": False}
    if source:
        where["rag:source"] = source
    payload = {
        "@context": config.CTX,
        "select": {"?c": ["*"]},
        "where": where,
    }
    return _rows(_post(f"/query/{config.LEDGER}", payload, timeout=120), "query")


def source_names() -> list[str]:
    """Renvoie les sources actives du graphe, sans charger les embeddings."""
    payload = {
        "@context": config.CTX,
        "select": {"?c": ["rag:source"]},
        "where": {"@id": "?c", "@type": "rag:Chunk", "rag:obsolete": False},
    }
    rows = _rows(_post(f"/query/{config.LEDGER}", payload, timeout=120), "query")
    return sorted({row["rag:source"] for row in rows if row.get("rag:source")})


def active_entity_names() -> dict[str, str]:
    """Noms des entités référencées par des chunks *actifs*.

    La requête part des chunks non obsolètes : les entités mentionnées
    uniquement par une version périmée d'un document ne polluent plus la liste.
    La clé est l'identifiant renvoyé par Fluree (`ent:<slug>` ou IRI) ; le slug
    est aussi indexé pour que l'appelant retrouve l'entité quelle que soit la
    forme utilisée.
    """
    payload = {
        "@context": config.CTX,
        "select": {"?e": ["ent:name"]},
        "where": {
            "@id": "?c",
            "@type": "rag:Chunk",
            "rag:obsolete": False,
            "rag:mentions": {"@id": "?e"},
        },
    }
    names: dict[str, str] = {}
    for row in _rows(_post(f"/query/{config.LEDGER}", payload, timeout=60), "query"):
        if not isinstance(row, dict):
            continue
        ent_id = row.get("@id")
        name = row.get("ent:name") or row.get(f"{config.ENT_NS}name")
        if not ent_id or not name:
            continue
        names[ent_id] = name
        names.setdefault(f"ent:{entity_slug(str(name))}", name)
    return names


def history(limit: int = 100) -> list[dict]:
    """Renvoie l'historique des transactions du ledger (API `/log` de Fluree).

    Fluree renvoie `{"ledger_id": ..., "commits": [...], "count": ...}` ; on
    n'expose que la liste des commits. Une liste vide est renvoyée si l'API
    d'historique est indisponible (le reste de l'application continue).
    """
    try:
        resp = _client().get(
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
