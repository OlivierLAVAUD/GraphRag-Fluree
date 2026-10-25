"""Recherche sémantique : similarité cosinus entre la question et les chunks
stockés dans le graphe Fluree, avec un index local optimisé.

- score_rows() accepte une matrice pré-formatée pour la recherche vectorielle
  (produit matriciel) — le point d'extension vers la recherche vectorielle
  native de Fluree.
- retrieve() est le pipeline complet : embed de la question, récupération des
  chunks, filtrage par seuil puis top_k.
- build_index() / invalidate_index() permettent de mettre en cache l'index
  quand on veut éviter de recalculer la matrice à chaque requête.
"""

import hashlib
import json
import logging
import os
import threading
import uuid

import numpy as np

from . import config, fluree, ollama_client as ollama

log = logging.getLogger(__name__)


def _get(row: dict, key: str):
    """Lit une valeur de prédicat qu'elle soit retournée en forme courte ou IRI,
    ou sous la forme de métadonnées standard (content, source, index)."""
    short = f"rag:{key}"
    if short in row:
        return row[short]
    full = f"{config.RAG_NS}{key}"
    if full in row:
        return row[full]
    # Métadonnées standard (content, source, index)
    if key in row:
        return row[key]
    return None


def _normalize(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _coerce_embedding(value):
    """Dénormalise un embedding selon sa forme de retour (liste, @list ou chaîne)."""
    if isinstance(value, dict):
        return value.get("@list")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return value


def _ent_display_name(mention: dict) -> str:
    """Nom d'une entité mentionnée, ou à défaut son slug lisible.

    Fluree renvoie normalement `ent:name`, mais peut ne renvoyer que la
    référence `ent:<slug>` : on retombe alors sur le slug (« alice bob ») pour
    que la recherche par entité continue de fonctionner.
    """
    name = mention.get("ent:name") or mention.get(f"{config.ENT_NS}name")
    if name:
        return str(name).strip().lower()
    ent_id = mention.get("@id")
    if not isinstance(ent_id, str):
        return ""
    for prefix in (config.ENT_NS, "ent:"):
        if ent_id.startswith(prefix):
            ent_id = ent_id[len(prefix) :]
            break
    return ent_id.replace("_", " ").replace("-", " ").strip().lower()


def _entity_names(row: dict) -> list[str]:
    """Entités mentionnées par un chunk, telles que persistées dans le graphe.

    L'index lit `rag:mentions` au lieu de relancer l'extracteur sur le texte de
    chaque candidat à chaque requête : le graphe reste la source de vérité (une
    amélioration de l'extracteur n'entre plus en contradiction silencieuse avec
    les chunks déjà indexés) et le coût par requête devient négligeable.
    """
    mentions = _get(row, "mentions") or []
    if isinstance(mentions, dict):
        mentions = [mentions]
    names: list[str] = []
    for mention in mentions if isinstance(mentions, list) else []:
        if not isinstance(mention, dict):
            continue
        name = _ent_display_name(mention)
        if name and name not in names:
            names.append(name)
    return names


def _build_index(rows: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Construit une matrice d'embeddings L2-normalisée et les métadonnées.

    Retourne (X, metas) — X est (n_chunks, dim), metas est la liste des
    métadonnées (content, source, index) des chunks, dans le même ordre que
    les lignes de X. Chaque vecteur est normalisé pour que le produit matriciel
    donne directement la similarité cosinus.
    """
    metas: list[dict] = []
    vectors: list[np.ndarray] = []
    dim: int | None = None
    for row in rows:
        embedding = _coerce_embedding(_get(row, "embedding"))
        if embedding is None:
            continue
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        # Un changement de EMBED_MODEL peut laisser des vecteurs de dimensions
        # différentes dans le graphe : on ignore les vecteurs incompatibles au
        # lieu de faire échouer toute la recherche.
        if dim is None:
            dim = arr.size
        elif arr.size != dim:
            log.warning(
                "Embedding de dimension %d ignoré (attendu %d) pour la source %s.",
                arr.size,
                dim,
                _get(row, "source"),
            )
            continue
        vectors.append(_normalize(arr))
        metas.append(
            {
                "content": _get(row, "content") or "",
                "source": _get(row, "source"),
                "index": _get(row, "chunkIndex"),
                "entities": _entity_names(row),
            }
        )
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), metas
    return np.vstack(vectors).astype(np.float32, copy=False), metas


def _fingerprint(content: str) -> str:
    """Empreinte exacte d'un chunk (évite de confondre deux chunks qui
    partagent seulement leurs 120 premiers caractères)."""
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def score_rows(
    query_embedding: list[float],
    rows: list[dict] | None = None,
    top_k: int | None = 7,
    min_score: float | None = None,
    matrix: np.ndarray | None = None,
    metas: list[dict] | None = None,
) -> list[dict]:
    """Calcule les similarités cosinus et renvoie les top_k chunks (dédoublonnés).

    Si *matrix* est fourni, la recherche s'effectue par produit matriciel
    (boucle vectorisée) — l'implémentation pensée pour être remplacée par la
    recherche vectorielle native de Fluree sans modifier le reste du pipeline.

    Si *matrix* est fourni, le paramètre *metas* doit être la liste des
    métadonnées (content, source, index) des chunks, dans le même ordre que les
    lignes de la matrice. Si *metas* est None, le calcul est fait ligne par ligne
    en utilisant *rows*.

    Si min_score est fourni, les chunks dont le score est strictement inférieur
    sont écartés avant la sélection (anti-bruit).
    """
    q = _normalize(query_embedding)
    if matrix is not None and matrix.shape[0] > 0 and matrix.shape[1] != q.size:
        log.warning(
            "Dimension d'embedding incompatible (index %d, question %d) : "
            "ré-indexez les documents après un changement de EMBED_MODEL.",
            matrix.shape[1],
            q.size,
        )
        return []
    scored: list[tuple[float, dict]] = []
    seen: set[str] = set()

    if matrix is not None and matrix.shape[0] > 0:
        sims = matrix @ q
        for i in np.argsort(-sims):
            if metas is None or i >= len(metas):
                continue
            meta = metas[i]
            content = meta.get("content") or ""
            if not content:
                continue
            fingerprint = _fingerprint(content)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            scored.append((float(sims[i]), meta))
    else:
        for row in rows or []:
            content = _get(row, "content")
            embedding = _coerce_embedding(_get(row, "embedding"))
            if not content or embedding is None:
                continue
            fingerprint = _fingerprint(content)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            score = float(np.dot(q, _normalize(embedding)))
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)

    if min_score is not None:
        scored = [item for item in scored if item[0] >= min_score]
    if top_k is None:
        selected = scored
    else:
        selected = scored[:top_k]
    return [
        {
            "content": _get(s[1], "content"),
            "source": _get(s[1], "source"),
            "index": _get(s[1], "index"),
            "score": round(s[0], 4),
        }
        for s in selected
    ]


# ---------------------------------------------------------------------------
# Cache de l'index local (matrice d'embeddings + métadonnées)
# ---------------------------------------------------------------------------
# Un seul index par ledger : les filtres (source, entité) sont appliqués sur
# les métadonnées au moment de la requête, et non plus en indexant le corpus
# une fois par combinaison de filtres.
_INDEX_KEY: tuple[str | None, str | None] = (None, None)
# Version du format des métadonnées : l'incrémenter invalide les caches disque
# écrits par une version antérieure du code.
_CACHE_FORMAT = "v2"

_index_cache: dict[
    tuple[str | None, str | None], tuple[str, np.ndarray, list[dict]]
] = {}
_index_lock = threading.Lock()


def _index_version() -> str:
    """Lit le marqueur partagé écrit à chaque ingestion (backend ou CLI)."""
    try:
        with open(config.INDEX_VERSION_FILE, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _write_index_version() -> None:
    """Signale aux autres processus que les chunks ont changé."""
    try:
        with open(config.INDEX_VERSION_FILE, "w", encoding="utf-8") as fh:
            fh.write(uuid.uuid4().hex)
    except OSError as exc:
        log.warning(
            "Impossible d'écrire le marqueur d'index %s : %s",
            config.INDEX_VERSION_FILE,
            exc,
        )


def invalidate_index() -> None:
    """Marque l'index comme obsolète (appelé après une insertion de chunks).

    Le cache mémoire est vidé et le marqueur partagé est renouvelé : les autres
    processus (ex. CLI `ingest.py` puis backend uvicorn) reconstruiront leur
    index à la prochaine requête, sans redémarrage. Le cache disque est versionné
    par le marqueur, les anciens fichiers deviennent donc inutilisés.
    """
    with _index_lock:
        _index_cache.clear()
    _write_index_version()


def _cache_stem(key: tuple[str | None, str | None], version: str) -> str:
    """Chemin de base du cache disque pour un index (ledger, format, version)."""
    raw = "\0".join((config.LEDGER, _CACHE_FORMAT, key[0] or "", key[1] or "", version))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return os.path.join(config.INDEX_CACHE_DIR, digest)


def _entry_bytes(matrix: np.ndarray, metas: list[dict]) -> int:
    """Taille approximative d'un index en mémoire (matrice + métadonnées)."""
    return int(matrix.nbytes) + len(json.dumps(metas, ensure_ascii=False).encode())


def _load_disk_index(
    key: tuple[str | None, str | None], version: str
) -> tuple[np.ndarray, list[dict]] | None:
    """Recharge un index depuis le disque (None si absent ou corrompu)."""
    stem = _cache_stem(key, version)
    try:
        with np.load(stem + ".npz") as data:
            matrix = data["matrix"]
        with open(stem + ".json", encoding="utf-8") as fh:
            metas = json.load(fh)
        if (
            not isinstance(metas, list)
            or matrix.ndim != 2
            or matrix.shape[0] != len(metas)
            or not all(isinstance(meta, dict) for meta in metas)
        ):
            return None
        return matrix.astype(np.float32, copy=False), metas
    except Exception:  # noqa: BLE001 - cache absent/corrompu = simple miss
        return None


def _prune_disk_cache() -> None:
    """Ne conserve que les INDEX_CACHE_MAX_ENTRIES index les plus récents."""
    try:
        names = os.listdir(config.INDEX_CACHE_DIR)
    except OSError:
        return

    def mtime(stem: str) -> float:
        path = os.path.join(config.INDEX_CACHE_DIR, stem + ".npz")
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    stems = sorted(
        {os.path.splitext(name)[0] for name in names}, key=mtime, reverse=True
    )
    for stem in stems[config.INDEX_CACHE_MAX_ENTRIES :]:
        for ext in (".npz", ".json"):
            try:
                os.remove(os.path.join(config.INDEX_CACHE_DIR, stem + ext))
            except OSError:
                pass


def _save_disk_index(
    key: tuple[str | None, str | None],
    version: str,
    matrix: np.ndarray,
    metas: list[dict],
) -> None:
    """Persiste un index pour éviter un rapatriement complet depuis Fluree."""
    stem = _cache_stem(key, version)
    tmp_npz = f"{stem}.{uuid.uuid4().hex}.tmp.npz"
    tmp_json = f"{stem}.{uuid.uuid4().hex}.tmp.json"
    try:
        os.makedirs(config.INDEX_CACHE_DIR, exist_ok=True)
        np.savez_compressed(tmp_npz, matrix=matrix)
        with open(tmp_json, "w", encoding="utf-8") as fh:
            json.dump(metas, fh, ensure_ascii=False)
        os.replace(tmp_npz, stem + ".npz")
        os.replace(tmp_json, stem + ".json")
        _prune_disk_cache()
    except OSError as exc:
        log.warning("Impossible d'écrire le cache d'index sur disque : %s", exc)
    finally:
        for path in (tmp_npz, tmp_json):
            try:
                os.remove(path)
            except OSError:
                pass


def _store_memory_index(
    key: tuple[str | None, str | None],
    version: str,
    matrix: np.ndarray,
    metas: list[dict],
) -> None:
    """Garde l'index en mémoire si son empreinte respecte le budget configuré.

    Il n'y a qu'un index par ledger : le budget `INDEX_CACHE_MAX_MB` sert de
    garde-fou mémoire (un ledger de plusieurs centaines de milliers de chunks
    ne doit pas faire enfler le processus sans limite). Au-delà, l'index n'est
    pas conservé : chaque requête le reconstruit.
    """
    size = _entry_bytes(matrix, metas)
    budget = config.INDEX_CACHE_MAX_MB * 1024 * 1024
    with _index_lock:
        _index_cache.clear()
        if size > budget:
            log.warning(
                "Index de %.1f Mo au-delà du budget INDEX_CACHE_MAX_MB=%d : "
                "non conservé en mémoire (reconstruction à chaque requête).",
                size / (1024 * 1024),
                config.INDEX_CACHE_MAX_MB,
            )
            return
        _index_cache[key] = (version, matrix, metas)


def build_index(
    rows: list[dict],
    version: str | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Construit (ou réutilise depuis le cache) la matrice d'embeddings.

    L'index est unique pour le ledger : les filtres de recherche sont appliqués
    ensuite sur les métadonnées (`retrieve`). *version* permet à l'appelant de
    fournir la version lue avant la récupération des chunks (évite de mettre en
    cache des données lues avant une ingestion concurrente).

    La construction se fait **hors du verrou** : une reconstruction après
    ingestion ne bloque pas les requêtes concurrentes qui utilisent un index
    déjà en cache.
    """
    if version is None:
        version = _index_version()
    matrix, metas = _build_index(rows)
    # Si une ingestion a eu lieu pendant la construction, on ne met pas en
    # cache un index déjà périmé.
    if _index_version() == version:
        _store_memory_index(_INDEX_KEY, version, matrix, metas)
        _save_disk_index(_INDEX_KEY, version, matrix, metas)
    return matrix, metas


def _full_index() -> tuple[np.ndarray, list[dict]]:
    """Index complet du ledger : cache mémoire, puis cache disque, puis Fluree."""
    version = _index_version()
    with _index_lock:
        cached = _index_cache.get(_INDEX_KEY)
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]
    disk = _load_disk_index(_INDEX_KEY, version)
    if disk is not None:
        matrix, metas = disk
        _store_memory_index(_INDEX_KEY, version, matrix, metas)
        return matrix, metas
    rows = fluree.all_chunks()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    return build_index(rows, version=version)


def _matches_filter(meta: dict, source: str | None, entity: str | None) -> bool:
    """Vrai si le chunk satisfait les filtres de recherche."""
    if source and meta.get("source") != source:
        return False
    if entity:
        wanted = entity.strip().lower()
        if not any(name == wanted for name in meta.get("entities") or ()):
            return False
    return True


def _apply_entity_bonus(hits: list[dict], metas: list[dict], question: str) -> None:
    """Ajoute un bonus aux chunks dont une entité du graphe est citée.

    Les entités viennent des `rag:mentions` persistées à l'ingestion (voir
    `_entity_names`) : plus aucune extraction n'est refaite à la requête, et le
    graphe reste la seule source de vérité. Le bonus total est borné pour ne
    pas dépasser l'échelle du cosinus.
    """
    question_lower = question.lower()
    entities_by_content = {
        meta.get("content") or "": meta.get("entities") or [] for meta in metas
    }
    for hit in hits:
        bonus = 0.0
        for ent_name in entities_by_content.get(hit.get("content") or "", ()):
            if ent_name and ent_name in question_lower:
                bonus += 0.1
        if bonus > 0:
            hit["score"] = round(min(1.0, hit["score"] + min(bonus, 0.3)), 4)


def retrieve(
    question: str,
    top_k: int | None = None,
    matrix: np.ndarray | None = None,
    metas: list[dict] | None = None,
    source: str | None = None,
    entity: str | None = None,
) -> list[dict]:
    """Pipeline complet : embed de la question, filtrage, scoring puis top_k.

    L'index est construit une seule fois pour le ledger (cache mémoire borné par
    `INDEX_CACHE_MAX_MB`, puis cache disque, puis Fluree) et réutilisé pour tous
    les filtres : *source* et *entity* ne déclenchent plus un index dédié, ils
    restreignent simplement les métadonnées avant le scoring.

    Si *matrix* / *metas* sont fournis (tests, pré-calcul), ils remplacent
    l'index automatique. Le cache est invalidé à chaque ingestion, y compris
    depuis un autre processus (CLI `ingest.py`) via un marqueur partagé ; les
    tests peuvent monkeypatcher fluree.all_chunks.
    """
    top_k = max(1, min(int(top_k or config.TOP_K), config.MAX_TOP_K))
    if matrix is None or metas is None:
        matrix, metas = _full_index()
    if matrix is None or matrix.shape[0] == 0 or not metas:
        return []

    candidates = matrix.shape[0]
    if source or entity:
        keep = [
            i for i, meta in enumerate(metas) if _matches_filter(meta, source, entity)
        ]
        if not keep:
            log.info("Retrieval : aucun chunk pour source=%r entity=%r", source, entity)
            return []
        matrix = matrix[keep]
        metas = [metas[i] for i in keep]

    query_embedding = ollama.embed_one(question)

    # On récupère *tous* les candidats au-dessus du seuil : le bonus d'entités
    # doit pouvoir promouvoir un chunk situé hors du top_k sémantique.
    hits = score_rows(
        query_embedding,
        None,
        None,
        min_score=config.RETRIEVAL_MIN_SCORE,
        matrix=matrix,
        metas=metas,
    )

    # Bonus hybride : une entité du graphe citée dans la question valorise le
    # chunk qui la mentionne (donnée lue depuis `rag:mentions`, sans ré-extraction).
    _apply_entity_bonus(hits, metas, question)

    # Re-trier après le bonus, puis tronquer au top_k demandé.
    hits.sort(key=lambda h: h["score"], reverse=True)
    hits = hits[:top_k]

    log.info(
        "Retrieval : %d chunks indexés, %d après filtres, %d retenus (seuil %.2f)",
        candidates,
        matrix.shape[0],
        len(hits),
        config.RETRIEVAL_MIN_SCORE,
    )
    return hits
