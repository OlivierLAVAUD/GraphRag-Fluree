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
# Clé : tuple (source, entity) pour différencier les index filtrés.
# Valeur : (version du marqueur partagé, matrice, métadonnées).
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
    """Chemin de base du cache disque pour un index (ledger, filtres, version)."""
    raw = "\0".join((config.LEDGER, key[0] or "", key[1] or "", version))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return os.path.join(config.INDEX_CACHE_DIR, digest)


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
    stems = sorted({os.path.splitext(name)[0] for name in names})
    stems.sort(
        key=lambda stem: (
            os.path.getmtime(os.path.join(config.INDEX_CACHE_DIR, stem + ".npz"))
            if os.path.exists(os.path.join(config.INDEX_CACHE_DIR, stem + ".npz"))
            else 0
        ),
        reverse=True,
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
    """Stocke l'index en mémoire en bornant le nombre d'entrées."""
    with _index_lock:
        _index_cache[key] = (version, matrix, metas)
        while len(_index_cache) > config.INDEX_CACHE_MAX_ENTRIES:
            _index_cache.pop(next(iter(_index_cache)))


def build_index(
    rows: list[dict],
    source: str | None = None,
    entity: str | None = None,
    version: str | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Construit (ou réutilise depuis le cache) la matrice d'embeddings.

    Le cache est indexé par (source, entity) pour éviter les collisions et
    invalidé dès que le marqueur partagé change. *version* permet à l'appelant
    de fournir la version lue avant la récupération des chunks (évite de mettre
    en cache des données lues avant une ingestion concurrente).

    La construction se fait **hors du verrou** : une reconstruction après
    ingestion ne bloque pas les requêtes concurrentes qui utilisent un index
    déjà en cache.
    """
    key = (source, entity)
    if version is None:
        version = _index_version()
    with _index_lock:
        cached = _index_cache.get(key)
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]
    matrix, metas = _build_index(rows)
    # Si une ingestion a eu lieu pendant la construction, on ne met pas en
    # cache un index déjà périmé.
    if _index_version() == version:
        _store_memory_index(key, version, matrix, metas)
        _save_disk_index(key, version, matrix, metas)
    return matrix, metas


def _extract_entity_names(text: str) -> list[str]:
    """Extrait les noms d'entités d'un texte (réutilise le module entities)."""
    try:
        from . import entities

        ents = entities.extract_entities(text)
        return [e["name"] for e in ents]
    except Exception:  # noqa: BLE001
        return []


def retrieve(
    question: str,
    top_k: int | None = None,
    matrix: np.ndarray | None = None,
    metas: list[dict] | None = None,
    source: str | None = None,
    entity: str | None = None,
) -> list[dict]:
    """Pipeline complet : embed de la question, récupération des chunks,
    filtrage par seuil puis top_k.

    Si *matrix* est fourni, la recherche s'effectue sur la matrice pré-formatée
    (index local) plutôt que sur une récupération ligne à ligne. Cela permet de
    préparer l'extension vers la recherche vectorielle native de Fluree sans
    changer l'API.

    Le paramètre *metas* doit être la liste des métadonnées (content, source,
    index) correspondantes à la matrice. Si None, le calcul est fait ligne par
    ligne avec les rows retournées par fluree.all_chunks().

    L'index est construit à la volée s'il n'est pas déjà en cache. Le cache est
    indexé par couple (source, entity) et invalidé à chaque ingestion, y compris
    depuis un autre processus (CLI `ingest.py`) via un marqueur partagé ; les
    tests peuvent monkeypatcher fluree.all_chunks.
    """
    top_k = max(1, min(int(top_k or config.TOP_K), config.MAX_TOP_K))
    key = (source, entity)
    version = _index_version()
    with _index_lock:
        cached = _index_cache.get(key)
    if cached is not None and cached[0] == version:
        _, matrix, metas = cached
        rows = []  # Pas besoin de recharger les rows
    else:
        disk = _load_disk_index(key, version)
        if disk is not None:
            matrix, metas = disk
            _store_memory_index(key, version, matrix, metas)
            rows = []
        else:
            rows = fluree.all_chunks(source=source, entity=entity)
            if not rows:
                return []
            matrix, metas = build_index(
                rows, source=source, entity=entity, version=version
            )

    query_embedding = ollama.embed_one(question)

    # On récupère *tous* les candidats au-dessus du seuil : le bonus d'entités
    # doit pouvoir promouvoir un chunk situé hors du top_k sémantique.
    if matrix is not None and matrix.shape[0] > 0 and metas is not None:
        hits = score_rows(
            query_embedding,
            None,
            None,
            min_score=config.RETRIEVAL_MIN_SCORE,
            matrix=matrix,
            metas=metas,
        )
    else:
        hits = score_rows(
            query_embedding, rows, None, min_score=config.RETRIEVAL_MIN_SCORE
        )

    # Hybrid bonus : si un chunk mentionne une entité présente dans la question,
    # on ajoute un bonus basé sur les relations du graphe. Le bonus total est
    # borné pour ne pas dépasser l'échelle du cosinus.
    question_lower = question.lower()
    for hit in hits:
        bonus = 0.0
        for ent_name in _extract_entity_names(hit.get("content", "")):
            if ent_name.lower() in question_lower:
                bonus += 0.1
        if bonus > 0:
            hit["score"] = round(min(1.0, hit["score"] + min(bonus, 0.3)), 4)

    # Re-trier après le bonus, puis tronquer au top_k demandé.
    hits.sort(key=lambda h: h["score"], reverse=True)
    hits = hits[:top_k]

    candidates = matrix.shape[0] if matrix is not None else len(rows)
    log.info(
        "Retrieval : %d candidats, %d retenus (seuil %.2f)",
        candidates,
        len(hits),
        config.RETRIEVAL_MIN_SCORE,
    )
    return hits
