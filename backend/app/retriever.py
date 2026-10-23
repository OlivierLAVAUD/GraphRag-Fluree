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
import heapq
import json
import logging

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
    les lignes de X. Chaque colonne est normalisée pour que le produit
    matriciel donne directement la similarité cosinus.
    """
    metas: list[dict] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        embedding = _coerce_embedding(_get(row, "embedding"))
        if embedding is None:
            continue
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if arr.size == 0:
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
    dim = vectors[0].size
    matrix = np.zeros((len(vectors), dim), dtype=np.float32)
    for i, v in enumerate(vectors):
        matrix[i] = v
    return matrix, metas


def score_rows(
    query_embedding: list[float],
    rows: list[dict] | None = None,
    top_k: int = 7,
    min_score: float | None = None,
    matrix: np.ndarray | None = None,
    metas: list[dict] | None = None,
) -> list[dict]:
    """Calcule les similarités cosinus et renvoie les top_k chunks (dédoublonnés).

    Si *matrix* est fourni, la recherche s'effectue par produit matriciel
    (boucle vectorisée) — l'implémentation pensée pour être remplacée par la
    recherche vectorielle native de Fluree sans modifier le reste du pipeline.

    Si *matrix* est fourni, le paramètre *metas* doit être la liste des
    métadonnées (content, source, index) des chunks, dans le même ordre que
    les lignes de la matrice. Si *metas* est None, le calcul est fait ligne par
    ligne (ancien comportement) en utilisant *rows*.

    Si min_score est fourni, les chunks dont le score est strictement inférieur
    sont écartés avant la sélection (anti-bruit).
    """
    q = _normalize(query_embedding)
    scored: list[tuple[float, dict]] = []
    seen: set[str] = set()

    if matrix is not None and matrix.shape[0] > 0:
        sims = matrix @ q
        for i, sim in enumerate(sims):
            if metas is None or i >= len(metas):
                continue
            meta = metas[i]
            content = meta.get("content") or ""
            if not content:
                continue
            fingerprint = content[:120]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            scored.append((float(sim), meta))
    else:
        for row in rows or []:
            content = _get(row, "content")
            embedding = _coerce_embedding(_get(row, "embedding"))
            if not content or embedding is None:
                continue
            fingerprint = content[:120]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            score = float(np.dot(q, _normalize(embedding)))
            scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    if min_score is not None:
        scored = [item for item in scored if item[0] >= min_score]
    return [
        {
            "content": _get(s[1], "content"),
            "source": _get(s[1], "source"),
            "index": _get(s[1], "index"),
            "score": round(s[0], 4),
        }
        for s in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# Cache de l'index local (matrice d'embeddings + métadonnées)
# ---------------------------------------------------------------------------
_index_cache: dict | None = None


def invalidate_index() -> None:
    """Marque l'index comme obsolète (appelé après une insertion de chunks)."""
    global _index_cache
    _index_cache = None


def build_index(rows: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Construit (ou réutilise depuis le cache) la matrice d'embeddings.

    Résultat : (X, metas) — X est (n_chunks, dim), metas est la liste des
    métadonnées (content, source, index) des chunks, dans le même ordre que
    les lignes de X. Chaque colonne est normalisée pour que le produit
    matriciel donne directement la similarité cosinus.

    Si _index_cache est utilisé, le résultat est réutilisé sans reconstruire.
    """
    global _index_cache
    if _index_cache is None:
        _index_cache = _build_index(rows)
    return _index_cache


def retrieve(question: str, top_k: int | None = None, matrix: np.ndarray | None = None, metas: list[dict] | None = None) -> list[dict]:
    """Pipeline complet : embed de la question, récupération des chunks,
    filtrage par seuil puis top_k.

    Si *matrix* est fourni, la recherche s'effectue sur la matrice pré-formatée
    (index local) plutôt que sur une récupération ligne à ligne. Cela permet de
    préparer l'extension vers la recherche vectorielle native de Fluree sans
    changer l'API.

    Le paramètre *metas* doit être la liste des métadonnées (content, source,
    index) correspondantes à la matrice. Si None, le calcul est fait ligne par
    ligne avec les rows retournées par fluree.all_chunks().

    Par défaut, l'index est construit à la volée (compatibilité avec les tests
    qui monkeypatchent fluree.all_chunks). Pour utiliser le cache, appelez
    build_index() explicitement puis passez (matrix, metas) à retrieve().
    """
    top_k = top_k or config.TOP_K
    rows = fluree.all_chunks()
    if not rows:
        return []
    query_embedding = ollama.embed_one(question)

    # Utiliser le cache d'index si disponible et à jour
    if matrix is None or matrix.shape[0] == 0:
        cached = build_index(rows)
        if cached is not None:
            matrix, metas = cached

    if matrix is not None and matrix.shape[0] > 0 and metas is not None:
        hits = score_rows(
            query_embedding, None, top_k, min_score=config.RETRIEVAL_MIN_SCORE,
            matrix=matrix, metas=metas
        )
    else:
        hits = score_rows(
            query_embedding, rows, top_k, min_score=config.RETRIEVAL_MIN_SCORE
        )

    log.info(
        "Retrieval : %d candidats, %d retenus (seuil %.2f)",
        len(rows),
        len(hits),
        config.RETRIEVAL_MIN_SCORE,
    )
    return hits
