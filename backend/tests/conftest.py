"""Fixtures partagées : isolation des caches d'index et pile RAG simulée.

La pile simulée (Fluree en mémoire + embeddings déterministes) est définie ici
plutôt que dans un module de test : les autres tests l'obtiennent par fixture,
sans `import` fragile dépendant du `PYTHONPATH`/rootdir de pytest.

Les embeddings simulés sont des n-grammes de caractères + tokens + paires de
tokens, pour un contrôle thématique proche d'un vrai modèle.
"""

import hashlib
import re

import pytest

from app import config, fluree, ollama_client as ollama, retriever

_DIM = 512
_STOP = frozenset(
    "le la les de des du et en un une pour avec est sont au aux qui que se sa son ses "
    "ce cette il elle ils elles dans par sur plus comme".split()
)


def _hash_idx(prefix: str, token: str) -> int:
    return int(hashlib.md5((prefix + token).encode()).hexdigest()[:8], 16) % _DIM


def fake_embed(text: str) -> list[float]:
    vec = [0.0] * _DIM
    t = text.lower()
    # n-grammes de caractères : thème approximatif
    for i in range(len(t) - 2):
        vec[_hash_idx("", t[i : i + 3])] += 1.0
    # tokens (entités, vocabulaire thématique)
    tokens = [
        tok
        for tok in re.findall(r"[a-zà-ÿ0-9]+", t)
        if tok not in _STOP and len(tok) >= 3
    ]
    for tok in tokens:
        vec[_hash_idx("t", tok)] += 4.0
    # paires de tokens consécutifs : expressions-clés (« calcul quantique », …)
    for a, b in zip(tokens, tokens[1:]):
        vec[_hash_idx("p", f"{a} {b}")] += 3.0
    return vec


def fake_embed_batch(texts: list[str]) -> list[list[float]]:
    return [fake_embed(t) for t in texts]


class FakeFluree:
    """Remplacement de app.fluree : garde les chunks en mémoire."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def ensure_ledger(self) -> str:
        return "rag/smoke"

    def insert_chunks(self, chunks: list[dict]) -> dict:
        for c in chunks:
            self.rows.append(
                {
                    "@id": f"rag:{c['chunk_id']}",
                    "@type": "rag:Chunk",
                    "rag:content": c["content"],
                    "rag:embedding": c["embedding"],
                    "rag:source": c["source"],
                    "rag:chunkIndex": c["index"],
                    "rag:obsolete": False,
                    "rag:mentions": [
                        {
                            "@id": f"ent:{fluree.entity_slug(e['name'])}",
                            "ent:name": e["name"],
                        }
                        for e in c.get("entities", [])
                    ],
                }
            )
        return {"ok": True}

    def mark_obsolete(self, source: str, keep_ids: set | None = None) -> dict:
        # Même normalisation que fluree.mark_obsolete : les keep_ids sont des
        # chunk_id nus alors que les lignes portent un @id préfixé.
        keep = {fluree.chunk_local_id(value) for value in (keep_ids or set())}
        for r in self.rows:
            if (
                r.get("rag:source") == source
                and fluree.chunk_local_id(r.get("@id")) not in keep
            ):
                r["rag:obsolete"] = True
        return {"ok": True}

    def mark_ids_obsolete(self, chunk_ids) -> dict:
        targets = {fluree.chunk_local_id(cid) for cid in chunk_ids}
        for r in self.rows:
            if fluree.chunk_local_id(r.get("@id")) in targets:
                r["rag:obsolete"] = True
        return {"ok": True}

    def all_chunks(self, source: str | None = None) -> list[dict]:
        rows = [r for r in self.rows if not r.get("rag:obsolete", False)]
        if source:
            rows = [r for r in rows if r.get("rag:source") == source]
        return rows

    def deindex_source(self, source: str) -> int:
        targets = {r["@id"] for r in self.all_chunks(source=source)}
        for row in self.rows:
            if row["@id"] in targets:
                row["rag:obsolete"] = True
        return len(targets)

    def deindex_all(self) -> int:
        targets = {r["@id"] for r in self.all_chunks()}
        for row in self.rows:
            if row["@id"] in targets:
                row["rag:obsolete"] = True
        return len(targets)

    def source_names(self) -> list[str]:
        return sorted(
            {r["rag:source"] for r in self.all_chunks() if r.get("rag:source")}
        )


@pytest.fixture(autouse=True)
def isolate_index_caches(tmp_path, monkeypatch):
    """Isole le marqueur de version et le cache disque entre tests.

    Ces deux réglages sont globaux au processus : sans isolation, un test
    pourrait réutiliser l'index d'un autre test (ou d'une exécution précédente).
    """
    monkeypatch.setattr(config, "INDEX_VERSION_FILE", str(tmp_path / "index.version"))
    monkeypatch.setattr(config, "INDEX_CACHE_DIR", str(tmp_path / "index-cache"))
    retriever.invalidate_index()
    yield
    retriever.invalidate_index()


@pytest.fixture
def fake_stack(monkeypatch):
    """Pile RAG simulée : Fluree en mémoire, embeddings déterministes."""
    store = FakeFluree()
    monkeypatch.setattr(fluree, "ensure_ledger", store.ensure_ledger)
    monkeypatch.setattr(fluree, "insert_chunks", store.insert_chunks)
    monkeypatch.setattr(fluree, "all_chunks", store.all_chunks)
    monkeypatch.setattr(fluree, "mark_obsolete", store.mark_obsolete)
    monkeypatch.setattr(fluree, "mark_ids_obsolete", store.mark_ids_obsolete)
    monkeypatch.setattr(fluree, "source_names", store.source_names)
    monkeypatch.setattr(fluree, "deindex_source", store.deindex_source)
    monkeypatch.setattr(fluree, "deindex_all", store.deindex_all)
    monkeypatch.setattr(ollama, "embed_batch", fake_embed_batch)
    monkeypatch.setattr(ollama, "embed_one", lambda t: fake_embed(t))
    # Les embeddings simulés ont une échelle de similarité inférieure aux vrais
    # (nomic-embed-text) : le seuil RETRIEVAL_MIN_SCORE est désactivé ici, sa
    # logique étant testée à part (test_min_score_filters_low_similarity).
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    return store
