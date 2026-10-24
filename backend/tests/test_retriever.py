from app import config, fluree, ollama_client as ollama, retriever
from app.retriever import _coerce_embedding, _get, score_rows


def test_coerce_embedding_handles_list_and_list_and_string():
    assert _coerce_embedding([0.1, 0.2]) == [0.1, 0.2]
    assert _coerce_embedding({"@list": [0.5, 0.1, 0.9]}) == [0.5, 0.1, 0.9]
    assert _coerce_embedding("[0.3, 0.4]") == [0.3, 0.4]
    assert _coerce_embedding("pas un json") is None
    assert _coerce_embedding(None) is None


def _row(content, embedding, source="doc.md", index=0):
    return {
        "@id": "rag:s123",
        "rag:content": content,
        "rag:embedding": embedding,
        "rag:source": source,
        "rag:chunkIndex": index,
    }


def test_get_supports_short_and_full_iris():
    short = {"rag:content": "c"}
    full = {"https://ns.flurag.dev/content": "c"}
    assert _get(short, "content") == "c"
    assert _get(full, "content") == "c"
    assert _get({}, "content") is None


def test_score_ranks_most_similar_first():
    query = [1.0, 0.0, 0.0]
    rows = [
        _row("orthogonal", [0.0, 1.0, 0.0]),
        _row("identical", [1.0, 0.0, 0.0]),
        _row("opposite", [-1.0, 0.0, 0.0]),
    ]
    hits = score_rows(query, rows, top_k=3)
    assert [h["content"] for h in hits] == ["identical", "orthogonal", "opposite"]
    assert hits[0]["score"] == 1.0
    assert hits[0]["source"] == "doc.md"


def test_top_k_and_dedup():
    query = [1.0, 0.0]
    rows = [
        _row("a" * 130 + " unique", [1.0, 0.0]),
        _row("a" * 130 + " unique", [1.0, 0.0]),  # doublon exact -> écarté
        # Même préfixe mais contenu différent : ne doit PAS être dédoublonné.
        _row("a" * 130 + " distinct", [0.95, 0.1]),
        _row("b", [0.9, 0.1]),
    ]
    hits = score_rows(query, rows, top_k=3)
    assert len(hits) == 3
    assert sum(h["content"].startswith("a" * 130) for h in hits) == 2


def test_bonus_can_promote_chunk_outside_semantic_top_k(monkeypatch):
    """Le bonus entité doit s'appliquer avant la troncature au top_k."""
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    query = [1.0, 0.0]
    rows = [
        _row("haut", [0.20, 0.98]),
        _row("milieu", [0.15, 0.99]),
        _row("Palantir a été remplacé par ChapsVision.", [0.14, 0.99]),
    ]
    monkeypatch.setattr(fluree, "all_chunks", lambda source=None, entity=None: rows)
    monkeypatch.setattr(ollama, "embed_one", lambda text: query)

    hits = retriever.retrieve("Palantir a-t-il été remplacé ?", top_k=2)
    assert any("Palantir" in h["content"] for h in hits)
    assert all(h["score"] <= 1.0 for h in hits)


def test_mixed_embedding_dimensions_are_ignored():
    """Un vecteur de dimension différente ne doit pas faire planter l'index."""
    rows = [
        _row("deux dimensions", [1.0, 0.0]),
        _row("trois dimensions", [1.0, 0.0, 0.0]),
    ]
    matrix, metas = retriever._build_index(rows)
    assert matrix.shape == (1, 2)
    assert [m["content"] for m in metas] == ["deux dimensions"]


def test_disk_cache_avoids_refetching_from_fluree(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    row = _row("contenu", [1.0, 0.0])
    calls = {"count": 0}

    def fake_all_chunks(source=None, entity=None):
        calls["count"] += 1
        return [row]

    monkeypatch.setattr(fluree, "all_chunks", fake_all_chunks)
    monkeypatch.setattr(ollama, "embed_one", lambda text: [1.0, 0.0])

    retriever.retrieve("question")
    assert calls["count"] == 1
    assert retriever._load_disk_index((None, None), retriever._index_version())

    # Simule un redémarrage : cache mémoire vide, cache disque conservé.
    retriever._index_cache.clear()
    retriever.retrieve("question")
    assert calls["count"] == 1


def test_min_score_filters_low_similarity():
    query = [1.0, 0.0]
    rows = [
        _row("fort", [1.0, 0.0]),  # cosinus 1.0
        _row("moyen", [0.3, 0.7]),  # cosinus 0.3 < 0.6 (filtré)
        _row("faible", [0.0, 1.0]),  # cosinus 0.0 (filtré)
    ]
    hits = score_rows(query, rows, top_k=5, min_score=0.6)
    assert len(hits) == 1
    assert hits[0]["content"] == "fort"
    # sans seuil, les 3 reviennent
    assert len(score_rows(query, rows, top_k=5)) == 3


def test_index_cache_invalidated_by_shared_version_file(monkeypatch, tmp_path):
    """Une ingestion dans un autre processus (CLI) doit invalider le cache local."""
    monkeypatch.setattr(config, "INDEX_VERSION_FILE", str(tmp_path / "index.version"))
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    row = _row("contenu", [1.0, 0.0])
    calls = {"count": 0}

    def fake_all_chunks(source=None, entity=None):
        calls["count"] += 1
        return [row]

    monkeypatch.setattr(fluree, "all_chunks", fake_all_chunks)
    monkeypatch.setattr(ollama, "embed_one", lambda text: [1.0, 0.0])

    retriever.invalidate_index()
    retriever.retrieve("question")
    retriever.retrieve("question")
    assert calls["count"] == 1  # le second appel utilise le cache

    # Simule l'ingestion CLI : elle met à jour le marqueur partagé.
    retriever.invalidate_index()
    retriever.retrieve("question")
    assert calls["count"] == 2  # le cache a bien été reconstruit


def test_missing_fields_are_skipped():
    rows = [
        {"@id": "rag:x", "rag:embedding": [1.0, 0.0]},  # pas de contenu
        {"@id": "rag:y", "rag:content": "ok", "rag:embedding": None},  # pas d'embedding
        _row("bon", [1.0, 0.0]),
    ]
    hits = score_rows([1.0, 0.0], rows, top_k=5)
    assert len(hits) == 1
    assert hits[0]["content"] == "bon"
