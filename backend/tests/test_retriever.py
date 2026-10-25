from app import config, fluree, ollama_client as ollama, retriever
from app.retriever import _coerce_embedding, _get, score_rows


def test_coerce_embedding_handles_list_and_list_and_string():
    assert _coerce_embedding([0.1, 0.2]) == [0.1, 0.2]
    assert _coerce_embedding({"@list": [0.5, 0.1, 0.9]}) == [0.5, 0.1, 0.9]
    assert _coerce_embedding("[0.3, 0.4]") == [0.3, 0.4]
    assert _coerce_embedding("pas un json") is None
    assert _coerce_embedding(None) is None


def _row(content, embedding, source="doc.md", index=0, mentions=None):
    return {
        "@id": "rag:s123",
        "rag:content": content,
        "rag:embedding": embedding,
        "rag:source": source,
        "rag:chunkIndex": index,
        "rag:mentions": mentions or [],
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


def test_entity_names_come_from_graph_mentions():
    """Les entités viennent de `rag:mentions`, sans ré-extraction du contenu."""
    row = _row(
        "Un texte sans nom propre exploitable.",
        [1.0, 0.0],
        mentions=[
            {"@id": "ent:palantir", "ent:name": "Palantir"},
            {"@id": "ent:alice_bob"},  # pas de nom : repli sur le slug
            {"ent:name": "Palantir"},  # doublon -> écarté
        ],
    )
    assert retriever._entity_names(row) == ["palantir", "alice bob"]


def test_entity_bonus_only_uses_graph_mentions():
    """Un nom propre présent dans le texte ne suffit plus : seul le graphe compte."""
    metas = [
        {"content": "Copilot Studio est intégré.", "source": "d.md", "index": 0},
        {
            "content": "Un extrait quelconque.",
            "source": "d.md",
            "index": 1,
            "entities": ["palantir"],
        },
    ]
    hits = [
        {"content": metas[0]["content"], "source": "d.md", "index": 0, "score": 0.4},
        {"content": metas[1]["content"], "source": "d.md", "index": 1, "score": 0.4},
    ]
    retriever._apply_entity_bonus(hits, metas, "Palantir et Copilot Studio ?")
    assert hits[0]["score"] == 0.4  # aucune mention enregistrée
    assert hits[1]["score"] == 0.5  # une entité du graphe citée


def test_bonus_can_promote_chunk_outside_semantic_top_k(monkeypatch):
    """Le bonus entité doit s'appliquer avant la troncature au top_k."""
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    query = [1.0, 0.0]
    rows = [
        _row("haut", [0.20, 0.98]),
        _row("milieu", [0.15, 0.99]),
        _row(
            "Palantir a été remplacé par ChapsVision.",
            [0.14, 0.99],
            mentions=[{"ent:name": "Palantir"}, {"ent:name": "ChapsVision"}],
        ),
    ]
    monkeypatch.setattr(fluree, "all_chunks", lambda source=None: rows)
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

    def fake_all_chunks(source=None):
        calls["count"] += 1
        return [row]

    monkeypatch.setattr(fluree, "all_chunks", fake_all_chunks)
    monkeypatch.setattr(ollama, "embed_one", lambda text: [1.0, 0.0])

    retriever.retrieve("question")
    assert calls["count"] == 1
    assert retriever._load_disk_index(retriever._INDEX_KEY, retriever._index_version())

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


def test_index_cache_invalidated_by_shared_version_file(monkeypatch):
    """Une ingestion dans un autre processus (CLI) doit invalider le cache local."""
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    row = _row("contenu", [1.0, 0.0])
    calls = {"count": 0}

    def fake_all_chunks(source=None):
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


# ---------------------------------------------------------------------------
# Filtres de recherche et budget de cache
# ---------------------------------------------------------------------------


def test_filters_reuse_the_single_ledger_index(monkeypatch):
    """`source`/`entity` filtrent les métadonnées au lieu de réindexer le corpus."""
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    rows = [
        _row(
            "Alice & Bob calcul quantique",
            [1.0, 0.0],
            source="a.md",
            index=0,
            mentions=[{"ent:name": "Alice & Bob"}],
        ),
        _row(
            "Palantir et ChapsVision",
            [0.9, 0.1],
            source="b.md",
            index=1,
            mentions=[{"ent:name": "Palantir"}],
        ),
        _row("Un autre extrait de b.md", [0.8, 0.2], source="b.md", index=2),
    ]
    calls = {"count": 0}

    def fake_all_chunks(source=None):
        calls["count"] += 1
        assert source is None  # le corpus complet est récupéré une seule fois
        return rows

    monkeypatch.setattr(fluree, "all_chunks", fake_all_chunks)
    monkeypatch.setattr(ollama, "embed_one", lambda text: [1.0, 0.0])

    assert len(retriever.retrieve("question", source="b.md")) == 2
    assert len(retriever.retrieve("question", entity="Palantir")) == 1
    assert len(retriever.retrieve("question", source="a.md", entity="Alice & Bob")) == 1
    assert len(retriever.retrieve("question", source="inexistant.md")) == 0
    assert calls["count"] == 1


def test_entity_filter_uses_graph_mentions(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    rows = [
        _row("texte sans mention", [1.0, 0.0], index=0),
        _row(
            "texte mentionnant",
            [0.5, 0.5],
            index=1,
            mentions=[{"ent:name": "Quandela"}],
        ),
    ]
    monkeypatch.setattr(fluree, "all_chunks", lambda source=None: rows)
    monkeypatch.setattr(ollama, "embed_one", lambda text: [1.0, 0.0])

    hits = retriever.retrieve("question", entity="quandela")
    assert [h["content"] for h in hits] == ["texte mentionnant"]


def test_index_cache_respects_memory_budget(monkeypatch):
    """Avec un budget nul, l'index n'est pas gardé en mémoire (pas de dérive RSS)."""
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 0.0)
    monkeypatch.setattr(config, "INDEX_CACHE_MAX_MB", 0)
    monkeypatch.setattr(config, "INDEX_CACHE_MAX_ENTRIES", 0)  # ni cache disque
    calls = {"count": 0}

    def fake_all_chunks(source=None):
        calls["count"] += 1
        return [_row("contenu", [1.0, 0.0])]

    monkeypatch.setattr(fluree, "all_chunks", fake_all_chunks)
    monkeypatch.setattr(ollama, "embed_one", lambda text: [1.0, 0.0])

    retriever.retrieve("question")
    assert retriever._index_cache == {}
    retriever.retrieve("question")
    assert calls["count"] == 2  # reconstruit à chaque fois


def test_disk_cache_rejects_inconsistent_files(monkeypatch):
    """Un cache disque dont la matrice et les métadonnées divergent est ignoré."""
    key = retriever._INDEX_KEY
    version = "v-test"
    retriever._save_disk_index(key, version, _matrix(3), [{"content": "a"}])
    assert retriever._load_disk_index(key, version) is None


def _matrix(rows: int):
    import numpy as np

    return np.zeros((rows, 2), dtype=np.float32)
