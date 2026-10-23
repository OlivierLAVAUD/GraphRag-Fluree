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
        _row("a" * 130 + " doublon", [1.0, 0.0]),  # même début -> dédoublonné
        _row("b", [0.9, 0.1]),
        _row("c", [0.8, 0.2]),
    ]
    hits = score_rows(query, rows, top_k=2)
    assert len(hits) == 2
    assert all(h["content"] != ("a" * 130 + " doublon") for h in hits)


def test_min_score_filters_low_similarity():
    query = [1.0, 0.0]
    rows = [
        _row("fort", [1.0, 0.0]),        # cosinus 1.0
        _row("moyen", [0.3, 0.7]),       # cosinus 0.3 < 0.6 (filtré)
        _row("faible", [0.0, 1.0]),      # cosinus 0.0 (filtré)
    ]
    hits = score_rows(query, rows, top_k=5, min_score=0.6)
    assert len(hits) == 1
    assert hits[0]["content"] == "fort"
    # sans seuil, les 3 reviennent
    assert len(score_rows(query, rows, top_k=5)) == 3


def test_missing_fields_are_skipped():
    rows = [
        {"@id": "rag:x", "rag:embedding": [1.0, 0.0]},      # pas de contenu
        {"@id": "rag:y", "rag:content": "ok", "rag:embedding": None},  # pas d'embedding
        _row("bon", [1.0, 0.0]),
    ]
    hits = score_rows([1.0, 0.0], rows, top_k=5)
    assert len(hits) == 1
    assert hits[0]["content"] == "bon"