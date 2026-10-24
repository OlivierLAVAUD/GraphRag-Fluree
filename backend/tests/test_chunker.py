from app.ingester import chunk_text, prepare_chunks


def test_empty_and_whitespace():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_single_chunk():
    texts = chunk_text("Bonjour. Ceci est un test. Merci.", size=500, overlap=50)
    assert texts == ["Bonjour. Ceci est un test. Merci."]


def test_chunks_respect_size_and_overlap():
    sentence = "Ceci est une phrase de test qui se termine. "
    text = sentence * 40
    chunks = chunk_text(text, size=200, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 250 for c in chunks)
    # le premier chunk est au début du texte
    assert text.startswith(chunks[0])


def test_overlap_contains_tail_of_previous():
    sentence = "Phrase numéro un. "
    text = sentence * 30
    chunks = chunk_text(text, size=120, overlap=40)
    # le second chunk doit reprendre une partie (au moins une phrase) du premier
    first_tail = chunks[0].split(". ")[-1]
    assert first_tail in chunks[1]


def test_prepare_chunks_generates_stable_ids():
    a = prepare_chunks("Un document. Une phrase.", "doc.md")
    b = prepare_chunks("Un document. Une phrase.", "doc.md")
    c = prepare_chunks("Autre contenu. Autre phrase.", "doc.md")
    assert [x["chunk_id"] for x in a] == [x["chunk_id"] for x in b]
    assert [x["chunk_id"] for x in a] != [x["chunk_id"] for x in c]
    assert all("embedding" not in x for x in a)
