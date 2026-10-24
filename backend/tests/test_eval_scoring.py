from eval_rag import is_absence_answer, score_overlap, tokenize


def test_tokenize_filters_stopwords():
    assert "dans" not in tokenize("dans la maison verte")
    assert "maison" in tokenize("dans la maison verte")
    assert "verte" in tokenize("dans la maison verte")
    assert "mistral" in tokenize("Mistral Medium 3.5")
    assert "medium" in tokenize("Mistral Medium 3.5")


def test_score_overlap_perfect_and_null():
    assert (
        score_overlap("Le chat dort sur le canapé.", "Le chat dort sur le canapé.")
        == 1.0
    )
    assert score_overlap("Bonjour le monde", "zzz yyy xxx") == 0.0


def test_score_overlap_partial():
    s = score_overlap(
        "Un million d'agents pour environ 700 000 euros",
        "Environ un million d'agents, pour un coût évalué entre 700 000 et 750 000 euros",
    )
    assert 0.0 < s <= 1.0


def test_absence_detection():
    assert is_absence_answer(
        "Je ne dispose pas de cette information dans ma base de connaissances."
    )
    assert is_absence_answer("L'information est absente du contexte fourni.")
    assert is_absence_answer("Je ne sais pas.")
    assert not is_absence_answer(
        "Le déploiement concerne un million d'agents pour 700 000 euros."
    )
    assert not is_absence_answer("")


def test_score_semantic_mocked(monkeypatch):
    """score_semantic retourne le cosinus entre deux vecteurs d'embedding."""
    import eval_rag

    def fake_embed_batch(texts):
        # Deux vecteurs orthogonaux pour "réponse", identiques pour "attendu"
        return [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    monkeypatch.setattr(eval_rag.ollama, "embed_batch", fake_embed_batch)
    score, hits = eval_rag.score_semantic("réponse test", "attendu test")
    assert score == 1.0
    assert hits == []


def test_score_semantic_orthogonal(monkeypatch):
    """score_semantic retourne 0.0 pour deux vecteurs orthogonaux."""
    import eval_rag

    def fake_embed_batch(texts):
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(eval_rag.ollama, "embed_batch", fake_embed_batch)
    score, hits = eval_rag.score_semantic("a", "b")
    assert score == 0.0


def test_score_semantic_fallback(monkeypatch):
    """score_semantic bascule sur le scoring lexical si Ollama échoue."""
    import eval_rag

    def fake_embed_batch(texts):
        raise RuntimeError("Ollama down")

    monkeypatch.setattr(eval_rag.ollama, "embed_batch", fake_embed_batch)
    score, hits = eval_rag.score_semantic("Le chat dort", "Le chat dort")
    assert score == 1.0  # recouvrement lexical parfait
