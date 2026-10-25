import eval_rag
from eval_rag import (
    ABSENCE_SPECULATION_SCORE,
    calibrate_semantic,
    is_absence_answer,
    is_clean_refusal,
    score_overlap,
    tokenize,
)


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


def test_semantic_scores_are_calibrated():
    """Un cosinus de plancher (0,6) vaut 0 : le modèle seul ne fait plus passer."""
    assert calibrate_semantic(0.6) == 0.0
    assert calibrate_semantic(0.4) == 0.0
    assert calibrate_semantic(0.8) == 0.5
    assert calibrate_semantic(0.9) == 0.75
    assert calibrate_semantic(1.0) == 1.0
    assert calibrate_semantic(1.0, floor=0.0) == 1.0


def test_clean_refusal_requires_no_extra_claim():
    assert is_clean_refusal(
        "Je ne dispose pas de cette information dans ma base de connaissances."
    )
    # Refus suivi d'une affirmation non vérifiée : ce n'est pas une réussite.
    assert not is_clean_refusal(
        "Je ne dispose pas de cette information, mais le déploiement a probablement "
        "réduit les effectifs du ministère."
    )
    assert not is_clean_refusal("Le déploiement a réduit les effectifs.")
    assert not is_clean_refusal(
        "Information non disponible : le texte ne mentionne aucun impact sur les "
        "effectifs du ministère de l'Intérieur."
    )


def _patch_llm(monkeypatch, answer: str) -> None:
    monkeypatch.setattr(eval_rag.retriever, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(
        eval_rag.ollama, "chat_stream", lambda messages, on_delta: on_delta(answer)
    )


def test_expected_absence_scores_one_for_clean_refusal(monkeypatch):
    _patch_llm(
        monkeypatch,
        "Je ne dispose pas de cette information dans ma base de connaissances.",
    )
    outcome = eval_rag.evaluate("Question ?", "peu importe", expected_absence=True)
    assert outcome["score"] == 1.0
    assert outcome["absence_clean"] is True


def test_expected_absence_penalizes_speculation(monkeypatch):
    _patch_llm(
        monkeypatch,
        "Je ne dispose pas de cette information, mais probablement une baisse "
        "des effectifs du ministère.",
    )
    outcome = eval_rag.evaluate("Question ?", "peu importe", expected_absence=True)
    assert outcome["score"] == ABSENCE_SPECULATION_SCORE
    assert outcome["absence_clean"] is False


def test_expected_absence_scores_zero_for_hallucination(monkeypatch):
    _patch_llm(monkeypatch, "Le déploiement a supprimé 2 000 postes au ministère.")
    outcome = eval_rag.evaluate("Question ?", "peu importe", expected_absence=True)
    assert outcome["score"] == 0.0


def test_refusal_on_answerable_question_scores_zero(monkeypatch):
    _patch_llm(
        monkeypatch,
        "Je ne dispose pas de cette information dans ma base de connaissances.",
    )
    outcome = eval_rag.evaluate(
        "Question ?", "réponse attendue", expected_absence=False
    )
    assert outcome["score"] == 0.0


def test_score_semantic_mocked(monkeypatch):
    """score_semantic retourne le cosinus calibré entre deux embeddings."""

    def fake_embed_batch(texts):
        return [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    monkeypatch.setattr(eval_rag.ollama, "embed_batch", fake_embed_batch)
    score, hits = eval_rag.score_semantic("réponse test", "attendu test")
    assert score == 1.0
    assert hits == []


def test_score_semantic_orthogonal(monkeypatch):
    """Deux vecteurs orthogonaux donnent 0 (et non le plancher du modèle)."""

    def fake_embed_batch(texts):
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(eval_rag.ollama, "embed_batch", fake_embed_batch)
    score, hits = eval_rag.score_semantic("a", "b")
    assert score == 0.0


def test_score_semantic_fallback(monkeypatch):
    """score_semantic bascule sur le scoring lexical si Ollama échoue."""

    def fake_embed_batch(texts):
        raise RuntimeError("Ollama down")

    monkeypatch.setattr(eval_rag.ollama, "embed_batch", fake_embed_batch)
    score, hits = eval_rag.score_semantic("Le chat dort", "Le chat dort")
    assert score == 1.0  # recouvrement lexical parfait
