"""Garde-fous sur le prompt système.

On vérifie des *propriétés* (répondre à partir du contexte, ne pas inventer,
citer, refuser honnêtement), pas les formulations d'une réponse attendue : le
prompt ne doit pas être ajusté au jeu d'évaluation, sinon le score ne mesure
plus la généralisation.
"""

from app.prompts import SYSTEM_PROMPT, build_messages


def test_prompt_restricts_answers_to_context():
    assert "uniquement à partir du contexte" in SYSTEM_PROMPT
    assert "n'ajoute aucun fait extérieur" in SYSTEM_PROMPT


def test_prompt_requires_combining_extracts():
    assert "combiner plusieurs extraits" in SYSTEM_PROMPT
    assert "n'est pas une invention" in SYSTEM_PROMPT


def test_prompt_requires_complete_lists():
    assert "donne tous ceux présents dans le contexte" in SYSTEM_PROMPT
    assert "n'omet aucun élément trouvé" in SYSTEM_PROMPT


def test_prompt_forbids_unstated_facts():
    """Règle générale : aucun fait non énoncé (juridique, capitalistique, …)."""
    assert "Ne déduis jamais un fait qui n'est pas énoncé" in SYSTEM_PROMPT
    assert "capitalistique" in SYSTEM_PROMPT
    assert "contractuel" in SYSTEM_PROMPT


def test_prompt_does_not_embed_expected_answers():
    """Aucune réponse du jeu d'évaluation ne doit être recopiée dans le prompt."""
    assert "implique que l'État lui fait confiance" not in SYSTEM_PROMPT
    assert "Mistral" not in SYSTEM_PROMPT
    assert "Palantir" not in SYSTEM_PROMPT
    assert "Q10" not in SYSTEM_PROMPT


def test_prompt_keeps_refusal_template_and_citations():
    assert (
        "Je ne dispose pas de cette information dans ma base de connaissances."
        in SYSTEM_PROMPT
    )
    assert "Cite tes sources" in SYSTEM_PROMPT
    assert "[1], [2]" in SYSTEM_PROMPT


def test_build_messages_embeds_question_and_hits():
    hits = [{"source": "doc.md", "index": 0, "content": "Le contexte pertinent."}]
    messages = build_messages("Question de test ?", hits)
    assert messages[0]["role"] == "system"
    assert "Question de test ?" in messages[1]["content"]
    assert "doc.md" in messages[1]["content"]
    assert "[1] Source : doc.md" in messages[1]["content"]


def test_build_messages_without_hits_signals_empty_context():
    """Sans extrait pertinent, le prompt reste valide et laisse le LLM répondre."""
    messages = build_messages("Bonjour ?", [])
    assert "Bonjour ?" in messages[1]["content"]
    assert "aucun extrait pertinent" in messages[1]["content"]
    assert "Source :" not in messages[1]["content"]
