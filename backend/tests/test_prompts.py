"""Garde-fous sur le prompt système : les règles anti-spéculation doivent
rester présentes (elles conditionnent le comportement Q6 de l'évaluation)."""

from app.prompts import SYSTEM_PROMPT, build_messages


def test_prompt_forbids_equity_stake_deduction():
    assert "prise de participation" in SYSTEM_PROMPT
    assert "n'implique pas" in SYSTEM_PROMPT.lower()
    assert "ne déduis jamais" in SYSTEM_PROMPT.lower()


def test_prompt_requires_context_reasoning():
    assert "relier les informations du contexte" in SYSTEM_PROMPT
    assert "comparer des dates" in SYSTEM_PROMPT
    assert "implique que l'État lui fait confiance" in SYSTEM_PROMPT
    assert "ne sont pas de l'invention" in SYSTEM_PROMPT


def test_prompt_requires_complete_lists():
    assert "Quelles entreprises fournissent X" in SYSTEM_PROMPT
    assert "extraits différents" in SYSTEM_PROMPT


def test_prompt_keeps_refusal_template():
    assert "Je ne dispose pas de cette information dans ma base de connaissances." in SYSTEM_PROMPT
    assert "Cite tes sources" in SYSTEM_PROMPT


def test_build_messages_embeds_question_and_hits():
    hits = [{"source": "doc.md", "index": 0, "content": "Le contexte pertinent."}]
    messages = build_messages("Question de test ?", hits)
    assert messages[0]["role"] == "system"
    assert "Question de test ?" in messages[1]["content"]
    assert "doc.md" in messages[1]["content"]