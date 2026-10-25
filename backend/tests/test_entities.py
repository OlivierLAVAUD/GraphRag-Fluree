"""Tests de l'extraction d'entités (module sans couverture jusqu'ici).

L'extracteur est volontairement simple (dictionnaire + noms propres de deux mots
ou plus) : ces tests documentent son comportement réel, y compris ses limites.
"""

from app.entities import extract_entities


def test_empty_text_returns_nothing():
    assert extract_entities("") == []
    assert extract_entities("   \n  ") == []


def test_known_entity_from_dictionary():
    found = {e["name"]: e for e in extract_entities("Microsoft investit en France")}
    assert found["Microsoft"]["type"] == "Company"
    assert found["Microsoft"]["confidence"] == 0.9


def test_multi_word_proper_noun_is_person_by_default():
    ents = extract_entities("Copilot Studio intègre un nouveau modèle.")
    assert [e["name"] for e in ents] == ["Copilot Studio"]
    assert ents[0]["type"] == "Person"
    assert ents[0]["confidence"] == 0.6


def test_company_suffix_marks_company():
    ents = extract_entities("Acme Corp a signé le contrat.")
    types = {e["name"]: e["type"] for e in ents}
    assert types["Acme Corp"] == "Company"


def test_single_word_names_are_not_extracted():
    """Limite connue : la regex exige deux mots capitalisés consécutifs.

    Les noms en un seul mot (Quandela, ChapsVision, GENCI) ne sont donc pas
    indexés comme entités — le bonus de recherche ne peut pas s'appuyer dessus.
    """
    assert extract_entities("Quandela développe Lucy avec GENCI.") == []


def test_duplicates_are_collapsed():
    ents = extract_entities("Microsoft et Microsoft encore Microsoft")
    assert [e["name"] for e in ents] == ["Microsoft"]
