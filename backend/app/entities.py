"""Extraction d'entités nommées légères pour le pipeline RAG.

Utilise une approche hybride :
- Regex pour les noms propres (majuscules consécutives)
- Dictionnaire d'entités connues
"""

import re

# Regex pour détecter les noms propres (2+ mots commençant par une majuscule)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Dictionnaire d'entités connues (extensible)
_KNOWN_ENTITIES: dict[str, str] = {
    "Palantir": "Company",
    "Microsoft": "Company",
    "Google": "Company",
    "OpenAI": "Company",
    "Anthropic": "Company",
    "Meta": "Company",
    "Apple": "Company",
    "Amazon": "Company",
}


def extract_entities(text: str) -> list[dict]:
    """Extrait les entités nommées d'un texte.

    Retourne une liste de dicts : {"name": str, "type": str, "confidence": float}.
    """
    if not text or not text.strip():
        return []

    found: dict[str, dict] = {}

    # 1. Dictionnaire connu
    for name, etype in _KNOWN_ENTITIES.items():
        if name in text:
            found[name] = {"name": name, "type": etype, "confidence": 0.9}

    # 2. Regex noms propres
    for match in _PROPER_NOUN_RE.finditer(text):
        name = match.group(0)
        if name not in found:
            # Heuristique : si le mot contient un mot-clé d'entreprise
            etype = (
                "Company"
                if any(k in name for k in ("Inc", "Corp", "LLC", "Ltd", "SA", "SAS"))
                else "Person"
            )
            found[name] = {"name": name, "type": etype, "confidence": 0.6}

    return list(found.values())
