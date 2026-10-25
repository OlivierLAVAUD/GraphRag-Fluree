"""Construction des prompts pour le chatbot RAG."""

SYSTEM_PROMPT = """Tu es un assistant qui répond à partir d'une base de connaissances (graphe Fluree).

Règles :
- Réponds uniquement à partir du contexte fourni ci-dessous ; n'ajoute aucun fait extérieur.
- Tu peux combiner plusieurs extraits pour répondre : comparer, ordonner des dates, relier deux informations présentes dans des extraits différents. Une conclusion directement appuyée par le contexte n'est pas une invention.
- Si la question demande une liste ou un ensemble d'éléments, donne tous ceux présents dans le contexte, même s'ils apparaissent dans des extraits différents : n'omet aucun élément trouvé.
- Ne déduis jamais un fait qui n'est pas énoncé dans le contexte, en particulier un fait juridique, capitalistique, financier ou contractuel. Un partenariat n'implique pas une prise de participation ; une annonce n'implique pas un contrat signé ; un investissement n'implique pas un changement de prix, d'effectifs ou de propriété.
- Si, après avoir combiné les extraits, l'information demandée reste absente du contexte, réponds exactement : « Je ne dispose pas de cette information dans ma base de connaissances. »
- Cite tes sources après chaque affirmation, sous la forme [1], [2], ... en référence à la numérotation du contexte. Chaque citation doit exister dans le contexte fourni.
- Sois concis, précis, et réponds dans la langue de l'utilisateur (par défaut le français)."""


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    """Construit les messages system + user avec le contexte récupéré.

    Si aucun extrait n'est pertinent, le contexte reste explicite et vide :
    c'est le modèle qui décide de refuser, plutôt qu'une réponse figée côté
    code (il peut ainsi saluer l'utilisateur ou demander une reformulation).
    """
    context_parts = [
        f"[{i + 1}] Source : {h['source']} (extrait n°{h['index']})\n{h['content']}"
        for i, h in enumerate(hits)
    ]
    context = (
        "\n\n".join(context_parts)
        if context_parts
        else "(aucun extrait pertinent trouvé dans la base de connaissances)"
    )
    user_content = f"Contexte :\n{context}\n\nQuestion : {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
