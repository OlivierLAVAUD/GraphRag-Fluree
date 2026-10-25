#!/usr/bin/env python3
"""Harnais d'évaluation du pipeline RAG.

Ingère un document, pose les questions du jeu de test, compare les réponses
aux réponses attendues (recouvrement lexical + sémantique) et génère un rapport JSON.

Usage (dans le conteneur backend, à la racine /app) :
    python eval_rag.py
    python eval_rag.py --qa tests/eval/qa_france2030.json --out eval_results.json

Le rapport inclut pour chaque question les extraits récupérés (sources +
scores), ce qui permet de distinguer un échec de RETRIEVAL d'un échec de
GENERATION. Les questions marquées expected_absence=true exigent une
information absente du document : la réponse correcte est un refus honnête,
sans affirmation supplémentaire non vérifiée.

Le score sémantique est *calibré* : un cosinus brut entre deux textes reste
élevé même hors sujet avec les modèles d'embeddings usuels, ce qui rendait le
seuil de réussite artificiellement facile. On retire donc le plancher
`SEMANTIC_FLOOR` avant de ramener le score sur [0, 1].
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from app import config, fluree, ingester, ollama_client as ollama, prompts, retriever

# ---------------------------------------------------------------------------
# Scoring lexical (heuristique simple : recouvrement de tokens)
# ---------------------------------------------------------------------------

STOPWORDS = frozenset(
    """le la les de des du et en un une pour avec est sont a au aux qui que se sa son
    ses ce cette il elle ils elles dans par sur plus comme quel quelle quels quelles
    pas pas ne n non oui ou d l m t s""".split()
)

ABSENCE_PATTERNS = [
    "ne dispose pas",
    "pas d'information",
    "aucune information",
    "information non disponible",
    "absente du contexte",
    "ne connais",
    "je ne sais pas",
    "pas en mesure",
]

# Vocabulaire d'un refus : les jetons restants après filtrage mesurent ce que la
# réponse ajoute *en plus* du refus (une affirmation inventée, par exemple).
REFUSAL_TOKENS = frozenset(
    """dispose information informations base connaissances contexte extrait extraits
    fourni fournis document documents disponible indisponible connais sais sait
    aucune aucun rien trouve trouvée trouvee préciser précisé malheureusement
    réponse repondre répondre question données donnee donnees pertinent
    pertinente malgré""".split()
)

# Plancher de similarité des embeddings : en dessous, deux textes sont
# considérés comme sans rapport. Calibrer évite qu'une réponse hors sujet
# obtienne 0,6 par simple effet du modèle d'embeddings.
SEMANTIC_FLOOR = 0.6
# Seuil de réussite d'une question (score calibré).
PASS_THRESHOLD = 0.55
# Score partiel d'une question d'absence : le modèle refuse, mais ajoute des
# affirmations non vérifiées (le cas dangereux, qui n'est pas une réussite).
ABSENCE_SPECULATION_SCORE = 0.5


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zà-ÿ0-9]+", (text or "").lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def score_overlap(answer: str, expected: str) -> float:
    """Recouvrement lexical entre la réponse et la réponse attendue."""
    a = tokenize(answer)
    e = tokenize(expected)
    if not e:
        return 1.0 if not a else 0.0
    return len(a & e) / len(e)


# ---------------------------------------------------------------------------
# Scoring sémantique (cosinus sur embeddings Ollama)
# ---------------------------------------------------------------------------


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embadge une liste de textes via Ollama (batch unique)."""
    resp = ollama.embed_batch(texts)
    return resp


def calibrate_semantic(cosine: float, floor: float | None = None) -> float:
    """Ramène un cosinus brut sur [0, 1] en retirant le plancher du modèle.

    Avec `floor = 0.6`, un cosinus de 0,60 (textes sans rapport) donne 0,0 et
    un cosinus de 0,8 donne 0,5 : le score mesure bien un recouvrement
    sémantique au-dessus du bruit, et non la proximité de base du modèle.
    """
    floor = SEMANTIC_FLOOR if floor is None else floor
    if cosine <= floor:
        return 0.0
    return round(min(1.0, (cosine - floor) / (1.0 - floor)), 3)


def score_semantic(
    answer: str, expected: str, top_k: int = 5, floor: float | None = None
) -> tuple[float, list[dict]]:
    """Scoring sémantique calibré entre l'answer et l'expected.

    Retourne (score, top_hits) où top_hits est la liste des chunks récupérés
    avec leur score de similarité, utile pour le rapport.
    """
    try:
        embeddings = _embed_texts([answer, expected])
        a_vec = np.asarray(embeddings[0], dtype=np.float32).reshape(-1)
        e_vec = np.asarray(embeddings[1], dtype=np.float32).reshape(-1)
        norm_a = np.linalg.norm(a_vec)
        norm_e = np.linalg.norm(e_vec)
        if norm_a == 0 or norm_e == 0:
            return 0.0, []
        sim = float(np.dot(a_vec, e_vec) / (norm_a * norm_e))
        return calibrate_semantic(sim, floor), []
    except Exception:
        # Fallback : scoring lexical si les embeddings échouent (déjà sur [0, 1])
        return score_overlap(answer, expected), []


def is_absence_answer(answer: str) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in ABSENCE_PATTERNS)


def is_clean_refusal(answer: str) -> bool:
    """Vrai si la réponse refuse *sans* ajouter d'affirmation non vérifiée.

    Un refus suivi d'une spéculation (« je ne dispose pas de cette information,
    mais probablement … ») ne doit pas être noté comme une réussite : on retire
    le vocabulaire de refus et on exige qu'il ne reste presque rien.
    """
    if not is_absence_answer(answer):
        return False
    leftover = {tok for tok in tokenize(answer) if tok not in REFUSAL_TOKENS}
    return len(leftover) <= 2


def evaluate(
    question: str, expected: str, expected_absence: bool, floor: float | None = None
) -> dict:
    """Pose une question au pipeline complet et calcule son score."""
    hits = retriever.retrieve(question, top_k=config.TOP_K)
    # Comme l'application, on interroge le modèle même sans extrait : c'est lui
    # qui décide de refuser (une réponse figée côté code masquerait le test).
    messages = prompts.build_messages(question, hits)
    parts: list[str] = []
    try:
        ollama.chat_stream(messages, on_delta=parts.append)
    except Exception as exc:  # noqa: BLE001
        return {
            "answer": f"Dans l'évaluation, la génération a échoué : {exc}",
            "score": 0.0,
            "absence_detected": False,
            "absence_clean": False,
            "top_hits": [],
            "error": str(exc),
        }
    answer = "".join(parts)

    absence_detected = is_absence_answer(answer)
    absence_clean = is_clean_refusal(answer)
    if expected_absence:
        if absence_clean:
            score = 1.0
        elif absence_detected:
            # Refus noyé dans des affirmations : le cas précis que le test
            # d'hallucination doit sanctionner sans le noter comme un succès.
            score = ABSENCE_SPECULATION_SCORE
        else:
            score = 0.0
    elif absence_detected:
        # Refuser une question couverte par le corpus est un échec du RAG.
        score = 0.0
    else:
        score, _ = score_semantic(answer, expected, floor=floor)

    return {
        "answer": answer,
        "score": score,
        "absence_detected": absence_detected,
        "absence_clean": absence_clean,
        "top_hits": [
            {
                "source": h["source"],
                "index": h["index"],
                "score": h["score"],
                "snippet": h["content"][:160],
            }
            for h in hits
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent


def _first_existing(paths: list[Path], label: str) -> Path:
    """Renvoie le premier chemin existant, sinon une erreur explicite.

    Les chemins sont résolus par rapport au fichier (et non au répertoire
    courant) : le harnais fonctionne donc aussi bien dans le conteneur
    (`/app/docs`, `/app/tests`) que depuis le dépôt (`backend/`, racine).
    """
    for path in paths:
        if path.exists():
            return path
    candidates = ", ".join(str(p) for p in paths)
    raise FileNotFoundError(f"{label} introuvable (chemins essayés : {candidates})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évalue le pipeline RAG sur un jeu de questions."
    )
    parser.add_argument(
        "--doc", default=None, help="Document à ingérer avant l'évaluation"
    )
    parser.add_argument("--qa", default=None, help="Fichier JSON du jeu de questions")
    parser.add_argument(
        "--out", default="eval_results.json", help="Fichier de rapport JSON"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PASS_THRESHOLD,
        help=(
            "Score minimal (calibré) pour compter une question comme réussie "
            f"(défaut : {PASS_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=SEMANTIC_FLOOR,
        help=(
            "Plancher de cosinus considéré comme hors sujet "
            f"(défaut : {SEMANTIC_FLOOR})"
        ),
    )
    args = parser.parse_args()
    threshold = args.threshold

    doc_path = (
        Path(args.doc)
        if args.doc
        else _first_existing(
            [BACKEND_DIR / "docs", REPO_DIR / "docs", Path("docs")],
            "Dossier de documents",
        )
    )
    qa_path = (
        Path(args.qa)
        if args.qa
        else _first_existing(
            [
                BACKEND_DIR / "tests/eval/qa_france2030.json",
                REPO_DIR / "tests/eval/qa_france2030.json",
                Path("tests/eval/qa_france2030.json"),
            ],
            "Jeu de questions",
        )
    )
    out_path = args.out

    with open(qa_path, encoding="utf-8") as fh:
        qa = json.load(fh)["questions"]

    # 1. Ingestion (idempotente au niveau des chunks : IDs stables).
    doc = Path(doc_path)
    print(f"[1/2] Ingestion de {doc} …")
    if doc.is_dir():
        report = ingester.ingest_path(doc)
        newly = sum(count for _, count in report)
        for source, count in report:
            print(f"      {source}: {count} chunks")
    else:
        chunks = ingester.ingest_text(doc.read_text(encoding="utf-8"), doc.name)
        newly = len(chunks)
        print(f"      {doc.name}: {newly} chunks")
    total = len(fluree.all_chunks())
    print(
        f"      {newly} nouveaux chunks → {total} chunks dans le ledger '{config.LEDGER}'."
    )

    # 2. Boucle de questions
    print(f"[2/2] {len(qa)} questions …")
    results = []
    for item in qa:
        outcome = evaluate(
            item["question"],
            item["expected"],
            item["expected_absence"],
            floor=args.floor,
        )
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "expected": item["expected"],
                "expected_absence": item["expected_absence"],
                **outcome,
            }
        )
        verdict = "OK    " if outcome["score"] >= threshold else "REVOIR"
        approx = (
            ""
            if outcome["absence_clean"] or not outcome["absence_detected"]
            else " (~)"
        )
        print(
            f"  Q{item['id']:>2} [{item['category']:<17}] {verdict} "
            f"score={outcome['score']:.2f}{approx}"
        )
        hits = outcome["top_hits"]
        if hits:
            top = hits[0]
            print(
                f"        top-hit: {top['source']}#{top['index']} (score {top['score']:.3f}) "
                f"— {top['snippet'][:90]}…"
            )

    # 3. Rapport
    scores = [r["score"] for r in results]
    summary = {
        "doc": str(doc),
        "questions": len(results),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "ok_count": sum(1 for s in scores if s >= threshold),
        "threshold": threshold,
        "semantic_floor": args.floor,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"summary": summary, "results": results}, fh, ensure_ascii=False, indent=2
        )
    print(f"\nRapport écrit dans {out_path}")
    print(
        f"Résumé : {summary['ok_count']}/{summary['questions']} OK, "
        f"score moyen {summary['mean_score']:.2f}"
    )
    print(
        f"Seuil de réussite : {threshold:.2f} (score calibré, plancher "
        f"d'embeddings {args.floor:.2f})."
    )
    print(
        "Interprétation : un score bas peut venir du RETRIEVAL (voir top-hit) ou de la"
    )
    print(
        "GENERATION (cas multi-hop/négation). Les questions expected_absence=true sont"
    )
    print("réussies seulement si le refus est *propre* : un refus suivi d'une")
    print("affirmation inventée ne vaut que 0.5.")


if __name__ == "__main__":
    main()
