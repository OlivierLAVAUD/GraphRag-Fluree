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
information absente du document : la réponse correcte est un refus honnête.
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


def score_semantic(
    answer: str, expected: str, top_k: int = 5
) -> tuple[float, list[dict]]:
    """Scoring sémantique par cosinus entre l'answer et l'expected.

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
        return round(max(0.0, min(1.0, sim)), 3), []
    except Exception:
        # Fallback : scoring lexical si les embeddings échouent
        return score_overlap(answer, expected), []


def is_absence_answer(answer: str) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in ABSENCE_PATTERNS)


def evaluate(question: str, expected: str, expected_absence: bool) -> dict:
    """Pose une question au pipeline complet et calcule son score."""
    hits = retriever.retrieve(question, top_k=config.TOP_K)
    if not hits:
        answer = (
            "Je ne dispose pas d'information pertinente dans ma base de connaissances "
            "pour répondre à cette question."
        )
    else:
        messages = prompts.build_messages(question, hits)
        parts: list[str] = []
        ollama.chat_stream(messages, on_delta=parts.append)
        answer = "".join(parts)

    absence_detected = is_absence_answer(answer)
    if expected_absence and absence_detected:
        score = 1.0
    else:
        score, _ = score_semantic(answer, expected)

    return {
        "answer": answer,
        "score": score,
        "absence_detected": absence_detected,
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


def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue le pipeline RAG sur un jeu de questions.")
    parser.add_argument("--doc", default=None, help="Document à ingérer avant l'évaluation")
    parser.add_argument("--qa", default=None, help="Fichier JSON du jeu de questions")
    parser.add_argument("--out", default="eval_results.json", help="Fichier de rapport JSON")
    args = parser.parse_args()

    doc_path = args.doc or _first_existing(["/app/docs", "docs"])
    qa_path = args.qa or _first_existing(
        ["/app/tests/eval/qa_france2030.json", "tests/eval/qa_france2030.json"]
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
    print(f"      {newly} nouveaux chunks → {total} chunks dans le ledger '{config.LEDGER}'.")

    # 2. Boucle de questions
    print(f"[2/2] {len(qa)} questions …")
    results = []
    for item in qa:
        outcome = evaluate(item["question"], item["expected"], item["expected_absence"])
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
        verdict = "OK    " if outcome["score"] >= 0.5 else "REVOIR"
        approx = "" if outcome["absence_detected"] else " (~)"
        print(
            f"  Q{item['id']:>2} [{item['category']:<17}] {verdict} "
            f"score={outcome['score']:.2f}{approx}"
        )
        hits = outcome["top_hits"]
        if hits:
            top = hits[0]
            print(f"        top-hit: {top['source']}#{top['index']} (score {top['score']:.3f}) "
                  f"— {top['snippet'][:90]}…")

    # 3. Rapport
    scores = [r["score"] for r in results]
    summary = {
        "doc": str(doc),
        "questions": len(results),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "ok_count": sum(1 for s in scores if s >= 0.5),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "results": results}, fh, ensure_ascii=False, indent=2)
    print(f"\nRapport écrit dans {out_path}")
    print(f"Résumé : {summary['ok_count']}/{summary['questions']} OK, "
          f"score moyen {summary['mean_score']:.2f}")
    print("\nInterprétation : un score bas peut venir du RETRIEVAL (voir top-hit) ou de la")
    print("GENERATION (cas multi-hop/négation). Les questions expected_absence=true sont")
    print("réussies si le modèle refuse de répondre plutôt que d'inventer.")


if __name__ == "__main__":
    main()
