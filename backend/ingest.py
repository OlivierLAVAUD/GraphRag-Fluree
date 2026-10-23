#!/usr/bin/env python3
"""CLI d'ingestion : insère des fichiers/dossiers (md, txt, pdf) dans Fluree.

Usage :
    python ingest.py docs/                  # dossier récursif
    python ingest.py docs/guide.md docs/faq.md
"""

import argparse
import logging
from pathlib import Path

from app import config
from app.ingester import ingest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingère des documents (Markdown/texte/PDF) dans le graphe Fluree du chatbot RAG."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Fichiers ou dossiers à ingérer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    total = 0
    for path in args.paths:
        report = ingest_path(path)
        for source, count in report:
            print(f"  - {source}: {count} chunks")
            total += count
    print(f"Terminé : {total} chunks insérés dans le ledger '{config.LEDGER}'.")


if __name__ == "__main__":
    main()