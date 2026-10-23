"""Pipeline d'ingestion : lecture de documents, découpage en chunks,
embedding via Ollama et stockage dans le graphe Fluree."""

import hashlib
import logging
import re
from io import BytesIO
from pathlib import Path

from . import config, fluree, ollama_client as ollama

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lecture des documents (Markdown / texte brut / PDF)
# ---------------------------------------------------------------------------

def read_bytes(data: bytes, filename: str) -> str:
    """Décode le contenu d'un fichier en texte brut (PDF inclus)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_document(path: Path) -> str:
    return read_bytes(path.read_bytes(), path.name)


# ---------------------------------------------------------------------------
# Découpage en chunks (approche par phrases, avec recouvrement)
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+")


def _sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(paragraph) if s.strip()]


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Découpe un texte en chunks ciblant ~size caractères, avec recouvrement."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        return []

    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        sentences.extend(_sentences(paragraph))

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            chunks.append(current)
            current = _overlap_tail(current, overlap)
        current = f"{current} {sentence}".strip() if current else sentence

    if current.strip():
        chunks.append(current.strip())
    return chunks


def _overlap_tail(previous: str, overlap: int) -> str:
    """Garde les dernières phrases du chunk précédent (recouvrement)."""
    if overlap <= 0:
        return ""
    tail_sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(previous) if s.strip()]
    keep = ""
    for s in reversed(tail_sentences):
        if keep and len(keep) + len(s) + 1 > overlap:
            break
        keep = f"{s} {keep}".strip() if keep else s
    return keep


# ---------------------------------------------------------------------------
# Préparation et insertion des chunks
# ---------------------------------------------------------------------------

def prepare_chunks(text: str, source: str) -> list[dict]:
    """Découpe un texte et construit les objets chunks (identifiants stables)."""
    chunks = []
    for index, content in enumerate(chunk_text(text)):
        chunk_id = hashlib.sha1(f"{source}::{index}::{content}".encode()).hexdigest()[:16]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "index": index,
                "source": source,
                "content": content,
            }
        )
    return chunks


def ingest_text(text: str, source: str) -> list[dict]:
    """Embedd les chunks d'un texte et les insère dans Fluree."""
    if not text or not text.strip():
        log.warning("Document vide ignoré : %s", source)
        return []
    chunks = prepare_chunks(text, source)
    if not chunks:
        return []
    fluree.ensure_ledger()
    embeddings = ollama.embed_batch([c["content"] for c in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    for start in range(0, len(chunks), 100):
        fluree.insert_chunks(chunks[start : start + 100])
    # Invalide l'index de recherche sémantique (les chunks ont changé)
    from . import retriever as _retriever

    _retriever.invalidate_index()
    return chunks


def ingest_path(path: Path) -> list[tuple[str, int]]:
    """Ingère un fichier ou tous les fichiers d'un dossier. Renvoie (source, nb chunks)."""
    files: list[Path]
    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
        )
    else:
        files = [path]

    report = []
    for file in files:
        text = read_document(file)
        chunks = ingest_text(text, file.name)
        report.append((file.name, len(chunks)))
        log.info("Ingéré : %s (%d caractères, %d chunks)", file.name, len(text), len(chunks))
    return report