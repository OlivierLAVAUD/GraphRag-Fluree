"""Pipeline d'ingestion : lecture de documents, découpage en chunks,
embedding via Ollama et stockage dans le graphe Fluree."""

import hashlib
import logging
import re
from io import BytesIO
from pathlib import Path

from . import config, entities, fluree, ollama_client as ollama

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


def chunk_text(
    text: str, size: int | None = None, overlap: int | None = None
) -> list[str]:
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
    tail_sentences = [
        s.strip() for s in _SENTENCE_BOUNDARY.split(previous) if s.strip()
    ]
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
        chunk_id = hashlib.sha1(f"{source}::{index}::{content}".encode()).hexdigest()[
            :16
        ]
        ents = entities.extract_entities(content)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "index": index,
                "source": source,
                "content": content,
                "entities": ents,
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
    # Collecter les IDs des nouveaux chunks avant le marquage d'obsolescence
    new_ids = {c["chunk_id"] for c in chunks}
    inserted_ids: list[str] = []
    try:
        for start in range(0, len(chunks), 100):
            batch = chunks[start : start + 100]
            fluree.insert_chunks(batch)
            inserted_ids.extend(c["chunk_id"] for c in batch)
    except Exception:
        # Ingestion partielle : on désactive les chunks déjà insérés pour que
        # l'ancienne version du document reste cohérente, puis on propage.
        if inserted_ids:
            fluree.mark_ids_obsolete(inserted_ids)
        raise
    # Marquer les anciens chunks de cette source comme obsolètes
    # (après insertion réussie pour éviter la perte de données)
    # On conserve les IDs des nouveaux chunks pour ne pas les marquer obsolètes.
    # Si ce marquage échoue, on annule les nouveaux chunks pour ne pas laisser
    # deux versions du document actives en même temps.
    try:
        fluree.mark_obsolete(source, keep_ids=new_ids)
    except Exception:
        fluree.mark_ids_obsolete(inserted_ids)
        raise
    # Invalide l'index de recherche sémantique (les chunks ont changé)
    from . import retriever as _retriever

    _retriever.invalidate_index()
    return chunks


def deindex_source(source: str) -> int:
    """Désindexe une source : ses chunks ne sont plus renvoyés par la recherche.

    Les chunks sont marqués obsolètes (le ledger reste immuable et l'historique
    conservé) et l'index de recherche est invalidé. Renvoie le nombre de chunks
    désindexés, 0 si la source n'existe pas.
    """
    removed = fluree.deindex_source(source)
    if removed:
        from . import retriever as _retriever

        _retriever.invalidate_index()
        log.info("Source désindexée : %s (%d chunks)", source, removed)
    return removed


def reset_knowledge_base() -> int:
    """Vide la base de connaissances sans détruire le ledger (audit conservé).

    Renvoie le nombre de chunks désindexés.
    """
    removed = fluree.deindex_all()
    from . import retriever as _retriever

    _retriever.invalidate_index()
    log.info("Base réinitialisée : %d chunks désindexés", removed)
    return removed


def source_for_path(path: Path) -> str:
    """Identifie un fichier par un chemin relatif stable, pas seulement son nom.

    Deux fichiers homonymes dans des sous-dossiers différents doivent rester
    distincts : sinon la ré-ingestion de l'un marque les chunks de l'autre comme
    obsolètes. On privilégie un chemin relatif au répertoire courant ; hors de
    ce répertoire, on retombe sur le nom du fichier.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def resolve_source(name: str) -> str:
    """Nom de source stable pour un upload (fichier sans chemin).

    La CLI identifie une source par son chemin relatif (`docs/faq.md`) alors
    qu'un upload ne fournit qu'un nom de fichier (`faq.md`) : sans
    réconciliation, le même document serait indexé deux fois (deux sources
    actives, donc deux jeux de chunks). On réutilise donc la source existante
    de même nom de fichier, en préférant le chemin le plus court (le plus
    stable d'une machine à l'autre).
    """
    try:
        existing = fluree.source_names()
    except Exception as exc:  # noqa: BLE001 - best effort : le nom suffit
        log.warning("Sources existantes indisponibles (%s) : source = %s", exc, name)
        return name
    matches = [src for src in existing if Path(src).name == Path(name).name]
    if not matches:
        return name
    # La CLI est la forme canonique (`docs/faq.md`) : si les deux formes
    # coexistent, l'upload rejoint la source qui porte un chemin. Plusieurs
    # chemins homonymes (`a/faq.md`, `docs/faq.md`) restent des documents
    # distincts : on retient le chemin le plus court, le plus stable.
    with_dir = [src for src in matches if Path(src).parent != Path(".")]
    return min(with_dir or matches, key=len)


# Longueur maximale d'une source construite depuis un upload : un chemin
# démesuré n'apporte rien et gonfle chaque chunk stocké dans le graphe.
_MAX_SOURCE_LENGTH = 400


def _clean_source_path(name: str) -> str:
    """Normalise un nom d'upload en chemin relatif (`/`), sans segments vides.

    Le navigateur envoie le chemin relatif d'un dossier sélectionné
    (`mon-dossier/sous/fichier.md`). Les segments `.` et `..` sont retirés : la
    source est une chaîne stockée dans le graphe, pas un chemin sur disque, mais
    on évite d'y propager une remontée de répertoire.
    """
    parts = [
        part
        for part in (name or "").replace("\\", "/").split("/")
        if part and part not in (".", "..")
    ]
    return "/".join(parts)


def resolve_upload_source(name: str) -> str:
    """Source d'un upload : conserve le chemin relatif quand il est fourni.

    Un dossier sélectionné envoie `dossier/sous/fichier.md` : le chemin doit
    rester la source, sinon deux fichiers homonymes de dossiers différents
    s'écraseraient mutuellement (le dernier ingéré marquerait les chunks de
    l'autre obsolètes). Un simple nom de fichier reste réconcilié avec les
    sources existantes (voir `resolve_source`).
    """
    path = _clean_source_path(name)
    if not path:
        return "fichier"
    if len(path) > _MAX_SOURCE_LENGTH:
        path = path.rsplit("/", 1)[-1][:_MAX_SOURCE_LENGTH]
    if "/" in path:
        return path
    return resolve_source(path)


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
        source = source_for_path(file)
        text = read_document(file)
        chunks = ingest_text(text, source)
        report.append((source, len(chunks)))
        log.info(
            "Ingéré : %s (%d caractères, %d chunks)", source, len(text), len(chunks)
        )
    return report
