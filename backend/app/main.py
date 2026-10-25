"""Point d'entrée de l'application FastAPI : API + interface web."""

import asyncio
import hmac
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, fluree, ingester, ollama_client as ollama, prompts, retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag")

STATIC_DIR = Path(__file__).parent / "static"


async def _wait_for(service: str, check, timeout: float = 120.0) -> None:
    """Attend qu'un service (Fluree/Ollama) soit prêt, avec backoff."""
    waited = 0.0
    while waited < timeout:
        if await asyncio.to_thread(check):
            log.info("Service %s prêt.", service)
            return
        await asyncio.sleep(2)
        waited += 2
    log.warning(
        "Service %s non prêt après %ss — l'application continuera.", service, timeout
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _wait_for("fluree", fluree.is_ready)
    await _wait_for("ollama", ollama.is_ready)
    await asyncio.to_thread(fluree.ensure_ledger)
    yield


app = FastAPI(title="GraphRag-Fluree", lifespan=lifespan)


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    """Force la revalidation des fichiers de l'interface.

    Sans en-tête de cache, le navigateur peut réutiliser un `index.html` ou un
    `app.js` d'une image Docker antérieure et afficher une UI périmée (boutons
    absents). `no-cache` conserve la validation par ETag : réponse 304 si le
    fichier n'a pas changé, contenu frais sinon.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class ChatRequest(BaseModel):
    question: str = Field(max_length=config.MAX_QUESTION_LENGTH)
    source: str | None = Field(default=None, max_length=500)
    entity: str | None = Field(default=None, max_length=200)
    top_k: int = Field(default=config.TOP_K, ge=1, le=config.MAX_TOP_K)


class SearchRequest(BaseModel):
    question: str = Field(max_length=config.MAX_QUESTION_LENGTH)
    top_k: int = Field(default=config.TOP_K, ge=1, le=config.MAX_TOP_K)
    source: str | None = Field(default=None, max_length=500)
    entity: str | None = Field(default=None, max_length=200)


def require_api_token(
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    authorization: str | None = Header(default=None),
) -> None:
    """Protège les endpoints sensibles si API_TOKEN est configuré (sinon no-op)."""
    if not config.API_TOKEN:
        return
    token = x_api_token
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    # Comparaison à temps constant : un `!=` laisse fuir la longueur du préfixe
    # correct via le temps de réponse.
    if not token or not hmac.compare_digest(token, config.API_TOKEN):
        raise HTTPException(status_code=401, detail="Jeton d'API manquant ou invalide.")


# Créneaux de génération occupés, par boucle asyncio (les tests instancient
# plusieurs boucles). Le compteur est incrémenté sans `await` : aucune course
# possible entre le test de disponibilité et la réservation, contrairement à
# `Semaphore.locked()` suivi d'un `acquire()` qui met la requête en attente.
_chat_slots: dict[object, int] = {}


def _try_acquire_chat_slot() -> bool:
    """Réserve un créneau de génération, ou renvoie False si le GPU est saturé."""
    loop = asyncio.get_running_loop()
    used = _chat_slots.get(loop, 0)
    if used >= config.MAX_CONCURRENT_CHATS:
        return False
    _chat_slots[loop] = used + 1
    return True


def _release_chat_slot() -> None:
    """Libère le créneau réservé pour la boucle courante."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # boucle déjà fermée (fin de test)
        return
    used = _chat_slots.get(loop, 0)
    if used <= 1:
        _chat_slots.pop(loop, None)
    else:
        _chat_slots[loop] = used - 1


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """Lit un upload par blocs en refusant de dépasser *limit* octets."""
    data = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Fichier trop volumineux (> {limit} octets, "
                    f"limite {config.MAX_UPLOAD_SIZE} par fichier / "
                    f"{config.MAX_TOTAL_UPLOAD_SIZE} par requête)."
                ),
            )
    return bytes(data)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status() -> dict:
    models = []
    try:
        models = ollama.list_models()
    except Exception as exc:  # noqa: BLE001
        log.warning("Ollama injoignable : %s", exc)
    return {
        "ledger": config.LEDGER,
        "embed_model": config.EMBED_MODEL,
        "chat_model": config.CHAT_MODEL,
        "top_k": config.TOP_K,
        "min_score": config.RETRIEVAL_MIN_SCORE,
        "ollama_models": models,
        "fluree": config.FLUREE_URL,
        # L'interface s'en sert pour valider une sélection (fichiers ou dossier)
        # avant d'envoyer l'ingestion, au lieu de découvrir la limite en 413.
        "limits": {
            "max_upload_files": config.MAX_UPLOAD_FILES,
            "max_upload_size": config.MAX_UPLOAD_SIZE,
            "max_total_upload_size": config.MAX_TOTAL_UPLOAD_SIZE,
            "supported_extensions": sorted(config.SUPPORTED_EXTENSIONS),
        },
    }


def _query_active_chunks(fields: list[str]) -> list[dict]:
    """Interroge les chunks actifs sans charger contenu ni embeddings."""
    payload = {
        "@context": config.CTX,
        "select": {"?c": fields},
        "where": {"@id": "?c", "@type": "rag:Chunk", "rag:obsolete": False},
    }
    out = fluree._post(f"/query/{config.LEDGER}", payload, 120)
    if isinstance(out, list):
        return out
    if isinstance(out, dict):
        for key in ("results", "result", "bindings", "rows"):
            if key in out:
                return out[key]
    raise RuntimeError(f"Réponse Fluree inattendue : {str(out)[:300]}")


@app.get("/api/sources", dependencies=[Depends(require_api_token)])
async def list_sources() -> dict:
    """Liste les sources actives du graphe et leur nombre de chunks."""
    rows = await asyncio.to_thread(_query_active_chunks, ["rag:source"])
    counts: dict[str, int] = {}
    for row in rows:
        src = row.get("rag:source")
        if src:
            counts[src] = counts.get(src, 0) + 1
    return {
        "sources": [{"name": name, "chunks": n} for name, n in sorted(counts.items())]
    }


@app.get("/api/entities", dependencies=[Depends(require_api_token)])
async def list_entities() -> dict:
    """Liste les entités mentionnées par les chunks et leur fréquence."""
    rows = await asyncio.to_thread(_query_active_chunks, ["rag:mentions"])
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    types: dict[str, str] = {}
    for row in rows:
        mentions = row.get("rag:mentions") or []
        if isinstance(mentions, dict):
            mentions = [mentions]
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            ent_id = mention.get("@id")
            if not ent_id:
                continue
            counts[ent_id] = counts.get(ent_id, 0) + 1
            if mention.get("ent:name"):
                names[ent_id] = mention["ent:name"]
            if mention.get("@type"):
                types[ent_id] = mention["@type"]

    if any(ent_id not in names for ent_id in counts):
        # Repli : noms des entités référencées par des chunks *actifs* (la
        # requête part des chunks non obsolètes, contrairement à un `select`
        # global qui ramenait aussi les entités de documents périmés).
        try:
            active = await asyncio.to_thread(fluree.active_entity_names)
        except Exception as exc:  # noqa: BLE001
            log.warning("Noms d'entités indisponibles : %s", exc)
            active = {}
        for ent_id in list(counts):
            if ent_id in names:
                continue
            name = active.get(ent_id) or active.get(
                f"ent:{fluree.entity_slug(str(ent_id).removeprefix('ent:'))}"
            )
            if name:
                names[ent_id] = name

    result = []
    for ent_id, chunks in counts.items():
        fallback = str(ent_id).removeprefix("ent:").replace("_", " ")
        result.append(
            {
                "name": names.get(ent_id) or fallback,
                "type": str(types.get(ent_id, "")).removeprefix("ent:"),
                "chunks": chunks,
            }
        )
    return {"entities": sorted(result, key=lambda e: -e["chunks"])}


@app.get("/api/gpu")
async def gpu() -> dict:
    """État GPU vu par Ollama (/api/ps) : vérifie que l'accélération est active."""
    try:
        return ollama.gpu_status()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"Ollama injoignable : {exc}"
        ) from exc


@app.post("/api/ingest", dependencies=[Depends(require_api_token)])
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Ingère des fichiers Markdown/texte/PDF dans le graphe Fluree."""
    if len(files) > config.MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Trop de fichiers ({len(files)}, limite {config.MAX_UPLOAD_FILES}).",
        )
    results = []
    total_bytes = 0
    for upload in files:
        filename = upload.filename or "fichier"
        suffix = Path(filename).suffix.lower()
        if suffix not in config.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Extension non supportée : {suffix or '(aucune)'}.",
            )
        remaining = config.MAX_TOTAL_UPLOAD_SIZE - total_bytes
        data = await _read_limited(upload, min(config.MAX_UPLOAD_SIZE, remaining))
        total_bytes += len(data)
        text = ingester.read_bytes(data, filename)
        # Réconcilie le nom de fichier de l'upload avec la source utilisée par
        # la CLI (`docs/faq.md`), pour ne pas indexer deux fois le même document.
        # Un upload de dossier fournit un chemin relatif (voir resolve_upload_source).
        source = await asyncio.to_thread(ingester.resolve_upload_source, filename)
        chunks = await asyncio.to_thread(ingester.ingest_text, text, source)
        results.append(
            {
                "file": filename,
                "source": source,
                "chars": len(text),
                "chunks": len(chunks),
            }
        )
    # `ingest_text` invalide déjà l'index de recherche après chaque insertion.
    return {"ingested": results}


@app.delete("/api/sources", dependencies=[Depends(require_api_token)])
async def deindex_source(
    source: str = Query(min_length=1, max_length=500),
) -> dict:
    """Désindexe une source : ses chunks ne sont plus renvoyés par la recherche.

    Le contenu n'est pas effacé du ledger (immuable) : les chunks sont marqués
    obsolètes, comme lors d'une ré-ingestion. Une nouvelle ingestion du même
    fichier réactive les chunks correspondants.
    """
    removed = await asyncio.to_thread(ingester.deindex_source, source)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Source introuvable : {source}")
    return {"source": source, "deindexed": removed, "ledger": config.LEDGER}


@app.post("/api/reset", dependencies=[Depends(require_api_token)])
async def reset_knowledge_base() -> dict:
    """Réinitialise la base de connaissances (tous les chunks désindexés).

    Le ledger Fluree n'est pas supprimé : l'historique d'audit reste consultable
    et le nom du ledger reste utilisable (un `drop` laisse un « tombeau »).
    """
    removed = await asyncio.to_thread(ingester.reset_knowledge_base)
    return {"deindexed": removed, "ledger": config.LEDGER}


@app.post("/api/chat", dependencies=[Depends(require_api_token)])
async def chat(request: ChatRequest) -> StreamingResponse:
    """Répond à une question : retrieval dans Fluree + génération streaming via Ollama."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question est vide.")

    if not _try_acquire_chat_slot():
        raise HTTPException(
            status_code=429,
            detail="Trop de générations simultanées, réessayez dans un instant.",
        )

    async def sse():
        try:
            try:
                hits = await asyncio.to_thread(
                    retriever.retrieve,
                    question,
                    top_k=request.top_k,
                    source=request.source,
                    entity=request.entity,
                )
            except Exception as exc:  # noqa: BLE001
                yield _event(
                    {"type": "error", "message": f"Recherche impossible : {exc}"}
                )
                return

            sources = [_source_payload(hit) for hit in hits]
            yield _event({"type": "sources", "sources": sources})

            # Aucun extrait pertinent : on interroge quand même le LLM. Le prompt
            # système lui demande de refuser honnêtement, ce qui permet aussi de
            # répondre à une salutation ou de demander une reformulation.
            messages = prompts.build_messages(question, hits)

            # La génération tourne dans un thread (client httpx synchrone), mais
            # le générateur SSE ne doit jamais bloquer la boucle asyncio.
            # On passe donc par une asyncio.Queue alimentée par
            # call_soon_threadsafe.
            loop = asyncio.get_running_loop()
            deltas: asyncio.Queue = asyncio.Queue()

            def emit(item) -> None:
                loop.call_soon_threadsafe(deltas.put_nowait, item)

            def producer():
                try:
                    ollama.chat_stream(messages, on_delta=emit)
                except Exception as exc:  # noqa: BLE001
                    emit(exc)
                finally:
                    emit(None)

            threading.Thread(target=producer, daemon=True).start()

            while True:
                item = await deltas.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    yield _event(
                        {"type": "error", "message": f"Génération interrompue : {item}"}
                    )
                    return
                yield _event({"type": "token", "content": item})

            yield _event({"type": "done", "sources": sources})
        finally:
            _release_chat_slot()

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/api/search", dependencies=[Depends(require_api_token)])
async def search(request: SearchRequest) -> dict:
    """Recherche sémantique pure (sans génération) : renvoie les chunks les plus pertinents."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question est vide.")
    hits = await asyncio.to_thread(
        retriever.retrieve,
        question,
        top_k=request.top_k,
        source=request.source,
        entity=request.entity,
    )
    return {
        "question": question,
        "top_k": request.top_k,
        "min_score": config.RETRIEVAL_MIN_SCORE,
        "results": hits,
    }


@app.get("/api/history", dependencies=[Depends(require_api_token)])
async def history() -> dict:
    """Consulte l'évolution des insertions dans Fluree."""
    try:
        entries = await asyncio.to_thread(fluree.history)
        return {"history": entries}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"Historique indisponible : {exc}"
        ) from exc


def _source_payload(hit: dict) -> dict:
    """Métadonnées de source + court extrait, sans renvoyer le chunk complet."""
    content = " ".join(str(hit.get("content") or "").split())
    if len(content) > 240:
        content = content[:240].rstrip() + "…"
    return {
        "source": hit["source"],
        "index": hit["index"],
        "score": hit["score"],
        "snippet": content,
    }


def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
