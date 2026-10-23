"""Point d'entrée de l'application FastAPI : API + interface web."""

import asyncio
import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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
    log.warning("Service %s non prêt après %ss — l'application continuera.", service, timeout)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _wait_for("fluree", fluree.is_ready)
    await _wait_for("ollama", ollama.is_ready)
    await asyncio.to_thread(fluree.ensure_ledger)
    yield


app = FastAPI(title="GraphRag-Fluree", lifespan=lifespan)


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
    }


@app.get("/api/gpu")
async def gpu() -> dict:
    """État GPU vu par Ollama (/api/ps) : vérifie que l'accélération est active."""
    try:
        return ollama.gpu_status()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Ollama injoignable : {exc}") from exc


@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Ingère des fichiers Markdown/texte/PDF dans le graphe Fluree."""
    results = []
    for upload in files:
        data = await upload.read()
        text = ingester.read_bytes(data, upload.filename or "fichier")
        chunks = await asyncio.to_thread(ingester.ingest_text, text, upload.filename or "fichier")
        results.append(
            {
                "file": upload.filename,
                "chars": len(text),
                "chunks": len(chunks),
            }
        )
    return {"ingested": results}


@app.post("/api/chat")
async def chat(request: dict) -> StreamingResponse:
    """Répond à une question : retrieval dans Fluree + génération streaming via Ollama."""
    question = (request.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question est vide.")

    async def sse():
        try:
            hits = await asyncio.to_thread(retriever.retrieve, question)
        except Exception as exc:  # noqa: BLE001
            yield _event({"type": "error", "message": f"Recherche impossible : {exc}"})
            return

        sources = [
            {"source": h["source"], "index": h["index"], "score": h["score"]} for h in hits
        ]
        yield _event({"type": "sources", "sources": sources})

        if not hits:
            for line in (
                "Je ne dispose pas d'information pertinente dans ma base de connaissances "
                "pour répondre à cette question. Ingérez d'abord des documents "
                "(onglet « Documents »)."
            ).split(" "):
                yield _event({"type": "token", "content": line + " "})
                await asyncio.sleep(0.02)
            yield _event({"type": "done", "sources": sources})
            return

        messages = prompts.build_messages(question, hits)
        deltas: queue.Queue = queue.Queue()

        def producer():
            try:
                ollama.chat_stream(messages, on_delta=deltas.put)
            except Exception as exc:  # noqa: BLE001
                deltas.put(exc)
            finally:
                deltas.put(None)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = deltas.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield _event({"type": "error", "message": f"Génération interrompue : {item}"})
                return
            yield _event({"type": "token", "content": item})

        yield _event({"type": "done", "sources": sources})

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/api/search")
async def search(request: dict) -> dict:
    """Recherche sémantique pure (sans génération) : renvoie les chunks les plus pertinents."""
    question = (request.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question est vide.")
    try:
        top_k = int(request.get("top_k", config.TOP_K))
    except (TypeError, ValueError):
        top_k = config.TOP_K
    hits = await asyncio.to_thread(retriever.retrieve, question, top_k=top_k)
    return {
        "question": question,
        "top_k": top_k,
        "min_score": config.RETRIEVAL_MIN_SCORE,
        "results": hits,
    }


def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")