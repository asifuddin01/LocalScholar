"""LocalScholar API entry point.

Run with:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import ask, documents, research, search
from backend.config import load_config
from backend.services.library import LibraryService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("localscholar")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    library = LibraryService(config)
    app.state.config = config
    app.state.library = library

    # Rebuilds the in-memory BM25 index and reports documents that were left
    # half-indexed by an interrupted run.
    stale = library.start()
    if stale:
        # A worker thread, so a large re-index never blocks the API from
        # starting. The documents involved already show as needing work.
        threading.Thread(
            target=lambda: [library.process_document(doc_id) for doc_id in stale],
            name="localscholar-reindex",
            daemon=True,
        ).start()

    # Pull the model into memory in the background so the first question the
    # user asks is not the one that pays for loading it.
    if hasattr(library.llm, "warm_up"):
        threading.Thread(target=library.llm.warm_up, name="llm-warmup", daemon=True).start()

    logger.info(
        "LocalScholar ready. Data: %s | %d chunks indexed",
        config.storage.data_dir, len(library.engine.bm25),
    )
    try:
        yield
    finally:
        library.close()


app = FastAPI(
    title="LocalScholar",
    description="Privacy-first local LLM research paper assistant.",
    version="0.1.0",
    lifespan=lifespan,
)

# The Vite dev server runs on a different port during development. In the
# Docker build the frontend is served as static files from this same origin,
# so this only matters for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ask.router)
app.include_router(research.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
