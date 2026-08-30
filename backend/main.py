"""LocalScholar API entry point.

Run with:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import documents
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
    app.state.config = config
    app.state.library = LibraryService(config)
    logger.info("LocalScholar ready. Data directory: %s", config.storage.data_dir)
    yield


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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
