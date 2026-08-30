"""Configuration loading.

One YAML file, one dataclass tree, loaded once at startup. Nothing reads
settings from the environment except the two that legitimately differ per
machine: the config file location and the data directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from backend.ingestion.chunking import ChunkingConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class StorageConfig:
    data_dir: Path = REPO_ROOT / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def catalogue_path(self) -> Path:
        return self.data_dir / "localscholar.db"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"


@dataclass
class IngestionConfig:
    max_upload_mb: int = 50

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@dataclass
class EmbeddingsConfig:
    provider: str = "fastembed"
    model: str = "snowflake/snowflake-arctic-embed-xs"
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 6
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0


@dataclass
class RerankerConfig:
    enabled: bool = True
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Same onnxruntime background-thread constraint as the embedder.
    batch_size: int = 16
    candidates: int = 24


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen2.5:3b-instruct"
    host: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 8192
    timeout_seconds: int = 180
    # Keeps the model resident between questions: without it, Ollama evicts it
    # after a few minutes idle and the next answer pays a ~30s reload.
    keep_alive: str = "30m"
    # Only used by the opt-in openai-compatible provider.
    base_url: str | None = None
    api_key: str | None = None
    # Hard cap on how much retrieved text is put in front of the model.
    context_char_budget: int = 6000


@dataclass
class Config:
    storage: StorageConfig = field(default_factory=StorageConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def ensure_directories(self) -> None:
        self.storage.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.storage.model_cache_dir.mkdir(parents=True, exist_ok=True)


def _resolve(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML, falling back to dataclass defaults."""
    config_path = Path(path or os.environ.get("LOCALSCHOLAR_CONFIG") or DEFAULT_CONFIG_PATH)
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}

    storage_raw = raw.get("storage", {}) or {}
    data_dir = os.environ.get("LOCALSCHOLAR_DATA_DIR") or storage_raw.get("data_dir")
    storage = StorageConfig(
        data_dir=_resolve(data_dir) if data_dir else StorageConfig().data_dir
    )

    ingestion_raw = raw.get("ingestion", {}) or {}
    ingestion = IngestionConfig(
        max_upload_mb=int(ingestion_raw.get("max_upload_mb", IngestionConfig().max_upload_mb))
    )

    chunking_defaults = ChunkingConfig()
    chunking_raw = raw.get("chunking", {}) or {}
    excluded = chunking_raw.get("exclude_sections")
    chunking = ChunkingConfig(
        chunk_size=int(chunking_raw.get("chunk_size", chunking_defaults.chunk_size)),
        overlap=int(chunking_raw.get("overlap", chunking_defaults.overlap)),
        min_chunk_size=int(
            chunking_raw.get("min_chunk_size", chunking_defaults.min_chunk_size)
        ),
        exclude_sections=(
            tuple(s.strip().lower() for s in excluded)
            if excluded is not None
            else chunking_defaults.exclude_sections
        ),
    )

    embeddings_raw = raw.get("embeddings", {}) or {}
    embeddings = EmbeddingsConfig(
        provider=str(embeddings_raw.get("provider", EmbeddingsConfig().provider)),
        model=str(embeddings_raw.get("model", EmbeddingsConfig().model)),
        batch_size=int(embeddings_raw.get("batch_size", EmbeddingsConfig().batch_size)),
    )

    retrieval_defaults = RetrievalConfig()
    retrieval_raw = raw.get("retrieval", {}) or {}
    retrieval = RetrievalConfig(
        dense_top_k=int(retrieval_raw.get("dense_top_k", retrieval_defaults.dense_top_k)),
        bm25_top_k=int(retrieval_raw.get("bm25_top_k", retrieval_defaults.bm25_top_k)),
        final_top_k=int(retrieval_raw.get("final_top_k", retrieval_defaults.final_top_k)),
    )

    retrieval = RetrievalConfig(
        dense_top_k=retrieval.dense_top_k,
        bm25_top_k=retrieval.bm25_top_k,
        final_top_k=retrieval.final_top_k,
        rrf_k=int(retrieval_raw.get("rrf_k", retrieval_defaults.rrf_k)),
        dense_weight=float(retrieval_raw.get("dense_weight", retrieval_defaults.dense_weight)),
        bm25_weight=float(retrieval_raw.get("bm25_weight", retrieval_defaults.bm25_weight)),
    )

    reranker_defaults = RerankerConfig()
    reranker_raw = raw.get("reranker", {}) or {}
    reranker = RerankerConfig(
        enabled=bool(reranker_raw.get("enabled", reranker_defaults.enabled)),
        model=str(reranker_raw.get("model", reranker_defaults.model)),
        batch_size=int(reranker_raw.get("batch_size", reranker_defaults.batch_size)),
        candidates=int(reranker_raw.get("candidates", reranker_defaults.candidates)),
    )

    llm_defaults = LLMConfig()
    llm_raw = raw.get("llm", {}) or {}
    llm = LLMConfig(
        provider=str(llm_raw.get("provider", llm_defaults.provider)),
        model=str(llm_raw.get("model", llm_defaults.model)),
        host=str(llm_raw.get("host", llm_defaults.host)),
        temperature=float(llm_raw.get("temperature", llm_defaults.temperature)),
        num_ctx=int(llm_raw.get("num_ctx", llm_defaults.num_ctx)),
        timeout_seconds=int(llm_raw.get("timeout_seconds", llm_defaults.timeout_seconds)),
        keep_alive=str(llm_raw.get("keep_alive", llm_defaults.keep_alive)),
        base_url=llm_raw.get("base_url") or os.environ.get("LOCALSCHOLAR_LLM_BASE_URL"),
        api_key=llm_raw.get("api_key") or os.environ.get("LOCALSCHOLAR_LLM_API_KEY"),
        context_char_budget=int(
            llm_raw.get("context_char_budget", llm_defaults.context_char_budget)
        ),
    )

    return Config(
        storage=storage,
        ingestion=ingestion,
        chunking=chunking,
        embeddings=embeddings,
        retrieval=retrieval,
        reranker=reranker,
        llm=llm,
    )
