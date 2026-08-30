"""Embedding providers.

The interface exists so the vector store and the indexing service never import
a specific model library. Today there is one implementation (local ONNX); the
point of the abstraction is that swapping it does not touch anything else.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

# Batches larger than this deadlock onnxruntime when inference runs off the
# main thread on macOS/ARM -- see FastEmbedProvider.embed_documents.
DEFAULT_BATCH_SIZE = 32

# Retrieval models in the BGE and Arctic families are trained with an
# instruction prefix on the *query* side only. fastembed exposes a
# `query_embed()` method that looks like it handles this, but for these models
# it is a straight alias for `embed()` -- verified by comparing the two vectors
# for the same string, which come back identical. So the prefix is applied
# here instead.
#
# It is not cosmetic. Measured over a 715-chunk corpus of four real papers:
#   "What are the limitations of this approach?"
#       without prefix -> an IEEE copyright banner and a running header
#       with prefix    -> "F. Limitation: One limitation of this study is..."
#   "How many images are in the dataset?"
#       without prefix -> a generic "overview of the models" paragraph
#       with prefix    -> "The dataset includes 13,599 fruit images..."
# Score spread between the best and third-best hit widened from 0.04 to 0.09,
# i.e. the ranking became meaningfully more discriminative.
_BGE_STYLE_PREFIX = "Represent this sentence for searching relevant passages: "
QUERY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("arctic-embed", _BGE_STYLE_PREFIX),
    ("bge-small-en", _BGE_STYLE_PREFIX),
    ("bge-base-en", _BGE_STYLE_PREFIX),
    ("bge-large-en", _BGE_STYLE_PREFIX),
)


def query_prefix_for(model_name: str) -> str:
    """The instruction prefix this model expects on queries, if any.

    Returns "" for models trained symmetrically (MiniLM and friends), where
    adding a prefix would hurt rather than help.
    """
    lowered = model_name.lower()
    for fragment, prefix in QUERY_PREFIXES:
        if fragment in lowered:
            return prefix
    return ""

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Documents and queries are embedded separately, on purpose.

    Retrieval models like BGE and E5 are trained asymmetrically: queries get a
    short instruction prefix that passages do not. Embedding a question the
    same way as a passage measurably degrades recall, so the two calls stay
    distinct rather than collapsing into one `embed()`.
    """

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedProvider(EmbeddingProvider):
    """Local embeddings via fastembed / onnxruntime.

    Runs on CPU with no PyTorch dependency. The model is downloaded from
    Hugging Face once (a few hundred MB) and cached; after that, indexing and
    search work with no network access at all.

    Loading is lazy so that starting the API, listing papers, or running tests
    that never embed anything doesn't pay for model initialisation.
    """

    def __init__(
        self,
        model_name: str = "snowflake/snowflake-arctic-embed-xs",
        cache_dir: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._batch_size = max(1, batch_size)
        self._model = None
        self._dimension: int | None = None
        # Serialises inference. Two documents indexing at once would otherwise
        # each allocate a batch of activations, and on an 8GB machine that is
        # the difference between working and swapping.
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("Loading embedding model %s", self._model_name)
            self._model = TextEmbedding(
                model_name=self._model_name, cache_dir=self._cache_dir
            )
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            from fastembed import TextEmbedding

            for spec in TextEmbedding.list_supported_models():
                if spec["model"].lower() == self._model_name.lower():
                    self._dimension = int(spec["dim"])
                    break
            else:
                # Unknown to the registry: ask the model itself.
                self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages, in deliberately small batches.

        The batch size is not a throughput knob, it is a correctness one.
        Indexing runs in a background thread so uploads stay responsive, and
        onnxruntime's default batch of 256 *deadlocks* inside `session.run()`
        when inference happens off the main thread on macOS/ARM -- no error, no
        timeout, the thread simply stops. Measured on an M1 with a 43-page
        paper (321 chunks):

            batch 256 -> hangs indefinitely
            batch 128 -> 24.5s
            batch  64 -> 18.7s
            batch  32 -> 20.5s
            batch  16 -> 20.1s

        32 is the default: within 10% of the best time, and far enough below
        the cliff to stay safe on a loaded machine. Configurable via
        `embeddings.batch_size` for anyone on hardware that behaves better.
        """
        if not texts:
            return []
        with self._lock:
            model = self._load()
            return [
                vector.tolist()
                for vector in model.embed(texts, batch_size=self._batch_size)
            ]

    def embed_query(self, text: str) -> list[float]:
        """Embed a question, with the model's query instruction prefix."""
        prefixed = f"{query_prefix_for(self._model_name)}{text}"
        with self._lock:
            return list(self._load().embed([prefixed], batch_size=1))[0].tolist()


def create_embedding_provider(
    provider: str,
    model: str,
    cache_dir: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingProvider:
    if provider in ("fastembed", "local"):
        return FastEmbedProvider(model_name=model, cache_dir=cache_dir, batch_size=batch_size)
    raise ValueError(
        f"Unknown embedding provider {provider!r}. Supported: 'fastembed' (local)."
    )
