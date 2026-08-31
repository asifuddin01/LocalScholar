"""Dense vector storage, backed by Qdrant in embedded mode.

Embedded mode means Qdrant runs inside this process against a local directory:
no server, no container, no port to configure. That is the whole reason it was
chosen over a hosted vector database -- a `git clone` should not require
standing up infrastructure before it can index a PDF.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from backend.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"

# Stable namespace so a chunk id always maps to the same point id. Re-indexing
# a document overwrites its points instead of duplicating them.
_POINT_NAMESPACE = uuid.UUID("6f1c7f2e-2c8b-4d3a-9a2f-1c0d5e7b4a91")


@dataclass
class ScoredChunk:
    """A retrieved chunk plus the score that retrieved it."""

    chunk_id: str
    document_id: str
    filename: str
    title: str | None
    text: str
    page_number: int
    section: str | None
    score: float

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> "ScoredChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            title=chunk.title,
            text=chunk.text,
            page_number=chunk.page_number,
            section=chunk.section,
            score=score,
        )


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class VectorStore:
    def __init__(self, path: Path, dimension: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.dimension = dimension
        self._ensure_collection()

    def recreate(self) -> None:
        """Drop and rebuild the collection from scratch.

        The repair path for a desynced local index -- see RetrievalEngine.repair.
        """
        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)
        self._ensure_collection()
        logger.info("Vector collection recreated")

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.dimension, distance=models.Distance.COSINE
                ),
            )
            logger.info("Created vector collection (dim=%d)", self.dimension)

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vector,
                    # The payload duplicates what SQLite holds so that a search
                    # result is self-contained: no second round trip is needed
                    # to build a citation.
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "title": chunk.title,
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ],
        )

    def search(
        self, vector: list[float], top_k: int, document_ids: list[str] | None = None
    ) -> list[ScoredChunk]:
        query_filter = None
        if document_ids:
            # Filtering happens inside the search rather than afterwards, so
            # "ask these three papers" still returns a full top_k from those
            # papers instead of whatever survives a global top_k.
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=document_ids)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk_id=point.payload["chunk_id"],
                document_id=point.payload["document_id"],
                filename=point.payload["filename"],
                title=point.payload.get("title"),
                text=point.payload["text"],
                page_number=point.payload["page_number"],
                section=point.payload.get("section"),
                score=float(point.score),
            )
            for point in response.points
        ]

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
        )

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION_NAME).count

    def healthy(self) -> bool:
        """Can this collection actually serve a filtered search?

        Qdrant's embedded mode keeps parallel arrays for vectors and deletion
        masks, and deleting a document can leave them different lengths. The
        failure surfaces much later as
        `operands could not be broadcast together with shapes (706,) (707,)`
        on an unrelated query, so it is worth provoking cheaply at startup
        rather than in front of a user.
        """
        try:
            self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=[0.0] * self.dimension,
                limit=1,
                query_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=["__healthcheck__"])
                    )]
                ),
                with_payload=False,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector index failed its health check: %s", exc)
            return False

    def close(self) -> None:
        self.client.close()
