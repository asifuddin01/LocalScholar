"""Unit tests for the two retrievers, independent of the API."""

from __future__ import annotations

import pytest

from backend.ingestion.chunking import Chunk
from backend.retrieval.bm25_index import BM25Index, tokenize
from backend.retrieval.vector_store import VectorStore
from tests.conftest import HashingEmbeddingProvider


def make_chunk(ordinal, text, document_id="docA", page=1, section="Method"):
    return Chunk(
        chunk_id=f"{document_id}:{ordinal}",
        document_id=document_id,
        filename=f"{document_id}.pdf",
        text=text,
        page_number=page,
        section=section,
        ordinal=ordinal,
        title=f"Title of {document_id}",
    )


CORPUS = [
    make_chunk(0, "We train on the KiTS19 dataset of 300 annotated CT scans.", page=4),
    make_chunk(1, "The U-Net baseline reaches a Dice score of 0.87.", page=6),
    make_chunk(2, "Optimisation uses AdamW with a cosine schedule.", page=5),
    make_chunk(3, "We evaluate on BraTS with 500 cases.", document_id="docB", page=2),
]


# --- tokenizer --------------------------------------------------------------

def test_tokenizer_keeps_dataset_names_with_digits():
    assert "kits19" in tokenize("We used KiTS19.")
    assert "cifar-10" in tokenize("CIFAR-10")


def test_tokenizer_indexes_hyphenated_terms_both_ways():
    """So "U-Net", "u net" and "net" all reach the same chunk."""
    tokens = tokenize("U-Net")
    assert {"u-net", "u", "net"} <= set(tokens)


def test_tokenizer_is_shared_by_index_and_query():
    assert tokenize("Dice Score") == tokenize("dice score")


# --- BM25 -------------------------------------------------------------------

@pytest.fixture
def bm25():
    index = BM25Index()
    index.build(CORPUS)
    return index


def test_bm25_finds_an_exact_rare_term(bm25):
    """The case dense retrieval is worst at: a literal dataset name."""
    results = bm25.search("KiTS19", top_k=3)
    assert results[0].text.startswith("We train on the KiTS19")
    assert results[0].page_number == 4


def test_bm25_result_carries_citation_metadata(bm25):
    result = bm25.search("Dice score", top_k=1)[0]
    assert (result.page_number, result.section) == (6, "Method")
    assert result.filename == "docA.pdf"
    assert result.chunk_id == "docA:1"


def test_bm25_filters_to_selected_documents(bm25):
    results = bm25.search("dataset cases", top_k=10, document_ids=["docB"])
    assert results
    assert {r.document_id for r in results} == {"docB"}


def test_bm25_returns_nothing_when_no_term_matches(bm25):
    """Padding out top_k with zero-scoring chunks would feed the LLM noise."""
    assert bm25.search("zebra helicopter", top_k=5) == []


def test_bm25_empty_index_is_safe():
    assert BM25Index().search("anything", top_k=5) == []


def test_bm25_rebuild_replaces_the_corpus(bm25):
    bm25.build(CORPUS[:1])
    assert len(bm25) == 1
    assert bm25.search("AdamW", top_k=5) == []


# --- vector store -----------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    provider = HashingEmbeddingProvider()
    store = VectorStore(path=tmp_path / "qdrant", dimension=provider.dimension)
    store.upsert_chunks(CORPUS, provider.embed_documents([c.search_text for c in CORPUS]))
    yield store, provider
    store.close()


def test_vector_search_returns_scored_chunks_with_metadata(store):
    vector_store, provider = store
    results = vector_store.search(provider.embed_query("KiTS19 dataset"), top_k=2)
    assert results
    assert results[0].page_number == 4
    assert results[0].section == "Method"
    assert results[0].score > 0


def test_vector_search_filters_inside_the_search(store):
    """Filtering after the fact would return fewer than top_k from the selection."""
    vector_store, provider = store
    results = vector_store.search(
        provider.embed_query("dataset"), top_k=5, document_ids=["docB"]
    )
    assert {r.document_id for r in results} == {"docB"}


def test_reindexing_the_same_chunks_does_not_duplicate_points(store):
    vector_store, provider = store
    before = vector_store.count()
    vector_store.upsert_chunks(CORPUS, provider.embed_documents([c.search_text for c in CORPUS]))
    assert vector_store.count() == before


def test_deleting_a_document_removes_only_its_points(store):
    vector_store, _ = store
    vector_store.delete_document("docB")
    assert vector_store.count() == 3


# --- query prefixes ---------------------------------------------------------

def test_asymmetric_models_get_a_query_prefix():
    """fastembed's query_embed() is an alias for embed() on these models.

    Without the prefix, "What are the limitations of this approach?" retrieved
    an IEEE copyright banner instead of the paper's Limitations section.
    """
    from backend.retrieval.embeddings import query_prefix_for

    assert query_prefix_for("snowflake/snowflake-arctic-embed-xs").startswith("Represent")
    assert query_prefix_for("BAAI/bge-small-en-v1.5").startswith("Represent")


def test_symmetric_models_get_no_prefix():
    """Adding one to a symmetrically-trained model would hurt, not help."""
    from backend.retrieval.embeddings import query_prefix_for

    assert query_prefix_for("sentence-transformers/all-MiniLM-L6-v2") == ""
