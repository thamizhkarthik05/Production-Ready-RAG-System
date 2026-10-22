
import numpy as np
import pytest

from app.document_processor import DocumentChunk
from app.rag_service import RAGService
from app.vector_store import VectorStore



class FakeEmbeddingService:
    """Deterministic 768-dimensional embeddings for fast, offline tests."""

    dimension = 768

    def embed_documents(self, texts):
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for index, text in enumerate(texts):
            lowered = text.lower()

            if "python" in lowered:
                vectors[index, 0] = 1.0
            elif "database" in lowered:
                vectors[index, 1] = 1.0
            else:
                vectors[index, 2] = 1.0

        return vectors

    def embed_queries(self, queries):
        return self.embed_documents(queries)


def make_chunk(
    document_id,
    chunk_id,
    text,
    chunk_index=0,
    page_number=1,
):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="sample.txt",
        chunk_index=chunk_index,
        text=text,
        token_count=10,
        page_number=page_number,
        page_numbers=(page_number,),
        token_start=0,
        token_end=10,
    )


@pytest.fixture
def service():
    return RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=VectorStore(dimension=768),
    )


def test_index_chunks_stores_vectors_and_metadata(service):
    chunks = [
        make_chunk("doc-1", "chunk-1", "Python programming"),
        make_chunk("doc-1", "chunk-2", "Database systems", 1, 2),
    ]

    count = service.index_chunks(chunks)

    assert count == 2
    assert service.indexed_document_count == 1
    assert service.indexed_chunk_count == 2


def test_search_returns_relevant_chunk(service):
    service.index_chunks(
        [
            make_chunk("doc-1", "chunk-1", "Python programming"),
            make_chunk("doc-1", "chunk-2", "Database systems", 1),
        ]
    )

    results = service.search("Python", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "chunk-1"
    assert results[0].metadata["text"] == "Python programming"


def test_search_preserves_source_page_metadata(service):
    service.index_chunks(
        [make_chunk("doc-1", "chunk-1", "Database systems", page_number=3)]
    )

    results = service.search("database")

    assert results[0].metadata["page_number"] == 3
    assert results[0].metadata["page_numbers"] == [3]


def test_search_can_be_limited_to_a_document(service):
    service.index_chunks(
        [make_chunk("doc-1", "chunk-1", "Python programming")]
    )
    service.index_chunks(
        [make_chunk("doc-2", "chunk-2", "Python language guide")]
    )

    results = service.search("Python", document_id="doc-2")

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc-2"


def test_reindexing_document_replaces_old_chunks(service):
    service.index_chunks(
        [
            make_chunk("doc-1", "old-1", "Python programming"),
            make_chunk("doc-1", "old-2", "Python development", 1),
        ]
    )

    service.index_chunks(
        [make_chunk("doc-1", "new-1", "Database systems")]
    )

    assert service.indexed_document_count == 1
    assert service.indexed_chunk_count == 1

    results = service.search("database")

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "new-1"


def test_empty_index_returns_no_results(service):
    assert service.search("Python") == []


def test_rejects_chunks_from_multiple_documents(service):
    chunks = [
        make_chunk("doc-1", "chunk-1", "Python"),
        make_chunk("doc-2", "chunk-2", "Database"),
    ]

    with pytest.raises(ValueError, match="same document"):
        service.index_chunks(chunks)


def test_rejects_empty_query(service):
    with pytest.raises(ValueError, match="non-empty"):
        service.search("   ")


def test_rejects_invalid_top_k(service):
    with pytest.raises(ValueError, match="top_k"):
        service.search("Python", top_k=0)
