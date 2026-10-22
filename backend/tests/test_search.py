
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.document_processor import DocumentChunk
from app.rag_service import RAGService
from app.vector_store import VectorStore


class FakeEmbeddingService:
    """Deterministic 768-dimensional embeddings for offline tests."""

    dimension = 768

    def embed_documents(self, texts):
        vectors = np.zeros(
            (len(texts), self.dimension),
            dtype=np.float32,
        )

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


@pytest.fixture
def rag_service(monkeypatch):
    """Create an isolated RAG service for each test."""

    service = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=VectorStore(dimension=768),
    )

    # The API route and test must use the same service instance.
    monkeypatch.setattr(
        "app.routes.search.rag_service",
        service,
    )

    return service


@pytest.fixture
def client(rag_service):
    """Create a test client using the isolated RAG service."""

    with TestClient(app) as test_client:
        yield test_client


def make_chunk(
    chunk_id: str,
    document_id: str,
    filename: str,
    text: str,
) -> DocumentChunk:
    """Build a document chunk for testing."""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        text=text,
        chunk_index=0,
        page_number=1,
        page_numbers=[1],
        token_count=len(text.split()),
        token_start=0,
        token_end=len(text.split()),
    )


def test_search_returns_results_and_source_metadata(client):
    """An empty index should return an empty result list."""

    response = client.post(
        "/api/search",
        json={
            "query": "Python programming",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "Python programming"
    assert body["result_count"] == 0
    assert body["results"] == []


def test_search_returns_indexed_document_chunks(client, rag_service):
    """Search should return matching text and source metadata."""

    rag_service.index_chunks([
        make_chunk(
            chunk_id="chunk-python",
            document_id="doc-python",
            filename="python.txt",
            text="Python is a programming language.",
        )
    ])

    response = client.post(
        "/api/search",
        json={
            "query": "Python programming",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "Python programming"
    assert body["result_count"] == 1

    result = body["results"][0]

    assert result["document_id"] == "doc-python"
    assert result["chunk_id"] == "chunk-python"
    assert result["filename"] == "python.txt"
    assert result["text"] == "Python is a programming language."
    assert result["page_number"] == 1
    assert result["page_numbers"] == [1]
    assert result["chunk_index"] == 0
    assert isinstance(result["score"], float)


def test_search_rejects_whitespace_only_query(client):
    """A query containing only whitespace should be rejected."""

    response = client.post(
        "/api/search",
        json={
            "query": "   ",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_search_rejects_invalid_top_k(client, top_k):
    """top_k must be between 1 and 20."""

    response = client.post(
        "/api/search",
        json={
            "query": "Python",
            "top_k": top_k,
        },
    )

    assert response.status_code == 422


def test_search_can_filter_by_document_id(client, rag_service):
    """Search restricted to a document must not return other documents."""

    rag_service.index_chunks([
        make_chunk(
            chunk_id="python-1",
            document_id="doc-python",
            filename="python.txt",
            text="Python is a programming language.",
        )
    ])

    rag_service.index_chunks([
        make_chunk(
            chunk_id="database-1",
            document_id="doc-database",
            filename="database.txt",
            text="Database management systems store information.",
        )
    ])

    response = client.post(
        "/api/search",
        json={
            "query": "Python",
            "top_k": 5,
            "document_id": "doc-database",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert all(
        result["document_id"] == "doc-database"
        for result in body["results"]
    )

    assert body["result_count"] == len(body["results"])
