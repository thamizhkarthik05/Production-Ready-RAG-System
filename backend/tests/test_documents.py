
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.document_processor import DocumentChunk
from app.rag_service import RAGService
from app.storage import SQLiteStorage
from app.vector_store import VectorStore


class FakeEmbeddingService:
    """Deterministic embeddings for offline tests."""

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
def rag_service(tmp_path, monkeypatch):
    """Create an isolated RAG service and SQLite database."""

    storage = SQLiteStorage(
        database_path=tmp_path / "test_documents.db"
    )

    

    service = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=VectorStore(dimension=768),
        storage_backend=storage,
        index_path=tmp_path / "test_index.faiss",
    )

    # All routes involved in document management must use
    # the same isolated RAG service.
    monkeypatch.setattr(
        "app.routes.documents.rag_service",
        service,
    )
    monkeypatch.setattr(
        "app.routes.search.rag_service",
        service,
    )
    monkeypatch.setattr(
        "app.routes.ask.rag_service",
        service,
    )

    # The list endpoint reads from SQLite.
    monkeypatch.setattr(
        "app.routes.documents.storage",
        storage,
    )

    return service


@pytest.fixture
def client(rag_service):
    """Create a test client for the FastAPI application."""

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


def test_list_documents_returns_empty_list(client):
    """An empty database should return zero documents."""

    response = client.get("/api/documents/")

    assert response.status_code == 200

    body = response.json()

    assert body["document_count"] == 0
    assert body["documents"] == []


def test_list_documents_returns_uploaded_document_metadata(
    client,
    rag_service,
):
    """The list endpoint should return indexed document metadata."""

    rag_service.index_chunks([
        make_chunk(
            chunk_id="chunk-python",
            document_id="doc-python",
            filename="python.txt",
            text="Python is a programming language.",
        )
    ])

    response = client.get("/api/documents/")

    assert response.status_code == 200

    body = response.json()

    assert body["document_count"] == 1
    assert len(body["documents"]) == 1

    document = body["documents"][0]

    assert document["document_id"] == "doc-python"
    assert document["filename"] == "python.txt"
    assert document["chunk_count"] == 1


def test_delete_document_removes_document_and_chunks(
    client,
    rag_service,
):
    """Deleting a document should remove it from storage and search."""

    rag_service.index_chunks([
        make_chunk(
            chunk_id="chunk-python",
            document_id="doc-python",
            filename="python.txt",
            text="Python is a programming language.",
        )
    ])

    response = client.delete("/api/documents/doc-python")

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-python"

    # The document should no longer appear in the list.
    list_response = client.get("/api/documents/")

    assert list_response.status_code == 200
    assert list_response.json()["document_count"] == 0
    assert list_response.json()["documents"] == []

    # Search should no longer find the deleted document.
    search_response = client.post(
        "/api/search",
        json={
            "query": "Python programming",
            "top_k": 5,
        },
    )

    assert search_response.status_code == 200
    assert search_response.json()["result_count"] == 0


def test_delete_missing_document_returns_404(client):
    """Deleting a nonexistent document should return 404."""

    response = client.delete("/api/documents/does-not-exist")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
