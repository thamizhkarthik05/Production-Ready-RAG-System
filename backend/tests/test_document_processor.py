
from io import BytesIO

import fitz
import pytest
from fastapi.testclient import TestClient

from app.document_processor import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    MAX_FILE_SIZE_BYTES,
    DocumentChunk,
    DocumentProcessingError,
    ExtractedPage,
    chunk_document,
    extract_document,
    process_document,
)
from app.main import app


client = TestClient(app)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def create_pdf(text: str = "Sample PDF document content.") -> bytes:
    """Create a small in-memory PDF for testing."""
    pdf_bytes = BytesIO()

    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), text)
        document.save(pdf_bytes)

    return pdf_bytes.getvalue()


# ---------------------------------------------------------------------
# Document extraction tests
# ---------------------------------------------------------------------

def test_extract_text_file():
    pages = extract_document("sample.txt", b"Hello document")

    assert len(pages) == 1
    assert pages[0].text == "Hello document"
    assert pages[0].page_number is None


def test_extract_markdown_file():
    content = b"# Heading\n\nMarkdown content."
    pages = extract_document("sample.md", content)

    assert len(pages) == 1
    assert "# Heading" in pages[0].text
    assert "Markdown content." in pages[0].text


def test_extract_pdf_file():
    pages = extract_document("sample.pdf", create_pdf())

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Sample PDF document content." in pages[0].text


def test_reject_unsupported_file_type():
    with pytest.raises(
        DocumentProcessingError,
        match="Unsupported file type",
    ):
        extract_document("sample.docx", b"Document content")


def test_reject_empty_file():
    with pytest.raises(
        DocumentProcessingError,
        match="empty",
    ):
        extract_document("sample.txt", b"")


def test_reject_file_over_size_limit():
    content = b"a" * (MAX_FILE_SIZE_BYTES + 1)

    with pytest.raises(
        DocumentProcessingError,
        match="10 MB",
    ):
        extract_document("sample.txt", content)


def test_reject_invalid_utf8_text():
    with pytest.raises(
        DocumentProcessingError,
        match="UTF-8",
    ):
        extract_document("sample.txt", b"\xff\xfe\xfa")


def test_reject_unreadable_pdf():
    with pytest.raises(
        DocumentProcessingError,
        match="PDF could not be read",
    ):
        extract_document("sample.pdf", b"not a valid PDF")


def test_reject_textless_pdf():
    with fitz.open() as document:
        document.new_page()
        pdf_content = document.tobytes()

    with pytest.raises(
        DocumentProcessingError,
        match="No readable text",
    ):
        extract_document("empty.pdf", pdf_content)


# ---------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------

def test_chunk_document_returns_chunks():
    pages = [
        ExtractedPage(
            text="This is a sample document. " * 100,
            page_number=1,
        )
    ]

    chunks = chunk_document("sample.txt", pages)

    assert len(chunks) >= 1
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)


def test_chunk_document_respects_chunk_size():
    pages = [
        ExtractedPage(
            text=" ".join(["document"] * 2000),
            page_number=1,
        )
    ]

    chunks = chunk_document(
        "sample.txt",
        pages,
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) > 1
    assert all(chunk.token_count <= 100 for chunk in chunks)


def test_chunk_document_rejects_invalid_chunk_size():
    pages = [ExtractedPage(text="Some content", page_number=1)]

    with pytest.raises(
        DocumentProcessingError,
        match="greater than zero",
    ):
        chunk_document("sample.txt", pages, chunk_size=0)


def test_chunk_document_rejects_invalid_overlap():
    pages = [ExtractedPage(text="Some content", page_number=1)]

    with pytest.raises(
        DocumentProcessingError,
        match="smaller than chunk size",
    ):
        chunk_document(
            "sample.txt",
            pages,
            chunk_size=100,
            overlap=100,
        )


def test_chunk_document_preserves_document_id():
    pages = [ExtractedPage(text="Sample content", page_number=1)]

    chunks = chunk_document(
        "sample.txt",
        pages,
        document_id="test-document",
    )

    assert chunks
    assert all(
        chunk.document_id == "test-document"
        for chunk in chunks
    )


def test_chunk_document_preserves_page_metadata():
    pages = [
        ExtractedPage(
            text="Content from the first page.",
            page_number=1,
        ),
        ExtractedPage(
            text="Content from the second page.",
            page_number=2,
        ),
    ]

    chunks = chunk_document("sample.pdf", pages)

    assert chunks
    assert any(
        1 in chunk.page_numbers
        for chunk in chunks
    )
    assert any(
        2 in chunk.page_numbers
        for chunk in chunks
    )


def test_chunk_document_empty_pages_returns_empty_list():
    chunks = chunk_document("sample.txt", [])

    assert chunks == []


def test_process_document_returns_id_and_chunks():
    document_id, chunks = process_document(
        "sample.txt",
        b"Sample document content for processing.",
    )

    assert document_id
    assert chunks
    assert all(
        chunk.document_id == document_id
        for chunk in chunks
    )


# ---------------------------------------------------------------------
# Upload endpoint tests
# ---------------------------------------------------------------------

def test_upload_endpoint_returns_chunks(monkeypatch):
    """Test upload without loading the real embedding model."""

    # Mock indexing so this endpoint test remains fast and offline.
    monkeypatch.setattr(
        "app.routes.documents.rag_service.index_chunks",
        lambda chunks: len(chunks),
    )

    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "sample.txt",
                b"This is a sample document for upload testing.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Document processed successfully."
    assert data["filename"] == "sample.txt"
    assert data["document_id"]
    assert data["chunk_count"] >= 1
    assert data["indexed_chunk_count"] == data["chunk_count"]
    assert len(data["chunks"]) == data["chunk_count"]

    first_chunk = data["chunks"][0]

    assert first_chunk["chunk_id"]
    assert first_chunk["document_id"] == data["document_id"]
    assert first_chunk["filename"] == "sample.txt"
    assert first_chunk["text"]
    assert first_chunk["token_count"] > 0
    assert "page_numbers" in first_chunk


def test_upload_endpoint_rejects_unsupported_file(monkeypatch):
    """Unsupported extensions should fail before indexing."""

    monkeypatch.setattr(
        "app.routes.documents.rag_service.index_chunks",
        lambda chunks: len(chunks),
    )

    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "sample.docx",
                b"Unsupported document",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_endpoint_rejects_empty_file(monkeypatch):
    """An empty file should produce a client error."""

    monkeypatch.setattr(
        "app.routes.documents.rag_service.index_chunks",
        lambda chunks: len(chunks),
    )

    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_endpoint_rejects_oversized_file():
    """Oversized uploads should be rejected before processing."""

    content = b"a" * (MAX_FILE_SIZE_BYTES + 1)

    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "large.txt",
                content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


def test_upload_endpoint_reports_indexing_failure(monkeypatch):
    """An indexing failure should return HTTP 500."""

    def fail_indexing(chunks):
        raise RuntimeError("Simulated indexing failure")

    monkeypatch.setattr(
        "app.routes.documents.rag_service.index_chunks",
        fail_indexing,
    )

    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "sample.txt",
                b"Content to test indexing failure.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "The document was processed but could not be indexed."
    )
