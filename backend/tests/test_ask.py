
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes import ask


@pytest.fixture
def client(monkeypatch):
    source = SimpleNamespace(
        score=0.91,
        metadata={
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "filename": "guide.pdf",
            "text": "RAG combines retrieval with text generation.",
            "page_number": 2,
            "page_numbers": [2],
            "chunk_index": 0,
        },
    )

    monkeypatch.setattr(ask.rag_service, "search", lambda **kwargs: [source])

    def fake_generate(prompt, temperature=0.1):
        return json.dumps({
            "answer": "RAG combines retrieval with text generation.",
            "citation_chunk_ids": ["chunk-1"],
            "insufficient_evidence": False,
        })

    monkeypatch.setattr(ask.llm_service, "generate", fake_generate)

    return TestClient(app)


def test_ask_returns_grounded_answer_and_citation(client):
    response = client.post(
        "/api/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == (
        "RAG combines retrieval with text generation."
    )
    assert body["insufficient_evidence"] is False
    assert body["citations"][0]["chunk_id"] == "chunk-1"
    assert body["citations"][0]["filename"] == "guide.pdf"
    assert body["citations"][0]["page_numbers"] == [2]
    assert body["sources"][0]["text"] == (
        "RAG combines retrieval with text generation."
    )


def test_ask_returns_insufficient_evidence_when_no_results(
    client, monkeypatch
):
    def no_results(**kwargs):
        return []

    def should_not_generate(*args, **kwargs):
        pytest.fail("LLM should not run without retrieved evidence")

    monkeypatch.setattr(ask.rag_service, "search", no_results)
    monkeypatch.setattr(ask.llm_service, "generate", should_not_generate)

    response = client.post(
        "/api/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["citations"] == []
    assert body["sources"] == []


def test_ask_rejects_whitespace_question(client):
    response = client.post(
        "/api/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k(client):
    response = client.post(
        "/api/ask",
        json={"question": "What is RAG?", "top_k": 21},
    )

    assert response.status_code == 422


def test_ask_passes_document_filter_to_retrieval(client, monkeypatch):
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(ask.rag_service, "search", fake_search)

    response = client.post(
        "/api/ask",
        json={
            "question": "What is RAG?",
            "top_k": 3,
            "document_id": "doc-123",
        },
    )

    assert response.status_code == 200
    assert captured["query"] == "What is RAG?"
    assert captured["top_k"] == 3
    assert captured["document_id"] == "doc-123"


def test_ask_rejects_unknown_citation(client, monkeypatch):
    monkeypatch.setattr(
        ask.llm_service,
        "generate",
        lambda *args, **kwargs: json.dumps({
            "answer": "Unsupported answer",
            "citation_chunk_ids": ["made-up-chunk"],
            "insufficient_evidence": False,
        }),
    )

    response = client.post(
        "/api/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 502


def test_ask_handles_invalid_model_json(client, monkeypatch):
    monkeypatch.setattr(
        ask.llm_service,
        "generate",
        lambda *args, **kwargs: "This is not JSON",
    )

    response = client.post(
        "/api/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 502


def test_ask_handles_model_connection_failure(client, monkeypatch):
    from app.llm_service import LLMServiceError

    def failed_generate(*args, **kwargs):
        raise LLMServiceError("Ollama unavailable")

    monkeypatch.setattr(
        ask.llm_service,
        "generate",
        failed_generate,
    )

    response = client.post(
        "/api/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 502


def test_ask_abstains_when_model_reports_insufficient_evidence(
    client, monkeypatch
):
    monkeypatch.setattr(
        ask.llm_service,
        "generate",
        lambda *args, **kwargs: json.dumps({
            "answer": "The document does not say.",
            "citation_chunk_ids": [],
            "insufficient_evidence": True,
        }),
    )

    response = client.post(
        "/api/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["citations"] == []


def test_ask_deduplicates_citations(client, monkeypatch):
    monkeypatch.setattr(
        ask.llm_service,
        "generate",
        lambda *args, **kwargs: json.dumps({
            "answer": "RAG combines retrieval with generation.",
            "citation_chunk_ids": ["chunk-1", "chunk-1"],
            "insufficient_evidence": False,
        }),
    )

    response = client.post(
        "/api/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200
    assert len(response.json()["citations"]) == 1
