
import numpy as np
import pytest

from app.embedding_service import (
    EMBEDDING_DIMENSION,
    QUERY_INSTRUCTION,
    EmbeddingService,
)


class FakeEmbeddingModel:
    """Predictable model for testing without loading the real AI model."""

    def __init__(self):
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append((sentences, kwargs))

        vectors = np.zeros(
            (len(sentences), EMBEDDING_DIMENSION),
            dtype=np.float32,
        )
        vectors[:, 0] = 1.0
        return vectors


def test_embed_documents_returns_normalized_vectors():
    model = FakeEmbeddingModel()
    service = EmbeddingService(model=model)

    vectors = service.embed_documents(
        ["First document chunk", "Second document chunk"]
    )

    assert vectors.shape == (2, 768)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1),
        [1.0, 1.0],
    )


def test_embed_queries_adds_bge_instruction():
    model = FakeEmbeddingModel()
    service = EmbeddingService(model=model)

    service.embed_queries(["What is RAG?"])

    encoded_texts, kwargs = model.calls[0]

    assert encoded_texts == [
        f"{QUERY_INSTRUCTION}What is RAG?"
    ]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True


def test_empty_documents_return_correct_shape():
    service = EmbeddingService(model=FakeEmbeddingModel())

    vectors = service.embed_documents([])

    assert vectors.shape == (0, 768)
    assert vectors.dtype == np.float32


def test_empty_queries_return_correct_shape():
    service = EmbeddingService(model=FakeEmbeddingModel())

    vectors = service.embed_queries([])

    assert vectors.shape == (0, 768)


@pytest.mark.parametrize(
    "texts",
    [
        [""],
        ["   "],
        ["valid text", ""],
    ],
)
def test_documents_reject_empty_text(texts):
    service = EmbeddingService(model=FakeEmbeddingModel())

    with pytest.raises(ValueError, match="non-empty"):
        service.embed_documents(texts)


def test_queries_reject_empty_text():
    service = EmbeddingService(model=FakeEmbeddingModel())

    with pytest.raises(ValueError, match="non-empty"):
        service.embed_queries(["  "])


def test_service_reuses_injected_model():
    model = FakeEmbeddingModel()
    service = EmbeddingService(model=model)

    service.embed_documents(["First"])
    service.embed_documents(["Second"])

    assert len(model.calls) == 2


def test_invalid_embedding_dimension_is_rejected():
    class WrongDimensionModel:
        def encode(self, sentences, **kwargs):
            return np.ones((len(sentences), 384), dtype=np.float32)

    service = EmbeddingService(model=WrongDimensionModel())

    with pytest.raises(ValueError, match="Expected embedding shape"):
        service.embed_documents(["Test document"])
