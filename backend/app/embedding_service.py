
"""Local embedding service for the Ask My Doc RAG system."""

from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

# BGE recommends this instruction prefix for retrieval queries.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel(Protocol):
    """Minimal interface required from an embedding model."""

    def encode(self, sentences, **kwargs) -> np.ndarray:
        ...


class EmbeddingService:
    """Generate normalized embeddings for documents and search queries."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model: EmbeddingModel | None = None,
    ) -> None:
        self.model_name = model_name
        # Dependency injection allows tests to use a lightweight fake model.
        self._model = model if model is not None else SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension."""
        return EMBEDDING_DIMENSION

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode text and return normalized float32 vectors."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All input texts must be non-empty strings.")

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(
                f"Expected embedding shape ({len(texts)}, "
                f"{self.dimension}), received {vectors.shape}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("The model returned non-finite embedding values.")

        return vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed document chunks for indexing."""
        return self._encode(texts)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed search queries using BGE's retrieval instruction."""
        if not queries:
            return np.empty((0, self.dimension), dtype=np.float32)

        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("All queries must be non-empty strings.")

        instructed_queries = [
            f"{QUERY_INSTRUCTION}{query.strip()}" for query in queries
        ]
        return self._encode(instructed_queries)
