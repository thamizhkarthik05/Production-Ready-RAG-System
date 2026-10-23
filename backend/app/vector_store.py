
"""Persistent-capable FAISS vector store for semantic retrieval."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its similarity score."""

    score: float
    metadata: dict[str, Any]


class VectorStore:
    """Exact cosine-similarity search over normalized embedding vectors."""

    def __init__(self, dimension: int = 768) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise ValueError("Dimension must be a positive integer.")
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")

        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return len(self._metadata)

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add embeddings and their corresponding chunk metadata."""
        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings with shape (n, {self.dimension})."
            )

        if len(metadata) != len(vectors):
            raise ValueError("Each embedding must have one metadata record.")

        if len(vectors) == 0:
            return

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values.")

        if any(not isinstance(item, dict) for item in metadata):
            raise ValueError("Every metadata record must be a dictionary.")

        vectors = np.ascontiguousarray(vectors.copy())
        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms <= 0):
            raise ValueError("Embedding vectors must have non-zero norms.")

        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._metadata.extend(dict(item) for item in metadata)

    def save(self, index_path: str | Path) -> None:
        """Persist the FAISS index and its aligned metadata."""
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata_path = path.with_suffix(path.suffix + ".json")
        temporary_index = path.with_suffix(path.suffix + ".tmp")
        temporary_metadata = metadata_path.with_suffix(
            metadata_path.suffix + ".tmp"
        )

        try:
            faiss.write_index(self._index, str(temporary_index))

            with temporary_metadata.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "dimension": self.dimension,
                        "metadata": self._metadata,
                    },
                    file,
                    ensure_ascii=False,
                )

            # Replace destination files only after writing both temporary files.
            temporary_index.replace(path)
            temporary_metadata.replace(metadata_path)
        finally:
            temporary_index.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)

    @classmethod
    def load(cls, index_path: str | Path) -> "VectorStore":
        """Restore a saved FAISS index and its aligned metadata."""
        path = Path(index_path)
        metadata_path = path.with_suffix(path.suffix + ".json")

        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "Both the FAISS index and metadata files must exist."
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        dimension = saved.get("dimension")
        metadata = saved.get("metadata")

        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise ValueError("Saved index has an invalid dimension.")

        if not isinstance(metadata, list) or any(
            not isinstance(item, dict) for item in metadata
        ):
            raise ValueError("Saved index has invalid metadata.")

        index = faiss.read_index(str(path))

        if index.d != dimension:
            raise ValueError("FAISS index dimension does not match metadata.")

        if index.ntotal != len(metadata):
            raise ValueError(
                "FAISS vector count does not match metadata count."
            )

        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Saved FAISS index has an unsupported index type.")

        store = cls(dimension=dimension)
        store._index = index
        store._metadata = metadata
        return store

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks, optionally for one document."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape != (1, self.dimension):
            raise ValueError(
                f"Expected one query vector with {self.dimension} dimensions."
            )

        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain finite values.")

        query = np.ascontiguousarray(query.copy())
        norm = np.linalg.norm(query)

        if norm <= 0:
            raise ValueError("Query embedding must have a non-zero norm.")

        faiss.normalize_L2(query)

        if self.size == 0:
            return []

        scores, indices = self._index.search(query, self.size)
        results: list[SearchResult] = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue

            record = self._metadata[int(index_position)]

            if (
                document_id is not None
                and record.get("document_id") != document_id
            ):
                continue

            results.append(
                SearchResult(
                    score=float(score),
                    metadata=dict(record),
                )
            )

            if len(results) >= top_k:
                break

        return results
