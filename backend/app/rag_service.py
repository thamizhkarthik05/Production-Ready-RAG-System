
"""Coordinate document embeddings, FAISS retrieval, and persistence."""

from pathlib import Path
from typing import Any

import numpy as np

from app.document_processor import DocumentChunk
from app.embedding_service import EMBEDDING_DIMENSION, EmbeddingService
from app.storage import SQLiteStorage, storage
from app.vector_store import SearchResult, VectorStore


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"


class RAGService:
    """Manage document embeddings and semantic retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        storage_backend: SQLiteStorage | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        # Keep the embedding model lazy-loaded.
        self._embedding_service = embedding_service
        self._vector_store = vector_store or VectorStore(
            dimension=EMBEDDING_DIMENSION
        )

        # Persistence is optional so unit tests remain isolated.
        self._storage = storage_backend
        self._index_path = Path(index_path) if index_path is not None else None

        self._document_vectors: dict[str, Any] = {}
        self._document_metadata: dict[str, list[dict[str, Any]]] = {}

    def _get_embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def indexed_document_count(self) -> int:
        return len(self._document_vectors)

    @property
    def indexed_chunk_count(self) -> int:
        return self._vector_store.size

    def load_persisted_data(self) -> None:
        """Restore saved chunks and FAISS index from local storage."""
        if self._storage is None or self._index_path is None:
            return

        documents = self._storage.list_documents()
        document_vectors: dict[str, Any] = {}
        document_metadata: dict[str, list[dict[str, Any]]] = {}

        for document in documents:
            document_id = document["document_id"]
            chunks = self._storage.get_document_chunks(document_id)

            if not chunks:
                continue

            vectors = []
            metadata = []

            for chunk in chunks:
                embedding_bytes = chunk["embedding"]
                if embedding_bytes is None:
                    raise ValueError(
                        f"Stored chunk {chunk['chunk_id']} has no embedding."
                    )

                vector = np.frombuffer(
                    embedding_bytes,
                    dtype=np.float32,
                ).copy()

                if vector.shape != (EMBEDDING_DIMENSION,):
                    raise ValueError(
                        f"Stored embedding for chunk {chunk['chunk_id']} "
                        "has an invalid dimension."
                    )

                vectors.append(vector)
                metadata.append(
                    {
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["chunk_id"],
                        "filename": chunk["filename"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "page_number": chunk["page_number"],
                        "page_numbers": chunk["page_numbers"],
                        "token_count": chunk["token_count"],
                        "token_start": chunk["token_start"],
                        "token_end": chunk["token_end"],
                    }
                )

            document_vectors[document_id] = np.vstack(vectors)
            document_metadata[document_id] = metadata

        self._document_vectors = document_vectors
        self._document_metadata = document_metadata

        # Reuse the saved FAISS index when its metadata matches SQLite.
        if self._index_path.exists():
            try:
                saved_store = VectorStore.load(self._index_path)
                saved_ids = [
                    item.get("chunk_id") for item in saved_store._metadata
                ]
                stored_ids = [
                    item["chunk_id"]
                    for records in document_metadata.values()
                    for item in records
                ]

                if (
                    saved_store.dimension == EMBEDDING_DIMENSION
                    and saved_store.size == len(stored_ids)
                    and len(saved_ids) == len(stored_ids)
                    and set(saved_ids) == set(stored_ids)
                ):
                    self._vector_store = saved_store
                    return
            except (OSError, ValueError, TypeError):
                # A missing, corrupt, or incompatible index can be rebuilt
                # from the embeddings already stored in SQLite.
                pass

        self._rebuild_index()
        self._save_faiss_index()

    def _persist_document(
        self,
        chunks: list[DocumentChunk],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        if self._storage is None:
            return

        records = []

        for chunk, vector in zip(chunks, vectors):
            records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "page_numbers": list(chunk.page_numbers),
                    "token_count": chunk.token_count,
                    "token_start": chunk.token_start,
                    "token_end": chunk.token_end,
                    "embedding": np.asarray(
                        vector, dtype=np.float32
                    ).tobytes(),
                }
            )

        self._storage.save_document(
            chunks[0].document_id,
            chunks[0].filename,
            records,
        )

    def _save_faiss_index(self) -> None:
        if self._index_path is not None:
            self._vector_store.save(self._index_path)

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and index chunks belonging to one document."""
        if not chunks:
            return 0

        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("All chunks must belong to the same document.")

        document_id = next(iter(document_ids))
        embedding_service = self._get_embedding_service()

        vectors = embedding_service.embed_documents(
            [chunk.text for chunk in chunks]
        )

        metadata = [
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "page_numbers": list(chunk.page_numbers),
                "token_count": chunk.token_count,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
            }
            for chunk in chunks
        ]

        # Save the document and its embeddings to SQLite first.
        self._persist_document(chunks, vectors, metadata)

        # Keep the existing replacement behavior.
        self._document_vectors[document_id] = vectors.copy()
        self._document_metadata[document_id] = metadata

        self._rebuild_index()
        self._save_faiss_index()

        return len(chunks)


    def delete_document(self, document_id: str) -> bool:
        """Delete a document from persistent storage and the search index."""
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID must be a non-empty string.")

        if document_id not in self._document_vectors:
            # If persistence is enabled, check whether the document exists
            # in SQLite even if it is not currently indexed in memory.
            if (
                self._storage is None
                or not self._storage.delete_document(document_id)
            ):
                return False
        elif self._storage is not None:
            # Delete persistent data before changing the in-memory index.
            if not self._storage.delete_document(document_id):
                return False

        self._document_vectors.pop(document_id, None)
        self._document_metadata.pop(document_id, None)

        self._rebuild_index()
        self._save_faiss_index()
        return True


    def _rebuild_index(self) -> None:
        """Rebuild FAISS from the current per-document records."""
        new_store = VectorStore(dimension=EMBEDDING_DIMENSION)

        for document_id, vectors in self._document_vectors.items():
            new_store.add(
                vectors,
                self._document_metadata[document_id],
            )

        self._vector_store = new_store

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Search indexed chunks using a natural-language query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self._vector_store.size == 0:
            return []

        query_vectors = self._get_embedding_service().embed_queries(
            [query.strip()]
        )

        return self._vector_store.search(
            query_vectors[0],
            top_k=top_k,
            document_id=document_id,
        )


# The API uses persistent storage. Restoration will be called during startup.
rag_service = RAGService(
    storage_backend=storage,
    index_path=FAISS_INDEX_PATH,
)
