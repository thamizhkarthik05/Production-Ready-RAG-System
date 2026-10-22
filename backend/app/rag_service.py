
"""Coordinate document embeddings and FAISS retrieval."""

from typing import Any

from app.document_processor import DocumentChunk
from app.embedding_service import EMBEDDING_DIMENSION, EmbeddingService
from app.vector_store import SearchResult, VectorStore


class RAGService:
    """Manage in-memory document embeddings and semantic retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        # Load the embedding model lazily, on the first indexing or search call.
        self._embedding_service = embedding_service
        self._vector_store = vector_store or VectorStore(
            dimension=EMBEDDING_DIMENSION
        )

        # Keep per-document data so the index can be rebuilt on replacement.
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

        # Prepare the replacement first. If embedding fails, the previous
        # indexed version of this document remains untouched.
        self._document_vectors[document_id] = vectors.copy()
        self._document_metadata[document_id] = metadata

        self._rebuild_index()
        return len(chunks)

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


# Shared instance used by API routes. The model is not loaded at import time.
rag_service = RAGService()
