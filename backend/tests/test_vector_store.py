
import numpy as np
import pytest

from app.vector_store import VectorStore


def metadata(document_id, chunk_id, text, page_number=1):
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "text": text,
        "page_number": page_number,
    }


def test_empty_store_returns_no_results():
    store = VectorStore(dimension=3)

    results = store.search(np.array([1, 0, 0], dtype=np.float32))

    assert results == []
    assert store.size == 0


def test_add_and_search_returns_best_match_first():
    store = VectorStore(dimension=3)

    vectors = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0.8, 0.2, 0],
        ],
        dtype=np.float32,
    )

    records = [
        metadata("doc-1", "chunk-1", "First chunk"),
        metadata("doc-1", "chunk-2", "Second chunk"),
        metadata("doc-2", "chunk-1", "Third chunk"),
    ]

    store.add(vectors, records)
    results = store.search(np.array([1, 0, 0]), top_k=2)

    assert len(results) == 2
    assert results[0].metadata["text"] == "First chunk"
    assert results[1].metadata["text"] == "Third chunk"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score >= results[1].score


def test_search_preserves_metadata():
    store = VectorStore(dimension=3)
    record = metadata("doc-7", "chunk-4", "Relevant evidence", page_number=3)

    store.add(np.array([[0, 0, 2]], dtype=np.float32), [record])
    results = store.search(np.array([0, 0, 1]), top_k=1)

    assert results[0].metadata == record
    assert results[0].metadata["page_number"] == 3
    assert results[0].score == pytest.approx(1.0)


def test_search_can_filter_by_document_id():
    store = VectorStore(dimension=3)

    store.add(
        np.array([[1, 0, 0], [0.9, 0.1, 0]], dtype=np.float32),
        [
            metadata("doc-a", "chunk-1", "Document A"),
            metadata("doc-b", "chunk-1", "Document B"),
        ],
    )

    results = store.search(
        np.array([1, 0, 0]),
        top_k=5,
        document_id="doc-b",
    )

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc-b"


def test_top_k_limits_results():
    store = VectorStore(dimension=3)

    store.add(
        np.eye(3, dtype=np.float32),
        [
            metadata("doc-1", "chunk-1", "A"),
            metadata("doc-1", "chunk-2", "B"),
            metadata("doc-1", "chunk-3", "C"),
        ],
    )

    results = store.search(np.array([1, 1, 1]), top_k=2)

    assert len(results) == 2


def test_add_rejects_wrong_embedding_dimension():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError, match="shape"):
        store.add(
            np.array([[1, 0]], dtype=np.float32),
            [metadata("doc-1", "chunk-1", "Text")],
        )


def test_add_rejects_metadata_count_mismatch():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError, match="metadata"):
        store.add(
            np.array([[1, 0, 0]], dtype=np.float32),
            [],
        )


def test_add_rejects_zero_vectors():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError, match="non-zero"):
        store.add(
            np.array([[0, 0, 0]], dtype=np.float32),
            [metadata("doc-1", "chunk-1", "Text")],
        )


def test_search_rejects_invalid_top_k():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError, match="top_k"):
        store.search(np.array([1, 0, 0]), top_k=0)


def test_search_rejects_wrong_query_dimension():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError, match="dimensions"):
        store.search(np.array([1, 0]))


def test_add_rejects_non_finite_vectors():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError, match="finite"):
        store.add(
            np.array([[np.nan, 0, 0]], dtype=np.float32),
            [metadata("doc-1", "chunk-1", "Text")],
        )


def test_add_empty_batch_does_not_change_store():
    store = VectorStore(dimension=3)

    store.add(
        np.empty((0, 3), dtype=np.float32),
        [],
    )

    assert store.size == 0


def test_invalid_dimension_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        VectorStore(dimension=0)


def test_save_and_load_preserves_search_results(tmp_path):
    import numpy as np

    from app.vector_store import VectorStore

    store = VectorStore(dimension=3)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"document_id": "doc-1", "chunk_id": "chunk-1"},
        {"document_id": "doc-2", "chunk_id": "chunk-2"},
    ]
    store.add(embeddings, metadata)

    index_path = tmp_path / "test.index"
    store.save(index_path)
    restored = VectorStore.load(index_path)

    assert restored.size == 2

    results = restored.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=2,
    )

    assert results[0].metadata["chunk_id"] == "chunk-1"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].metadata["chunk_id"] == "chunk-2"


def test_load_rejects_missing_persistence_files(tmp_path):
    from app.vector_store import VectorStore

    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "missing.index")


def test_load_rejects_mismatched_metadata_count(tmp_path):
    import json

    import faiss

    from app.vector_store import VectorStore

    index_path = tmp_path / "broken.index"
    index = faiss.IndexFlatIP(3)
    index.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    faiss.write_index(index, str(index_path))

    metadata_path = index_path.with_suffix(".index.json")
    metadata_path.write_text(
        json.dumps({"dimension": 3, "metadata": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vector count"):
        VectorStore.load(index_path)

