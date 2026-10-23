
import numpy as np
import pytest

from app.storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


def make_chunk(
    document_id="doc-1",
    chunk_id="chunk-1",
    chunk_index=0,
    text="Example document text.",
    embedding=None,
):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "filename": "example.txt",
        "chunk_index": chunk_index,
        "text": text,
        "page_number": 1,
        "page_numbers": [1, 2],
        "token_count": 4,
        "token_start": 0,
        "token_end": 4,
        "embedding": embedding,
    }


def test_empty_database_has_no_documents(storage):
    assert storage.list_documents() == []


def test_save_document_and_retrieve_chunks(storage):
    chunks = [
        make_chunk(),
        make_chunk(
            chunk_id="chunk-2",
            chunk_index=1,
            text="Second chunk.",
        ),
    ]

    storage.save_document("doc-1", "example.txt", chunks)

    documents = storage.list_documents()
    saved_chunks = storage.get_document_chunks("doc-1")

    assert len(documents) == 1
    assert documents[0]["document_id"] == "doc-1"
    assert documents[0]["filename"] == "example.txt"
    assert documents[0]["chunk_count"] == 2

    assert [c["chunk_id"] for c in saved_chunks] == [
        "chunk-1",
        "chunk-2",
    ]
    assert saved_chunks[0]["page_numbers"] == [1, 2]
    assert saved_chunks[1]["text"] == "Second chunk."


def test_save_and_restore_embedding_bytes(storage):
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    chunk = make_chunk(embedding=vector.tobytes())

    storage.save_document("doc-1", "example.txt", [chunk])

    restored = storage.get_document_chunks("doc-1")[0]["embedding"]
    restored_vector = np.frombuffer(restored, dtype=np.float32)

    np.testing.assert_allclose(restored_vector, vector)


def test_saving_document_again_replaces_its_chunks(storage):
    storage.save_document(
        "doc-1",
        "example.txt",
        [make_chunk()],
    )

    replacement = make_chunk(
        chunk_id="replacement",
        chunk_index=0,
        text="Updated text.",
    )
    storage.save_document("doc-1", "example.txt", [replacement])

    chunks = storage.get_document_chunks("doc-1")

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "replacement"
    assert chunks[0]["text"] == "Updated text."
    assert storage.list_documents()[0]["chunk_count"] == 1


def test_reject_chunks_from_another_document(storage):
    invalid_chunk = make_chunk(document_id="different-doc")

    with pytest.raises(ValueError, match="All chunks must belong"):
        storage.save_document("doc-1", "example.txt", [invalid_chunk])

    assert storage.list_documents() == []


def test_missing_document_returns_no_chunks(storage):
    assert storage.get_document_chunks("missing-doc") == []




def test_delete_document_removes_document_and_chunks(storage):
    storage.save_document(
        "doc-1",
        "example.txt",
        [make_chunk(document_id="doc-1")],
    )

    deleted = storage.delete_document("doc-1")

    assert deleted is True
    assert storage.list_documents() == []
    assert storage.get_document_chunks("doc-1") == []


def test_delete_missing_document_returns_false(storage):
    assert storage.delete_document("missing-doc") is False


def test_delete_document_preserves_other_documents(storage):
    storage.save_document(
        "doc-1",
        "first.txt",
        [make_chunk(document_id="doc-1", chunk_id="chunk-1")],
    )
    storage.save_document(
        "doc-2",
        "second.txt",
        [
            make_chunk(
                document_id="doc-2",
                chunk_id="chunk-2",
            )
        ],
    )

    assert storage.delete_document("doc-1") is True

    documents = storage.list_documents()
    assert [document["document_id"] for document in documents] == ["doc-2"]
    assert len(storage.get_document_chunks("doc-2")) == 1
