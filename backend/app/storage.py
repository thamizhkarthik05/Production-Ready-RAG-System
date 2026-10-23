
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
DATABASE_PATH = DATA_DIR / "ask_my_doc.db"


class SQLiteStorage:
    """Local SQLite persistence for documents, chunks, and embeddings."""

    def __init__(self, database_path: str | Path = DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_number INTEGER,
                    page_numbers_json TEXT NOT NULL DEFAULT '[]',
                    token_count INTEGER NOT NULL,
                    token_start INTEGER NOT NULL,
                    token_end INTEGER NOT NULL,
                    embedding BLOB,
                    FOREIGN KEY (document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE,
                    UNIQUE (document_id, chunk_index)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id)
                """
            )

    def save_document(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[dict[str, Any]],
    ) -> None:
        """Atomically save a document and all its chunks."""
        if not document_id or not filename:
            raise ValueError("Document ID and filename are required.")

        if any(chunk.get("document_id") != document_id for chunk in chunks):
            raise ValueError("All chunks must belong to the specified document.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (document_id, filename, chunk_count)
                VALUES (?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename = excluded.filename,
                    chunk_count = excluded.chunk_count
                """,
                (document_id, filename, len(chunks)),
            )

            connection.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )

            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, document_id, filename, chunk_index,
                        text, page_number, page_numbers_json, token_count,
                        token_start, token_end, embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["chunk_id"],
                        document_id,
                        chunk["filename"],
                        chunk["chunk_index"],
                        chunk["text"],
                        chunk.get("page_number"),
                        json.dumps(chunk.get("page_numbers", [])),
                        chunk["token_count"],
                        chunk["token_start"],
                        chunk["token_end"],
                        chunk.get("embedding"),
                    ),
                )

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, filename, chunk_count, created_at
                FROM documents
                ORDER BY created_at, document_id
                """
            ).fetchall()
        return [dict(row) for row in rows]


    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks. Return whether it existed."""
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID must be a non-empty string.")

        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            return cursor.rowcount > 0


    def get_document_chunks(
        self,
        document_id: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunk_id, document_id, filename, chunk_index, text,
                    page_number, page_numbers_json, token_count,
                    token_start, token_end, embedding
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            item["page_numbers"] = json.loads(
                item.pop("page_numbers_json")
            )
            results.append(item)
        return results




storage = SQLiteStorage()
