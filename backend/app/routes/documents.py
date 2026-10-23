
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.document_processor import (
    DocumentProcessingError,
    MAX_FILE_SIZE_BYTES,
    process_document,
)
from app.rag_service import rag_service
from app.storage import storage


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


@router.get("/", status_code=status.HTTP_200_OK)
async def list_documents():
    """Return all documents stored in the database."""
    try:
        documents = storage.list_documents()

        return {
            "document_count": len(documents),
            "documents": documents,
        }

    except Exception as exc:
        logger.exception("Failed to list documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve documents.",
        ) from exc


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    # Read at most one byte beyond the limit.
    content = await file.read(MAX_FILE_SIZE_BYTES + 1)
    await file.close()

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File exceeds the 10 MB upload limit.",
        )

    # Step 1: Extract text and split it into chunks.
    try:
        document_id, chunks = process_document(filename, content)

    except DocumentProcessingError as exc:
        message = str(exc)

        if message.startswith("Unsupported file type"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=message,
            ) from exc

        if message == "The uploaded file is empty.":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message,
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=message,
        ) from exc

    # Step 2: Generate embeddings and index the chunks in FAISS.
    try:
        indexed_count = rag_service.index_chunks(chunks)

    except Exception as exc:
        logger.exception(
            "Failed to index uploaded document %s",
            document_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document was processed but could not be indexed.",
        ) from exc

    # Step 3: Return document and indexing details.
    return {
        "message": "Document processed successfully.",
        "document_id": document_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "indexed_chunk_count": indexed_count,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "page_numbers": list(chunk.page_numbers),
                "token_count": chunk.token_count,
                "text": chunk.text,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
            }
            for chunk in chunks
        ],
    }


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(document_id: str):
    """Delete a document from persistent storage and the search index."""
    try:
        deleted = rag_service.delete_document(document_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' was not found.",
            )

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to delete document %s",
            document_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete the document.",
        ) from exc
