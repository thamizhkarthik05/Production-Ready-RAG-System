
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import re

import fitz
import tiktoken


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

CHUNK_SIZE_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 100

_ENCODING = tiktoken.get_encoding("cl100k_base")


class DocumentProcessingError(ValueError):
    """Raised when a document cannot be processed."""


@dataclass(frozen=True)
class ExtractedPage:
    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    token_count: int
    page_number: int | None = None
    page_numbers: tuple[int, ...] = ()
    token_start: int = 0
    token_end: int = 0


def _clean_text(text: str) -> str:
    """Normalize whitespace without removing paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def extract_document(
    filename: str,
    content: bytes,
) -> list[ExtractedPage]:
    """Extract text from a PDF, Markdown, or plain-text file."""
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError(
            f"Unsupported file type: {extension or 'unknown'}"
        )

    if not content:
        raise DocumentProcessingError("The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise DocumentProcessingError(
            "The uploaded file exceeds the 10 MB limit."
        )

    pages: list[ExtractedPage] = []

    if extension == ".pdf":
        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                if pdf.is_encrypted:
                    raise DocumentProcessingError(
                        "Password-protected PDFs are not supported."
                    )

                for page_index, page in enumerate(pdf):
                    text = _clean_text(page.get_text("text"))
                    if text:
                        pages.append(
                            ExtractedPage(
                                text=text,
                                page_number=page_index + 1,
                            )
                        )
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(
                "The PDF could not be read."
            ) from exc
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentProcessingError(
                "Text files must use UTF-8 encoding."
            ) from exc

        text = _clean_text(text)
        if text:
            pages.append(ExtractedPage(text=text, page_number=None))

    if not pages:
        raise DocumentProcessingError(
            "No readable text was found in the document. "
            "Scanned PDFs may require OCR."
        )

    return pages


def _choose_chunk_end(
    token_ids: list[int],
    start: int,
    hard_end: int,
) -> int:
    """
    Prefer a paragraph or sentence boundary near the end of the
    allowed window. Always return an end within the token limit.
    """
    if hard_end >= len(token_ids):
        return len(token_ids)

    # Search the latter half of the window for a natural boundary.
    minimum_end = start + max(1, (hard_end - start) // 2)

    candidates = [
        index
        for index in range(minimum_end, hard_end)
        if _ENCODING.decode(token_ids[index:index + 1]).endswith(
            ("\n", ".", "!", "?", ";", ":")
        )
    ]

    return candidates[-1] + 1 if candidates else hard_end


def chunk_document(
    filename: str,
    pages: list[ExtractedPage],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    document_id: str | None = None,
) -> list[DocumentChunk]:
    """Create overlapping token-bounded chunks with page attribution."""
    if chunk_size <= 0:
        raise DocumentProcessingError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise DocumentProcessingError(
            "Overlap must be at least zero and smaller than chunk size."
        )

    document_id = document_id or str(uuid4())

    # Build one continuous token stream and track each token's page.
    all_tokens: list[int] = []
    token_pages: list[int | None] = []

    for page in pages:
        text = _clean_text(page.text)
        if not text:
            continue

        page_tokens = _ENCODING.encode(text)
        all_tokens.extend(page_tokens)
        token_pages.extend([page.page_number] * len(page_tokens))

        # Preserve a separator between pages in the continuous stream.
        separator = _ENCODING.encode("\n\n")
        all_tokens.extend(separator)
        token_pages.extend([page.page_number] * len(separator))

    if not all_tokens:
        return []

    chunks: list[DocumentChunk] = []
    start = 0

    while start < len(all_tokens):
        hard_end = min(start + chunk_size, len(all_tokens))
        end = _choose_chunk_end(all_tokens, start, hard_end)

        chunk_tokens = all_tokens[start:end]
        chunk_text = _ENCODING.decode(chunk_tokens).strip()
        source_pages = tuple(
            sorted({
                page_number
                for page_number in token_pages[start:end]
                if page_number is not None
            })
        )

        if chunk_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    filename=filename,
                    chunk_index=len(chunks),
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    page_number=source_pages[0] if source_pages else None,
                    page_numbers=source_pages,
                    token_start=start,
                    token_end=end,
                )
            )

        if end >= len(all_tokens):
            break

        # The next chunk begins before the previous chunk ends.
        start = max(start + 1, end - overlap)

    return chunks


def process_document(
    filename: str,
    content: bytes,
) -> tuple[str, list[DocumentChunk]]:
    """Extract and chunk an uploaded document."""
    pages = extract_document(filename, content)
    document_id = str(uuid4())

    chunks = chunk_document(
        filename=filename,
        pages=pages,
        document_id=document_id,
    )

    if not chunks:
        raise DocumentProcessingError(
            "No text chunks could be created."
        )

    return document_id, chunks
