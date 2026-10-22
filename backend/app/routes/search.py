
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.rag_service import rag_service

router = APIRouter(prefix="/api", tags=["Search"])


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question or search query",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of relevant chunks to retrieve",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional document ID to restrict the search",
    )


class SearchResultItem(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    text: str
    page_number: int | None = None
    page_numbers: list[int] = Field(default_factory=list)
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[SearchResultItem]


@router.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query must not be empty or contain only whitespace.",
        )

    try:
        matches = rag_service.search(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    results: list[SearchResultItem] = []

    for match in matches:
        metadata: dict[str, Any] = match.metadata

        results.append(
            SearchResultItem(
                document_id=metadata["document_id"],
                chunk_id=metadata["chunk_id"],
                filename=metadata["filename"],
                text=metadata["text"],
                page_number=metadata.get("page_number"),
                page_numbers=metadata.get("page_numbers", []),
                chunk_index=metadata["chunk_index"],
                score=float(match.score),
            )
        )

    return SearchResponse(
        query=query,
        result_count=len(results),
        results=results,
    )
