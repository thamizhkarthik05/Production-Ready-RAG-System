
# Ask My Doc

A production-oriented, evidence-aware Retrieval-Augmented
Generation (RAG) application.

## Technology

- Python and FastAPI
- React and Vite
- FAISS and BM25 (planned)
- Sentence Transformers (planned)
- Ragas and pytest (evaluation)

## Backend

Open a terminal in the backend directory:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    Copy-Item .env.example .env
    python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

## Frontend

Open a separate terminal in the frontend directory:

    npm install
    npm run dev

## Testing

From the backend directory:

    python -m pytest -v

## Current milestone

Phase 1, Step 1: application foundation and health checks.
