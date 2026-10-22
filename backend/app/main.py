from app.routes.documents import router as documents_router
import logging
from contextlib import asynccontextmanager
from app.routes import search
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.ask import router as ask_router
from app.config import get_settings
from app.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", settings.app_name)
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend foundation for the Ask My Doc RAG system.",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(search.router)
app.include_router(ask_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Welcome to Ask My Doc API",
        "docs": "/docs",
    }


@app.get("/api/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ask-my-doc-api",
        "version": settings.app_version,
    }
