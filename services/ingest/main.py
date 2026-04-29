# services/ingest/main.py
# NDI Ingest-Service – FastAPI Hauptapplikation

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.api.routes import ingest, health
from app.core.config import settings
from app.services.ingest_service import IngestService

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: Startup / Shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: IngestService einmalig initialisieren (Embedder lädt Modell).
    Shutdown: Verbindungen sauber schließen.
    """
    logger.info("NDI Ingest-Service startet", version="0.1.0")

    # Singleton anlegen – Embedder-Modell wird einmalig geladen
    service = IngestService()
    app.state.ingest_service = service
    logger.info(
        "IngestService initialisiert",
        embedding_model=service.embedder.active_name,
        device=service.embedder.device,
    )

    yield  # ← Service läuft

    # Shutdown
    logger.info("NDI Ingest-Service wird beendet")
    await service.close()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NDI Ingest-Service",
    description=(
        "Dokument-Ingestion, Chunking und Embedding "
        "für das Meta-Normen-Register (MNR)"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # In Produktion einschränken
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router einbinden
app.include_router(health.router,  prefix="/health",          tags=["Health"])
app.include_router(ingest.router,  prefix="/api/v1/ingest",   tags=["Ingest"])
