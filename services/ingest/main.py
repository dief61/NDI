# services/ingest/main.py
# NDI Ingest-Service – FastAPI Hauptapplikation

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.api.routes import ingest, health
from app.core.config import settings

logger = structlog.get_logger()

app = FastAPI(
    title="NDI Ingest-Service",
    description="Dokument-Ingestion, Chunking und Embedding für das Meta-Normen-Register",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router einbinden
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])


@app.on_event("startup")
async def startup():
    logger.info("NDI Ingest-Service gestartet", version="0.1.0")


@app.on_event("shutdown")
async def shutdown():
    logger.info("NDI Ingest-Service beendet")
