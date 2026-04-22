# services/ingest/app/api/routes/ingest.py

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import date
import uuid
import structlog

from app.services.ingest_service import IngestService

logger = structlog.get_logger()
router = APIRouter()


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadaten die beim Upload mitgegeben werden."""
    source_type: str                        # gesetz | verordnung | standard | ...
    title: str
    jurisdiction: Optional[str] = None     # z.B. "BW"
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    norm_reference: Optional[str] = None   # z.B. "§ 1 MeldeG BW"
    version: Optional[str] = None
    language: str = "de"
    register_scope: Optional[list[str]] = None


class IngestResponse(BaseModel):
    job_id: str
    doc_id: str
    status: str
    message: str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.post("/document", response_model=IngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str = "gesetz",
    title: str = "Unbekanntes Dokument",
    jurisdiction: Optional[str] = None,
    norm_reference: Optional[str] = None,
    version: Optional[str] = None,
    language: str = "de",
):
    """
    Dokument hochladen und Ingest-Pipeline starten.

    Der eigentliche Ingest (Parsing, Chunking, Embedding) läuft
    als Background-Task – der Endpoint antwortet sofort mit einer job_id.
    """
    # Dateiformat prüfen
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/html",
        "text/plain",
        "application/rtf",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Dateiformat '{file.content_type}' wird nicht unterstützt. "
                   f"Erlaubt: PDF, DOCX, DOC, HTML, TXT, RTF"
        )

    doc_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    metadata = DocumentMetadata(
        source_type=source_type,
        title=title,
        jurisdiction=jurisdiction,
        norm_reference=norm_reference,
        version=version,
        language=language,
    )

    # Dateiinhalt lesen (vor dem Background-Task, da Stream danach geschlossen)
    file_content = await file.read()
    filename = file.filename

    logger.info(
        "Ingest gestartet",
        doc_id=doc_id,
        job_id=job_id,
        filename=filename,
        source_type=source_type,
    )

    # Pipeline als Background-Task starten
    service = IngestService()
    background_tasks.add_task(
        service.run_pipeline,
        doc_id=doc_id,
        job_id=job_id,
        file_content=file_content,
        filename=filename,
        metadata=metadata,
    )

    return IngestResponse(
        job_id=job_id,
        doc_id=doc_id,
        status="queued",
        message=f"Dokument '{filename}' wird verarbeitet. Status über /api/v1/ingest/status/{job_id} abrufbar.",
    )


@router.get("/status/{job_id}")
async def get_ingest_status(job_id: str):
    """Verarbeitungsstatus eines Ingest-Jobs abfragen."""
    # TODO: Status aus PostgreSQL-Job-Tabelle lesen (M1 Iteration 2)
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Job-Status-Tracking wird in M1 Iteration 2 implementiert.",
    }
