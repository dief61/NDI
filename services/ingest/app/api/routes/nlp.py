# services/ingest/app/api/routes/nlp.py
#
# FastAPI-Endpoints für die M2 NLP-Pipeline.
# Ermöglicht NLP-Jobs über die REST-API zu starten und zu überwachen.

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import uuid

from app.services.nlp.nlp_service import NLPService

router = APIRouter()


class NLPJobRequest(BaseModel):
    doc_id:             Optional[str] = None    # None = alle Dokumente
    overwrite_existing: bool = True             # Alte Ergebnisse überschreiben


class NLPJobResponse(BaseModel):
    job_id:  str
    status:  str
    message: str


@router.post("/run", response_model=NLPJobResponse)
async def run_nlp(
    body:             NLPJobRequest,
    request:          Request,
    background_tasks: BackgroundTasks,
):
    """
    NLP-Pipeline starten.
    Läuft als Background-Task – antwortet sofort mit job_id.

    Bei overwrite_existing=true werden alte SVO/NER-Ergebnisse
    für die betroffenen Chunks gelöscht und neu berechnet.
    """
    service: NLPService = request.app.state.nlp_service
    job_id = str(uuid.uuid4())

    background_tasks.add_task(service.run, doc_id=body.doc_id, job_id=job_id)

    return NLPJobResponse(
        job_id=job_id,
        status="queued",
        message=(
            f"NLP-Job gestartet. "
            f"Status: GET /api/v1/nlp/status/{job_id}"
        ),
    )


@router.get("/status/{job_id}")
async def get_nlp_status(job_id: str, request: Request):
    """NLP-Job-Status abfragen."""
    service: NLPService = request.app.state.nlp_service
    status = await service.get_job_status(job_id)
    if not status:
        raise HTTPException(404, detail=f"Job '{job_id}' nicht gefunden.")
    return status


@router.get("/jobs")
async def list_nlp_jobs(request: Request, limit: int = 20):
    """Letzte NLP-Jobs auflisten."""
    service: NLPService = request.app.state.nlp_service
    return await service.list_jobs(limit)
