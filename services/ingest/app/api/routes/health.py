# services/ingest/app/api/routes/health.py

from fastapi import APIRouter
import asyncpg
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basis-Health-Check – prüft ob der Service läuft."""
    return {"status": "ok", "service": "ndi-ingest"}


@router.get("/ready")
async def readiness_check():
    """
    Readiness-Check – prüft ob alle abhängigen Services
    (PostgreSQL, Tika) erreichbar sind.
    """
    checks = {}

    # PostgreSQL prüfen
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"fehler: {str(e)}"

    # Tika prüfen
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.tika_server_url}/tika")
            checks["tika"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        checks["tika"] = f"fehler: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }
