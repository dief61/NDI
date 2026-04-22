# services/ingest/ – Verzeichnisstruktur

services/ingest/
│
├── main.py                          # FastAPI App-Einstiegspunkt
├── requirements.txt                 # Python-Abhängigkeiten
│
└── app/
    ├── __init__.py
    │
    ├── core/
    │   ├── __init__.py
    │   └── config.py                # Konfiguration via pydantic-settings
    │
    ├── api/
    │   ├── __init__.py
    │   └── routes/
    │       ├── __init__.py
    │       ├── health.py            # GET /health, GET /health/ready
    │       └── ingest.py            # POST /api/v1/ingest/document
    │
    └── services/
        ├── __init__.py
        ├── ingest_service.py        # Pipeline-Orchestrierung (dieser Stand)
        ├── parser.py                # Tika-Parser          → nächster Schritt
        ├── chunker.py               # Chunking-Router A/B/C → nächster Schritt
        ├── embedder.py              # sentence-transformers → nächster Schritt
        └── storage.py              # PostgreSQL + MinIO    → nächster Schritt
