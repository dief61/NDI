-- NDI – Job-Status-Tabelle
-- Datei: infra/postgres/init/02_jobs.sql
-- Wird beim ersten Container-Start automatisch eingespielt.
-- Für bestehende Instanzen: manuell ausführen via
--   docker exec -i mnr-postgres psql -U mnr -d mnr_db < infra/postgres/init/02_jobs.sql

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          TEXT UNIQUE NOT NULL,
    doc_id          TEXT NOT NULL,
    filename        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN (
                            'queued','parsing','chunking',
                            'embedding','storing','done','error'
                        )),
    doc_class       CHAR(1),                        -- A | B | C (nach Parsing bekannt)
    chunk_count     INT,                            -- nach Chunking bekannt
    error_message   TEXT,                           -- bei status=error
    started_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    finished_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ingest_jobs_job_id_idx  ON ingest_jobs (job_id);
CREATE INDEX IF NOT EXISTS ingest_jobs_doc_id_idx  ON ingest_jobs (doc_id);
CREATE INDEX IF NOT EXISTS ingest_jobs_status_idx  ON ingest_jobs (status);
