-- NDI – PostgreSQL Initialisierungsschema
-- Wird beim ersten Start des Containers automatisch ausgeführt.
-- Datei: infra/postgres/init/01_schema.sql

-- ─────────────────────────────────────────────
-- Extension
-- ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────
-- Quelldokumente
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS norm_documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          TEXT UNIQUE NOT NULL,           -- externe Versionierungs-ID
    source_type     TEXT NOT NULL
                        CHECK (source_type IN (
                            'gesetz','verordnung','standard',
                            'fachkonzept','leitfaden','lastenheft','auslegung'
                        )),
    title           TEXT NOT NULL,
    jurisdiction    TEXT,
    valid_from      DATE,
    valid_to        DATE,
    norm_reference  TEXT,
    version         TEXT,
    language        TEXT DEFAULT 'de',
    register_scope  TEXT[],
    approved_by     TEXT,
    ingest_ts       TIMESTAMPTZ DEFAULT now(),
    metadata        JSONB
);

-- ─────────────────────────────────────────────
-- Chunk-Store (Kernschema aus Architekturkonzept)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS norm_chunks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id              UUID NOT NULL REFERENCES norm_documents(id) ON DELETE CASCADE,
    doc_class           CHAR(1) NOT NULL CHECK (doc_class IN ('A','B','C')),

    -- Klasse A
    norm_reference      TEXT,
    cross_references    UUID[],

    -- Klasse B
    section_path        TEXT,
    heading_breadcrumb  TEXT,
    requirement_id      TEXT,

    -- Gemeinsam
    chunk_type          TEXT
                            CHECK (chunk_type IN (
                                'tatbestand','rechtsfolge','definition','ausnahme',
                                'anforderung','tabelle','verweis','zustaendigkeit'
                            )),
    hierarchy_level     INT,
    parent_id           UUID REFERENCES norm_chunks(id),
    overlap_with_prev   NUMERIC(4,2),
    confidence_weight   NUMERIC(3,2) NOT NULL DEFAULT 1.0,

    content             TEXT NOT NULL,
    content_hash        TEXT GENERATED ALWAYS AS (md5(content)) STORED,
    token_count         INT,
    entities            JSONB,
    metadata            JSONB,

    -- Vektor-Embedding (multilingual-e5-large: 1024 Dimensionen)
    embedding           VECTOR(1024),

    version             TEXT,
    valid_from          DATE,
    valid_to            DATE,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─────────────────────────────────────────────
-- Indizes (aus Architekturkonzept)
-- ─────────────────────────────────────────────

-- Vektor-Index (IVFFlat für cosine similarity)
CREATE INDEX IF NOT EXISTS norm_chunks_embedding_idx
    ON norm_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- GIN-Index für JSONB-Metadaten
CREATE INDEX IF NOT EXISTS norm_chunks_metadata_idx
    ON norm_chunks USING gin (metadata);

-- Basis-Lookup-Indizes
CREATE INDEX IF NOT EXISTS norm_chunks_doc_class_idx
    ON norm_chunks (doc_id, doc_class);

CREATE INDEX IF NOT EXISTS norm_chunks_norm_reference_idx
    ON norm_chunks (norm_reference)
    WHERE norm_reference IS NOT NULL;

CREATE INDEX IF NOT EXISTS norm_chunks_section_path_idx
    ON norm_chunks (section_path)
    WHERE section_path IS NOT NULL;

CREATE INDEX IF NOT EXISTS norm_chunks_requirement_id_idx
    ON norm_chunks (requirement_id)
    WHERE requirement_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS norm_chunks_cross_references_idx
    ON norm_chunks USING gin (cross_references)
    WHERE cross_references IS NOT NULL;

-- ─────────────────────────────────────────────
-- Informationsmodell (IM) – Versionstabelle
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS information_models (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    register_name   TEXT NOT NULL,
    im_version      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'draft'
                        CHECK (status IN ('draft','review','approved','rejected')),
    content_yaml    TEXT,                           -- serialisiertes YAML
    minio_path      TEXT,                           -- Pfad in MinIO
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- ─────────────────────────────────────────────
-- Audit-Log (Human-in-the-Middle Freigabe-Workflow)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS im_review_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    im_id       UUID NOT NULL REFERENCES information_models(id),
    user_id     TEXT NOT NULL,
    action      TEXT NOT NULL
                    CHECK (action IN ('submitted','approved','rejected','comment')),
    comment     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- unveränderlich: kein UPDATE/DELETE erlaubt
CREATE RULE im_review_log_no_update AS ON UPDATE TO im_review_log DO INSTEAD NOTHING;
CREATE RULE im_review_log_no_delete AS ON DELETE TO im_review_log DO INSTEAD NOTHING;

