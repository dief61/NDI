-- NDI – Alle Tabellen löschen
-- Datei: infra/postgres/maintenance/drop_all_tables.sql
--
-- WARNUNG: Löscht alle Daten unwiderruflich!
-- Nur in Entwicklung verwenden.
--
-- Aufruf:
--   docker exec -i mnr-postgres psql -U mnr -d mnr_db \
--     < infra/postgres/maintenance/drop_all_tables.sql

-- Reihenfolge beachten: erst abhängige Tabellen (FK-Constraints)
DROP TABLE IF EXISTS im_review_log        CASCADE;
DROP TABLE IF EXISTS information_models   CASCADE;
DROP TABLE IF EXISTS ingest_jobs          CASCADE;
DROP TABLE IF EXISTS norm_chunks          CASCADE;
DROP TABLE IF EXISTS norm_documents       CASCADE;

-- Bestätigung
SELECT 'Alle Tabellen gelöscht.' AS status;
