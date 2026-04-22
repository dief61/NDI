# services/ingest/app/services/ingest_service.py
# Orchestriert: Parsing → Chunking-Router → Embedding → Speicherung

import structlog
from app.services.parser import TikaParser
from app.services.chunker import ChunkingRouter
from app.services.embedder import Embedder
from app.services.storage import DocumentStorage

logger = structlog.get_logger()


class IngestService:

    def __init__(self):
        self.parser   = TikaParser()
        self.chunker  = ChunkingRouter()
        self.embedder = Embedder()
        self.storage  = DocumentStorage()

    async def run_pipeline(
        self,
        doc_id: str,
        job_id: str,
        file_content: bytes,
        filename: str,
        metadata,
    ):
        """
        Vollständige Ingest-Pipeline:
        1. Dokument in MinIO speichern
        2. Tika-Parsing (Text + Struktur)
        3. Chunking-Router (Klasse A / B / C)
        4. Embedding je Chunk
        5. Chunks + Embeddings in PostgreSQL speichern
        """
        log = logger.bind(doc_id=doc_id, job_id=job_id, filename=filename)

        try:
            # Schritt 1: Rohdokument in MinIO ablegen
            log.info("Schritt 1: Dokument in MinIO speichern")
            minio_path = await self.storage.store_raw_document(
                doc_id=doc_id,
                filename=filename,
                content=file_content,
            )

            # Schritt 2: Dokument-Metadaten in PostgreSQL anlegen
            log.info("Schritt 2: Dokument-Metadaten speichern")
            await self.storage.create_document_record(
                doc_id=doc_id,
                metadata=metadata,
                minio_path=minio_path,
            )

            # Schritt 3: Tika-Parsing
            log.info("Schritt 3: Tika-Parsing")
            parsed = await self.parser.parse(
                content=file_content,
                filename=filename,
            )
            log.info("Parsing abgeschlossen", char_count=len(parsed.text))

            # Schritt 4: Chunking-Router
            log.info("Schritt 4: Chunking-Router")
            chunks = self.chunker.route_and_chunk(
                text=parsed.text,
                structure=parsed.structure,
                doc_id=doc_id,
                metadata=metadata,
            )
            log.info("Chunking abgeschlossen", chunk_count=len(chunks))

            # Schritt 5: Embeddings erzeugen
            log.info("Schritt 5: Embeddings erzeugen")
            chunks_with_embeddings = await self.embedder.embed_chunks(chunks)

            # Schritt 6: In PostgreSQL speichern
            log.info("Schritt 6: Chunks in PostgreSQL speichern")
            await self.storage.store_chunks(chunks_with_embeddings)

            log.info("Pipeline erfolgreich abgeschlossen", chunk_count=len(chunks))

        except Exception as e:
            log.error("Pipeline fehlgeschlagen", error=str(e))
            raise
