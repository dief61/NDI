#!/usr/bin/env python3
# services/ingest/nlp_worker.py
#
# CLI für die M2 NLP-Pipeline.
# Kann jederzeit neu gestartet werden – erzeugt immer neue Ergebnisse.
#
# Aufruf:
#   python nlp_worker.py --run                    # alle Chunks verarbeiten
#   python nlp_worker.py --run --doc-id <doc_id>  # nur ein Dokument
#   python nlp_worker.py --jobs                   # letzte Jobs anzeigen
#   python nlp_worker.py --status <job_id>        # Job-Status
#   python nlp_worker.py --stats                  # DB-Statistik
#   python nlp_worker.py --config nlp_config.yaml # andere Config-Datei

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.services.nlp.nlp_service import NLPService


def parse_args():
    parser = argparse.ArgumentParser(
        description="NDI NLP-Worker – M2 NLP & SVO-Extraktion",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run",    action="store_true",
                        help="NLP-Pipeline starten (alle oder ein Dokument)")
    action.add_argument("--jobs",   action="store_true",
                        help="Letzte NLP-Jobs anzeigen")
    action.add_argument("--status", type=str, metavar="JOB_ID",
                        help="Status eines Jobs abfragen")
    action.add_argument("--stats",  action="store_true",
                        help="DB-Statistik anzeigen")

    parser.add_argument("--doc-id", type=str, default=None,
                        help="Nur dieses Dokument verarbeiten (doc_id aus norm_documents)")
    parser.add_argument("--config", type=str, default=None,
                        metavar="YAML",
                        help="Pfad zur nlp_config.yaml (Standard: services/ingest/nlp_config.yaml)")
    parser.add_argument("--jobs-limit", type=int, default=20,
                        help="Anzahl Jobs bei --jobs (Standard: 20)")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Bestehende Ergebnisse NICHT überschreiben")
    return parser.parse_args()


def sep(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


async def run_nlp(args):
    """NLP-Pipeline starten."""
    config_path = Path(args.config) if args.config else None

    # overwrite_existing temporär überschreiben wenn --no-overwrite
    if args.no_overwrite and config_path is None:
        import yaml
        default = Path(__file__).parent / "nlp_config.yaml"
        with open(default, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("worker", {})["overwrite_existing"] = False
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        yaml.dump(cfg, tmp)
        tmp.close()
        config_path = Path(tmp.name)

    service = NLPService(config_path=config_path)

    sep("NDI NLP-Worker – M2")
    if args.doc_id:
        print(f"  Modus:   Einzeldokument")
        print(f"  doc_id:  {args.doc_id}")
    else:
        print(f"  Modus:   Alle Dokumente")
    print(f"  Config:  {config_path or 'nlp_config.yaml (Standard)'}")
    print(f"  Überschreiben: {'Nein' if args.no_overwrite else 'Ja'}")
    print(f"{'─'*65}")

    try:
        result = await service.run(doc_id=args.doc_id)
        _print_job_status(result)
    finally:
        await service.close()
        # Temp-Datei aufräumen
        if args.no_overwrite and config_path and "tmp" in str(config_path):
            import os
            os.unlink(config_path)


async def show_jobs(limit: int):
    """Letzte NLP-Jobs anzeigen."""
    service = NLPService()
    try:
        jobs = await service.list_jobs(limit)
        sep(f"Letzte {limit} NLP-Jobs")

        ICONS = {
            "done":"✅","error":"❌","running":"🔄","queued":"⏳"
        }
        for j in jobs:
            icon = ICONS.get(j["status"], "❓")
            dur  = ""
            if j.get("started_at") and j.get("finished_at"):
                from datetime import datetime
                s = j["started_at"]
                e = j["finished_at"]
                if hasattr(s, "total_seconds"):
                    dur = f"  {(e-s).total_seconds():.0f}s"

            print(f"\n  {icon} {j['status']:<10} doc: {j['doc_id'] or 'alle'}")
            print(f"     job_id: {j['job_id']}")
            print(
                f"     Chunks: {j['chunks_done']}/{j['chunks_total']}  |  "
                f"SVO: {j['svo_count']}  |  NER: {j['ner_count']}{dur}"
            )
        if not jobs:
            print("\n  Keine Jobs gefunden.")
    finally:
        await service.close()


async def show_status(job_id: str):
    """Status eines einzelnen Jobs."""
    service = NLPService()
    try:
        status = await service.get_job_status(job_id)
        if not status:
            print(f"\n  Job '{job_id}' nicht gefunden.")
            return
        sep("NLP-Job Status")
        _print_job_status(status)
    finally:
        await service.close()


async def show_stats():
    """DB-Statistik der NLP-Ergebnisse."""
    sep("NLP Datenbank-Statistik")

    conn = await asyncpg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db,
    )

    # SVO-Statistik
    svo = await conn.fetch("""
        SELECT norm_type, COUNT(*) as cnt,
               AVG(confidence)::NUMERIC(4,3) as avg_conf
        FROM svo_extractions
        GROUP BY norm_type ORDER BY cnt DESC
    """)

    print(f"\n  SVO-Extraktionen ({sum(r['cnt'] for r in svo)} gesamt):")
    print(f"  {'─'*50}")
    for r in svo:
        bar = "█" * min(30, int(r["cnt"] / max(1, sum(
            x["cnt"] for x in svo)) * 30))
        print(f"  {r['norm_type']:<15} {r['cnt']:>6}  {bar}  ø{r['avg_conf']}")

    # NER-Statistik
    ner = await conn.fetch("""
        SELECT label, COUNT(*) as cnt,
               AVG(confidence)::NUMERIC(4,3) as avg_conf
        FROM ner_entities
        GROUP BY label ORDER BY cnt DESC
    """)

    print(f"\n  NER-Entitäten ({sum(r['cnt'] for r in ner)} gesamt):")
    print(f"  {'─'*50}")
    for r in ner:
        print(f"  {r['label']:<15} {r['cnt']:>6}  ø{r['avg_conf']}")

    # Top-Subjekte
    top_subj = await conn.fetch("""
        SELECT subject, subject_type, COUNT(*) as cnt
        FROM svo_extractions
        WHERE subject IS NOT NULL
        GROUP BY subject, subject_type
        ORDER BY cnt DESC LIMIT 10
    """)

    print(f"\n  Top-10 Subjekte (Akteure):")
    print(f"  {'─'*50}")
    for r in top_subj:
        print(f"  {r['subject']:<35} [{r['subject_type']}]  {r['cnt']}×")

    # Top-Objekte
    top_obj = await conn.fetch("""
        SELECT object, object_type, COUNT(*) as cnt
        FROM svo_extractions
        WHERE object IS NOT NULL
        GROUP BY object, object_type
        ORDER BY cnt DESC LIMIT 10
    """)

    print(f"\n  Top-10 Objekte (Datenobjekte/Anforderungen):")
    print(f"  {'─'*50}")
    for r in top_obj:
        print(f"  {r['object']:<35} [{r['object_type']}]  {r['cnt']}×")

    await conn.close()
    print()


def _print_job_status(s: dict):
    ICONS = {"done":"✅","error":"❌","running":"🔄","queued":"⏳"}
    icon  = ICONS.get(s.get("status",""), "❓")
    print(f"\n  {icon} Status:        {s.get('status')}")
    print(f"  job_id:          {s.get('job_id')}")
    print(f"  Dokument:        {s.get('doc_id','alle')}")
    done  = s.get("chunks_done",  0) or 0
    total = s.get("chunks_total", 0) or 0
    pct   = f"{done/total*100:.0f}%" if total > 0 else "–"
    print(f"  Fortschritt:     {done}/{total}  ({pct})")
    print(f"  SVO-Tripel:      {s.get('svo_count', 0)}")
    print(f"  NER-Entitäten:   {s.get('ner_count',  0)}")
    if s.get("error_message"):
        print(f"  Fehler:          {s['error_message']}")
    print(f"  Gestartet:       {s.get('started_at','–')}")
    print(f"  Abgeschlossen:   {s.get('finished_at') or 'läuft noch'}")


async def main():
    args = parse_args()
    if args.run:
        await run_nlp(args)
    elif args.jobs:
        await show_jobs(args.jobs_limit)
    elif args.status:
        await show_status(args.status)
    elif args.stats:
        await show_stats()

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    asyncio.run(main())
