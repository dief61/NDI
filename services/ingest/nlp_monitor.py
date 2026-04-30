#!/usr/bin/env python3
# services/ingest/nlp_monitor.py
#
# Live-Monitor für laufende NLP-Jobs.
# Fragt die Datenbank alle N Sekunden ab und zeigt den Fortschritt.
#
# Aufruf:
#   python nlp_monitor.py               # letzten Job überwachen
#   python nlp_monitor.py --job <job_id> # bestimmten Job
#   python nlp_monitor.py --interval 3  # Abfrageintervall in Sekunden
#   python nlp_monitor.py --once        # nur einmal abfragen, kein Loop

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent))
from app.core.config import settings


def parse_args():
    parser = argparse.ArgumentParser(description="NDI NLP Live-Monitor")
    parser.add_argument("--job",      type=str,   default=None,
                        help="Job-ID überwachen (Standard: letzter Job)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Abfrageintervall in Sekunden (Standard: 5)")
    parser.add_argument("--once",     action="store_true",
                        help="Nur einmal abfragen, kein Live-Loop")
    return parser.parse_args()


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def progress_bar(done: int, total: int, width: int = 40) -> str:
    if total == 0:
        return f"[{'─' * width}]   –%"
    pct   = done / total
    filled = int(width * pct)
    bar   = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:5.1f}%"


def elapsed(started_at) -> str:
    if not started_at:
        return "–"
    delta = datetime.now(started_at.tzinfo) - started_at
    secs  = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s"
    return f"{secs//60}m {secs%60}s"


def eta(done: int, total: int, started_at) -> str:
    if not started_at or done == 0 or total == 0:
        return "–"
    delta    = datetime.now(started_at.tzinfo) - started_at
    secs     = delta.total_seconds()
    rate     = done / secs if secs > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    if remaining < 60:
        return f"~{int(remaining)}s"
    return f"~{int(remaining)//60}m {int(remaining)%60}s"


async def get_latest_job_id(conn) -> str | None:
    row = await conn.fetchrow(
        "SELECT job_id FROM nlp_jobs ORDER BY started_at DESC LIMIT 1"
    )
    return row["job_id"] if row else None


async def get_job(conn, job_id: str) -> dict | None:
    row = await conn.fetchrow(
        """
        SELECT job_id, doc_id, status, chunks_total, chunks_done,
               svo_count, ner_count, error_message,
               started_at, updated_at, finished_at
        FROM nlp_jobs WHERE job_id = $1
        """,
        job_id,
    )
    return dict(row) if row else None


async def get_recent_svos(conn, job_id: str, limit: int = 5) -> list:
    rows = await conn.fetch(
        """
        SELECT s.subject, s.predicate, s.object, s.norm_type,
               s.confidence
        FROM svo_extractions s
        WHERE s.nlp_job_id = $1
          AND s.subject IS NOT NULL
          AND s.object  IS NOT NULL
        ORDER BY s.created_at DESC
        LIMIT $2
        """,
        job_id, limit,
    )
    return [dict(r) for r in rows]


async def get_ner_stats(conn, job_id: str) -> list:
    rows = await conn.fetch(
        """
        SELECT label, COUNT(*) as cnt
        FROM ner_entities
        WHERE nlp_job_id = $1
        GROUP BY label ORDER BY cnt DESC
        """,
        job_id,
    )
    return [dict(r) for r in rows]


def render(job: dict, svos: list, ner_stats: list, interval: float):
    done  = job["chunks_done"]  or 0
    total = job["chunks_total"] or 0
    svo   = job["svo_count"]    or 0
    ner   = job["ner_count"]    or 0

    STATUS_ICONS = {
        "done":    "✅",
        "error":   "❌",
        "running": "🔄",
        "queued":  "⏳",
    }
    icon = STATUS_ICONS.get(job["status"], "❓")

    print(f"\n{'='*65}")
    print(f"  NDI NLP-Monitor  –  Aktualisierung alle {interval:.0f}s  [CTRL+C zum Beenden]")
    print(f"{'='*65}")
    print(f"  {icon} Status:     {job['status']}")
    print(f"  job_id:       {job['job_id']}")
    print(f"  Dokument:     {job['doc_id'] or 'alle'}")
    print(f"\n  Fortschritt:  {progress_bar(done, total)}")
    print(f"  Chunks:       {done:,} / {total:,}")
    print(f"  Vergangen:    {elapsed(job['started_at'])}")
    if job["status"] == "running":
        print(f"  ETA:          {eta(done, total, job['started_at'])}")

    # Durchsatz
    if job["started_at"] and done > 0:
        delta = (datetime.now(job["started_at"].tzinfo) - job["started_at"]).total_seconds()
        rate  = done / delta if delta > 0 else 0
        print(f"  Durchsatz:    {rate:.1f} Chunks/s")

    print(f"\n  SVO-Tripel:   {svo:,}")
    print(f"  NER-Entit.:   {ner:,}")

    # NER-Verteilung
    if ner_stats:
        print(f"\n  NER-Verteilung:")
        max_cnt = max(r["cnt"] for r in ner_stats) or 1
        for r in ner_stats:
            bar = "█" * int(r["cnt"] / max_cnt * 20)
            print(f"    {r['label']:<14} {r['cnt']:>5}  {bar}")

    # Letzte SVOs
    if svos:
        print(f"\n  Letzte SVO-Extraktionen:")
        for s in svos:
            subj = (s["subject"]   or "–")[:25]
            pred = (s["predicate"] or "–")[:15]
            obj  = (s["object"]    or "–")[:25]
            print(f"    [{s['norm_type']:<10}] {subj:<25} | {pred:<15} | {obj}")

    if job["status"] == "error":
        print(f"\n  ❌ Fehler: {job['error_message']}")

    if job["finished_at"]:
        print(f"\n  Abgeschlossen: {job['finished_at'].strftime('%H:%M:%S')}")

    print(f"\n  Zuletzt aktualisiert: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*65}")


async def main():
    args = parse_args()

    conn = await asyncpg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db,
    )

    # Job-ID bestimmen
    job_id = args.job
    if not job_id:
        job_id = await get_latest_job_id(conn)
        if not job_id:
            print("\n  Kein NLP-Job gefunden. Zuerst: python nlp_worker.py --run\n")
            await conn.close()
            return

    print(f"\n  Überwache Job: {job_id}")

    try:
        while True:
            job = await get_job(conn, job_id)
            if not job:
                print(f"\n  Job '{job_id}' nicht gefunden.")
                break

            svos      = await get_recent_svos(conn, job_id)
            ner_stats = await get_ner_stats(conn, job_id)

            if not args.once:
                clear()

            render(job, svos, ner_stats, args.interval)

            # Abbrechen wenn Job fertig
            if job["status"] in ("done", "error"):
                print("\n  Job abgeschlossen – Monitor beendet.\n")
                break

            if args.once:
                break

            await asyncio.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n  Monitor beendet.\n")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
