"""Lightweight audit trail for AI generation and clinician edits."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src.board_store import DB_PATH, connect_db, init_db


def _ensure_audit_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            patient_id TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT '',
            prompt_version TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )


def log_audit_event(
    event_type: str,
    *,
    patient_id: str = "",
    detail: str = "",
    prompt_version: str = "",
) -> None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connect_db() as conn:
        _ensure_audit_table(conn)
        conn.execute(
            """
            INSERT INTO audit_events (event_type, patient_id, detail, prompt_version, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_type, patient_id, detail[:2000], prompt_version, now),
        )
        conn.commit()
