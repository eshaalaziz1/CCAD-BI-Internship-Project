"""Persist PDF report briefings linked to patient records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from src.board_store import connect_db, init_db

MAX_STORED_REPORT_TEXT = 50_000


@dataclass
class PatientReportLink:
    patient_id: str
    report_name: str
    report_fingerprint: str
    analysis: dict
    report_text: str
    linked_at: str
    prompt_version: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_patient_report_link(
    patient_id: str,
    *,
    report_name: str,
    report_fingerprint: str,
    analysis: dict,
    report_text: str,
    prompt_version: str,
) -> None:
    init_db()
    text = report_text[:MAX_STORED_REPORT_TEXT]
    payload = json.dumps(analysis, ensure_ascii=False)
    now = _now_iso()
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO patient_report_links (
                patient_id, report_name, report_fingerprint,
                analysis_json, report_text, linked_at, prompt_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                report_name = excluded.report_name,
                report_fingerprint = excluded.report_fingerprint,
                analysis_json = excluded.analysis_json,
                report_text = excluded.report_text,
                linked_at = excluded.linked_at,
                prompt_version = excluded.prompt_version
            """,
            (
                patient_id,
                report_name,
                report_fingerprint,
                payload,
                text,
                now,
                prompt_version,
            ),
        )
        conn.commit()


def load_patient_report_link(patient_id: str) -> PatientReportLink | None:
    init_db()
    with connect_db() as conn:
        row = conn.execute(
            "SELECT * FROM patient_report_links WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        analysis = json.loads(row["analysis_json"])
    except json.JSONDecodeError:
        analysis = {}
    if not isinstance(analysis, dict):
        analysis = {}
    return PatientReportLink(
        patient_id=row["patient_id"],
        report_name=row["report_name"],
        report_fingerprint=row["report_fingerprint"],
        analysis=analysis,
        report_text=row["report_text"],
        linked_at=row["linked_at"],
        prompt_version=row["prompt_version"],
    )


def delete_patient_report_link(patient_id: str) -> None:
    init_db()
    with connect_db() as conn:
        conn.execute(
            "DELETE FROM patient_report_links WHERE patient_id = ?",
            (patient_id,),
        )
        conn.commit()


def find_patient_by_report_fingerprint(fingerprint: str) -> str | None:
    init_db()
    with connect_db() as conn:
        row = conn.execute(
            "SELECT patient_id FROM patient_report_links WHERE report_fingerprint = ?",
            (fingerprint,),
        ).fetchone()
    return row["patient_id"] if row else None
