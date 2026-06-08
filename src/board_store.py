"""SQLite persistence for tumor board meetings, decisions, and action items."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "oncoboard.db"


@dataclass
class ActionItem:
    task: str = ""
    owner: str = ""
    due_date: str = ""


@dataclass
class MeetingCase:
    patient_id: str
    sort_order: int
    status: str = "Ready for board"
    discussion_question: str = ""
    recommendation: str = ""
    rationale: str = ""
    follow_up_date: str = ""
    action_items: list[ActionItem] = field(default_factory=list)


@dataclass
class MeetingState:
    meeting_date: str
    board_title: str = ""
    active_idx: int = 0
    cases: list[MeetingCase] = field(default_factory=list)


def connect_db() -> sqlite3.Connection:
    """Open the shared OncoBoard SQLite database."""
    return _connect()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_date TEXT NOT NULL UNIQUE,
                board_title TEXT NOT NULL DEFAULT '',
                active_idx INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meeting_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                patient_id TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'Queued',
                discussion_question TEXT NOT NULL DEFAULT '',
                recommendation TEXT NOT NULL DEFAULT '',
                rationale TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
                UNIQUE (meeting_id, patient_id)
            );

            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_case_id INTEGER NOT NULL,
                sort_order INTEGER NOT NULL,
                task TEXT NOT NULL DEFAULT '',
                owner TEXT NOT NULL DEFAULT '',
                due_date TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (meeting_case_id) REFERENCES meeting_cases(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS patient_report_links (
                patient_id TEXT PRIMARY KEY,
                report_name TEXT NOT NULL DEFAULT '',
                report_fingerprint TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                report_text TEXT NOT NULL DEFAULT '',
                linked_at TEXT NOT NULL,
                prompt_version TEXT NOT NULL DEFAULT ''
            );
            """
        )
        _migrate_schema(conn)
        conn.commit()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(meeting_cases)").fetchall()
    }
    if "follow_up_date" not in columns:
        conn.execute(
            "ALTER TABLE meeting_cases ADD COLUMN follow_up_date TEXT NOT NULL DEFAULT ''"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_meeting(meeting_date: str) -> MeetingState:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM meetings WHERE meeting_date = ?",
            (meeting_date,),
        ).fetchone()
        if row is None:
            return MeetingState(meeting_date=meeting_date)

        cases_rows = conn.execute(
            """
            SELECT * FROM meeting_cases
            WHERE meeting_id = ?
            ORDER BY sort_order ASC, id ASC
            """,
            (row["id"],),
        ).fetchall()

        cases: list[MeetingCase] = []
        for case_row in cases_rows:
            action_rows = conn.execute(
                """
                SELECT task, owner, due_date FROM action_items
                WHERE meeting_case_id = ?
                ORDER BY sort_order ASC, id ASC
                """,
                (case_row["id"],),
            ).fetchall()
            actions = [
                ActionItem(task=ar["task"], owner=ar["owner"], due_date=ar["due_date"])
                for ar in action_rows
            ]
            cases.append(
                MeetingCase(
                    patient_id=case_row["patient_id"],
                    sort_order=case_row["sort_order"],
                    status=case_row["status"],
                    discussion_question=case_row["discussion_question"],
                    recommendation=case_row["recommendation"],
                    rationale=case_row["rationale"],
                    follow_up_date=case_row["follow_up_date"]
                    if "follow_up_date" in case_row.keys()
                    else "",
                    action_items=actions,
                )
            )

        return MeetingState(
            meeting_date=meeting_date,
            board_title=row["board_title"],
            active_idx=row["active_idx"],
            cases=cases,
        )


def save_meeting(state: MeetingState) -> None:
    init_db()
    now = _now_iso()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM meetings WHERE meeting_date = ?",
            (state.meeting_date,),
        ).fetchone()

        if existing is None:
            cursor = conn.execute(
                """
                INSERT INTO meetings (meeting_date, board_title, active_idx, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (state.meeting_date, state.board_title, state.active_idx, now, now),
            )
            meeting_id = cursor.lastrowid
        else:
            meeting_id = existing["id"]
            conn.execute(
                """
                UPDATE meetings
                SET board_title = ?, active_idx = ?, updated_at = ?
                WHERE id = ?
                """,
                (state.board_title, state.active_idx, now, meeting_id),
            )
            conn.execute("DELETE FROM meeting_cases WHERE meeting_id = ?", (meeting_id,))

        for case in state.cases:
            cursor = conn.execute(
                """
                INSERT INTO meeting_cases (
                    meeting_id, patient_id, sort_order, status,
                    discussion_question, recommendation, rationale, follow_up_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meeting_id,
                    case.patient_id,
                    case.sort_order,
                    case.status,
                    case.discussion_question,
                    case.recommendation,
                    case.rationale,
                    case.follow_up_date,
                ),
            )
            case_id = cursor.lastrowid
            for idx, action in enumerate(case.action_items):
                conn.execute(
                    """
                    INSERT INTO action_items (meeting_case_id, sort_order, task, owner, due_date)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (case_id, idx, action.task.strip(), action.owner.strip(), action.due_date.strip()),
                )
        conn.commit()


def list_meeting_dates() -> list[str]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT meeting_date FROM meetings ORDER BY meeting_date DESC"
        ).fetchall()
    return [row["meeting_date"] for row in rows]


def default_meeting_date() -> str:
    return date.today().isoformat()
