"""Today's board queue with SQLite persistence."""

from __future__ import annotations

from datetime import date

import streamlit as st

from src.board_store import (
    ActionItem,
    MeetingCase,
    MeetingState,
    default_meeting_date,
    load_meeting,
    save_meeting,
)

BOARD_STATUSES = (
    "Ready for board",
    "Missing imaging",
    "Needs pathology",
    "Needs molecular results",
    "Urgent",
    "Discussed",
    "Decision recorded",
)

_LEGACY_STATUS_MAP = {
    "Queued": "Ready for board",
    "Deferred": "Needs pathology",
}


def normalize_board_status(status: str) -> str:
    cleaned = (status or "").strip()
    if cleaned in BOARD_STATUSES:
        return cleaned
    return _LEGACY_STATUS_MAP.get(cleaned, "Ready for board")


def get_meeting_date() -> str:
    return st.session_state.setdefault("board_meeting_date", default_meeting_date())


def set_meeting_date(meeting_date: str) -> None:
    st.session_state["board_meeting_date"] = meeting_date
    st.session_state.pop("_board_hydrated_date", None)


def ensure_board_state() -> None:
    st.session_state.setdefault("board_queue", [])
    st.session_state.setdefault("board_status", {})
    st.session_state.setdefault("board_active_idx", 0)
    st.session_state.setdefault("board_questions", {})
    st.session_state.setdefault("board_recommendations", {})
    st.session_state.setdefault("board_rationale", {})
    st.session_state.setdefault("board_actions", {})
    st.session_state.setdefault("board_follow_up", {})
    st.session_state.setdefault("board_title", "")


def _state_from_session() -> MeetingState:
    ensure_board_state()
    queue = st.session_state["board_queue"]
    cases: list[MeetingCase] = []
    for order, pid in enumerate(queue):
        raw_actions = st.session_state["board_actions"].get(pid, [])
        actions = [
            ActionItem(
                task=str(a.get("task", "")),
                owner=str(a.get("owner", "")),
                due_date=str(a.get("due_date", "")),
            )
            for a in raw_actions
            if isinstance(a, dict)
        ]
        cases.append(
            MeetingCase(
                patient_id=pid,
                sort_order=order,
                status=normalize_board_status(st.session_state["board_status"].get(pid, "Ready for board")),
                discussion_question=st.session_state["board_questions"].get(pid, ""),
                recommendation=st.session_state["board_recommendations"].get(pid, ""),
                rationale=st.session_state["board_rationale"].get(pid, ""),
                follow_up_date=st.session_state["board_follow_up"].get(pid, ""),
                action_items=actions,
            )
        )
    return MeetingState(
        meeting_date=get_meeting_date(),
        board_title=st.session_state.get("board_title", ""),
        active_idx=st.session_state.get("board_active_idx", 0),
        cases=cases,
    )


def _apply_state_to_session(state: MeetingState) -> None:
    st.session_state["board_title"] = state.board_title
    st.session_state["board_queue"] = [c.patient_id for c in sorted(state.cases, key=lambda x: x.sort_order)]
    st.session_state["board_status"] = {
        c.patient_id: normalize_board_status(c.status) for c in state.cases
    }
    st.session_state["board_questions"] = {c.patient_id: c.discussion_question for c in state.cases}
    st.session_state["board_recommendations"] = {c.patient_id: c.recommendation for c in state.cases}
    st.session_state["board_rationale"] = {c.patient_id: c.rationale for c in state.cases}
    st.session_state["board_follow_up"] = {c.patient_id: getattr(c, "follow_up_date", "") for c in state.cases}
    st.session_state["board_actions"] = {
        c.patient_id: [
            {"task": a.task, "owner": a.owner, "due_date": a.due_date}
            for a in c.action_items
        ]
        for c in state.cases
    }
    st.session_state["board_active_idx"] = max(
        0,
        min(state.active_idx, max(len(st.session_state["board_queue"]) - 1, 0)),
    )


def hydrate_meeting(meeting_date: str | None = None) -> None:
    meeting_date = meeting_date or get_meeting_date()
    state = load_meeting(meeting_date)
    _apply_state_to_session(state)
    st.session_state["_board_hydrated_date"] = meeting_date


def persist_meeting() -> None:
    save_meeting(_state_from_session())


def _persist_after(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        persist_meeting()
        return result

    return wrapper


def prune_board_queue(df) -> None:
    """Drop board entries for patients that no longer exist."""
    hydrate_meeting()
    ensure_board_state()
    valid = set(df["patient_id"].astype(str))
    queue = [pid for pid in st.session_state["board_queue"] if pid in valid]
    st.session_state["board_queue"] = queue
    for pid in list(st.session_state["board_status"].keys()):
        if pid not in valid:
            for store in (
                st.session_state["board_status"],
                st.session_state["board_questions"],
                st.session_state["board_recommendations"],
                st.session_state["board_rationale"],
                st.session_state["board_follow_up"],
                st.session_state["board_actions"],
            ):
                store.pop(pid, None)
    if queue:
        st.session_state["board_active_idx"] = min(
            st.session_state["board_active_idx"],
            len(queue) - 1,
        )
    else:
        st.session_state["board_active_idx"] = 0
    persist_meeting()


def add_patients_to_board(patient_ids: list[str]) -> None:
    hydrate_meeting()
    ensure_board_state()
    queue = st.session_state["board_queue"]
    for pid in patient_ids:
        if pid and pid not in queue:
            queue.append(pid)
            st.session_state["board_status"].setdefault(pid, "Ready for board")
            st.session_state["board_questions"].setdefault(pid, "")
            st.session_state["board_recommendations"].setdefault(pid, "")
            st.session_state["board_rationale"].setdefault(pid, "")
            st.session_state["board_follow_up"].setdefault(pid, "")
            st.session_state["board_actions"].setdefault(pid, [])
    st.session_state["board_queue"] = queue
    persist_meeting()


def add_patient_to_board(patient_id: str) -> bool:
    if not patient_id:
        return False
    add_patients_to_board([patient_id])
    return patient_id in st.session_state.get("board_queue", [])


def remove_from_board(patient_id: str) -> None:
    hydrate_meeting()
    ensure_board_state()
    queue = st.session_state["board_queue"]
    if patient_id not in queue:
        return
    queue.remove(patient_id)
    st.session_state["board_queue"] = queue
    for store in (
        st.session_state["board_status"],
        st.session_state["board_questions"],
        st.session_state["board_recommendations"],
        st.session_state["board_rationale"],
        st.session_state["board_actions"],
    ):
        store.pop(patient_id, None)
    if queue:
        st.session_state["board_active_idx"] = min(
            st.session_state["board_active_idx"],
            len(queue) - 1,
        )
    else:
        st.session_state["board_active_idx"] = 0
    persist_meeting()


def move_case(patient_id: str, direction: int) -> None:
    hydrate_meeting()
    ensure_board_state()
    queue = st.session_state["board_queue"]
    if patient_id not in queue:
        return
    idx = queue.index(patient_id)
    new_idx = idx + direction
    if 0 <= new_idx < len(queue):
        queue[idx], queue[new_idx] = queue[new_idx], queue[idx]
        reorder_board_queue(queue)


def reorder_board_queue(new_queue: list[str]) -> None:
    hydrate_meeting()
    ensure_board_state()
    valid = set(st.session_state["board_queue"])
    queue = [patient_id for patient_id in new_queue if patient_id in valid]
    for patient_id in st.session_state["board_queue"]:
        if patient_id not in queue:
            queue.append(patient_id)
    active_pid = get_active_patient_id()
    st.session_state["board_queue"] = queue
    if active_pid and active_pid in queue:
        st.session_state["board_active_idx"] = queue.index(active_pid)
    elif queue:
        st.session_state["board_active_idx"] = min(
            st.session_state.get("board_active_idx", 0),
            len(queue) - 1,
        )
    else:
        st.session_state["board_active_idx"] = 0
    persist_meeting()


def set_active_index(idx: int) -> None:
    hydrate_meeting()
    ensure_board_state()
    queue = st.session_state["board_queue"]
    if not queue:
        st.session_state["board_active_idx"] = 0
    else:
        st.session_state["board_active_idx"] = max(0, min(idx, len(queue) - 1))
    persist_meeting()


def get_active_patient_id() -> str | None:
    hydrate_meeting()
    ensure_board_state()
    queue = st.session_state["board_queue"]
    if not queue:
        return None
    idx = st.session_state["board_active_idx"]
    return queue[min(idx, len(queue) - 1)]


def set_status(patient_id: str, status: str) -> None:
    normalized = normalize_board_status(status)
    if normalized not in BOARD_STATUSES:
        return
    hydrate_meeting()
    ensure_board_state()
    st.session_state["board_status"][patient_id] = normalized
    persist_meeting()


def update_case_notes(
    patient_id: str,
    *,
    discussion_question: str | None = None,
    recommendation: str | None = None,
    rationale: str | None = None,
    follow_up_date: str | None = None,
    action_items: list[dict] | None = None,
) -> None:
    hydrate_meeting()
    ensure_board_state()
    if discussion_question is not None:
        st.session_state["board_questions"][patient_id] = discussion_question
    if recommendation is not None:
        st.session_state["board_recommendations"][patient_id] = recommendation
    if rationale is not None:
        st.session_state["board_rationale"][patient_id] = rationale
    if follow_up_date is not None:
        st.session_state["board_follow_up"][patient_id] = follow_up_date
    if action_items is not None:
        st.session_state["board_actions"][patient_id] = action_items
    persist_meeting()


def get_meeting_state() -> MeetingState:
    hydrate_meeting()
    return _state_from_session()


def row_for_patient_id(df, patient_id: str):
    matches = df[df["patient_id"].astype(str) == patient_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def board_progress() -> tuple[int, int, int]:
    hydrate_meeting()
    ensure_board_state()
    queue = st.session_state["board_queue"]
    statuses = st.session_state["board_status"]
    total = len(queue)
    discussed = sum(
        1
        for pid in queue
        if normalize_board_status(statuses.get(pid, "")) in ("Discussed", "Decision recorded")
    )
    remaining = total - discussed
    return total, discussed, remaining
