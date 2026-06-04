"""Session queue for Today's board (in-memory, per Streamlit session)."""

from __future__ import annotations

import streamlit as st

BOARD_STATUSES = ("Queued", "Discussed", "Deferred")


def ensure_board_state() -> None:
    st.session_state.setdefault("board_queue", [])
    st.session_state.setdefault("board_status", {})
    st.session_state.setdefault("board_active_idx", 0)
    st.session_state.setdefault("board_questions", {})


def prune_board_queue(df) -> None:
    """Drop board entries for patients that no longer exist."""
    ensure_board_state()
    valid = set(df["patient_id"].astype(str))
    queue = [pid for pid in st.session_state["board_queue"] if pid in valid]
    st.session_state["board_queue"] = queue
    for pid in list(st.session_state["board_status"].keys()):
        if pid not in valid:
            del st.session_state["board_status"][pid]
    if queue:
        st.session_state["board_active_idx"] = min(
            st.session_state["board_active_idx"],
            len(queue) - 1,
        )
    else:
        st.session_state["board_active_idx"] = 0


def add_patients_to_board(patient_ids: list[str]) -> None:
    ensure_board_state()
    queue = st.session_state["board_queue"]
    for pid in patient_ids:
        if pid and pid not in queue:
            queue.append(pid)
            st.session_state["board_status"].setdefault(pid, "Queued")
    st.session_state["board_queue"] = queue


def add_patient_to_board(patient_id: str) -> bool:
    if not patient_id:
        return False
    ensure_board_state()
    if patient_id in st.session_state["board_queue"]:
        return False
    st.session_state["board_queue"].append(patient_id)
    st.session_state["board_status"].setdefault(patient_id, "Queued")
    return True


def remove_from_board(patient_id: str) -> None:
    ensure_board_state()
    queue = st.session_state["board_queue"]
    if patient_id not in queue:
        return
    queue.remove(patient_id)
    st.session_state["board_queue"] = queue
    st.session_state["board_status"].pop(patient_id, None)
    st.session_state["board_questions"].pop(patient_id, None)
    if queue:
        st.session_state["board_active_idx"] = min(
            st.session_state["board_active_idx"],
            len(queue) - 1,
        )
    else:
        st.session_state["board_active_idx"] = 0


def move_case(patient_id: str, direction: int) -> None:
    ensure_board_state()
    queue = st.session_state["board_queue"]
    if patient_id not in queue:
        return
    idx = queue.index(patient_id)
    new_idx = idx + direction
    if 0 <= new_idx < len(queue):
        queue[idx], queue[new_idx] = queue[new_idx], queue[idx]
        st.session_state["board_queue"] = queue
        if st.session_state.get("board_active_idx") == idx:
            st.session_state["board_active_idx"] = new_idx
        elif st.session_state.get("board_active_idx") == new_idx:
            st.session_state["board_active_idx"] = idx


def set_active_index(idx: int) -> None:
    ensure_board_state()
    queue = st.session_state["board_queue"]
    if not queue:
        st.session_state["board_active_idx"] = 0
        return
    st.session_state["board_active_idx"] = max(0, min(idx, len(queue) - 1))


def get_active_patient_id() -> str | None:
    ensure_board_state()
    queue = st.session_state["board_queue"]
    if not queue:
        return None
    idx = st.session_state["board_active_idx"]
    return queue[min(idx, len(queue) - 1)]


def set_status(patient_id: str, status: str) -> None:
    if status not in BOARD_STATUSES:
        return
    ensure_board_state()
    st.session_state["board_status"][patient_id] = status


def row_for_patient_id(df, patient_id: str):
    matches = df[df["patient_id"].astype(str) == patient_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def board_progress() -> tuple[int, int, int]:
    """Return total, discussed, remaining counts."""
    ensure_board_state()
    queue = st.session_state["board_queue"]
    statuses = st.session_state["board_status"]
    total = len(queue)
    discussed = sum(1 for pid in queue if statuses.get(pid) == "Discussed")
    remaining = total - discussed
    return total, discussed, remaining
