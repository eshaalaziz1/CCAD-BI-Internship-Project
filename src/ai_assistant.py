"""Floating MDT assistant (MedGemma via Ollama) on every workspace tab."""

from __future__ import annotations

import re
from collections.abc import Iterator

import ollama
import pandas as pd
import streamlit as st

from src.constants import CHAT_OPTIONS, KEEP_ALIVE, MODEL
from src.patient_reports import load_patient_report_link
from src.patients import format_patient_data, patient_profile_label
from src.summarizer import check_ollama_reachable

ASSISTANT_SYSTEM = """You are an MDT clinical assistant for oncology tumor board review in OncoBoard.
Use the patient context below for case-specific questions. If information is not in the context,
say "Not found in chart" and do not guess. Phrase treatment ideas as possibilities, not final decisions.
Do not reveal chain-of-thought, hidden reasoning, analysis notes, planning steps, internal checklists,
or tokens such as <unused94>, thought, analysis, or identify the core task.

You can also answer questions about how to use OncoBoard (tabs, workflow, exporting minutes, etc.)
using the app guide in the context. Do not invent clinical facts about patients when explaining the app."""

APP_GUIDE = """
OncoBoard tabs: Home, Today's board, Patients, Report analysis, Add patient, Synthetic intake.
Workflow: review chart → optional PDF report analysis → add to Today's board → record decision → export minutes.
Today's board: queue cases, set status, MDT decision, action items, PDF/Markdown minutes export.
Report analysis: upload PDF, AI briefing, review fields, merge into chart or create new patient.
Requires Ollama with medgemma1.5 running locally.
"""

QUICK_PROMPTS = [
    "Summarize this patient for MDT.",
    "What data is missing before we can decide?",
    "What treatment options should we discuss?",
    "List biomarkers and molecular results on file.",
    "Draft bullet points for board minutes.",
    "How do I add this case to today's board?",
    "How does report analysis work?",
]


def _history_key() -> str:
    return "ai_assistant_messages"


def _context_key() -> str:
    return "ai_assistant_context_id"


def _patient_key() -> str:
    return "ai_assistant_patient_id"


def _panel_key() -> str:
    return "ai_assistant_panel"


def _pending_key() -> str:
    return "ai_assistant_pending"


def reset_chat_if_context_changed(context_id: str) -> None:
    if st.session_state.get(_context_key()) != context_id:
        st.session_state[_context_key()] = context_id
        st.session_state[_history_key()] = []


def _patient_context_for_id(df: pd.DataFrame, patient_id: str) -> str:
    matches = df[df["patient_id"].astype(str) == patient_id]
    if matches.empty:
        return f"Patient {patient_id} not found in registry."
    row = matches.iloc[0]
    link = load_patient_report_link(patient_id)
    extra = f"\n\nLinked PDF excerpt:\n{link.report_text[:4000]}" if link else ""
    return format_patient_data(row) + extra


def _patient_label_map(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    labels = [patient_profile_label(row) for _, row in df.iterrows()]
    label_to_id = {patient_profile_label(row): str(row["patient_id"]) for _, row in df.iterrows()}
    return labels, label_to_id


def _build_context(df: pd.DataFrame, patient_id: str, fallback_context: str) -> tuple[str, str]:
    if patient_id:
        patient_ctx = _patient_context_for_id(df, patient_id)
        combined = f"{patient_ctx}\n\n--- OncoBoard app guide ---\n{APP_GUIDE}"
        return combined, f"patient:{patient_id}"
    if fallback_context.strip():
        combined = f"{fallback_context}\n\n--- OncoBoard app guide ---\n{APP_GUIDE}"
        return combined, "workspace"
    return f"No patient selected.\n\n--- OncoBoard app guide ---\n{APP_GUIDE}", "none"


def _assistant_prompt(user_message: str, context: str) -> str:
    return (
        f"{ASSISTANT_SYSTEM}\n\n"
        "Return only the user-facing answer. Start directly with the answer.\n\n"
        f"Context:\n{context[:12000]}\n\nQuestion: {user_message}"
    )


def _clean_assistant_reply(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<unused\d+>\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:thought|analysis|reasoning)\s*[:\-]*\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^\s*(?:the user wants|i need to|we need to|identify the core task|identify key information|scan the context).*?(?=\n\s*(?:[-*]\s+|\d+[.)]\s+|patient id|diagnosis|stage|summary|key facts|missing|recommendation)|$)",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    cleaned = re.sub(
        r"^\s*\d+[.)]\s*\*\?(?:identify|scan|construct|formulate|review|draft)[^:\n]*:?\*?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*\d+[.)]\s*(?:identify|scan|construct|formulate|review|draft)[^:\n]*:?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    answer_match = re.search(
        r"(?:^|\n)\s*(?:final answer|answer|response)\s*[:\-]\s*(.+)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        cleaned = answer_match.group(1)

    reasoning_markers = [
        "identify the core task",
        "scan the context",
        "the user wants",
        "identify key information",
        "provided patient context",
        "construct the answer",
        "formulate the response",
    ]
    lowered = cleaned.lower()
    if any(marker in lowered for marker in reasoning_markers):
        return (
            "MedGemma returned internal reasoning instead of a usable assistant answer. "
            "Please ask again, or try one of the quick prompts."
        )

    cleaned = re.sub(r"</?[^>\s]+>", "", cleaned)
    return cleaned.strip()


def _run_assistant_stream(user_message: str, context: str) -> Iterator[str]:
    prompt = _assistant_prompt(user_message, context)
    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=CHAT_OPTIONS,
        keep_alive=KEEP_ALIVE,
        stream=True,
    )
    for chunk in stream:
        piece = (chunk.get("message", {}) or {}).get("content", "")
        if piece:
            yield piece


def _append_exchange(history: list[dict[str, str]], user_message: str, assistant_message: str) -> None:
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": _clean_assistant_reply(assistant_message)})
    st.session_state[_history_key()] = history[-12:]


def _process_pending_prompt(history: list[dict[str, str]], context: str, ollama_ok: bool) -> None:
    pending = st.session_state.pop(_pending_key(), None)
    if not pending or not ollama_ok:
        return

    with st.chat_message("user"):
        st.write(pending)

    with st.chat_message("assistant"):
        try:
            placeholder = st.empty()
            raw_reply = ""
            with st.spinner("Drafting"):
                for piece in _run_assistant_stream(pending, context):
                    raw_reply += piece
            reply = _clean_assistant_reply(raw_reply) or "No usable response."
            placeholder.write(reply)
        except Exception as exc:
            reply = f"Assistant error: {exc}"
            st.write(reply)

    _append_exchange(history, pending, reply)


def _assistant_panel_body(
    df: pd.DataFrame,
    *,
    default_patient_id: str | None,
    fallback_context: str,
    ollama_ok: bool,
    expanded: bool,
) -> None:
    title_col, min_col, expand_col, close_col = st.columns([5.5, 0.5, 0.5, 0.5])
    with title_col:
        st.markdown("**MDT Assistant**")
        st.caption(f"MedGemma ({MODEL}) · Ollama")
    with min_col:
        if st.button("−", key="ai_panel_minimize", help="Minimize"):
            st.session_state[_panel_key()] = "minimized"
            st.rerun()
    with expand_col:
        if st.button("⤢", key="ai_panel_expand", help="Expand"):
            st.session_state[_panel_key()] = "expanded"
            st.rerun()
    with close_col:
        if st.button("×", key="ai_panel_close", help="Close"):
            st.session_state[_panel_key()] = "closed"
            st.rerun()

    if not ollama_ok:
        st.warning("Start Ollama to use the assistant.")
        return

    labels, label_to_id = _patient_label_map(df)
    if default_patient_id and default_patient_id in label_to_id.values():
        st.session_state.setdefault(_patient_key(), default_patient_id)
    elif labels:
        st.session_state.setdefault(_patient_key(), label_to_id[labels[0]])

    patient_id = ""
    if labels:
        id_to_label = {pid: lab for lab, pid in label_to_id.items()}
        current_id = str(st.session_state.get(_patient_key(), label_to_id[labels[0]]))
        if current_id not in id_to_label:
            current_id = label_to_id[labels[0]]
        picked_label = st.selectbox(
            "Patient",
            labels,
            index=labels.index(id_to_label[current_id]),
            key="ai_assistant_patient_select",
        )
        patient_id = label_to_id[picked_label]
        st.session_state[_patient_key()] = patient_id
    else:
        st.info("No patients in the registry yet — you can still ask about OncoBoard.")

    context, context_id = _build_context(df, patient_id, fallback_context)
    reset_chat_if_context_changed(context_id)

    history: list[dict[str, str]] = st.session_state.setdefault(_history_key(), [])

    for msg in history[-8:]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.write(_clean_assistant_reply(msg["content"]))
            else:
                st.write(msg["content"])

    quick_cols = st.columns(2)
    for idx, prompt in enumerate(QUICK_PROMPTS):
        with quick_cols[idx % 2]:
            if st.button(prompt, key=f"ai_quick_{idx}", use_container_width=True):
                st.session_state[_pending_key()] = prompt

    user_input = st.chat_input("Ask about this patient or how OncoBoard works…", key="ai_assistant_chat")
    if user_input:
        st.session_state[_pending_key()] = user_input

    if history and st.button("Clear chat", key="ai_assistant_clear"):
        st.session_state[_history_key()] = []
        st.session_state.pop(_pending_key(), None)

    _process_pending_prompt(history, context, ollama_ok)


def render_ai_assistant_fab(
    df: pd.DataFrame,
    *,
    default_patient_id: str | None = None,
    fallback_context: str = "",
    ollama_ok: bool | None = None,
) -> None:
    """Hidden launcher used by the fixed OB decoy button."""
    del df, default_patient_id, fallback_context
    if ollama_ok is None:
        ollama_ok = check_ollama_reachable()
    del ollama_ok

    panel_state = st.session_state.get(_panel_key(), "closed")
    if panel_state != "closed":
        return

    if st.button("OB", key="ai_assistant_fab", type="primary", help="Open MDT assistant"):
        st.session_state[_panel_key()] = "minimized"
        st.rerun()


def render_ai_assistant_panel(
    df: pd.DataFrame,
    *,
    default_patient_id: str | None = None,
    fallback_context: str = "",
    ollama_ok: bool | None = None,
) -> None:
    """Persistent bottom-right assistant panel (expand / minimize / close)."""
    if ollama_ok is None:
        ollama_ok = check_ollama_reachable()

    panel_state = st.session_state.get(_panel_key(), "closed")
    if panel_state == "closed":
        return

    panel_class = "is-expanded" if panel_state == "expanded" else "is-minimized"
    st.markdown(
        f'<div class="ob-assistant-state {panel_class}" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )

    with st.container(key="ai_assistant_panel"):
        _assistant_panel_body(
            df,
            default_patient_id=default_patient_id,
            fallback_context=fallback_context,
            ollama_ok=ollama_ok,
            expanded=panel_state == "expanded",
        )
