"""OncoBoard clinical review workspace."""

from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from io import BytesIO

import pandas as pd
import streamlit as st
from fpdf import FPDF
from pypdf import PdfReader

from src.constants import MODEL, PROMPT_VERSION
from src.patient_insights import (
    render_profile_clinical_summary,
    render_profile_visual_insights,
)
from src.ai_assistant import render_ai_assistant_fab, render_ai_assistant_panel
from src.audit_log import log_audit_event
from src.mdt_case_summary import render_mdt_case_summary
from src.patients import (
    PATIENT_COLUMNS,
    add_custom_patient,
    coerce_int,
    delete_custom_patient,
    format_patient_data,
    load_all_patients,
    next_patient_id,
    normalize_record,
    patient_profile_label,
    update_custom_patient,
)
from src.summarizer import (
    analyze_report,
    analyze_report_fast,
    analyze_report_with_mode,
    check_ollama_reachable,
    heuristic_report_analysis,
    repair_report_analysis,
    summarize_patient,
)
from src.board_queue_sorter import board_queue_sorter
from src.board_session import (
    BOARD_STATUSES,
    add_patient_to_board,
    add_patients_to_board,
    board_progress,
    create_new_board,
    display_meeting_date,
    ensure_board_state,
    get_active_patient_id,
    get_meeting_date,
    get_meeting_state,
    hydrate_meeting,
    init_board_session,
    normalize_board_status,
    open_meeting,
    persist_meeting,
    prune_board_queue,
    remove_from_board,
    reorder_board_queue,
    row_for_patient_id,
    set_active_index,
    set_meeting_date,
    set_status,
    update_case_notes,
)
from src.board_store import list_meeting_summaries
from src.patient_reports import (
    delete_patient_report_link,
    load_patient_report_link,
    save_patient_report_link,
)
from src.meeting_minutes import format_meeting_minutes_markdown, format_meeting_minutes_pdf
from src.report_charts import render_report_charts
from src.synthetic_generator import generate_synthetic_patient
from src.ui_components import (
    display_summary,
    inject_styles,
    render_footer,
    render_profile_hero,
    render_profile_sections,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


NEW_REPORT_PATIENT_ID = "__new_from_report__"


def record_fingerprint(patient_data: str) -> str:
    return hashlib.sha256(patient_data.encode("utf-8")).hexdigest()[:16]


def bump_patient_data_version() -> None:
    st.session_state["patients_version"] = (
        st.session_state.get("patients_version", 0) + 1
    )


@st.cache_data
def get_patients_df(version: int) -> pd.DataFrame:
    return load_all_patients()


@st.cache_data(show_spinner=False)
def get_cached_summary(
    patient_id: str,
    patient_data: str,
    cache_bust: int,
    prompt_version: str,
    precomputed: str | None = None,
) -> str:
    if precomputed is not None:
        return precomputed
    return summarize_patient(patient_data)


def _summary_cache_key(patient_id: str, fingerprint: str, cache_bust: int) -> str:
    return f"{PROMPT_VERSION}:{patient_id}:{fingerprint}:{cache_bust}"


def _mark_summary_warmed(cache_key: str) -> None:
    warmed = set(st.session_state.get("summary_warmed_keys", set()))
    warmed.add(cache_key)
    st.session_state["summary_warmed_keys"] = warmed


def render_generation_loader(progress: int = 12, stage: str = "Context") -> None:
    progress = max(8, min(progress, 100))
    context_class = "active" if progress >= 12 else ""
    structure_class = "active" if progress >= 42 else ""
    brief_class = "active" if progress >= 72 else ""
    st.markdown(
        f"""
        <style>
          @keyframes tbaLoaderSweep {{
            0% {{ transform: translateX(-120%); }}
            100% {{ transform: translateX(120%); }}
          }}
          @keyframes tbaNodePulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.62; }}
            50% {{ transform: scale(1.22); opacity: 1; }}
          }}
          @keyframes tbaLoaderFloat {{
            from {{ opacity: 0; transform: translateY(10px) scale(0.985); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
          }}
          .tba-live-loader {{
            position: relative;
            overflow: hidden;
            margin: 1rem 0 0.85rem;
            padding: 1.05rem 1.1rem;
            border-radius: 10px;
            border: 1px solid rgba(190, 212, 226, 0.95);
            background:
              radial-gradient(circle at 18% 22%, rgba(31, 138, 131, 0.17), transparent 10rem),
              linear-gradient(135deg, rgba(255,255,255,0.96), rgba(238,246,251,0.92));
            box-shadow: 0 18px 40px rgba(34, 57, 86, 0.13);
            animation: tbaLoaderFloat 220ms ease-out both;
          }}
          .tba-live-loader::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(110deg, transparent, rgba(255,255,255,0.75), transparent);
            transform: translateX(-120%);
            animation: tbaLoaderSweep 2.1s ease-in-out infinite;
          }}
          .tba-loader-top {{
            position: relative;
            z-index: 2;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 0.85rem;
          }}
          .tba-loader-title {{
            color: #172033;
            font-weight: 850;
            font-size: 1rem;
            letter-spacing: 0;
            line-height: 1.2;
          }}
          .tba-loader-subtitle {{
            color: #65748b;
            font-weight: 650;
            font-size: 0.84rem;
            margin-top: 0.25rem;
            line-height: 1.35;
          }}
          .tba-loader-percent {{
            color: #255e7e;
            font-weight: 900;
            font-size: 1.28rem;
            font-variant-numeric: tabular-nums;
            line-height: 1;
            min-width: 3.8rem;
            text-align: right;
          }}
          .tba-loader-track {{
            position: relative;
            z-index: 2;
            height: 0.72rem;
            border-radius: 999px;
            overflow: hidden;
            background: #dfeaf3;
            border: 1px solid #cfddea;
            box-shadow: inset 0 1px 3px rgba(23, 32, 51, 0.08);
          }}
          .tba-loader-fill {{
            height: 100%;
            width: {progress}%;
            border-radius: 999px;
            background: linear-gradient(90deg, #1f8a83, #2f7b8f, #b88020);
            box-shadow: 0 0 18px rgba(31,138,131,0.32);
            transition: width 260ms cubic-bezier(.2,.8,.2,1);
          }}
          .tba-loader-nodes {{
            position: relative;
            z-index: 2;
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            margin-top: 0.82rem;
          }}
          .tba-loader-node {{
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            color: #65748b;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1;
          }}
          .tba-loader-node::before {{
            content: "";
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 999px;
            background: #9dafbf;
          }}
          .tba-loader-node.active {{
            color: #255e7e;
          }}
          .tba-loader-node.active::before {{
            background: #1f8a83;
            animation: tbaNodePulse 1.55s ease-in-out infinite;
          }}
        </style>
        <div class="tba-live-loader">
          <div class="tba-loader-top">
            <div>
              <div class="tba-loader-title">MedGemma is composing the MDT brief</div>
              <div class="tba-loader-subtitle">{stage} · local model inference in progress</div>
            </div>
            <div class="tba-loader-percent">{progress}%</div>
          </div>
          <div class="tba-loader-track"><div class="tba-loader-fill"></div></div>
          <div class="tba-loader-nodes">
            <div class="tba-loader-node {context_class}">Context</div>
            <div class="tba-loader-node {structure_class}">Structure</div>
            <div class="tba-loader-node {brief_class}">Brief</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report_loader(progress: int = 12, stage: str = "Reading report") -> None:
    progress = max(8, min(progress, 100))
    st.markdown(
        f"""
        <div class="tba-live-loader">
          <div class="tba-loader-top">
            <div>
              <div class="tba-loader-title">MedGemma is preparing the board briefing</div>
              <div class="tba-loader-subtitle">{stage} · extracting meeting-critical signal</div>
            </div>
            <div class="tba-loader-percent">{progress}%</div>
          </div>
          <div class="tba-loader-track"><div class="tba-loader-fill" style="width:{progress}%"></div></div>
          <div class="tba-loader-nodes">
            <div class="tba-loader-node active">Extract</div>
            <div class="tba-loader-node {'active' if progress >= 45 else ''}">Prioritize</div>
            <div class="tba-loader-node {'active' if progress >= 75 else ''}">Board view</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_summary_with_loader(patient_data: str) -> str:
    loader = st.empty()
    stages = [
        (12, "Reading patient context"),
        (28, "Identifying clinical facts"),
        (45, "Checking missing information"),
        (62, "Drafting MDT questions"),
        (78, "Formatting treatment considerations"),
        (90, "Finalizing brief"),
    ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(summarize_patient, patient_data)
        tick = 0
        while not future.done():
            base_progress, stage = stages[min(tick // 5, len(stages) - 1)]
            shimmer = min(tick % 5, 4)
            with loader.container():
                render_generation_loader(base_progress + shimmer, stage)
            time.sleep(0.18)
            tick += 1

        summary = future.result()

    with loader.container():
        render_generation_loader(100, "Brief ready")
    time.sleep(0.25)
    loader.empty()
    return summary


@st.cache_data(show_spinner=False)
def get_cached_report_analysis(
    fingerprint: str,
    report_text: str,
    prompt_version: str,
    cache_bust: int,
) -> dict:
    return analyze_report(report_text)


def _render_report_preview_banner(analysis: dict) -> None:
    snapshot = analysis.get("patient_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    name = str(snapshot.get("name_or_id", "Uploaded report"))
    problem = str(snapshot.get("presenting_problem", "Reviewing report…"))
    objective = str(analysis.get("meeting_objective", ""))
    st.markdown(
        f"""
        <div class="report-brief-hero" style="margin-bottom:0.75rem;">
          <p class="report-eyebrow">Quick preview</p>
          <h2 style="font-size:1.15rem; margin:0.2rem 0 0.45rem;">{html.escape(name)}</h2>
          <p style="margin:0; opacity:0.92;">{html.escape(problem[:220])}</p>
          <p style="margin:0.55rem 0 0; font-size:0.88rem; opacity:0.85;">{html.escape(objective[:220])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def analyze_report_with_loader(
    report_text: str,
    fingerprint: str,
    patient_id: str,
    *,
    force_refresh: bool = False,
) -> tuple[dict, bool]:
    """Return (analysis, from_cache). Shows preview and progress while MedGemma runs."""
    cache_bust = _get_report_field(patient_id, "cache_bust", 0)

    if not force_refresh:
        cached_key = _get_report_field(patient_id, "cached_key")
        if (
            cached_key == f"{PROMPT_VERSION}:{fingerprint}:{cache_bust}"
            and _get_report_field(patient_id, "analysis")
            and _get_report_field(patient_id, "text") == report_text
        ):
            return _get_report_field(patient_id, "analysis"), True

    preview_slot = st.empty()
    loader = st.empty()
    preview = repair_report_analysis(heuristic_report_analysis(report_text), report_text)
    with preview_slot.container():
        _render_report_preview_banner(preview)

    stages = [
        (18, "Reading report"),
        (42, "Extracting clinical signal"),
        (68, "Building meeting view"),
        (88, "Finalizing board view"),
    ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(analyze_report_fast, report_text)
        tick = 0
        while not future.done():
            if tick > 420:
                future.cancel()
                loader.empty()
                preview_slot.empty()
                fallback = repair_report_analysis(heuristic_report_analysis(report_text), report_text)
                _set_report_field(patient_id, "analysis_mode", "timeout_heuristic")
                return fallback, False
            base_progress, stage = stages[min(tick // 4, len(stages) - 1)]
            with loader.container():
                render_report_loader(base_progress + min(tick % 4, 3), stage)
            time.sleep(0.12)
            tick += 1
        analysis, mode = future.result()

    get_cached_report_analysis(fingerprint, report_text, PROMPT_VERSION, cache_bust)

    preview_slot.empty()
    with loader.container():
        render_report_loader(100, "Board view ready")
    time.sleep(0.15)
    loader.empty()
    _set_report_field(patient_id, "cached_key", f"{PROMPT_VERSION}:{fingerprint}:{cache_bust}")
    _set_report_field(patient_id, "analysis_mode", mode)
    return analysis, False


def load_summary(
    patient_id: str,
    patient_data: str,
    fingerprint: str,
    cache_bust: int,
    *,
    force_refresh: bool,
) -> tuple[str, bool, float | None]:
    cache_key = _summary_cache_key(patient_id, fingerprint, cache_bust)
    warmed_keys = st.session_state.get("summary_warmed_keys", set())

    if force_refresh:
        start = time.perf_counter()
        summary = generate_summary_with_loader(patient_data)
        latency = time.perf_counter() - start
        get_cached_summary(
            patient_id,
            patient_data,
            cache_bust,
            PROMPT_VERSION,
            precomputed=summary,
        )
        _mark_summary_warmed(cache_key)
        return summary, False, latency

    from_cache = cache_key in warmed_keys
    summary = get_cached_summary(patient_id, patient_data, cache_bust, PROMPT_VERSION)
    if not from_cache:
        _mark_summary_warmed(cache_key)
    return summary, from_cache, None


def summary_to_markdown(patient_id: str, summary: str) -> str:
    return "\n".join(
        [
            f"# OncoBoard MDT summary - {patient_id}",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            summary,
        ]
    )


def summary_to_pdf(patient_id: str, summary: str) -> bytes:
    return _markdown_pdf_bytes(
        f"OncoBoard MDT brief - {patient_id}",
        summary_to_markdown(patient_id, summary),
    )


def _sync_patient_ids(df: pd.DataFrame) -> dict[str, str]:
    """Map profile labels to patient_id and ensure session has a valid selection."""
    labels, label_to_id = _patient_label_map(df)
    id_to_label = {pid: label for label, pid in label_to_id.items()}
    preferred_id = str(st.session_state.get("selected_patient_id", ""))
    if preferred_id not in id_to_label:
        preferred_id = str(df.iloc[0]["patient_id"])
        st.session_state["selected_patient_id"] = preferred_id
        st.session_state["selected_patient_label"] = id_to_label[preferred_id]
    return id_to_label


def get_selected_row(df: pd.DataFrame) -> pd.Series:
    id_to_label = _sync_patient_ids(df)
    patient_id = str(st.session_state["selected_patient_id"])
    row = row_for_patient_id(df, patient_id)
    if row is not None:
        return row
    st.session_state["selected_patient_id"] = str(df.iloc[0]["patient_id"])
    st.session_state["selected_patient_label"] = id_to_label[st.session_state["selected_patient_id"]]
    return df.iloc[0]


def patient_selector_ui(df: pd.DataFrame, *, widget_key: str, label: str = "Patient") -> pd.Series:
    """Dropdown to pick any registered patient; keeps selection in session by patient_id."""
    labels, label_to_id = _patient_label_map(df)
    if not labels:
        st.warning("No patients in the registry.")
        return df.iloc[0]

    id_to_label = _sync_patient_ids(df)
    preferred_id = str(st.session_state["selected_patient_id"])
    picked_label = st.selectbox(
        label,
        labels,
        index=labels.index(id_to_label[preferred_id]),
        key=widget_key,
    )
    patient_id = label_to_id[picked_label]
    st.session_state["selected_patient_id"] = patient_id
    st.session_state["selected_patient_label"] = picked_label
    row = row_for_patient_id(df, patient_id)
    return row if row is not None else df.iloc[0]


def sidebar_patient_picker(df: pd.DataFrame) -> pd.Series:
    st.markdown("##### Patient panel")
    search = st.text_input("Search", placeholder="ID, diagnosis, stage…", label_visibility="collapsed")
    labels, label_to_id = _patient_label_map(df)
    filtered_labels = labels
    if search.strip():
        term = search.lower()
        filtered_labels = [
            label
            for label in labels
            if term in label.lower() or term in label_to_id[label].lower()
        ]

    if not filtered_labels:
        st.caption("No patients match your search.")
        return get_selected_row(df)

    preferred_id = str(st.session_state.get("selected_patient_id", ""))
    if preferred_id not in label_to_id.values():
        preferred_id = label_to_id[labels[0]]
    if preferred_id not in {label_to_id[lab] for lab in filtered_labels}:
        default_label = filtered_labels[0]
    else:
        default_label = next(
            lab for lab in filtered_labels if label_to_id[lab] == preferred_id
        )

    picked_label = st.selectbox(
        "Patients",
        filtered_labels,
        index=filtered_labels.index(default_label),
        label_visibility="collapsed",
        key="sidebar_patient_select",
    )
    patient_id = label_to_id[picked_label]
    st.session_state["selected_patient_id"] = patient_id
    st.session_state["selected_patient_label"] = picked_label

    idx = labels.index(picked_label)
    p_col, n_col = st.columns(2)
    with p_col:
        if st.button("Prev", disabled=idx <= 0, key="patient_prev"):
            st.session_state["selected_patient_id"] = label_to_id[labels[idx - 1]]
            st.session_state["selected_patient_label"] = labels[idx - 1]
            st.rerun()
    with n_col:
        if st.button("Next", disabled=idx >= len(labels) - 1, key="patient_next"):
            st.session_state["selected_patient_id"] = label_to_id[labels[idx + 1]]
            st.session_state["selected_patient_label"] = labels[idx + 1]
            st.rerun()

    row = row_for_patient_id(df, patient_id)
    return row if row is not None else df.iloc[0]


def render_top_nav() -> str:
    nav_options = [
        "Home",
        "Today's board",
        "Patients",
        "Report analysis",
        "Add patient",
        "Synthetic intake",
    ]
    st.session_state.setdefault("workspace_nav", nav_options[0])
    active = st.session_state["workspace_nav"]
    if active == "Report intake":
        active = "Report analysis"
        st.session_state["workspace_nav"] = "Report analysis"

    # Defensive spacing: some Streamlit layouts clip first-row controls near the header.
    st.markdown("<div class='top-nav-safe-offset'></div>", unsafe_allow_html=True)
    cols = st.columns(len(nav_options))
    for idx, option in enumerate(nav_options):
        with cols[idx]:
            if st.button(
                option,
                key=f"top_nav_{option.lower().replace(' ', '_')}",
                type="primary" if option == active else "secondary",
                use_container_width=True,
            ):
                st.session_state["workspace_nav"] = option
                st.rerun()
    return st.session_state["workspace_nav"]


def _board_status_html(status: str) -> str:
    normalized = normalize_board_status(status)
    css = re.sub(r"[^a-z0-9]", "", normalized.lower())
    return f'<span class="board-status-pill {css}">{html.escape(normalized)}</span>'


def _assistant_context(df: pd.DataFrame) -> tuple[str, str]:
    """Return (context_text, context_id) for the floating AI assistant."""
    nav = st.session_state.get("workspace_nav", "Home")
    if nav == "Today's board":
        patient_id = get_active_patient_id()
        if patient_id:
            row = row_for_patient_id(df, patient_id)
            if row is not None:
                link = load_patient_report_link(patient_id)
                extra = f"\n\nLinked PDF excerpt:\n{link.report_text[:4000]}" if link else ""
                return format_patient_data(row) + extra, f"board:{patient_id}"
    if nav in ("Patients", "Report analysis"):
        patient_id = str(st.session_state.get("selected_patient_id", "") or "")
        if patient_id and patient_id != NEW_REPORT_PATIENT_ID:
            row = row_for_patient_id(df, patient_id)
            if row is not None:
                link = load_patient_report_link(patient_id)
                extra = f"\n\nLinked PDF excerpt:\n{link.report_text[:4000]}" if link else ""
                return format_patient_data(row) + extra, f"patient:{patient_id}"

    preview_cols = [c for c in ("patient_id", "diagnosis", "stage", "age") if c in df.columns]
    preview = df[preview_cols].head(8).to_string(index=False) if preview_cols else ""
    workspace = (
        f"Current view: {nav}\n"
        f"Patients in registry: {len(df)}\n"
        f"Sample cases:\n{preview}"
    )
    return workspace, f"workspace:{nav}"


def _profile_summary_for_patient(patient_id: str, row: pd.Series) -> str | None:
    patient_data = format_patient_data(row)
    fingerprint = record_fingerprint(patient_data)
    meta = st.session_state.get("summary_meta", {})
    summary = st.session_state.get("summary")
    if (
        summary
        and meta.get("patient_id") == patient_id
        and meta.get("fingerprint") == fingerprint
        and meta.get("prompt_version") == PROMPT_VERSION
    ):
        return summary
    return None


_REPORT_SESSION_FIELDS = (
    "pdf_bytes",
    "pdf_name",
    "text",
    "analysis",
    "fingerprint",
    "cached_key",
    "name",
    "from_cache",
    "analysis_mode",
    "cache_bust",
)


def _report_key(patient_id: str, field: str) -> str:
    return f"report_{patient_id}_{field}"


def _get_report_field(patient_id: str, field: str, default=None):
    return st.session_state.get(_report_key(patient_id, field), default)


def _set_report_field(patient_id: str, field: str, value) -> None:
    st.session_state[_report_key(patient_id, field)] = value


def _clear_report_workbench(patient_id: str) -> None:
    for field in _REPORT_SESSION_FIELDS:
        st.session_state.pop(_report_key(patient_id, field), None)


def _hydrate_report_workbench_from_patient(patient_id: str) -> None:
    """Load this patient's saved PDF briefing into the report intake workbench."""
    _clear_report_workbench(patient_id)
    link = load_patient_report_link(patient_id)
    if not link:
        return
    _set_report_field(patient_id, "text", link.report_text)
    _set_report_field(patient_id, "analysis", link.analysis)
    _set_report_field(patient_id, "fingerprint", link.report_fingerprint)
    _set_report_field(patient_id, "name", link.report_name)


def _save_report_to_patient(
    patient_id: str,
    *,
    report_name: str,
    report_fingerprint: str,
    analysis: dict,
    report_text: str,
) -> None:
    save_patient_report_link(
        patient_id,
        report_name=report_name,
        report_fingerprint=report_fingerprint,
        analysis=analysis,
        report_text=report_text,
        prompt_version=PROMPT_VERSION,
    )


def _patient_label_map(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    label_to_id = {
        patient_profile_label(row): str(row["patient_id"])
        for _, row in df.iterrows()
    }
    return list(label_to_id.keys()), label_to_id


def render_patient_context_strip(row: pd.Series) -> None:
    """One-line patient context without duplicating the full profile layout."""
    label = patient_profile_label(row)
    stage = str(row.get("stage", "—"))
    ecog = str(row.get("ecog", "—"))
    diagnosis = str(row.get("diagnosis", "—"))
    st.markdown(
        f"<p style='margin:0.35rem 0 0.85rem; color:#41556b;'>"
        f"<strong style='color:#172033;'>{html.escape(label)}</strong>"
        f" · {html.escape(diagnosis)} · Stage {html.escape(stage)} · ECOG {html.escape(ecog)}"
        f"</p>",
        unsafe_allow_html=True,
    )


def render_linked_report_summary(link, *, compact: bool = False) -> None:
    """Linked PDF briefing — compact on the board, fuller in the patient chart."""
    analysis = repair_report_analysis(link.analysis, link.report_text)
    snapshot = analysis.get("patient_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}

    linked_display = link.linked_at[:10] if link.linked_at else ""
    meta = f"**{link.report_name}**"
    if linked_display:
        meta += f" · saved {linked_display}"
    st.caption(meta)

    presenting = str(snapshot.get("presenting_problem", "")).strip()
    if presenting:
        st.markdown(presenting)

    if compact:
        render_list_items(_as_list(analysis.get("critical_facts"))[:5], "No critical facts extracted.")
        objective = str(analysis.get("meeting_objective", "")).strip()
        if objective:
            st.markdown(f"**Meeting objective:** {objective}")
        return

    objective = str(analysis.get("meeting_objective", "")).strip()
    if objective:
        st.markdown(f"**Meeting objective:** {objective}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Critical facts**")
        render_list_items(analysis.get("critical_facts"))
        st.markdown("**Red flags**")
        render_list_items(analysis.get("red_flags"), "None extracted.")
    with c2:
        st.markdown("**Gaps**")
        render_list_items(analysis.get("missing_data"))

    problems = _as_list(analysis.get("priority_problems"))
    rows = [p for p in problems if isinstance(p, dict)]
    if rows:
        st.markdown("**Priority problems**")
        render_problem_cards(rows[:3])




def _board_action_row_count(patient_id: str) -> int:
    """How many action-item rows to show (includes blank rows being filled in)."""
    ensure_board_state()
    items = st.session_state["board_actions"].get(patient_id, [])
    row_key = f"board_action_rows_{patient_id}"
    stored = max(len(items), 1)
    count = max(st.session_state.get(row_key, stored), stored)
    st.session_state[row_key] = count
    return count


def _add_board_action_row(patient_id: str) -> None:
    row_key = f"board_action_rows_{patient_id}"
    ensure_board_state()
    count = _board_action_row_count(patient_id) + 1
    st.session_state[row_key] = count
    items = list(st.session_state["board_actions"].get(patient_id, []))
    if not items:
        items = [{"task": "", "owner": "", "due_date": ""}]
    while len(items) < count:
        items.append({"task": "", "owner": "", "due_date": ""})
    update_case_notes(patient_id, action_items=items)


def _render_action_items_editor(patient_id: str) -> list[dict]:
    ensure_board_state()
    row_count = _board_action_row_count(patient_id)
    items = list(st.session_state["board_actions"].get(patient_id, []))
    if not items:
        items = [{"task": "", "owner": "", "due_date": ""}]
    while len(items) < row_count:
        items.append({"task": "", "owner": "", "due_date": ""})

    updated: list[dict] = []
    for idx in range(row_count):
        item = items[idx] if idx < len(items) else {"task": "", "owner": "", "due_date": ""}
        c1, c2, c3 = st.columns([2.2, 1, 1])
        with c1:
            task = st.text_input(
                "Task",
                value=item.get("task", ""),
                key=f"board_action_task_{patient_id}_{idx}",
                placeholder="e.g. Order PET-CT",
            )
        with c2:
            owner = st.text_input(
                "Owner",
                value=item.get("owner", ""),
                key=f"board_action_owner_{patient_id}_{idx}",
                placeholder="Role or name",
            )
        with c3:
            due = st.text_input(
                "Due",
                value=item.get("due_date", ""),
                key=f"board_action_due_{patient_id}_{idx}",
                placeholder="e.g. 1 week",
            )
        updated.append({"task": task, "owner": owner, "due_date": due})

    st.button(
        "Add action item",
        key=f"board_add_action_{patient_id}",
        on_click=_add_board_action_row,
        args=(patient_id,),
    )

    return updated


def _format_saved_board_label(summary: dict[str, str | int]) -> str:
    board_key = str(summary["meeting_date"])
    title = str(summary["board_title"]).strip()
    case_count = int(summary["case_count"])
    label = date.fromisoformat(display_meeting_date(board_key)).strftime("%b %d, %Y")
    if title:
        label = f"{title} — {label}"
    elif "#" in board_key:
        label += f" · board {board_key.split('#', 1)[1][:4]}"
    noun = "case" if case_count == 1 else "cases"
    return f"{label} ({case_count} {noun})"


def render_saved_board_sessions() -> None:
    summaries = list_meeting_summaries()
    if not summaries:
        return

    current = get_meeting_date()
    with st.expander("Saved boards", expanded=False):
        for summary in summaries:
            board_key = str(summary["meeting_date"])
            label = _format_saved_board_label(summary)
            is_current = board_key == current
            label_col, action_col = st.columns([4, 1])
            with label_col:
                suffix = " · open now" if is_current else ""
                st.markdown(f"**{label}**{suffix}")
            with action_col:
                if not is_current and st.button(
                    "Open",
                    key=f"open_board_{board_key.replace('#', '_')}",
                    use_container_width=True,
                ):
                    open_meeting(board_key)
                    st.rerun()


def page_todays_board(df: pd.DataFrame, ollama_ok: bool) -> None:
    prune_board_queue(df)
    ensure_board_state()
    render_saved_board_sessions()

    meta1, meta2, meta3, meta4 = st.columns([1.1, 1.35, 0.55, 0.4])
    with meta1:
        st.text_input(
            "Board date",
            value=date.fromisoformat(display_meeting_date(get_meeting_date())).strftime("%b %d, %Y"),
            disabled=True,
            key="board_display_date",
        )
    with meta2:
        board_title = st.text_input(
            "Board title",
            value=st.session_state.get("board_title", ""),
            key="board_title_input",
            placeholder="Optional meeting title",
        )
        if board_title != st.session_state.get("board_title"):
            st.session_state["board_title"] = board_title
            persist_meeting()
    with meta3:
        state = get_meeting_state()
        patient_labels = {
            str(row["patient_id"]): patient_profile_label(row)
            for _, row in df.iterrows()
        }
        minutes_pdf = format_meeting_minutes_pdf(state, patient_labels)
        minutes_md = format_meeting_minutes_markdown(state, patient_labels)
        dl_pdf, dl_md = st.columns(2)
        with dl_pdf:
            st.download_button(
                "📄",
                data=minutes_pdf,
                file_name=f"board_minutes_{display_meeting_date(get_meeting_date())}.pdf",
                mime="application/pdf",
                disabled=not state.cases,
                use_container_width=True,
                help="Export minutes (PDF)",
            )
        with dl_md:
            st.download_button(
                "📝",
                data=minutes_md,
                file_name=f"board_minutes_{display_meeting_date(get_meeting_date())}.md",
                mime="text/markdown",
                disabled=not state.cases,
                use_container_width=True,
                help="Export minutes (Markdown)",
            )
    with meta4:
        with st.popover("+", use_container_width=True, help="New board"):
            new_date = st.date_input(
                "Meeting date",
                value=date.today(),
                key="board_new_date_picker",
            )
            new_title = st.text_input(
                "Board title",
                placeholder="Optional",
                key="board_new_title_input",
            )
            if st.button("Create board", type="primary", use_container_width=True):
                create_new_board(new_date.isoformat(), new_title.strip())
                st.rerun()

    total, discussed, remaining = board_progress()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("On board", total)
    m2.metric("Discussed", discussed)
    m3.metric("Remaining", remaining)
    m4.metric("Meeting", display_meeting_date(get_meeting_date()))

    with st.expander("Add cases", expanded=not st.session_state["board_queue"]):
        label_to_id = {
            patient_profile_label(row): str(row["patient_id"])
            for _, row in df.iterrows()
        }
        on_board = set(st.session_state["board_queue"])
        available = [label for label, pid in label_to_id.items() if pid not in on_board]
        picked = st.multiselect(
            "Select patients",
            available,
            placeholder="Choose cases for this board…",
            key="board_add_multiselect",
        )
        if st.button("Add to board", type="primary", disabled=not picked, use_container_width=True):
            added = add_patients_to_board([label_to_id[label] for label in picked])
            if added:
                st.session_state.pop("board_add_multiselect", None)
            st.rerun()

    if not st.session_state["board_queue"]:
        return

    queue = st.session_state["board_queue"]
    idx = st.session_state["board_active_idx"]
    patient_id = get_active_patient_id()
    row = row_for_patient_id(df, patient_id) if patient_id else None

    queue_col, case_col = st.columns([0.32, 0.68], gap="large")

    with queue_col:
        st.markdown('<div class="board-panel">', unsafe_allow_html=True)
        st.markdown("##### Case list")
        board_cases = []
        for pid in queue:
            case_row = row_for_patient_id(df, pid)
            if case_row is None:
                continue
            status = normalize_board_status(
                st.session_state["board_status"].get(pid, "Ready for board")
            )
            board_cases.append(
                {
                    "id": pid,
                    "title": f"{pid} · {str(case_row['diagnosis'])[:28]}",
                    "status": status,
                    "status_class": re.sub(r"[^a-z0-9]", "", status.lower()),
                }
            )

        queue_event = board_queue_sorter(
            board_cases,
            active_id=patient_id or "",
            key=f"board_queue_sorter_{get_meeting_date()}",
        )
        if queue_event and queue_event.get("event"):
            if queue_event["event"] == "reorder":
                new_order = [pid for pid in queue_event.get("order", []) if pid in queue]
                if new_order and new_order != queue:
                    reorder_board_queue(new_order)
                    st.rerun()
            elif queue_event["event"] == "select":
                selected = str(queue_event.get("selected", ""))
                if selected in queue and selected != patient_id:
                    set_active_index(queue.index(selected))
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with case_col:
        if row is None:
            st.warning("Selected case is no longer available.")
            return

        prev_col, count_col, next_col = st.columns([0.45, 1.1, 0.45])
        with prev_col:
            if st.button(
                "←",
                disabled=idx <= 0,
                use_container_width=True,
                help="Previous case",
            ):
                set_active_index(idx - 1)
                st.rerun()
        with count_col:
            st.markdown(
                f"<p style='text-align:center; color:#65748b; margin:0.55rem 0 0;'>"
                f"Case <strong>{idx + 1}</strong> of <strong>{len(queue)}</strong></p>",
                unsafe_allow_html=True,
            )
        with next_col:
            if st.button(
                "→",
                disabled=idx >= len(queue) - 1,
                use_container_width=True,
                help="Next case",
            ):
                set_active_index(idx + 1)
                st.rerun()

        report_link = load_patient_report_link(patient_id)
        profile_summary = _profile_summary_for_patient(patient_id, row)
        analysis = report_link.analysis if report_link else None
        report_text = report_link.report_text if report_link else None
        render_mdt_case_summary(
            row,
            profile_summary=profile_summary,
            analysis=analysis,
            report_text=report_text,
        )

        current_status = normalize_board_status(
            st.session_state["board_status"].get(patient_id, "Ready for board")
        )
        status_index = (
            list(BOARD_STATUSES).index(current_status)
            if current_status in BOARD_STATUSES
            else 0
        )
        new_status = st.selectbox(
            "Case status",
            BOARD_STATUSES,
            index=status_index,
            key=f"board_status_{patient_id}",
        )
        if new_status != current_status:
            set_status(patient_id, new_status)
            log_audit_event("board_status", patient_id=patient_id, detail=new_status)
            st.rerun()

        with st.expander("Record decision", expanded=True):
            question = st.text_input(
                "Discussion question",
                value=st.session_state["board_questions"].get(patient_id, ""),
                placeholder="What should the board decide today?",
                key=f"board_question_{patient_id}",
            )

            recommendation = st.text_area(
                "MDT decision",
                value=st.session_state["board_recommendations"].get(patient_id, ""),
                placeholder="Board decision (plan, treatment path, further workup…)",
                key=f"board_recommendation_{patient_id}",
                height=100,
            )
            rationale = st.text_area(
                "Rationale",
                value=st.session_state["board_rationale"].get(patient_id, ""),
                placeholder="Why this decision; key factors from the discussion",
                key=f"board_rationale_{patient_id}",
                height=80,
            )
            follow_up = st.text_input(
                "Follow-up review date",
                value=st.session_state["board_follow_up"].get(patient_id, ""),
                placeholder="e.g. 2026-07-15 or 6 weeks",
                key=f"board_follow_up_{patient_id}",
            )

            action_items = _render_action_items_editor(patient_id)
            update_case_notes(
                patient_id,
                discussion_question=question,
                recommendation=recommendation,
                rationale=rationale,
                follow_up_date=follow_up,
                action_items=action_items,
            )

        if report_link:
            with st.expander(f"PDF briefing — {report_link.report_name}", expanded=False):
                render_linked_report_summary(report_link, compact=True)

        with st.expander("Chart details", expanded=False):
            render_patient_overview(row)

        if st.button("Remove from board", use_container_width=True):
            remove_from_board(patient_id)
            st.rerun()

        with st.expander("Full MDT brief (profile)", expanded=False):
            patient_data = format_patient_data(row)
            fingerprint = record_fingerprint(patient_data)
            if profile_summary:
                display_summary(profile_summary)
            if ollama_ok:
                if st.button("Generate MDT brief", type="primary", use_container_width=True):
                    try:
                        summary, from_cache, latency = load_summary(
                            patient_id,
                            patient_data,
                            fingerprint,
                            st.session_state.get(f"cache_bust_{patient_id}", 0),
                            force_refresh=True,
                        )
                        st.session_state["summary"] = summary
                        st.session_state["summary_meta"] = {
                            "patient_id": patient_id,
                            "fingerprint": fingerprint,
                            "from_cache": from_cache,
                            "latency_sec": latency,
                            "prompt_version": PROMPT_VERSION,
                        }
                        log_audit_event(
                            "mdt_brief_generated",
                            patient_id=patient_id,
                            prompt_version=PROMPT_VERSION,
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Brief generation failed: {exc}")


def page_home(df: pd.DataFrame) -> None:
    custom_count = int((df.get("source", "") == "custom").sum()) if "source" in df else 0
    synthetic_count = int((df.get("source", "") == "synthetic").sum()) if "source" in df else 0
    reference_count = max(len(df) - custom_count - synthetic_count, 0)

    stat_a, stat_b, stat_c = st.columns(3)
    with stat_a:
        st.markdown(
            f"<div class='home-stat-card'><p>Patients</p><h2>{len(df)}</h2></div>",
            unsafe_allow_html=True,
        )
    with stat_b:
        st.markdown(
            f"<div class='home-stat-card'><p>Reference</p><h2>{reference_count}</h2></div>",
            unsafe_allow_html=True,
        )
    with stat_c:
        st.markdown(
            f"<div class='home-stat-card'><p>Custom</p><h2>{custom_count}</h2></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Case mix")
    preview_cols = ["patient_id", "diagnosis", "stage", "age", "sex", "source"]
    available_cols = [col for col in preview_cols if col in df.columns]
    st.dataframe(
        df[available_cols],
        use_container_width=True,
        hide_index=True,
    )


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            text = page.extract_text() or ""
        if text.strip():
            pages.append(f"Page {idx}\n{text.strip()}")
    return clean_extracted_report_text("\n\n".join(pages))


def clean_extracted_report_text(text: str) -> str:
    """Remove teaching/commentary fragments that PDF extraction can merge into notes."""
    annotation_markers = [
        "Define the",
        "specifically as possible",
        "Convey the",
        "establish a chronology",
        "circumstances; exacerbating factors",
        "associated symptoms",
        "resolution; alleviating factors",
        "Describe the natural history",
        "Change or new circumstances",
        "New duration",
        "Reason she come in",
        "What has patient tried",
        "Relevant positive",
        "Review of systems for the relevant",
        "Relevant risk factor",
        "This highly relevant",
        "trivial detail",
        "Always use generic names",
        "Always list",
        "Quantity",
        "Include over-the-counter",
        "Comment specifically",
        "Separate each ROS",
        "OK to refer",
        "List positive and negative",
        "Check for orthostatic",
        "Description may give",
        "Comment on all organ systems",
        "List specific normal",
        "This patient needs",
        "More precise",
        "Always include these exams",
        "Although you can",
        "shown below",
        "to keep track",
        "This list regroups",
        "suspect are related",
        "In the assessment",
        "You should",
        "As in the previous problem",
        "Follow this pattern",
        "You are expected",
    ]
    cleaned_lines = []
    for raw_line in text.splitlines():
        raw_line = raw_line.rstrip()
        line = " ".join(raw_line.split())
        if not line:
            continue

        cut_at = None
        for marker in annotation_markers:
            idx = raw_line.find(marker)
            if idx == 0:
                cut_at = 0
                break
            if idx > 0 and re.search(r"\s{2,}$", raw_line[:idx]):
                # Side-column comments are separated from note text by PDF layout spacing.
                cut_at = idx if cut_at is None else min(cut_at, idx)

        if cut_at == 0:
            continue
        if cut_at is not None:
            line = " ".join(raw_line[:cut_at].split())

        line = re.sub(r"\s+(onset|character|location|radiation|duration)$", "", line, flags=re.IGNORECASE)

        # Remove short standalone teaching labels that survive extraction.
        if line.lower() == "history and physical examination comments":
            continue
        if line.lower() in {"onset", "character", "location", "radiation", "duration"}:
            continue
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def render_list_items(items, empty_text: str = "Not specified in the report.") -> None:
    items = [str(item).strip() for item in _as_list(items) if str(item).strip()]
    items = [
        item
        for item in items
        if not re.search(r"\b(thought|identify the goal|scan the report|the user wants)\b", item, re.IGNORECASE)
    ]
    if not items:
        st.caption(empty_text)
        return
    for item in items:
        st.markdown(f"- {item}")


def _clean_card_text(value, fallback: str) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "..."}:
        return fallback
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<unused\d+>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>\s]+>", "", text)
    if re.search(r"\b(thought|identify the goal|scan the report|the user wants|i need to)\b", text, re.IGNORECASE):
        return fallback
    return " ".join(text.split())


def render_problem_cards(rows: list[dict]) -> None:
    cards = []
    for idx, row in enumerate(rows, start=1):
        problem = html.escape(_clean_card_text(row.get("problem", row.get("Problem")), "Priority problem"))
        evidence = html.escape(_clean_card_text(row.get("evidence", row.get("Evidence")), "Not specified in the report."))
        why = html.escape(
            _clean_card_text(
                row.get("why_it_matters", row.get("Why it matters", row.get("why"))),
                "Review during meeting.",
            )
        )
        cards.append(
            (
                '<article class="problem-card">'
                f'<div class="problem-card-index">{idx}</div>'
                '<div>'
                f'<h4>{problem}</h4>'
                f'<p><b>Evidence</b>{evidence}</p>'
                f'<p><b>Why it matters</b>{why}</p>'
                '</div>'
                '</article>'
            )
        )
    st.markdown(f"<div class='problem-card-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_report_clinical_briefing(analysis: dict, report_text: str) -> None:
    """AI briefing sections derived from an uploaded PDF (no charts)."""
    analysis = repair_report_analysis(analysis, report_text)
    snapshot = analysis.get("patient_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}

    m1, m2, m3 = st.columns(3)
    m1.metric("Problems", len(_as_list(analysis.get("priority_problems"))))
    m2.metric("Red flags", len(_as_list(analysis.get("red_flags"))))
    m3.metric("Gaps", len(_as_list(analysis.get("missing_data"))))

    presenting = str(snapshot.get("presenting_problem", "")).strip()
    likely = str(snapshot.get("likely_primary_issue", "")).strip()
    objective = str(analysis.get("meeting_objective", "")).strip()
    if presenting:
        st.markdown(f"**Presenting problem** — {presenting}")
    if likely:
        st.markdown(f"**Likely issue** — {likely}")
    if objective:
        st.markdown(f"**Meeting objective** — {objective}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Critical facts**")
        render_list_items(analysis.get("critical_facts"))
        st.markdown("**Red flags**")
        render_list_items(analysis.get("red_flags"), "None extracted.")
    with right:
        st.markdown("**Missing data**")
        render_list_items(analysis.get("missing_data"))
        st.markdown("**Decision points**")
        render_list_items(analysis.get("decision_points"))

    problems = _as_list(analysis.get("priority_problems"))
    rows = [p for p in problems if isinstance(p, dict)]
    if rows:
        st.markdown("**Priority problems**")
        render_problem_cards(rows)
    else:
        render_list_items(problems, "No priority problems extracted.")

    flow = [str(item).strip() for item in _as_list(analysis.get("meeting_flow")) if str(item).strip()]
    if flow:
        st.markdown("**Discussion order**")
        for idx, item in enumerate(flow, start=1):
            st.markdown(f"{idx}. {item}")

    focus = _as_list(analysis.get("specialist_focus"))
    focus_rows = [item for item in focus if isinstance(item, dict)]
    if focus_rows:
        with st.expander("Specialist focus (table)", expanded=False):
            st.dataframe(pd.DataFrame(focus_rows), use_container_width=True, hide_index=True)
    elif focus:
        with st.expander("Specialist focus", expanded=False):
            render_list_items(focus)

    with st.expander("Source text from PDF", expanded=False):
        st.text_area(
            "Extracted report",
            value=report_text[:30000],
            height=280,
            label_visibility="collapsed",
        )


def _legacy_clean_display_text(value, fallback: str = "Not specified") -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "..."}:
        return fallback
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<unused\d+>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>\s]+>", "", text)
    if re.search(
        r"\b(thought|identify the goal|scan the report|the user wants|i need to)\b",
        text,
        re.IGNORECASE,
    ):
        return fallback
    return " ".join(text.split())


def _legacy_report_packet_markdown(analysis: dict, report_text: str) -> str:
    snapshot = analysis.get("patient_snapshot") if isinstance(analysis.get("patient_snapshot"), dict) else {}
    lines = [
        "# OncoBoard report briefing",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"## Patient: {_legacy_clean_display_text(snapshot.get('name_or_id'), 'Uploaded report')}",
        f"- Age: {_legacy_clean_display_text(snapshot.get('age'))}",
        f"- Sex: {_legacy_clean_display_text(snapshot.get('sex'))}",
        f"- Presenting problem: {_legacy_clean_display_text(snapshot.get('presenting_problem'))}",
        f"- Likely issue: {_legacy_clean_display_text(snapshot.get('likely_primary_issue'))}",
        "",
        f"## Meeting objective\n{_legacy_clean_display_text(analysis.get('meeting_objective'), 'Clarify the key decision for this case.')}",
        "",
    ]
    for title, values in [
        ("Critical facts", analysis.get("critical_facts")),
        ("Red flags", analysis.get("red_flags")),
        ("Missing data", analysis.get("missing_data")),
        ("Priority problems", analysis.get("priority_problems")),
        ("Specialist focus", analysis.get("specialist_focus")),
        ("Decision points", analysis.get("decision_points")),
        ("Suggested meeting flow", analysis.get("meeting_flow")),
    ]:
        lines.append(f"## {title}")
        items = _as_list(values)
        if not items:
            lines.append("- Not specified")
        for item in items:
            if isinstance(item, dict):
                text = " | ".join(
                    f"{key.replace('_', ' ').title()}: {_legacy_clean_display_text(value, '')}"
                    for key, value in item.items()
                    if _legacy_clean_display_text(value, "")
                )
            else:
                text = _legacy_clean_display_text(item, "")
            if text:
                lines.append(f"- {text}")
        lines.append("")
    lines.append("## Source excerpt")
    lines.append(report_text[:3000])
    return "\n".join(lines)


def _markdown_pdf_bytes(title: str, body: str) -> bytes:
    def safe_pdf_text(value: str) -> str:
        return (
            str(value)
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\t", "    ")
            .replace("\u200b", "")
            .replace("\ufeff", "")
            .encode("latin-1", errors="replace")
            .decode("latin-1")
        )

    safe_body = (
        safe_pdf_text(body)
    )
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=18, top=18, right=18)
    pdf.add_page()
    content_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(content_width, 9, safe_pdf_text(title)[:90])
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    for line in safe_body.splitlines():
        chunks = textwrap.wrap(
            line,
            width=72,
            break_long_words=True,
            break_on_hyphens=True,
            replace_whitespace=False,
            drop_whitespace=True,
        ) or [" "]
        for chunk in chunks:
            pdf.multi_cell(content_width, 5.6, chunk[:120])
    return bytes(pdf.output())


def _legacy_list(items, empty_text: str = "Not specified in the report.") -> None:
    clean_items = [
        _legacy_clean_display_text(item, "")
        for item in _as_list(items)
        if str(item).strip()
    ]
    clean_items = [item for item in clean_items if item]
    if not clean_items:
        st.caption(empty_text)
        return
    for item in clean_items:
        st.markdown(f"- {item}")


def _legacy_specialist_cards(items) -> None:
    rows = [item for item in _as_list(items) if isinstance(item, dict)]
    if not rows:
        _legacy_list(items, "No specialist focus items were extracted.")
        return
    cards = []
    for row in rows:
        specialist = _legacy_clean_display_text(row.get("specialist"), "Specialist")
        review = _legacy_clean_display_text(
            row.get("what_they_need_to_review", row.get("review")),
            "Review decision-relevant details for this case.",
        )
        cards.append(
            "<article class='specialist-card'>"
            f"<span>{html.escape(specialist)}</span>"
            f"<p>{html.escape(review)}</p>"
            "</article>"
        )
    st.markdown(f"<div class='specialist-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _legacy_report_analysis_with_loader(report_text: str) -> dict:
    loader = st.empty()
    stages = [
        (12, "Reading report"),
        (24, "Building patient snapshot"),
        (36, "Extracting clinical facts"),
        (48, "Prioritizing problems"),
        (62, "Mapping specialist focus"),
        (76, "Finding gaps and red flags"),
        (88, "Building meeting agenda"),
        (90, "Finalizing board view"),
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(analyze_report_fast, report_text)
        tick = 0
        while not future.done():
            if tick > 300:
                future.cancel()
                loader.empty()
                return repair_report_analysis(heuristic_report_analysis(report_text), report_text)
            base_progress, stage = stages[min(tick // 5, len(stages) - 1)]
            with loader.container():
                render_report_loader(base_progress + min(tick % 5, 4), stage)
            time.sleep(0.18)
            tick += 1
        analysis, _mode = future.result()

    with loader.container():
        render_report_loader(100, "Board view ready")
    time.sleep(0.25)
    loader.empty()
    return repair_report_analysis(analysis, report_text)


def render_legacy_report_analysis(analysis: dict, report_text: str) -> None:
    analysis = repair_report_analysis(analysis, report_text)
    snapshot = analysis.get("patient_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}

    name = _legacy_clean_display_text(snapshot.get("name_or_id"), "Uploaded report")
    age = _legacy_clean_display_text(snapshot.get("age"), "Not specified")
    sex = _legacy_clean_display_text(snapshot.get("sex"), "Not specified")
    presenting_problem = _legacy_clean_display_text(snapshot.get("presenting_problem"), "Not specified in the report.")
    likely_issue = _legacy_clean_display_text(snapshot.get("likely_primary_issue"), "Not specified in the report.")
    objective = _legacy_clean_display_text(analysis.get("meeting_objective"), "Clarify the key clinical decision for this case.")
    page_count = report_text.count("Page ")
    problem_count = len(_as_list(analysis.get("priority_problems")))
    red_flag_count = len(_as_list(analysis.get("red_flags")))
    missing_count = len(_as_list(analysis.get("missing_data")))

    st.markdown(
        f"""
        <section class="report-brief-hero">
          <div class="report-brief-top">
            <div>
              <p class="report-eyebrow">Board briefing</p>
              <h2>{html.escape(name)}</h2>
              <div class="report-chip-row">
                <span>{html.escape(age)} yrs</span>
                <span>{html.escape(sex)}</span>
                <span>{page_count} pages read</span>
              </div>
            </div>
            <div class="report-risk-stack">
              <div><b>{problem_count}</b><span>Problems</span></div>
              <div><b>{red_flag_count}</b><span>Red flags</span></div>
              <div><b>{missing_count}</b><span>Gaps</span></div>
            </div>
          </div>
          <div class="report-brief-grid">
            <article><span>Presenting problem</span><p>{html.escape(presenting_problem)}</p></article>
            <article><span>Likely primary issue</span><p>{html.escape(likely_issue)}</p></article>
          </div>
          <article class="meeting-objective-card">
            <span>Meeting objective</span>
            <p>{html.escape(objective)}</p>
          </article>
        </section>
        """,
        unsafe_allow_html=True,
    )

    packet = _legacy_report_packet_markdown(analysis, report_text)
    pdf_col, md_col = st.columns(2)
    pdf_col.download_button(
        "Export report briefing PDF",
        data=_markdown_pdf_bytes("OncoBoard report briefing", packet),
        file_name=f"oncoboard_report_briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    md_col.download_button(
        "Export report briefing",
        data=packet,
        file_name=f"oncoboard_report_briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    tab_signal, tab_charts, tab_team, tab_agenda, tab_source = st.tabs(
        ["Clinical signal", "Vitals & timeline", "Team focus", "Meeting agenda", "Source text"]
    )
    with tab_signal:
        left, right = st.columns([0.52, 0.48])
        with left:
            st.markdown("#### Critical facts")
            _legacy_list(analysis.get("critical_facts"))
            st.markdown("#### Red flags")
            _legacy_list(analysis.get("red_flags"), "No urgent red flags were extracted.")
        with right:
            st.markdown("#### Missing data")
            _legacy_list(analysis.get("missing_data"), "No missing data was extracted.")

        problems = _as_list(analysis.get("priority_problems"))
        rows = [p for p in problems if isinstance(p, dict)]
        st.markdown("#### Priority problems")
        if rows:
            render_problem_cards(rows)
        else:
            _legacy_list(problems, "No priority problems were extracted.")

    with tab_charts:
        chart_key = hashlib.sha256(report_text[:4000].encode("utf-8")).hexdigest()[:12]
        render_report_charts(
            analysis,
            report_text,
            section_label=None,
            chart_key_prefix=f"legacy_{chart_key}",
        )

    with tab_team:
        _legacy_specialist_cards(analysis.get("specialist_focus"))
        st.markdown("#### Decision points")
        _legacy_list(analysis.get("decision_points"))

    with tab_agenda:
        st.markdown("#### Suggested discussion order")
        flow = [
            _legacy_clean_display_text(item, "")
            for item in _as_list(analysis.get("meeting_flow"))
            if _legacy_clean_display_text(item, "")
        ]
        if flow:
            for idx, item in enumerate(flow, start=1):
                st.markdown(f"**{idx}.** {item}")
        else:
            st.caption("No meeting flow was extracted.")

    with tab_source:
        st.text_area("Extracted report text", value=report_text[:30000], height=360)


def page_legacy_report_intake(ollama_ok: bool) -> None:
    st.subheader("Report intake")
    st.caption(
        "Upload a clinical PDF and convert it into a meeting-ready board briefing. "
        "MedGemma analyzes each section separately to reduce cutoff and improve completeness."
    )

    if not ollama_ok:
        st.warning("Start Ollama locally to enable report analysis.")

    uploaded = st.file_uploader("Upload report PDF", type=["pdf"], key="legacy_report_pdf")
    if not uploaded:
        st.info("Upload a report to extract a patient snapshot, key facts, priority problems, specialist focus, and meeting agenda.")
        return

    try:
        report_text = extract_pdf_text(uploaded)
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
        return

    if not report_text:
        st.error("No readable text was found in this PDF.")
        return

    report_fingerprint = hashlib.sha256(f"{PROMPT_VERSION}\n{report_text}".encode("utf-8")).hexdigest()
    if st.session_state.get("legacy_report_fingerprint") != report_fingerprint:
        st.session_state.pop("legacy_report_analysis", None)
        st.session_state.pop("legacy_report_text", None)
        st.session_state["legacy_report_fingerprint"] = report_fingerprint

    st.success(f"Extracted {len(report_text):,} characters from {uploaded.name}.")

    action_col, reset_col = st.columns([0.72, 0.28])
    button_label = "Regenerate board briefing" if st.session_state.get("legacy_report_analysis") else "Generate board briefing"
    if action_col.button(button_label, type="primary", disabled=not ollama_ok, use_container_width=True):
        try:
            st.session_state["legacy_report_analysis"] = _legacy_report_analysis_with_loader(report_text)
            st.session_state["legacy_report_text"] = report_text
            st.session_state["legacy_report_name"] = uploaded.name
            st.session_state["legacy_report_fingerprint"] = report_fingerprint
        except Exception as exc:
            st.error(f"Report analysis failed: {exc}")

    if reset_col.button("Clear briefing", use_container_width=True):
        st.session_state.pop("legacy_report_analysis", None)
        st.session_state.pop("legacy_report_text", None)
        st.rerun()

    if st.session_state.get("legacy_report_analysis") and st.session_state.get("legacy_report_text"):
        repaired_analysis = repair_report_analysis(
            st.session_state["legacy_report_analysis"],
            st.session_state["legacy_report_text"],
        )
        st.session_state["legacy_report_analysis"] = repaired_analysis
        st.caption("Briefing checked against source text before display.")
        render_legacy_report_analysis(repaired_analysis, st.session_state["legacy_report_text"])


def render_layered_analysis(
    row: pd.Series,
    analysis: dict | None,
    report_text: str | None,
) -> None:
    """Chart-record baseline plus optional PDF report layer."""
    has_report = bool(analysis and report_text)
    tab_visual, tab_clinical = st.tabs(["Visual insights", "Clinical briefing"])

    patient_id = str(row.get("patient_id", "patient"))
    with tab_visual:
        render_profile_visual_insights(row, chart_key_prefix=f"layered_profile_{patient_id}")
        if has_report:
            st.divider()
            render_report_charts(
                analysis,
                report_text,
                section_label="Added from uploaded report",
                chart_key_prefix=f"layered_report_{patient_id}",
            )
    with tab_clinical:
        st.markdown("#### Chart record")
        render_profile_clinical_summary(row)
        if has_report:
            st.divider()
            st.markdown("#### Added from uploaded report")
            render_report_clinical_briefing(analysis, report_text)


def _record_from_report_snapshot(
    snapshot: dict,
    analysis: dict,
    report_text: str,
    df: pd.DataFrame,
    *,
    patient_id: str = "",
) -> dict:
    """Build a registry record from PDF analysis (editable before save)."""
    age_raw = str(snapshot.get("age", ""))
    age_match = re.search(r"\d+", age_raw)
    age = int(age_match.group()) if age_match else 60
    sex = str(snapshot.get("sex", "U")).strip().upper()[:1]
    if sex not in "FMU":
        sex = "U"
    presenting = str(snapshot.get("presenting_problem", "")).strip()
    likely = str(snapshot.get("likely_primary_issue", "")).strip()
    diagnosis = likely or presenting or "Diagnosis pending (from report)"
    facts = [str(f).strip() for f in _as_list(analysis.get("critical_facts")) if str(f).strip()]
    intake_parts = [
        presenting,
        likely,
        "",
        "Critical facts from PDF briefing:",
        *[f"- {fact}" for fact in facts[:12]],
        "",
        "--- Report excerpt ---",
        report_text[:6000],
    ]
    return normalize_record(
        {
            "patient_id": patient_id or next_patient_id(df),
            "age": age,
            "sex": sex,
            "diagnosis": diagnosis[:200],
            "stage": "Pending staging",
            "ecog": 1,
            "biomarkers": "",
            "imaging": "",
            "pathology": "",
            "comorbidities": "",
            "medications": "",
            "pending_tests": "; ".join(
                str(x).strip() for x in _as_list(analysis.get("missing_data"))[:5] if str(x).strip()
            ),
            "prior_treatment": "",
            "notes": str(analysis.get("meeting_objective", ""))[:500],
            "intake_text": "\n".join(part for part in intake_parts if part),
            "source": "custom",
        },
        source="custom",
    )


def render_patient_report_status(patient_id: str) -> None:
    """Show linked PDF briefing status on the patient chart."""
    link = load_patient_report_link(patient_id)
    if not link:
        return
    st.markdown(
        f"<div class='info-card' style='margin:0.5rem 0 1rem; padding:0.75rem 1rem;'>"
        f"<strong>PDF analyzed</strong> — {html.escape(link.report_name)}"
        f"</div>",
        unsafe_allow_html=True,
    )


def run_report_analysis_workbench(
    patient_id: str,
    ollama_ok: bool,
    *,
    persist_link: bool = True,
) -> tuple[dict | None, str | None]:
    """Upload PDF, generate AI briefing. Returns (analysis, report_text) when ready."""
    if patient_id != NEW_REPORT_PATIENT_ID:
        st.session_state["report_intake_patient_id"] = patient_id
        st.session_state["report_link_patient_id"] = patient_id

    saved_link = load_patient_report_link(patient_id) if persist_link else None
    if saved_link and not _get_report_field(patient_id, "text"):
        _hydrate_report_workbench_from_patient(patient_id)

    if not ollama_ok:
        st.warning("Start Ollama to generate briefings.")

    uploaded = st.file_uploader(
        "Clinical PDF",
        type=["pdf"],
        key=f"report_pdf_uploader_{patient_id}",
    )
    if saved_link:
        if st.button("Clear saved analysis", key=f"pdf_clear_{patient_id}"):
            delete_patient_report_link(patient_id)
            _clear_report_workbench(patient_id)
            st.rerun()

    if uploaded is not None:
        _set_report_field(patient_id, "pdf_bytes", uploaded.getvalue())
        _set_report_field(patient_id, "pdf_name", uploaded.name)

    pdf_bytes = _get_report_field(patient_id, "pdf_bytes")
    if not pdf_bytes:
        if saved_link:
            _set_report_field(patient_id, "text", saved_link.report_text)
            _set_report_field(patient_id, "analysis", saved_link.analysis)
            _set_report_field(patient_id, "fingerprint", saved_link.report_fingerprint)
            _set_report_field(patient_id, "name", saved_link.report_name)
        elif not _get_report_field(patient_id, "text"):
            return None, None
    else:
        report_name = _get_report_field(patient_id, "pdf_name", "uploaded_report.pdf")
        try:
            report_text = extract_pdf_text(BytesIO(pdf_bytes))
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")
            return None, None
        if not report_text:
            st.error("No readable text was found in this PDF.")
            return None, None
        report_fingerprint = hashlib.sha256(f"{PROMPT_VERSION}\n{report_text}".encode("utf-8")).hexdigest()
        if _get_report_field(patient_id, "fingerprint") != report_fingerprint:
            _set_report_field(patient_id, "analysis", None)
            _set_report_field(patient_id, "cached_key", None)
            _set_report_field(patient_id, "fingerprint", report_fingerprint)
        _set_report_field(patient_id, "text", report_text)
        _set_report_field(patient_id, "name", report_name)

    report_text = _get_report_field(patient_id, "text")
    report_name = _get_report_field(patient_id, "name", "uploaded_report.pdf")
    report_fingerprint = _get_report_field(patient_id, "fingerprint", "")

    has_briefing = bool(_get_report_field(patient_id, "analysis"))
    if st.button(
        "Regenerate briefing" if has_briefing else "Generate briefing",
        type="primary",
        disabled=not ollama_ok or not report_text,
        use_container_width=True,
        key=f"report_generate_{patient_id}",
    ):
        try:
            force_refresh = has_briefing
            if force_refresh:
                _set_report_field(
                    patient_id,
                    "cache_bust",
                    _get_report_field(patient_id, "cache_bust", 0) + 1,
                )
                _set_report_field(patient_id, "cached_key", None)
                get_cached_report_analysis.clear()
            analysis, from_cache = analyze_report_with_loader(
                report_text,
                report_fingerprint,
                patient_id,
                force_refresh=force_refresh,
            )
            _set_report_field(patient_id, "analysis", analysis)
            _set_report_field(patient_id, "from_cache", from_cache)
            repaired = repair_report_analysis(analysis, report_text)
            _set_report_field(patient_id, "analysis", repaired)
            if persist_link and patient_id != NEW_REPORT_PATIENT_ID:
                st.session_state[f"report_pending_review_{patient_id}"] = True
            elif persist_link:
                _save_report_to_patient(
                    patient_id,
                    report_name=report_name,
                    report_fingerprint=report_fingerprint,
                    analysis=repaired,
                    report_text=report_text,
                )
            log_audit_event(
                "report_analysis_generated",
                patient_id=patient_id if patient_id != NEW_REPORT_PATIENT_ID else "",
                prompt_version=PROMPT_VERSION,
            )
            if not from_cache:
                st.success("Report analysis ready for review.")
            st.rerun()
        except Exception as exc:
            st.error(f"Report analysis failed: {exc}")
            logging.exception("report_analysis_failed")

    analysis = _get_report_field(patient_id, "analysis")
    return analysis, report_text


def render_report_chart_review(
    patient_id: str,
    row: pd.Series,
    analysis: dict,
    report_text: str,
    df: pd.DataFrame,
) -> None:
    """Review extracted PDF fields before saving briefing and optionally merging chart."""
    snapshot = analysis.get("patient_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    draft = _record_from_report_snapshot(snapshot, analysis, report_text, df, patient_id=patient_id)

    st.markdown("#### Save briefing to patient")
    st.caption(
        "Stores the board briefing unchanged. Optionally merge suggested chart fields below."
    )
    chart_editable = str(row.get("source", "")) in ("custom", "synthetic")
    with st.form(f"report_review_{patient_id}", clear_on_submit=False):
        merge_chart = st.checkbox(
            "Merge suggested fields into patient chart",
            value=chart_editable,
            disabled=not chart_editable,
            help="Reference patients keep chart read-only; briefing still saves.",
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            diag = st.text_input("Diagnosis", value=str(draft.get("diagnosis", row.get("diagnosis", ""))))
        with c2:
            stage = st.text_input("Stage", value=str(draft.get("stage", row.get("stage", ""))))
        with c3:
            ecog = st.number_input(
                "ECOG",
                min_value=0,
                max_value=4,
                value=coerce_int(draft.get("ecog", row.get("ecog", 1)), 1),
            )
        with c4:
            pending = st.text_input(
                "Pending tests",
                value=str(draft.get("pending_tests", row.get("pending_tests", ""))),
            )
        biomarkers = st.text_area(
            "Biomarkers",
            value=str(draft.get("biomarkers", row.get("biomarkers", ""))),
            height=68,
        )
        imaging = st.text_area(
            "Imaging",
            value=str(draft.get("imaging", row.get("imaging", ""))),
            height=68,
        )
        pathology = st.text_area(
            "Pathology",
            value=str(draft.get("pathology", row.get("pathology", ""))),
            height=68,
        )
        prior = st.text_input(
            "Prior treatment",
            value=str(draft.get("prior_treatment", row.get("prior_treatment", ""))),
        )
        mdt_question = st.text_input(
            "MDT question for today",
            value=str(analysis.get("meeting_objective", ""))[:240],
        )
        if st.form_submit_button("Save briefing to patient", type="primary"):
            report_name = _get_report_field(patient_id, "name", "uploaded_report.pdf")
            report_fingerprint = _get_report_field(patient_id, "fingerprint", "")
            _save_report_to_patient(
                patient_id,
                report_name=report_name,
                report_fingerprint=report_fingerprint,
                analysis=analysis,
                report_text=report_text,
            )
            if merge_chart:
                record = {col: row.get(col, "") for col in PATIENT_COLUMNS}
                record.update(
                    {
                        "patient_id": patient_id,
                        "diagnosis": diag,
                        "stage": stage,
                        "ecog": ecog,
                        "biomarkers": biomarkers,
                        "imaging": imaging,
                        "pathology": pathology,
                        "pending_tests": pending,
                        "prior_treatment": prior,
                        "notes": mdt_question or str(record.get("notes", "")),
                    }
                )
                try:
                    update_custom_patient(record)
                    bump_patient_data_version()
                    log_audit_event("chart_merged_from_report", patient_id=patient_id)
                except ValueError as exc:
                    st.warning(str(exc))
            st.session_state.pop(f"report_pending_review_{patient_id}", None)
            log_audit_event("report_briefing_saved", patient_id=patient_id)
            st.success("PDF briefing saved.")
            st.rerun()


def render_new_patient_from_report(df: pd.DataFrame, ollama_ok: bool) -> None:
    """Analyze a PDF for someone not in the registry, then create their chart record."""
    patient_id = NEW_REPORT_PATIENT_ID
    analysis, report_text = run_report_analysis_workbench(
        patient_id,
        ollama_ok,
        persist_link=False,
    )
    if analysis and report_text:
        st.session_state["new_report_draft_analysis"] = analysis
        st.session_state["new_report_draft_text"] = report_text
    else:
        analysis = st.session_state.get("new_report_draft_analysis")
        report_text = st.session_state.get("new_report_draft_text")

    if not analysis or not report_text:
        return

    analysis = repair_report_analysis(analysis, report_text)
    snapshot = analysis.get("patient_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    draft_record = _record_from_report_snapshot(snapshot, analysis, report_text, df)

    st.divider()
    render_legacy_report_analysis(analysis, report_text)

    st.divider()
    st.markdown("#### Add to patient registry")
    with st.form("create_patient_from_report", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            new_id = st.text_input("Patient ID", value=draft_record["patient_id"])
        with c2:
            new_age = st.number_input("Age", min_value=0, max_value=120, value=coerce_int(draft_record["age"], 60))
        with c3:
            sex_val = str(draft_record.get("sex", "U"))
            if sex_val not in ("F", "M", "U"):
                sex_val = "U"
            new_sex = st.selectbox("Sex", ["F", "M", "U"], index=["F", "M", "U"].index(sex_val))
        with c4:
            new_ecog = st.number_input("ECOG", min_value=0, max_value=4, value=coerce_int(draft_record["ecog"], 1))
        new_diagnosis = st.text_input("Diagnosis", value=str(draft_record["diagnosis"]))
        new_stage = st.text_input("Stage", value=str(draft_record["stage"]))
        new_intake = st.text_area("Clinical narrative", value=str(draft_record["intake_text"]), height=160)
        if st.form_submit_button("Create patient & save report", type="primary"):
            if not new_diagnosis.strip() or not new_stage.strip():
                st.error("Diagnosis and stage are required.")
            else:
                record = normalize_record(
                    {
                        "patient_id": new_id,
                        "age": new_age,
                        "sex": new_sex,
                        "diagnosis": new_diagnosis,
                        "stage": new_stage,
                        "ecog": new_ecog,
                        "biomarkers": draft_record.get("biomarkers", ""),
                        "imaging": draft_record.get("imaging", ""),
                        "pathology": draft_record.get("pathology", ""),
                        "comorbidities": draft_record.get("comorbidities", ""),
                        "medications": draft_record.get("medications", ""),
                        "pending_tests": draft_record.get("pending_tests", ""),
                        "prior_treatment": draft_record.get("prior_treatment", ""),
                        "notes": draft_record.get("notes", ""),
                        "intake_text": new_intake,
                        "source": "custom",
                    },
                    source="custom",
                )
                try:
                    add_custom_patient(record)
                    bump_patient_data_version()
                    pid = str(record["patient_id"])
                    report_name = _get_report_field(NEW_REPORT_PATIENT_ID, "name", "uploaded_report.pdf")
                    report_fingerprint = _get_report_field(NEW_REPORT_PATIENT_ID, "fingerprint", "")
                    _save_report_to_patient(
                        pid,
                        report_name=report_name,
                        report_fingerprint=report_fingerprint,
                        analysis=analysis,
                        report_text=report_text,
                    )
                    _clear_report_workbench(NEW_REPORT_PATIENT_ID)
                    st.session_state.pop("new_report_draft_analysis", None)
                    st.session_state.pop("new_report_draft_text", None)
                    st.session_state["report_analysis_mode"] = "Existing patient"
                    st.session_state["selected_patient_id"] = pid
                    st.session_state["selected_patient_label"] = patient_profile_label(pd.Series(record))
                    st.session_state["report_intake_patient_id"] = pid
                    st.success(f"Created **{pid}** and linked the PDF briefing.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


def page_report_analysis(df: pd.DataFrame, ollama_ok: bool) -> None:
    st.subheader("Report analysis")
    st.caption("Upload a clinical PDF for a meeting-ready board briefing.")

    if not ollama_ok:
        st.warning("Start Ollama locally to enable report analysis.")

    _mode_options = ("Existing patient", "New patient")
    _legacy_mode = st.session_state.get("report_analysis_mode")
    if _legacy_mode == "Patient on file":
        st.session_state["report_analysis_mode"] = "Existing patient"
    elif _legacy_mode == "New patient from report":
        st.session_state["report_analysis_mode"] = "New patient"
    elif _legacy_mode not in _mode_options:
        st.session_state["report_analysis_mode"] = "Existing patient"

    mode = st.radio(
        "Save to",
        list(_mode_options),
        horizontal=True,
        key="report_analysis_mode",
        label_visibility="collapsed",
    )

    if mode == "New patient":
        render_new_patient_from_report(df, ollama_ok)
        return

    row = patient_selector_ui(df, widget_key="report_analysis_patient_select", label="Select patient")
    patient_id = str(row["patient_id"])
    render_patient_context_strip(row)

    analysis, report_text = run_report_analysis_workbench(patient_id, ollama_ok)
    if not analysis or not report_text:
        return

    repaired = repair_report_analysis(analysis, report_text)
    render_legacy_report_analysis(repaired, report_text)

    if st.session_state.get(f"report_pending_review_{patient_id}"):
        st.divider()
        render_report_chart_review(patient_id, row, repaired, report_text, df)


def _header_title(nav: str) -> str:
    return "OncoBoard" if nav == "Home" else nav


def render_app_header(nav: str, ollama_ok: bool) -> None:
    title = html.escape(_header_title(nav))
    dot_color = "#b8f5d8" if ollama_ok else "#ffb3b8"
    label = "Ollama online" if ollama_ok else "Ollama offline"

    st.markdown(
        f"""
        <div class="home-hero app-page-header ob-header-shell">
          <div class="ob-header-status-wrap" title="{label}" aria-label="{label}">
            <span class="ob-header-status-dot" style="background:{dot_color};"></span>
          </div>
          <div class="brand-lockup hero-brand app-header-brand">
            <div class="brand-mark">OB</div>
            <div><h1>{title}</h1></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_sidebar_visibility(show: bool) -> None:
    if show:
        return
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] { display: none; }
          button[data-testid="collapsedControl"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_patient_overview(row: pd.Series) -> None:
    st.markdown("#### Case overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stage", str(row.get("stage", "—")))
    c2.metric("Age", str(row.get("age", "—")))
    c3.metric("ECOG", str(row.get("ecog", "—")))
    c4.metric("Sex", str(row.get("sex", "—")))

    left, right = st.columns(2)
    with left:
        st.markdown("**Diagnosis**")
        st.write(str(row.get("diagnosis", "—")))
        st.markdown("**Biomarkers**")
        st.write(str(row.get("biomarkers", "—")))
    with right:
        st.markdown("**Current workup**")
        st.write(str(row.get("imaging", "—")))
        st.markdown("**Pending tests**")
        st.write(str(row.get("pending_tests", "—")))

    narrative = str(row.get("intake_text", "")).strip()
    if narrative and narrative != "nan":
        with st.expander("Clinical narrative", expanded=False):
            st.write(narrative)


def page_patient_chart(df: pd.DataFrame, ollama_ok: bool) -> None:
    hydrate_meeting()
    ensure_board_state()
    row = get_selected_row(df)
    patient_id = str(row["patient_id"])
    patient_data = format_patient_data(row)
    fingerprint = record_fingerprint(patient_data)
    bust_key = f"cache_bust_{patient_id}"

    render_profile_hero(row)

    report_link = load_patient_report_link(patient_id)
    profile_summary = _profile_summary_for_patient(patient_id, row)
    render_mdt_case_summary(
        row,
        profile_summary=profile_summary,
        analysis=report_link.analysis if report_link else None,
        report_text=report_link.report_text if report_link else None,
    )

    on_board = patient_id in st.session_state.get("board_queue", [])
    if st.button(
        "On today's board" if on_board else "Add to today's board",
        disabled=on_board,
        use_container_width=True,
        key=f"patient_add_board_{patient_id}",
    ):
        add_patient_to_board(patient_id)
        st.rerun()

    left, right = st.columns([0.52, 0.48])

    with left:
        render_patient_report_status(patient_id)
        tab_overview, tab_profile = st.tabs(["Overview", "Full profile"])
        with tab_overview:
            render_patient_overview(row)
    with tab_profile:
        render_profile_sections(row, list(row.index))
        if str(row.get("source", "")) in ("custom", "synthetic"):
            if st.button("Remove this patient", type="secondary"):
                try:
                    delete_custom_patient(patient_id)
                    bump_patient_data_version()
                    st.session_state.pop("summary", None)
                    st.session_state.pop("summary_meta", None)
                    st.success(f"Removed {patient_id}.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with right:
        st.markdown("#### MDT brief")
        if not ollama_ok:
            st.warning("Start Ollama to generate briefs.")

        generate = st.button(
            "Generate from profile",
            type="primary",
            disabled=not ollama_ok,
            key="generate_mdt_brief",
            use_container_width=True,
        )

        with st.expander("Brief options", expanded=False):
            regenerate = st.button("Regenerate", disabled=not ollama_ok, key="regenerate_mdt_brief")
            clear_cache = st.button("Reset cache", key="reset_mdt_cache")
        if clear_cache:
            st.session_state[bust_key] = st.session_state.get(bust_key, 0) + 1
            get_cached_summary.clear()
            st.session_state["summary_warmed_keys"] = set()
            st.session_state.pop("summary", None)
            st.session_state.pop("summary_meta", None)
            st.rerun()
        if regenerate:
            st.session_state[bust_key] = st.session_state.get(bust_key, 0) + 1

        if generate or regenerate:
            st.session_state.pop("summary", None)
            st.session_state.pop("summary_meta", None)
            try:
                summary, from_cache, latency = load_summary(
                    patient_id,
                    patient_data,
                    fingerprint,
                    st.session_state.get(bust_key, 0),
                    force_refresh=True,
                )
                st.session_state["summary"] = summary
                st.session_state["summary_meta"] = {
                    "patient_id": patient_id,
                    "fingerprint": fingerprint,
                    "from_cache": from_cache,
                    "latency_sec": latency,
                    "prompt_version": PROMPT_VERSION,
                }
                log_audit_event(
                    "mdt_brief_generated",
                    patient_id=patient_id,
                    prompt_version=PROMPT_VERSION,
                )
            except Exception as exc:
                st.error(f"Summary generation failed: {exc}")

        meta = st.session_state.get("summary_meta", {})
        summary = st.session_state.get("summary")
        if (
            summary
            and meta.get("patient_id") == patient_id
            and meta.get("fingerprint") == fingerprint
            and meta.get("prompt_version") == PROMPT_VERSION
        ):
            display_summary(summary)
            brief_markdown = summary_to_markdown(patient_id, summary)
            brief_pdf, brief_md = st.columns(2)
            with brief_pdf:
                st.download_button(
                    "Export brief (PDF)",
                    data=summary_to_pdf(patient_id, summary),
                    file_name=f"{patient_id}_mdt_brief.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with brief_md:
                st.download_button(
                    "Export brief (Markdown)",
                    data=brief_markdown,
                    file_name=f"{patient_id}_mdt_brief.md",
                    mime="text/markdown",
                    use_container_width=True,
                )


def page_add_patient(df: pd.DataFrame) -> None:
    suggested_id = next_patient_id(df)
    with st.form("add_patient_form", clear_on_submit=False):
        st.markdown("##### Identity & disease")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            patient_id = st.text_input("Patient ID", value=suggested_id)
        with c2:
            age = st.number_input("Age", min_value=0, max_value=120, value=60)
        with c3:
            sex = st.selectbox("Sex", ["F", "M", "U"])
        with c4:
            ecog = st.selectbox("ECOG", [0, 1, 2, 3, 4], index=1)

        c5, c6 = st.columns(2)
        with c5:
            diagnosis = st.text_input("Diagnosis", placeholder="e.g. Lung adenocarcinoma")
        with c6:
            stage = st.text_input("Stage", placeholder="e.g. IIIA")

        st.markdown("##### Structured clinical data")
        biomarkers = st.text_area("Biomarkers", height=68)
        imaging = st.text_area("Imaging", height=68)
        pathology = st.text_area("Pathology", height=68)
        pending_tests = st.text_area("Pending tests", height=68)
        comorbidities = st.text_input("Comorbidities")
        medications = st.text_input("Medications")
        prior_treatment = st.text_input("Prior treatment")
        notes = st.text_area("Care team notes", height=80)

        st.markdown("##### Free-text intake")
        intake_text = st.text_area(
            "Clinical narrative",
            height=140,
            placeholder="Paste or type history, symptoms, referral reason, open questions…",
        )

        submitted = st.form_submit_button("Save patient", type="primary")

    if submitted:
        if not diagnosis.strip() or not stage.strip():
            st.error("Diagnosis and stage are required.")
            return
        record = normalize_record(
            {
                "patient_id": patient_id,
                "age": age,
                "sex": sex,
                "diagnosis": diagnosis,
                "stage": stage,
                "ecog": ecog,
                "biomarkers": biomarkers,
                "imaging": imaging,
                "pathology": pathology,
                "comorbidities": comorbidities,
                "medications": medications,
                "pending_tests": pending_tests,
                "prior_treatment": prior_treatment,
                "notes": notes,
                "intake_text": intake_text,
                "source": "custom",
            },
            source="custom",
        )
        try:
            add_custom_patient(record)
            bump_patient_data_version()
            st.session_state.selected_patient_label = patient_profile_label(
                pd.Series(record)
            )
            st.success(f"Patient {record['patient_id']} saved.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def page_synthetic_intake(df: pd.DataFrame, ollama_ok: bool) -> None:
    provider = os.getenv("SYNTHETIC_GENERATOR_PROVIDER", "ollama").lower()

    if provider != "synthia" and not ollama_ok:
        st.warning("Start Ollama to generate synthetic cases.")
        return

    cancer_hint = st.text_input("Cancer type or focus", placeholder="e.g. Glioblastoma, NSCLC")
    constraints = st.text_area(
        "Constraints (optional)",
        placeholder="e.g. ECOG 2, include pending molecular tests",
        height=60,
    )

    if st.button("Generate synthetic patient", type="primary"):
        with st.spinner("Generating fictional case…"):
            try:
                draft = generate_synthetic_patient(cancer_hint, constraints)
                st.session_state["synthetic_draft"] = draft
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return

    draft = st.session_state.get("synthetic_draft")
    if not draft:
        return

    st.markdown("##### Review draft")
    draft["patient_id"] = next_patient_id(df)

    long_fields = {
        "biomarkers",
        "imaging",
        "pathology",
        "pending_tests",
        "notes",
        "intake_text",
    }

    with st.form("save_synthetic_form"):
        edited: dict = {}
        for col in PATIENT_COLUMNS:
            if col == "source":
                continue
            label = col.replace("_", " ").title()
            raw = draft.get(col, "")
            if col == "age":
                edited[col] = st.number_input(label, min_value=0, max_value=120, value=int(raw or 60))
            elif col == "ecog":
                edited[col] = st.selectbox(label, [0, 1, 2, 3, 4], index=int(raw or 0))
            elif col == "sex":
                s = str(raw or "F")[:1].upper()
                edited[col] = st.selectbox(label, ["F", "M", "U"], index=["F", "M", "U"].index(s) if s in "FMU" else 0)
            elif col in long_fields:
                edited[col] = st.text_area(label, value=str(raw), height=80)
            else:
                edited[col] = st.text_input(label, value=str(raw))

        if st.form_submit_button("Save to patient panel", type="primary"):
            record = normalize_record({**draft, **edited, "source": "synthetic"}, source="synthetic")
            try:
                add_custom_patient(record)
                bump_patient_data_version()
                st.session_state.pop("synthetic_draft", None)
                st.session_state.selected_patient_label = patient_profile_label(pd.Series(record))
                st.success(f"Saved synthetic patient {record['patient_id']}.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def main() -> None:
    st.set_page_config(
        page_title="OncoBoard",
        page_icon="OB",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    init_board_session()

    version = st.session_state.get("patients_version", 0)
    df = get_patients_df(version)
    ollama_ok = check_ollama_reachable()

    nav = render_top_nav()
    render_app_header(nav, ollama_ok)

    assistant_context, assistant_id = _assistant_context(df)
    default_assistant_patient = None
    if assistant_id.startswith(("board:", "patient:")):
        default_assistant_patient = assistant_id.split(":", 1)[1]

    render_ai_assistant_fab(
        df,
        default_patient_id=default_assistant_patient,
        fallback_context=assistant_context,
        ollama_ok=ollama_ok,
    )

    set_sidebar_visibility(show=nav == "Patients")

    with st.sidebar:
        if nav == "Patients":
            sidebar_patient_picker(df)

    if nav == "Home":
        page_home(df)
    elif nav == "Today's board":
        page_todays_board(df, ollama_ok)
    elif nav == "Patients":
        page_patient_chart(df, ollama_ok)
    elif nav == "Report analysis":
        page_report_analysis(df, ollama_ok)
    elif nav == "Add patient":
        page_add_patient(df)
    else:
        page_synthetic_intake(df, ollama_ok)

    render_footer()

    render_ai_assistant_panel(
        df,
        default_patient_id=default_assistant_patient,
        fallback_context=assistant_context,
        ollama_ok=ollama_ok,
    )


if __name__ == "__main__":
    main()
