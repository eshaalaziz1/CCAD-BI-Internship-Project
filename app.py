"""OncoBoard clinical review workspace."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from src.constants import MODEL, PROMPT_VERSION
from src.patients import (
    PATIENT_COLUMNS,
    add_custom_patient,
    delete_custom_patient,
    format_patient_data,
    load_all_patients,
    next_patient_id,
    normalize_record,
    patient_profile_label,
)
from src.summarizer import (
    analyze_report,
    check_ollama_reachable,
    summarize_patient,
)
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


def analyze_report_with_loader(report_text: str) -> dict:
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
        future = executor.submit(analyze_report, report_text)
        tick = 0
        while not future.done():
            base_progress, stage = stages[min(tick // 5, len(stages) - 1)]
            with loader.container():
                render_report_loader(base_progress + min(tick % 5, 4), stage)
            time.sleep(0.18)
            tick += 1
        analysis = future.result()

    with loader.container():
        render_report_loader(100, "Board view ready")
    time.sleep(0.25)
    loader.empty()
    return analysis


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


def get_selected_row(df: pd.DataFrame) -> pd.Series:
    labels = [patient_profile_label(row) for _, row in df.iterrows()]
    if "selected_patient_label" not in st.session_state:
        st.session_state.selected_patient_label = labels[0]
    if st.session_state.selected_patient_label not in labels:
        st.session_state.selected_patient_label = labels[0]
    idx = labels.index(st.session_state.selected_patient_label)
    return df.iloc[idx]


def sidebar_patient_picker(df: pd.DataFrame) -> pd.Series:
    st.markdown("##### Patient panel")
    search = st.text_input("Search", placeholder="ID, diagnosis, stage…", label_visibility="collapsed")
    filtered = df.copy()
    if search.strip():
        mask = filtered.apply(
            lambda r: search.lower()
            in " ".join(str(r[c]) for c in ["patient_id", "diagnosis", "stage"]).lower(),
            axis=1,
        )
        filtered = filtered[mask]

    if filtered.empty:
        st.caption("No patients match your search.")
        return df.iloc[0]

    labels = [patient_profile_label(row) for _, row in filtered.iterrows()]
    current = st.session_state.get("selected_patient_label")
    if current not in labels:
        current = labels[0]
        st.session_state.selected_patient_label = current
    idx = labels.index(current)

    p_col, n_col = st.columns(2)
    with p_col:
        if st.button("Prev", disabled=idx <= 0, key="patient_prev"):
            st.session_state.selected_patient_label = labels[idx - 1]
            st.rerun()
    with n_col:
        if st.button("Next", disabled=idx >= len(labels) - 1, key="patient_next"):
            st.session_state.selected_patient_label = labels[idx + 1]
            st.rerun()

    choice = st.radio(
        "Patients",
        labels,
        index=idx,
        label_visibility="collapsed",
        key="selected_patient_label",
    )
    return filtered.iloc[labels.index(choice)]


def render_top_nav() -> str:
    nav_options = ["Home", "Patients", "Report intake", "Add patient", "Synthetic intake"]
    st.session_state.setdefault("workspace_nav", nav_options[0])
    active = st.session_state["workspace_nav"]

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


def page_home(df: pd.DataFrame) -> None:
    custom_count = int((df.get("source", "") == "custom").sum()) if "source" in df else 0
    synthetic_count = int((df.get("source", "") == "synthetic").sum()) if "source" in df else 0
    reference_count = max(len(df) - custom_count - synthetic_count, 0)

    st.markdown(
        """
        <div class="home-hero">
          <div class="brand-lockup hero-brand">
            <div class="brand-mark">OB</div>
            <div>
              <h1>OncoBoard</h1>
              <p>Clinical review workspace for synthetic oncology cases, MDT briefs, and patient intake.</p>
            </div>
          </div>
          <div class="workflow-strip">
            <div class="workflow-step"><strong>Ingest</strong><span>Upload a long report and extract the decision-critical signal.</span></div>
            <div class="workflow-step"><strong>Review</strong><span>Scan patient context, problems, gaps, and specialty questions.</span></div>
            <div class="workflow-step"><strong>Decide</strong><span>Use an MDT-ready agenda to move through cases faster.</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        df[available_cols].head(10),
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
    if not items:
        st.caption(empty_text)
        return
    for item in items:
        st.markdown(f"- {item}")


def render_report_analysis(analysis: dict, report_text: str) -> None:
    snapshot = analysis.get("patient_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {"summary": str(snapshot)}

    st.markdown("### Board briefing")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", str(snapshot.get("age", "N/A")))
    c2.metric("Sex", str(snapshot.get("sex", "N/A")))
    c3.metric("Pages read", str(report_text.count("Page ")))
    c4.metric("Problems", str(len(_as_list(analysis.get("priority_problems")))))

    st.markdown(
        f"""
        <article class="summary-panel">
          <div class="summary-header">
            <h3>{str(snapshot.get("name_or_id", "Uploaded report"))}</h3>
            <span>Board view</span>
          </div>
          <section class="summary-section">
            <h4>Presenting problem</h4>
            <p>{str(snapshot.get("presenting_problem", "Not specified in the report."))}</p>
          </section>
          <section class="summary-section">
            <h4>Likely primary issue</h4>
            <p>{str(snapshot.get("likely_primary_issue", "Not specified in the report."))}</p>
          </section>
          <section class="summary-section">
            <h4>Meeting objective</h4>
            <p>{str(analysis.get("meeting_objective", "Clarify the key clinical decision for this case."))}</p>
          </section>
        </article>
        """,
        unsafe_allow_html=True,
    )

    tab_signal, tab_charts, tab_team, tab_agenda, tab_source = st.tabs(
        ["Clinical signal", "Visual insights", "Team focus", "Meeting agenda", "Source text"]
    )

    with tab_signal:
        left, right = st.columns([0.52, 0.48])
        with left:
            st.markdown("#### Critical facts")
            render_list_items(analysis.get("critical_facts"))
            st.markdown("#### Red flags")
            render_list_items(analysis.get("red_flags"), "No urgent red flags were extracted.")
        with right:
            st.markdown("#### Missing data")
            render_list_items(analysis.get("missing_data"), "No missing data was extracted.")

        problems = _as_list(analysis.get("priority_problems"))
        rows = [p for p in problems if isinstance(p, dict)]
        if rows:
            st.markdown("#### Priority problem table")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("#### Priority problems")
            render_list_items(problems, "No priority problems were extracted.")

    with tab_charts:
        render_report_charts(analysis, report_text)

    with tab_team:
        focus = _as_list(analysis.get("specialist_focus"))
        rows = [item for item in focus if isinstance(item, dict)]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            render_list_items(focus, "No specialist focus items were extracted.")

        st.markdown("#### Decision points")
        render_list_items(analysis.get("decision_points"))

    with tab_agenda:
        st.markdown("#### Suggested discussion order")
        flow = [str(item).strip() for item in _as_list(analysis.get("meeting_flow")) if str(item).strip()]
        if flow:
            for idx, item in enumerate(flow, start=1):
                st.markdown(f"**{idx}.** {item}")
        else:
            st.caption("No meeting flow was extracted.")

    with tab_source:
        st.text_area("Extracted report text", value=report_text[:30000], height=360)


def page_report_intake(ollama_ok: bool) -> None:
    st.subheader("Report intake")
    st.caption("Upload a clinical PDF and convert it into a meeting-ready board briefing. MedGemma analyzes each section separately to reduce cutoff and improve completeness.")

    if not ollama_ok:
        st.warning("Start Ollama locally to enable report analysis.")

    uploaded = st.file_uploader("Upload report PDF", type=["pdf"])
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

    report_fingerprint = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
    if st.session_state.get("report_fingerprint") != report_fingerprint:
        st.session_state.pop("report_analysis", None)
        st.session_state.pop("report_text", None)
        st.session_state["report_fingerprint"] = report_fingerprint

    st.success(f"Extracted {len(report_text):,} characters from {uploaded.name}.")

    button_label = "Regenerate board briefing" if st.session_state.get("report_analysis") else "Generate board briefing"
    if st.button(button_label, type="primary", disabled=not ollama_ok):
        try:
            st.session_state["report_analysis"] = analyze_report_with_loader(report_text)
            st.session_state["report_text"] = report_text
            st.session_state["report_name"] = uploaded.name
            st.session_state["report_fingerprint"] = report_fingerprint
        except Exception as exc:
            st.error(f"Report analysis failed: {exc}")

    if st.session_state.get("report_analysis") and st.session_state.get("report_text"):
        render_report_analysis(st.session_state["report_analysis"], st.session_state["report_text"])


def render_ollama_status_light(ollama_ok: bool) -> None:
    color = "#1f8a83" if ollama_ok else "#b45c63"
    label = "Ollama online" if ollama_ok else "Ollama offline"
    st.markdown(
        (
            "<div style='display:flex; justify-content:flex-end; padding-top:0.65rem;'>"
            f"<div class='status-pill status-dot-only' title='{label}' aria-label='{label}'>"
            f"<span class='status-dot' style='background:{color};'></span>"
            "</div>"
            "</div>"
        ),
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
    row = get_selected_row(df)
    patient_id = str(row["patient_id"])
    patient_data = format_patient_data(row)
    fingerprint = record_fingerprint(patient_data)
    bust_key = f"cache_bust_{patient_id}"

    left, right = st.columns([0.52, 0.48])

    with left:
        render_profile_hero(row)
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
        st.subheader("MDT summary")
        if not ollama_ok:
            st.warning("AI summaries unavailable — start Ollama locally to enable generation.")

        c1, c2, c3 = st.columns(3)
        with c1:
            generate = st.button(
                "Generate MDT brief",
                type="primary",
                disabled=not ollama_ok,
                key="generate_mdt_brief",
            )
        with c2:
            regenerate = st.button("Regenerate", disabled=not ollama_ok, key="regenerate_mdt_brief")
        with c3:
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
            if meta.get("from_cache"):
                st.caption("Cached brief for this record.")
            elif meta.get("latency_sec"):
                st.caption(f"Generated in {meta['latency_sec']:.1f}s")
            display_summary(summary)
            st.download_button(
                "Export brief (Markdown)",
                data=summary_to_markdown(patient_id, summary),
                file_name=f"{patient_id}_mdt_brief.md",
                mime="text/markdown",
            )
        else:
            st.info("Generate an MDT brief from this patient's profile and clinical text.")


def page_add_patient(df: pd.DataFrame) -> None:
    st.subheader("Register patient")
    st.caption("Add a synthetic case manually. Imaging upload will be supported in a future release.")

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
        imaging = st.text_area("Imaging", height=68, placeholder="Future: link to imaging studies")
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
    st.subheader("Synthetic case generator")
    st.caption(f"Provider: {provider.upper()} · Review and edit before saving.")

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
        st.info("Describe a cancer focus and generate a draft case.")
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

    version = st.session_state.get("patients_version", 0)
    df = get_patients_df(version)
    ollama_ok = check_ollama_reachable()

    # Top bar with page sections first, then title and compact status light.
    nav = render_top_nav()
    c_brand, c_status = st.columns([1.8, 0.6])
    with c_brand:
        st.markdown(
            """
            <div class="app-titlebar">
              <div class="brand-lockup">
                <div class="brand-mark">OB</div>
                <div>
                  <div class="product-kicker">Local MedGemma MDT workspace</div>
                  <h2>OncoBoard</h2>
                  <p>Multidisciplinary oncology review workspace</p>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_status:
        render_ollama_status_light(ollama_ok)

    set_sidebar_visibility(show=nav == "Patients")

    with st.sidebar:
        if nav == "Patients":
            sidebar_patient_picker(df)
            st.divider()
            st.caption("Use Prev/Next for fast case switching.")

    if nav == "Home":
        page_home(df)
    elif nav == "Patients":
        page_patient_chart(df, ollama_ok)
    elif nav == "Report intake":
        page_report_intake(ollama_ok)
    elif nav == "Add patient":
        page_add_patient(df)
    else:
        page_synthetic_intake(df, ollama_ok)

    render_footer()


if __name__ == "__main__":
    main()
