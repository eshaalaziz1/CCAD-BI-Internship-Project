"""Shared UI helpers and styling."""

from __future__ import annotations

import streamlit as st

from src.constants import MODEL

SECTION_ORDER = [
    "One-line case summary",
    "Key clinical facts",
    "Missing or unclear information",
    "MDT discussion questions",
    "Treatment considerations",
]

PROFILE_SECTIONS = {
    "Demographics": ["patient_id", "age", "sex", "ecog"],
    "Disease": ["diagnosis", "stage", "biomarkers", "pathology"],
    "Workup": ["imaging", "pending_tests"],
    "Care context": ["comorbidities", "medications", "prior_treatment", "notes"],
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; max-width: 1200px; }
          .profile-hero {
            background: linear-gradient(135deg, #0f3d5c 0%, #1a5f8f 100%);
            color: #fff;
            padding: 1.25rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
          }
          .profile-hero h1 { color: #fff; font-size: 1.5rem; margin: 0 0 0.35rem 0; }
          .profile-hero p { margin: 0; opacity: 0.92; font-size: 0.95rem; }
          .badge {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-right: 0.35rem;
          }
          .info-card {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
          }
          .info-card h4 {
            margin: 0 0 0.5rem 0;
            color: #0f3d5c;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
          }
          .info-row { margin: 0.25rem 0; font-size: 0.92rem; }
          .info-label { color: #64748b; font-weight: 600; }
          div[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
          }
          .platform-footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.75rem;
            padding: 1.5rem 0 0.5rem;
            border-top: 1px solid #e2e8f0;
            margin-top: 2rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f'<div class="platform-footer">MedGemma 1.5 4B · Synthetic training data only · Not for clinical use</div>',
        unsafe_allow_html=True,
    )


def render_profile_hero(row) -> None:
    source = str(row.get("source", "reference")).replace("_", " ").title()
    st.markdown(
        f"""
        <div class="profile-hero">
          <h1>{row['patient_id']} — {row['diagnosis']}</h1>
          <p>
            <span class="badge">Stage {row['stage']}</span>
            <span class="badge">{row['age']} yrs · {row['sex']}</span>
            <span class="badge">ECOG {row['ecog']}</span>
            <span class="badge">{source}</span>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_value(value) -> str:
    if value is None or (isinstance(value, float) and str(value) == "nan"):
        return "—"
    text = str(value).strip()
    return text if text else "—"


def render_profile_sections(row, columns: list[str]) -> None:
    cols = st.columns(2)
    sections = list(PROFILE_SECTIONS.items())
    for idx, (title, fields) in enumerate(sections):
        with cols[idx % 2]:
            st.markdown(f'<div class="info-card"><h4>{title}</h4>', unsafe_allow_html=True)
            for field in fields:
                if field not in columns:
                    continue
                label = field.replace("_", " ").title()
                st.markdown(
                    f'<div class="info-row"><span class="info-label">{label}:</span> '
                    f"{_display_value(row.get(field))}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    intake = row.get("intake_text", "")
    if intake and str(intake).strip() and str(intake) != "nan":
        st.markdown('<div class="info-card"><h4>Clinical narrative</h4>', unsafe_allow_html=True)
        st.markdown(str(intake))
        st.markdown("</div>", unsafe_allow_html=True)


def parse_summary_sections(text: str) -> dict[str, str]:
    import re

    sections: dict[str, str] = {}
    for block in re.split(r"\n(?=\d+\.\s+)", text.strip()):
        match = re.match(r"^\d+\.\s+([^\n]+)\n?(.*)", block.strip(), re.DOTALL)
        if match:
            sections[match.group(1).strip()] = match.group(2).strip()
    return sections


def display_summary(text: str) -> None:
    sections = parse_summary_sections(text)
    if not sections:
        st.markdown(text)
        return
    for title in SECTION_ORDER:
        if sections.get(title):
            st.markdown(f"#### {title}")
            st.markdown(sections[title])
    for title in sections:
        if title not in SECTION_ORDER:
            st.markdown(f"#### {title}")
            st.markdown(sections[title])
