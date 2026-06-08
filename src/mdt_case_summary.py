"""Board-ready one-page MDT case summary."""

from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from src.summarizer import repair_report_analysis


def _chart_value(row: pd.Series, field: str) -> tuple[str, bool]:
    raw = row.get(field, "")
    text = str(raw).strip() if raw is not None else ""
    if not text or text.lower() == "nan":
        return "Not found in chart", False
    return text, True


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _infer_confidence(found_in_chart: bool, has_pdf: bool, has_evidence: bool) -> str:
    if has_evidence and (found_in_chart or has_pdf):
        return "Moderate"
    if found_in_chart or has_pdf:
        return "Low–moderate"
    return "Uncertain"


def render_evidence_card(
    *,
    recommendation: str,
    reasoning: str = "",
    source: str = "Patient chart",
    confidence: str = "Moderate",
    missing: str = "",
) -> None:
    missing_html = (
        f'<p class="evidence-missing"><b>Before deciding:</b> {html.escape(missing)}</p>'
        if missing.strip()
        else ""
    )
    reasoning_html = (
        f'<p class="evidence-reasoning">{html.escape(reasoning)}</p>' if reasoning.strip() else ""
    )
    st.markdown(
        f"""
        <article class="evidence-card">
          <div class="evidence-card-head">
            <span class="evidence-confidence">{html.escape(confidence)}</span>
            <span class="evidence-source">{html.escape(source)}</span>
          </div>
          <p class="evidence-rec"><b>Consideration</b> — {html.escape(recommendation)}</p>
          {reasoning_html}
          {missing_html}
        </article>
        """,
        unsafe_allow_html=True,
    )


def _evidence_cards_from_analysis(analysis: dict, report_text: str, row: pd.Series) -> None:
    analysis = repair_report_analysis(analysis, report_text)
    missing_items = [
        str(x).strip() for x in _as_list(analysis.get("missing_data")) if str(x).strip()
    ]
    missing_joined = "; ".join(missing_items[:3])

    problems = [p for p in _as_list(analysis.get("priority_problems")) if isinstance(p, dict)]
    for problem in problems[:4]:
        rec = str(problem.get("problem", "")).strip()
        evidence = str(problem.get("evidence", "")).strip()
        why = str(problem.get("why_it_matters", "")).strip()
        if not rec:
            continue
        render_evidence_card(
            recommendation=rec,
            reasoning=why or evidence,
            source="PDF report" if evidence else "PDF briefing (no cited snippet)",
            confidence="Moderate" if evidence else "Low",
            missing=missing_joined,
        )

    for point in _as_list(analysis.get("decision_points"))[:3]:
        text = str(point).strip()
        if not text:
            continue
        render_evidence_card(
            recommendation=text,
            reasoning="Open MDT question extracted from uploaded report.",
            source="PDF report",
            confidence=_infer_confidence(False, True, True),
            missing=missing_joined,
        )

    for item in _as_list(analysis.get("treatment_considerations"))[:3]:
        text = str(item).strip()
        if not text:
            continue
        render_evidence_card(
            recommendation=text,
            reasoning="Possibility only — requires clinician verification.",
            source="PDF briefing",
            confidence="Low–moderate",
            missing=missing_joined or "Confirm staging, fitness, and molecular results.",
        )


def _profile_summary_sections(summary: str) -> dict[str, str]:
    if not summary or not summary.strip():
        return {}
    sections: dict[str, str] = {}
    current = ""
    lines = summary.splitlines()
    for line in lines:
        match = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line.strip())
        if match:
            current = match.group(1).strip().lower()
            sections[current] = ""
            continue
        if current:
            sections[current] += line + "\n"
    return sections


def render_mdt_case_summary(
    row: pd.Series,
    *,
    profile_summary: str | None = None,
    analysis: dict | None = None,
    report_text: str | None = None,
    show_evidence: bool = True,
) -> None:
    """One-page board-ready view: chart facts, AI discussion points, evidence cards."""
    st.markdown('<div class="mdt-case-summary">', unsafe_allow_html=True)

    diagnosis, _ = _chart_value(row, "diagnosis")
    stage, _ = _chart_value(row, "stage")
    ecog, _ = _chart_value(row, "ecog")
    comorbidities, comorb_ok = _chart_value(row, "comorbidities")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diagnosis", diagnosis[:48] + ("…" if len(diagnosis) > 48 else ""))
    c2.metric("Stage", stage[:24])
    c3.metric("ECOG", ecog[:8])
    c4.metric("Comorbidities", "On chart" if comorb_ok else "Not found")

    left, right = st.columns(2)
    with left:
        st.markdown("**Recent imaging**")
        imaging, imaging_ok = _chart_value(row, "imaging")
        st.markdown(
            f'<span class="{"chart-hit" if imaging_ok else "chart-miss"}">{html.escape(imaging)}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("**Pathology & biomarkers**")
        pathology, path_ok = _chart_value(row, "pathology")
        biomarkers, bio_ok = _chart_value(row, "biomarkers")
        path_line = pathology if path_ok else "Not found in chart"
        bio_line = biomarkers if bio_ok else "Not found in chart"
        st.markdown(
            f'<span class="{"chart-hit" if path_ok else "chart-miss"}">{html.escape(path_line)}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="{"chart-hit" if bio_ok else "chart-miss"}">Biomarkers: {html.escape(bio_line)}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("**Prior treatment**")
        prior, prior_ok = _chart_value(row, "prior_treatment")
        st.markdown(
            f'<span class="{"chart-hit" if prior_ok else "chart-miss"}">{html.escape(prior)}</span>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("**Open clinical questions**")
        pending, pending_ok = _chart_value(row, "pending_tests")
        notes, notes_ok = _chart_value(row, "notes")
        if pending_ok:
            st.markdown(f"- {html.escape(pending)}")
        if notes_ok and "?" in notes:
            st.markdown(f"- {html.escape(notes[:280])}")
        if not pending_ok and not (notes_ok and "?" in notes):
            st.markdown('<span class="chart-miss">Not found in chart</span>', unsafe_allow_html=True)

        if profile_summary:
            sections = _profile_summary_sections(profile_summary)
            questions = (
                sections.get("mdt discussion questions", "")
                or sections.get("discussion questions", "")
            )
            treatment = sections.get("treatment considerations", "")
            missing = sections.get("missing or unclear information", "")
            if questions.strip():
                st.markdown("**MDT discussion points (profile brief)**")
                st.markdown(questions.strip())
            if treatment.strip():
                st.markdown("**Possible options (profile brief)**")
                st.markdown(treatment.strip())
            if missing.strip():
                st.markdown("**Missing / unclear (profile brief)**")
                st.markdown(missing.strip())
        elif analysis and report_text:
            repaired = repair_report_analysis(analysis, report_text)
            st.markdown("**MDT discussion points (PDF)**")
            for item in _as_list(repaired.get("decision_points"))[:5]:
                text = str(item).strip()
                if text:
                    st.markdown(f"- {html.escape(text)}")
            flags = [str(x).strip() for x in _as_list(repaired.get("red_flags")) if str(x).strip()]
            if flags:
                st.markdown("**Red flags**")
                for flag in flags[:4]:
                    st.markdown(f"- {html.escape(flag)}")

    if show_evidence and analysis and report_text:
        st.markdown("**Evidence-aware AI considerations**")
        _evidence_cards_from_analysis(analysis, report_text, row)

    st.markdown("</div>", unsafe_allow_html=True)
