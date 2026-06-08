"""Baseline patient-chart insights (structured record, before any PDF)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.patients import format_patient_data
from src.report_charts import (
    _numeric_measurements,
    chart_history_durations,
    chart_timeline,
    chart_vitals_and_labs,
    render_measurement_cards,
)


def patient_chart_text(row: pd.Series) -> str:
    """Build a text corpus from the on-file chart for numeric extraction."""
    parts = [format_patient_data(row)]
    intake = str(row.get("intake_text", "")).strip()
    if intake and intake != "nan":
        parts.append(intake)
    return "\n\n".join(parts)


def render_profile_visual_insights(
    row: pd.Series,
    *,
    label: str = "Patient chart (on file)",
    chart_key_prefix: str | None = None,
) -> bool:
    """Charts and measurement cards from structured profile + narrative."""
    st.markdown(f"#### {label}")
    text = patient_chart_text(row)
    patient_id = str(row.get("patient_id", "patient"))
    prefix = chart_key_prefix or f"profile_{patient_id}"
    numeric_df = _numeric_measurements(text)
    if numeric_df.empty:
        timeline = chart_timeline(text)
        history = chart_history_durations(text)
        if timeline is not None:
            st.plotly_chart(timeline, use_container_width=True, key=f"{prefix}_timeline")
            return True
        if history is not None:
            st.plotly_chart(history, use_container_width=True, key=f"{prefix}_history")
            return True
        st.caption("No chartable vitals, labs, or timelines in the patient record yet.")
        return False

    vitals_chart = chart_vitals_and_labs(text)
    if vitals_chart is not None:
        render_measurement_cards(numeric_df)
        st.plotly_chart(vitals_chart, use_container_width=True, key=f"{prefix}_vitals")
    timeline = chart_timeline(text)
    if timeline is not None:
        st.plotly_chart(timeline, use_container_width=True, key=f"{prefix}_timeline")
    history = chart_history_durations(text)
    if history is not None:
        st.plotly_chart(history, use_container_width=True, key=f"{prefix}_history")
    return True


def render_profile_clinical_summary(row: pd.Series) -> None:
    """Compact clinical summary from the structured patient record."""
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
        st.markdown("**Pathology**")
        st.write(str(row.get("pathology", "—")))
    with right:
        st.markdown("**Imaging**")
        st.write(str(row.get("imaging", "—")))
        st.markdown("**Pending tests**")
        st.write(str(row.get("pending_tests", "—")))
        st.markdown("**Prior treatment**")
        st.write(str(row.get("prior_treatment", "—")))

    narrative = str(row.get("intake_text", "")).strip()
    if narrative and narrative != "nan":
        with st.expander("Chart narrative", expanded=False):
            st.write(narrative)
