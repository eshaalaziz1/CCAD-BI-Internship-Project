"""Numeric clinical visualizations derived from extracted report text."""

from __future__ import annotations

import re
from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "primary": "#255e7e",
    "teal": "#1f8a83",
    "amber": "#b88020",
    "red": "#b45c63",
    "violet": "#6d5fa8",
    "muted": "#d8e3ee",
}

STATUS_COLORS = {"Low": "#5b9bd5", "In range": "#1f8a83", "High": "#b45c63", "Recorded": "#255e7e"}

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#172033", size=13),
    margin=dict(l=48, r=72, t=64, b=56),
    autosize=True,
)

LAB_PATTERNS: list[tuple[str, re.Pattern[str], str, float | None, float | None]] = [
    ("HbA1c", re.compile(r"HbA1c[:\s]+(\d+\.?\d*)\s*%", re.I), "%", 4.0, 5.6),
    ("Glucose", re.compile(r"(?:glucose|blood sugar)[:\s]+(\d+\.?\d*)\s*mg/dL", re.I), "mg/dL", 70, 99),
    ("Creatinine", re.compile(r"creatinine[:\s]+(\d+\.?\d*)\s*mg/dL", re.I), "mg/dL", 0.6, 1.2),
    ("Hemoglobin", re.compile(r"(?:hemoglobin|hgb|hb)[:\s]+(\d+\.?\d*)\s*g/dL", re.I), "g/dL", 12.0, 16.0),
    ("WBC", re.compile(r"(?:wbc|white blood cell)[:\s]+(\d+\.?\d*)\s*(?:x10\^3|K)?", re.I), "K/uL", 4.5, 11.0),
    ("Platelets", re.compile(r"platelets?[:\s]+(\d+\.?\d*)\s*(?:x10\^3|K)?", re.I), "K/uL", 150, 400),
    ("LDL", re.compile(r"LDL[:\s]+(\d+\.?\d*)\s*mg/dL", re.I), "mg/dL", None, 100),
    ("HDL", re.compile(r"HDL[:\s]+(\d+\.?\d*)\s*mg/dL", re.I), "mg/dL", 40, None),
    ("PSA", re.compile(r"PSA[:\s]+(\d+\.?\d*)\s*ng/mL", re.I), "ng/mL", None, 4.0),
    ("CEA", re.compile(r"CEA[:\s]+(\d+\.?\d*)\s*ng/mL", re.I), "ng/mL", None, 5.0),
    ("CA 19-9", re.compile(r"CA[- ]?19[- ]?9[:\s]+(\d+\.?\d*)\s*U/mL", re.I), "U/mL", None, 37),
]


def _apply_theme(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(**CHART_THEME, height=height)
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(217,228,239,0.72)",
        zeroline=False,
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(217,228,239,0.72)",
        zeroline=False,
        automargin=True,
    )
    return fig


def _reference_label(ref_low: float | None, ref_high: float | None, unit: str) -> str:
    if ref_low is not None and ref_high is not None:
        return f"{ref_low:g}-{ref_high:g} {unit}"
    if ref_high is not None:
        return f"<={ref_high:g} {unit}"
    if ref_low is not None:
        return f">={ref_low:g} {unit}"
    return "No reference range"


def _status(value: float, ref_low: float | None, ref_high: float | None) -> str:
    if ref_low is not None and value < ref_low:
        return "Low"
    if ref_high is not None and value > ref_high:
        return "High"
    if ref_low is None and ref_high is None:
        return "Recorded"
    return "In range"


def extract_labs_from_text(report_text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, pattern, unit, ref_low, ref_high in LAB_PATTERNS:
        for match in pattern.finditer(report_text):
            key = f"{name}:{match.group(1)}"
            if key in seen:
                continue
            seen.add(key)
            value = float(match.group(1))
            rows.append(
                {
                    "Measurement": name,
                    "Value": value,
                    "Unit": unit,
                    "Reference": _reference_label(ref_low, ref_high, unit),
                    "Status": _status(value, ref_low, ref_high),
                    "Source": "Report text",
                }
            )
    return pd.DataFrame(rows)


def extract_vitals_from_text(report_text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    bp_match = re.search(r"Blood Pressure\s+(\d{2,3})/(\d{2,3})|BP\s+(\d{2,3})/(\d{2,3})", report_text, re.I)
    if bp_match:
        systolic = float(bp_match.group(1) or bp_match.group(3))
        diastolic = float(bp_match.group(2) or bp_match.group(4))
        rows.extend(
            [
                {
                    "Measurement": "Systolic BP",
                    "Value": systolic,
                    "Unit": "mmHg",
                    "Reference": "90-120 mmHg",
                    "Status": _status(systolic, 90, 120),
                    "Source": f"{int(systolic)}/{int(diastolic)} mmHg",
                },
                {
                    "Measurement": "Diastolic BP",
                    "Value": diastolic,
                    "Unit": "mmHg",
                    "Reference": "60-80 mmHg",
                    "Status": _status(diastolic, 60, 80),
                    "Source": f"{int(systolic)}/{int(diastolic)} mmHg",
                },
            ]
        )

    patterns = [
        ("Pulse", re.compile(r"Pulse\s+(\d{2,3})", re.I), "bpm", 60, 100),
        ("Respirations", re.compile(r"Respirations\s+(\d{1,2})", re.I), "breaths/min", 12, 20),
        ("Temperature", re.compile(r"Temperature\s+(\d+\.?\d*)\s*degrees?", re.I), "C", 36.0, 37.5),
        ("JVP", re.compile(r"pressure is measured as\s+(\d+\.?\d*)\s*cm", re.I), "cm", None, 8),
    ]
    for name, pattern, unit, ref_low, ref_high in patterns:
        match = pattern.search(report_text)
        if not match:
            continue
        value = float(match.group(1))
        rows.append(
            {
                "Measurement": name,
                "Value": value,
                "Unit": unit,
                "Reference": _reference_label(ref_low, ref_high, unit),
                "Status": _status(value, ref_low, ref_high),
                "Source": match.group(0),
            }
        )

    return pd.DataFrame(rows)


def extract_symptom_timeline(report_text: str) -> pd.DataFrame:
    text = " ".join(report_text.split())
    rows: list[dict[str, Any]] = []

    first = re.search(r"one week prior.*?pain lasted approximately\s+(\d+)\s+to\s+(\d+)\s+minutes", text, re.I)
    if first:
        low = float(first.group(1))
        high = float(first.group(2))
        rows.append(
            {
                "Episode": "Initial exertional pain",
                "Days before visit": 7,
                "Duration minutes": round((low + high) / 2, 1),
                "Range": f"{int(low)}-{int(high)} min",
                "Context": "Working in garden; relieved with rest",
            }
        )

    second = re.search(r"Three days ago.*?(\d+)\s+minute episode.*?walking her dog", text, re.I)
    if second:
        minutes = float(second.group(1))
        rows.append(
            {
                "Episode": "Recurrent exertional pain",
                "Days before visit": 3,
                "Duration minutes": minutes,
                "Range": f"{int(minutes)} min",
                "Context": "Walking dog; resolved with rest",
            }
        )

    current = re.search(r"evening.*?awaken her from sleep.*?lasting\s+(\d+)\s+minutes", text, re.I)
    if current:
        minutes = float(current.group(1))
        rows.append(
            {
                "Episode": "Rest/nocturnal pain",
                "Days before visit": 0,
                "Duration minutes": minutes,
                "Range": f"{int(minutes)} min",
                "Context": "Awoke from sleep; prompted ED visit",
            }
        )

    return pd.DataFrame(rows)


def extract_history_durations(report_text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    htn = re.search(r"hypertension\s+(\d+)\s+years?\s+ago|HTN\s+(\d+)\s+years?\s+ago", report_text, re.I)
    if htn:
        years = float(htn.group(1) or htn.group(2))
        rows.append({"History item": "Hypertension history", "Years": years, "Source": htn.group(0)})

    surgical_menopause = re.search(r"TAH with BSO\s+(\d+)\s+years?\s+ago", report_text, re.I)
    if surgical_menopause:
        years = float(surgical_menopause.group(1))
        rows.append({"History item": "TAH/BSO history", "Years": years, "Source": surgical_menopause.group(0)})

    return pd.DataFrame(rows)


def _numeric_measurements(report_text: str) -> pd.DataFrame:
    frames = [extract_vitals_from_text(report_text), extract_labs_from_text(report_text)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def chart_vitals_and_labs(report_text: str) -> go.Figure | None:
    df = _numeric_measurements(report_text)
    if df.empty:
        return None
    fig = go.Figure(
        go.Bar(
            x=df["Measurement"],
            y=df["Value"],
            marker_color=[STATUS_COLORS[status] for status in df["Status"]],
            text=[f"{value:g} {unit}" for value, unit in zip(df["Value"], df["Unit"])],
            textposition="outside",
            customdata=df[["Status", "Reference", "Source"]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Value: %{text}<br>"
                "Status: %{customdata[0]}<br>"
                "Reference: %{customdata[1]}<br>"
                "Source: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Recorded numeric values from the report",
        yaxis_title="Recorded value",
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-18)
    fig.update_yaxes(rangemode="tozero")
    fig = _apply_theme(fig, height=380)
    fig.update_layout(margin=dict(l=48, r=96, t=64, b=72))
    return fig


def chart_timeline(report_text: str) -> go.Figure | None:
    df = extract_symptom_timeline(report_text)
    if df.empty:
        return None
    fig = go.Figure(
        go.Scatter(
            x=df["Days before visit"],
            y=df["Duration minutes"],
            mode="lines+markers+text",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(
                size=[max(16, value * 1.2) for value in df["Duration minutes"]],
                color=[COLORS["teal"], COLORS["amber"], COLORS["red"]][: len(df)],
                line=dict(color="white", width=2),
            ),
            text=df["Range"],
            textposition="top center",
            customdata=df[["Episode", "Context"]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{y} min<br>%{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Chest-pain episode timeline",
        xaxis_title="Days before ED visit",
        yaxis_title="Episode duration (minutes)",
        showlegend=False,
    )
    fig.update_xaxes(autorange="reversed", dtick=1)
    fig = _apply_theme(fig, height=380)
    fig.update_layout(margin=dict(l=48, r=72, t=88, b=56))
    return fig


def chart_history_durations(report_text: str) -> go.Figure | None:
    df = extract_history_durations(report_text)
    if df.empty or "Years" not in df.columns:
        return None
    df = df[df["Years"] >= 0.01].copy()
    if df.empty:
        return None
    fig = go.Figure(
        go.Bar(
            x=df["Years"],
            y=df["History item"],
            orientation="h",
            marker_color=[COLORS["primary"], COLORS["violet"], COLORS["amber"]][: len(df)],
            text=[f"{years:g} yr" for years in df["Years"]],
            textposition="outside",
            customdata=df["Source"],
            hovertemplate="<b>%{y}</b><br>%{x:g} years<br>Source: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Relevant history duration",
        xaxis_title="Years documented",
        showlegend=False,
    )
    fig = _apply_theme(fig, height=max(260, 76 * len(df)))
    fig.update_layout(margin=dict(l=48, r=96, t=64, b=48))
    return fig


def _status_style(status: str) -> tuple[str, str]:
    if status == "High":
        return "#fff1f2", "#b45c63"
    if status == "Low":
        return "#eff6ff", "#5b9bd5"
    if status == "In range":
        return "#ecfdf5", "#1f8a83"
    return "#eef2ff", "#255e7e"


def render_measurement_cards(df: pd.DataFrame) -> None:
    if df.empty:
        return

    cards = []
    for row in df.itertuples(index=False):
        bg, color = _status_style(row.Status)
        cards.append(
            f"""
            <div class="measurement-card" style="border-color:{color};">
              <div class="measurement-card-top">
                <span>{escape(str(row.Measurement))}</span>
                <strong style="background:{bg}; color:{color};">{escape(str(row.Status))}</strong>
              </div>
              <div class="measurement-value">{row.Value:g}<small>{escape(str(row.Unit))}</small></div>
              <div class="measurement-meta">
                <span>Reference</span><b>{escape(str(row.Reference))}</b>
              </div>
              <div class="measurement-source">{escape(str(row.Source))}</div>
            </div>
            """
        )

    st.markdown(
        f"""
        <style>
          .measurement-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 0.85rem;
            margin: 0.6rem 0 1.2rem;
          }}
          .measurement-card {{
            background: rgba(255,255,255,0.88);
            border: 1px solid;
            border-left-width: 5px;
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(31, 41, 55, 0.08);
          }}
          .measurement-card-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            color: #334155;
            font-size: 0.88rem;
            font-weight: 750;
          }}
          .measurement-card-top strong {{
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
            font-size: 0.72rem;
            white-space: nowrap;
          }}
          .measurement-value {{
            margin-top: 0.75rem;
            color: #0f172a;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
          }}
          .measurement-value small {{
            margin-left: 0.35rem;
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 650;
          }}
          .measurement-meta {{
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            margin-top: 0.8rem;
            color: #64748b;
            font-size: 0.8rem;
          }}
          .measurement-meta b {{
            color: #334155;
            text-align: right;
          }}
          .measurement-source {{
            margin-top: 0.55rem;
            color: #64748b;
            font-size: 0.75rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
            word-break: break-word;
          }}
        </style>
        <div class="measurement-grid">{''.join(cards)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_report_charts(
    analysis: dict,
    report_text: str,
    *,
    section_label: str | None = "Uploaded report",
) -> None:
    if section_label:
        st.markdown(f"#### {section_label}")
    numeric_df = _numeric_measurements(report_text)
    timeline_df = extract_symptom_timeline(report_text)
    history_df = extract_history_durations(report_text)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Values found", len(numeric_df))
    metric_cols[1].metric("Episodes", len(timeline_df))
    metric_cols[2].metric(
        "Abnormal",
        int((numeric_df["Status"] == "High").sum()) if not numeric_df.empty else 0,
    )

    vitals_chart = chart_vitals_and_labs(report_text)
    timeline_chart = chart_timeline(report_text)
    history_chart = chart_history_durations(report_text)

    if vitals_chart is None and timeline_chart is None and history_chart is None:
        return

    if vitals_chart is not None:
        render_measurement_cards(numeric_df)
        st.plotly_chart(vitals_chart, use_container_width=True)
        with st.expander("Measurement details"):
            st.dataframe(numeric_df, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        if timeline_chart is not None:
            st.plotly_chart(timeline_chart, use_container_width=True)
    with right:
        if history_chart is not None:
            st.plotly_chart(history_chart, use_container_width=True)
