"""Clinical visualizations derived from report analysis and extracted text."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COLORS = ["#255e7e", "#1f8a83", "#b88020", "#b45c63", "#5b9bd5", "#6bb89a", "#8b7bb8", "#65748b"]
STATUS_COLORS = {"Low": "#5b9bd5", "In range": "#1f8a83", "High": "#b45c63"}

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#172033", size=13),
    margin=dict(l=16, r=16, t=52, b=16),
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

RED_FLAG_THEMES = {
    "Cardiovascular": r"chest pain|mi|stemi|heart failure|arrhythmia|hypertension|murmur|bruit|cad",
    "Respiratory": r"dyspnea|hypoxia|pe|pulmonary|respiratory failure|pneumonia",
    "Oncology": r"malignan|metast|tumor|cancer|chemo|radiation|stage|biopsy",
    "Neurologic": r"stroke|seizure|altered mental|weakness|neuro",
    "Infection": r"sepsis|fever|infection|abscess",
    "Hematologic": r"anemia|bleed|transfusion|thrombocyt",
}

SPECIALTY_KEYWORDS = {
    "Medical oncology": r"oncolog|chemo|immunotherapy|targeted therapy|tumor board",
    "Radiation oncology": r"radiation|radiotherapy|sbrt|srs",
    "Surgery": r"surg|resection|lobectomy|mastectomy|prostatectomy",
    "Radiology": r"imaging|ct scan|mri|pet|ultrasound|radiolog",
    "Pathology": r"pathology|biopsy|histolog|immunohistochem",
    "Palliative care": r"palliative|symptom control|hospice|pain control",
    "Primary care": r"primary care|family medicine|internal medicine|gp",
}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("problem", "Problem", "specialist", "Specialist", "evidence", "review"):
            val = item.get(key)
            if val:
                return str(val).strip()
        return " ".join(str(v) for v in item.values() if v).strip()
    return str(item).strip()


def _short_code(index: int, prefix: str) -> str:
    return f"{prefix} {index}"


def _apply_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(**CHART_THEME, height=height)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(217,228,239,0.6)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(217,228,239,0.6)", zeroline=False)
    return fig


def compute_readiness_score(analysis: dict) -> dict[str, int | float]:
    facts = len(_as_list(analysis.get("critical_facts")))
    problems = len(_as_list(analysis.get("priority_problems")))
    missing = len(_as_list(analysis.get("missing_data")))
    red_flags = len(_as_list(analysis.get("red_flags")))
    decisions = len(_as_list(analysis.get("decision_points")))
    specialists = len(_as_list(analysis.get("specialist_focus")))

    signal = min(100, facts * 8 + problems * 6 + specialists * 5)
    gaps = min(100, missing * 12 + red_flags * 8)
    readiness = max(0, min(100, signal - gaps * 0.45 + decisions * 4))

    return {
        "readiness": round(readiness, 1),
        "facts": facts,
        "problems": problems,
        "missing": missing,
        "red_flags": red_flags,
        "decisions": decisions,
        "specialists": specialists,
    }


def extract_labs_from_text(report_text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, pattern, unit, ref_low, ref_high in LAB_PATTERNS:
        match = pattern.search(report_text)
        if not match:
            continue
        key = f"{name}:{match.group(1)}"
        if key in seen:
            continue
        seen.add(key)
        value = float(match.group(1))
        status = "In range"
        if ref_low is not None and value < ref_low:
            status = "Low"
        elif ref_high is not None and value > ref_high:
            status = "High"
        rows.append(
            {
                "Analyte": name,
                "Value": value,
                "Unit": unit,
                "Ref low": ref_low,
                "Ref high": ref_high,
                "Status": status,
            }
        )

    bp_match = re.search(r"(\d{2,3})/(\d{2,3})\s*mmHg", report_text, re.I)
    if bp_match:
        systolic = int(bp_match.group(1))
        diastolic = int(bp_match.group(2))
        rows.extend(
            [
                {
                    "Analyte": "Systolic BP",
                    "Value": systolic,
                    "Unit": "mmHg",
                    "Ref low": 90,
                    "Ref high": 120,
                    "Status": "High" if systolic > 120 else ("Low" if systolic < 90 else "In range"),
                },
                {
                    "Analyte": "Diastolic BP",
                    "Value": diastolic,
                    "Unit": "mmHg",
                    "Ref low": 60,
                    "Ref high": 80,
                    "Status": "High" if diastolic > 80 else ("Low" if diastolic < 60 else "In range"),
                },
            ]
        )
    return pd.DataFrame(rows)


def _signal_donut_df(analysis: dict) -> pd.DataFrame:
    scores = compute_readiness_score(analysis)
    rows = [
        ("Critical facts", scores["facts"]),
        ("Priority problems", scores["problems"]),
        ("Red flags", scores["red_flags"]),
        ("Missing data", scores["missing"]),
        ("Decision points", scores["decisions"]),
        ("Specialists", scores["specialists"]),
    ]
    df = pd.DataFrame(rows, columns=["Category", "Count"])
    return df[df["Count"] > 0]


def _infer_specialties(analysis: dict, report_text: str) -> pd.DataFrame:
    blob = " ".join(
        _item_text(item)
        for key in ("specialist_focus", "critical_facts", "priority_problems", "decision_points")
        for item in _as_list(analysis.get(key))
    )
    blob = f"{blob} {report_text[:8000]}".lower()
    rows = []
    for specialty, pattern in SPECIALTY_KEYWORDS.items():
        hits = len(re.findall(pattern, blob, re.I))
        if hits:
            rows.append({"Specialty": specialty, "Weight": hits})
    if not rows:
        focus = [_item_text(item) for item in _as_list(analysis.get("specialist_focus"))]
        rows = [{"Specialty": f"Team {i + 1}", "Weight": 1} for i in range(min(len(focus), 6))]
    return pd.DataFrame(rows)


def _problem_chart_df(analysis: dict) -> pd.DataFrame:
    problems = _as_list(analysis.get("priority_problems"))
    rows = []
    for idx, item in enumerate(problems[:8], start=1):
        full = _item_text(item)
        if not full:
            continue
        rows.append(
            {
                "Code": _short_code(idx, "Problem"),
                "Weight": len(problems) - idx + 1,
                "Detail": full,
            }
        )
    return pd.DataFrame(rows)


def _facts_chart_df(analysis: dict) -> pd.DataFrame:
    facts = [_item_text(item) for item in _as_list(analysis.get("critical_facts")) if _item_text(item)]
    rows = []
    for idx, fact in enumerate(facts[:8], start=1):
        rows.append({"Code": _short_code(idx, "Fact"), "Weight": len(facts) - idx + 1, "Detail": fact})
    return pd.DataFrame(rows)


def _missing_chart_df(analysis: dict) -> pd.DataFrame:
    missing = [_item_text(item) for item in _as_list(analysis.get("missing_data")) if _item_text(item)]
    if not missing:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Code": [_short_code(i, "Gap") for i in range(1, len(missing[:8]) + 1)],
            "Weight": list(range(len(missing[:8]), 0, -1)),
            "Detail": missing[:8],
        }
    )


def _red_flag_df(analysis: dict) -> pd.DataFrame:
    red_flags = _as_list(analysis.get("red_flags"))
    counts: dict[str, int] = {theme: 0 for theme in RED_FLAG_THEMES}
    text_blob = " ".join(_item_text(item) for item in red_flags).lower()
    for theme, pattern in RED_FLAG_THEMES.items():
        if re.search(pattern, text_blob, re.I):
            counts[theme] += max(1, len(re.findall(pattern, text_blob, re.I)))
    if not any(counts.values()) and red_flags:
        return pd.DataFrame({"Theme": ["Clinical risk"], "Count": [len(red_flags)]})
    rows = [{"Theme": k, "Count": v} for k, v in counts.items() if v > 0]
    return pd.DataFrame(rows)


def _flow_chart_df(analysis: dict) -> pd.DataFrame:
    flow = [_item_text(item) for item in _as_list(analysis.get("meeting_flow")) if _item_text(item)]
    if not flow:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Step": [f"Step {i}" for i in range(1, len(flow[:6]) + 1)],
            "Weight": list(range(len(flow[:6]), 0, -1)),
            "Detail": flow[:6],
        }
    )


def _decisions_df(analysis: dict) -> pd.DataFrame:
    decisions = [_item_text(item) for item in _as_list(analysis.get("decision_points")) if _item_text(item)]
    if not decisions:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Code": [_short_code(i, "Decision") for i in range(1, len(decisions[:6]) + 1)],
            "Weight": list(range(len(decisions[:6]), 0, -1)),
            "Detail": decisions[:6],
        }
    )


def chart_signal_donut(analysis: dict) -> go.Figure:
    df = _signal_donut_df(analysis)
    if df.empty:
        df = pd.DataFrame({"Category": ["No signal"], "Count": [1]})
    fig = px.pie(
        df,
        names="Category",
        values="Count",
        hole=0.52,
        color_discrete_sequence=COLORS,
        title="Case signal mix",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_theme(fig, height=340)


def chart_readiness_gauge(analysis: dict) -> go.Figure:
    score = compute_readiness_score(analysis)["readiness"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "Board readiness", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#255e7e", "thickness": 0.22},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#fde2e4"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d8f3dc"},
                ],
            },
        )
    )
    return _apply_theme(fig, height=340)


def chart_clinical_radar(analysis: dict) -> go.Figure:
    scores = compute_readiness_score(analysis)
    categories = ["Facts", "Problems", "Red flags", "Missing", "Decisions", "Specialists"]
    values = [
        scores["facts"],
        scores["problems"],
        scores["red_flags"],
        scores["missing"],
        scores["decisions"],
        scores["specialists"],
    ]
    max_val = max(values) or 1
    norm = [v / max_val * 100 for v in values]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=norm + [norm[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(37, 94, 126, 0.25)",
            line=dict(color="#255e7e", width=3),
            name="Case profile",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#d9e4ef")),
        title="Clinical profile radar",
    )
    return _apply_theme(fig, height=360)


def chart_lollipop(df: pd.DataFrame, title: str, color: str = "#255e7e") -> go.Figure | None:
    if df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Weight"],
            y=df["Code"],
            mode="markers",
            marker=dict(size=18, color=color, line=dict(width=2, color="white")),
            customdata=df["Detail"],
            hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>",
        )
    )
    for _, row in df.iterrows():
        fig.add_shape(
            type="line",
            x0=0,
            x1=row["Weight"],
            y0=row["Code"],
            y1=row["Code"],
            line=dict(color="#cbd5e1", width=3),
        )
    fig.update_layout(title=title, xaxis_title="Priority weight", yaxis=dict(categoryorder="array", categoryarray=df["Code"].tolist()[::-1]))
    return _apply_theme(fig, height=max(280, 56 * len(df)))


def chart_specialty_pie(analysis: dict, report_text: str) -> go.Figure:
    df = _infer_specialties(analysis, report_text)
    if df.empty:
        df = pd.DataFrame({"Specialty": ["Multidisciplinary team"], "Weight": [1]})
    fig = px.pie(
        df,
        names="Specialty",
        values="Weight",
        hole=0.4,
        color_discrete_sequence=COLORS,
        title="Specialty involvement",
    )
    fig.update_traces(textposition="inside", textinfo="label+percent")
    return _apply_theme(fig, height=340)


def chart_red_flags(analysis: dict) -> go.Figure | None:
    df = _red_flag_df(analysis)
    if df.empty:
        return None
    fig = px.bar(
        df,
        x="Theme",
        y="Count",
        color="Theme",
        color_discrete_sequence=COLORS,
        title="Red-flag categories",
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-20)
    return _apply_theme(fig, height=320)


def chart_readiness_stack(analysis: dict) -> go.Figure:
    scores = compute_readiness_score(analysis)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Documented", x=["Case"], y=[scores["facts"] + scores["problems"]], marker_color="#1f8a83"))
    fig.add_trace(go.Bar(name="Gaps", x=["Case"], y=[scores["missing"] + scores["red_flags"]], marker_color="#b45c63"))
    fig.add_trace(go.Bar(name="Decisions", x=["Case"], y=[scores["decisions"]], marker_color="#b88020"))
    fig.update_layout(barmode="stack", title="Readiness composition", showlegend=True, legend=dict(orientation="h", y=-0.15))
    return _apply_theme(fig, height=320)


def chart_labs_bar(report_text: str) -> go.Figure | None:
    df = extract_labs_from_text(report_text)
    if df.empty:
        return None
    fig = px.bar(
        df,
        x="Analyte",
        y="Value",
        color="Status",
        color_discrete_map=STATUS_COLORS,
        title="Labs & vitals",
    )
    fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(tickangle=-15)
    return _apply_theme(fig, height=340)


def chart_lab_status_pie(report_text: str) -> go.Figure | None:
    df = extract_labs_from_text(report_text)
    if df.empty:
        return None
    grouped = df.groupby("Status", as_index=False)["Analyte"].count().rename(columns={"Analyte": "Count"})
    fig = px.pie(
        grouped,
        names="Status",
        values="Count",
        hole=0.45,
        color="Status",
        color_discrete_map=STATUS_COLORS,
        title="Lab status distribution",
    )
    fig.update_traces(textposition="inside", textinfo="label+value+percent")
    return _apply_theme(fig, height=320)


def chart_meeting_steps(analysis: dict) -> go.Figure | None:
    df = _flow_chart_df(analysis)
    if df.empty:
        return None
    fig = px.bar(
        df,
        x="Step",
        y="Weight",
        color="Weight",
        color_continuous_scale=["#dbeafe", "#255e7e"],
        custom_data=["Detail"],
        title="Meeting agenda flow",
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{customdata[0]}<extra></extra>")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return _apply_theme(fig, height=320)


def chart_age_band(snapshot: dict) -> go.Figure | None:
    age_raw = str(snapshot.get("age", "")).strip()
    match = re.search(r"(\d{1,3})", age_raw)
    if not match:
        return None
    age = int(match.group(1))
    bands = ["18-39", "40-59", "60-74", "75+"]
    shares = [25, 30, 30, 15]
    patient_band = "18-39" if age < 40 else "40-59" if age < 60 else "60-74" if age < 75 else "75+"
    colors = ["#255e7e" if band == patient_band else "#dbe7f3" for band in bands]
    fig = go.Figure(go.Bar(x=bands, y=shares, marker_color=colors))
    fig.add_vline(x=bands.index(patient_band), line_dash="dot", line_color="#b45c63", line_width=2)
    fig.update_layout(title=f"Age band highlight ({age} yrs)", yaxis_title="Cohort share (%)", showlegend=False)
    return _apply_theme(fig, height=300)


def _plot(fig: go.Figure | None, placeholder: str) -> None:
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(
            _apply_theme(
                go.Figure(
                    go.Pie(labels=["No data"], values=[1], hole=0.55, marker_colors=["#e2e8f0"])
                ).update_layout(title=placeholder),
                height=280,
            ),
            use_container_width=True,
        )


def render_report_charts(analysis: dict, report_text: str) -> None:
    snapshot = analysis.get("patient_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}

    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.plotly_chart(chart_signal_donut(analysis), use_container_width=True)
    with row1_right:
        st.plotly_chart(chart_readiness_gauge(analysis), use_container_width=True)

    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.plotly_chart(chart_clinical_radar(analysis), use_container_width=True)
    with row2_right:
        st.plotly_chart(chart_specialty_pie(analysis, report_text), use_container_width=True)

    row3_left, row3_right = st.columns(2)
    with row3_left:
        _plot(chart_lollipop(_problem_chart_df(analysis), "Priority problems", "#255e7e"), "Priority problems")
    with row3_right:
        _plot(chart_lollipop(_facts_chart_df(analysis), "Critical facts", "#1f8a83"), "Critical facts")

    row4_left, row4_right = st.columns(2)
    with row4_left:
        _plot(chart_red_flags(analysis), "Red-flag categories")
    with row4_right:
        st.plotly_chart(chart_readiness_stack(analysis), use_container_width=True)

    row5_left, row5_right = st.columns(2)
    with row5_left:
        _plot(chart_lollipop(_missing_chart_df(analysis), "Missing data gaps", "#b88020"), "Missing data gaps")
    with row5_right:
        _plot(chart_lollipop(_decisions_df(analysis), "Decision points", "#8b7bb8"), "Decision points")

    row6_left, row6_right = st.columns(2)
    with row6_left:
        _plot(chart_labs_bar(report_text), "Labs & vitals")
    with row6_right:
        _plot(chart_lab_status_pie(report_text), "Lab status distribution")

    row7_left, row7_right = st.columns(2)
    with row7_left:
        _plot(chart_meeting_steps(analysis), "Meeting agenda flow")
    with row7_right:
        age_fig = chart_age_band(snapshot)
        if age_fig:
            st.plotly_chart(age_fig, use_container_width=True)
        else:
            st.plotly_chart(chart_readiness_stack(analysis), use_container_width=True)
