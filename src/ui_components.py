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
          :root {
            --ink: #172033;
            --muted: #65748b;
            --line: #d9e4ef;
            --paper: #ffffff;
            --surface: #f7fbff;
            --wash: #eef6fb;
            --brand: #255e7e;
            --brand-2: #2f7b8f;
            --teal: #1f8a83;
            --gold: #b88020;
            --rose: #b45c63;
            --shadow: 0 16px 45px rgba(34, 57, 86, 0.10);
            --soft-shadow: 0 7px 20px rgba(34, 57, 86, 0.07);
          }

          @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes headerSheen {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
          }
          @keyframes statusPulse {
            0%, 100% { box-shadow: 0 0 0 3px rgba(31,138,131,0.10); }
            50% { box-shadow: 0 0 0 6px rgba(31,138,131,0.16); }
          }
          @keyframes auroraDrift {
            0% { transform: translate3d(-2%, -1%, 0) scale(1); opacity: 0.72; }
            50% { transform: translate3d(3%, 2%, 0) scale(1.04); opacity: 0.92; }
            100% { transform: translate3d(0%, 0%, 0) scale(1); opacity: 0.78; }
          }
          @keyframes slowGlow {
            0%, 100% { filter: saturate(1) brightness(1); }
            50% { filter: saturate(1.12) brightness(1.04); }
          }
          @keyframes shimmerLine {
            0% { transform: translateX(-110%); }
            100% { transform: translateX(110%); }
          }
          @keyframes floatIn {
            from { opacity: 0; transform: translateY(14px) scale(0.985); }
            to { opacity: 1; transform: translateY(0) scale(1); }
          }

          /* Extra top padding prevents first-row controls from clipping under Streamlit header. */
          .block-container {
            padding-top: 2.25rem;
            padding-bottom: 1.5rem;
            max-width: 1380px;
          }
          .top-nav-safe-offset { height: 0.85rem; }
          html, body, .stApp {
            background:
              radial-gradient(circle at 12% 0%, rgba(47, 123, 143, 0.12), transparent 28rem),
              radial-gradient(circle at 78% 8%, rgba(184, 128, 32, 0.10), transparent 22rem),
              radial-gradient(circle at 92% 72%, rgba(180, 92, 99, 0.08), transparent 24rem),
              linear-gradient(180deg, #f7fbff 0%, #eef5f8 48%, #f8fafc 100%);
            color: var(--ink);
          }
          .stApp::before {
            content: "";
            position: fixed;
            inset: -18rem -10rem auto -10rem;
            height: 34rem;
            pointer-events: none;
            background:
              radial-gradient(circle at 22% 32%, rgba(31, 138, 131, 0.20), transparent 16rem),
              radial-gradient(circle at 52% 18%, rgba(37, 94, 126, 0.16), transparent 18rem),
              radial-gradient(circle at 76% 42%, rgba(184, 128, 32, 0.14), transparent 14rem);
            filter: blur(12px);
            animation: auroraDrift 12s ease-in-out infinite alternate;
            z-index: 0;
          }
          .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
              linear-gradient(rgba(23,32,51,0.035) 1px, transparent 1px),
              linear-gradient(90deg, rgba(23,32,51,0.03) 1px, transparent 1px);
            background-size: 38px 38px;
            mask-image: linear-gradient(to bottom, rgba(0,0,0,0.38), transparent 58%);
            z-index: 0;
          }
          .block-container {
            position: relative;
            z-index: 1;
          }
          .block-container p, .block-container li {
            font-size: 0.98rem;
            line-height: 1.62;
          }
          .block-container > div {
            animation: fadeUp 260ms ease-out both;
          }
          .block-container h1,
          .block-container h2,
          .block-container h3,
          .block-container h4 {
            letter-spacing: 0;
            color: var(--ink);
          }
          .stMarkdown, .stText, div[data-testid="stMarkdownContainer"] {
            color: var(--ink);
          }

          .app-titlebar {
            background: rgba(255, 255, 255, 0.82);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(217, 228, 239, 0.92);
            border-radius: 8px;
            padding: 0.95rem 1.1rem;
            box-shadow: var(--soft-shadow);
            margin: 0.4rem 0 1rem;
            position: relative;
            overflow: hidden;
            transition: transform 170ms ease, box-shadow 170ms ease, border-color 170ms ease;
          }
          .app-titlebar:hover {
            transform: translateY(-2px);
            border-color: rgba(31,138,131,0.34);
            box-shadow: 0 18px 44px rgba(34,57,86,0.12);
          }
          .app-titlebar::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: linear-gradient(180deg, var(--teal), var(--gold));
          }
          .app-titlebar::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(31,138,131,0.55), transparent);
          }
          .app-titlebar h2 {
            margin: 0;
            font-size: 1.45rem;
            line-height: 1.15;
          }
          .app-titlebar p {
            margin: 0.25rem 0 0;
            color: var(--muted);
            font-size: 0.92rem;
          }
          .product-kicker {
            color: var(--teal);
            font-weight: 800;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.22rem;
          }
          .status-pill {
            display: inline-flex;
            justify-content: flex-end;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.62rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.82);
            border: 1px solid var(--line);
            color: var(--muted);
            font-weight: 800;
            font-size: 0.84rem;
            box-shadow: var(--soft-shadow);
          }
          .status-dot-only {
            width: 2.25rem;
            height: 2.25rem;
            justify-content: center;
            padding: 0;
            border-radius: 999px;
            backdrop-filter: blur(14px);
            cursor: default;
            transition: transform 160ms ease, box-shadow 160ms ease;
          }
          .status-dot-only:hover {
            transform: scale(1.08);
            box-shadow: 0 12px 24px rgba(34,57,86,0.14);
          }
          .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
            animation: statusPulse 2.4s ease-in-out infinite;
          }

          .profile-hero {
            background:
              linear-gradient(135deg, rgba(23, 32, 51, 0.12) 0%, rgba(255,255,255,0) 42%),
              repeating-linear-gradient(115deg, rgba(255,255,255,0.08) 0 1px, transparent 1px 18px),
              linear-gradient(135deg, #245a78 0%, #2f7b8f 58%, #1f8a83 100%);
            background-size: auto, auto, 180% 180%;
            color: #fff;
            padding: 1.3rem 1.45rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255, 255, 255, 0.28);
            animation: fadeUp 320ms ease-out both, headerSheen 9s ease-in-out infinite alternate;
            position: relative;
            overflow: hidden;
          }
          .profile-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(110deg, transparent 0%, rgba(255,255,255,0.18) 42%, transparent 62%);
            transform: translateX(-120%);
            animation: shimmerLine 7s ease-in-out infinite;
          }
          .profile-hero h1 {
            color: #fff;
            font-size: clamp(1.35rem, 2.2vw, 2rem);
            line-height: 1.18;
            margin: 0 0 0.65rem 0;
            font-weight: 750;
          }
          .profile-hero p { margin: 0; opacity: 0.94; font-size: 0.95rem; }
          .badge {
            display: inline-block;
            background: rgba(255,255,255,0.17);
            border: 1px solid rgba(255,255,255,0.18);
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.35rem;
            margin-bottom: 0.25rem;
          }
          .info-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.95rem 1.05rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--soft-shadow);
            transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease;
            animation: floatIn 280ms ease-out both;
          }
          .info-card:hover {
            transform: translateY(-2px);
            border-color: #bdd6df;
            box-shadow: 0 12px 28px rgba(34, 57, 86, 0.10);
          }
          .info-card h4 {
            margin: 0 0 0.65rem 0;
            color: var(--brand);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
          }
          .info-row {
            margin: 0.38rem 0;
            font-size: 0.94rem;
            line-height: 1.45;
          }
          .info-label { color: var(--muted); font-weight: 700; }
          div[data-testid="stSidebar"] {
            background:
              linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,251,255,0.98));
            border-right: 1px solid var(--line);
            box-shadow: 10px 0 25px rgba(34, 57, 86, 0.06);
          }
          div[data-testid="stSidebar"] h5,
          div[data-testid="stSidebar"] p,
          div[data-testid="stSidebar"] label {
            color: var(--ink);
          }
          div[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(255,255,255,0.72);
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 0.34rem 0.45rem;
            margin-bottom: 0.18rem;
          }
          div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: #e7f2f4;
            border-color: #bcd7dd;
            box-shadow: inset 3px 0 0 var(--teal);
          }

          /* Make actions feel consistent and scan-friendly. */
          .stButton > button {
            border-radius: 8px;
            min-height: 2.65rem;
            font-weight: 700;
            box-shadow: 0 2px 0 rgba(23, 32, 51, 0.04);
            transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
          }
          .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(34, 57, 86, 0.10);
          }
          .stButton > button:active {
            transform: translateY(0) scale(0.99);
          }
          .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%);
            border-color: var(--brand);
            position: relative;
            overflow: hidden;
          }
          .stButton > button[kind="primary"]::before {
            content: "";
            position: absolute;
            inset: 0;
            transform: translateX(-120%);
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
            transition: transform 420ms ease;
          }
          .stButton > button[kind="primary"]:hover::before { transform: translateX(120%); }
          .stButton > button[kind="secondary"] {
            background: rgba(255,255,255,0.86);
            border-color: #cbd8e5;
            color: var(--brand);
          }
          .top-nav-safe-offset + div .stButton > button {
            border-radius: 999px;
            min-height: 2.55rem;
            backdrop-filter: blur(14px);
          }
          .top-nav-safe-offset + div .stButton > button[kind="secondary"] {
            background: rgba(255,255,255,0.62);
          }
          .top-nav-safe-offset + div .stButton > button[kind="primary"] {
            box-shadow: 0 12px 24px rgba(37,94,126,0.18);
          }

          /* Radio inputs should use the same brand accent. */
          div[data-testid="stRadio"] input[type="radio"] { accent-color: var(--teal); }
          div[data-testid="stRadio"] label { font-weight: 620; color: var(--ink); }

          /* Add breathing room between main vertical blocks. */
          div[data-testid="stVerticalBlock"] > div:has(> .element-container) {
            margin-bottom: 0.25rem;
          }

          /* Tabs for dense content: cleaner and lighter. */
          button[data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            font-weight: 700;
            color: var(--muted);
            transition: color 140ms ease, background 140ms ease, transform 140ms ease;
          }
          button[data-baseweb="tab"]:hover {
            transform: translateY(-1px);
            background: rgba(255,255,255,0.62);
          }
          button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--brand);
          }

          .home-stat-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: var(--soft-shadow);
            margin-bottom: 0.55rem;
            min-height: 6.1rem;
            position: relative;
            overflow: hidden;
            transition: transform 150ms ease, box-shadow 150ms ease;
            animation: floatIn 320ms ease-out both;
          }
          .home-stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 16px 34px rgba(34,57,86,0.12);
          }
          .home-stat-card::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 5px;
            background: linear-gradient(180deg, var(--teal), var(--brand-2));
          }
          .home-stat-card::after {
            content: "";
            position: absolute;
            right: -2.5rem;
            top: -2.5rem;
            width: 6rem;
            height: 6rem;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(31,138,131,0.12), transparent 68%);
            transition: transform 200ms ease, opacity 200ms ease;
          }
          .home-stat-card:hover::after {
            transform: scale(1.25);
            opacity: 0.9;
          }
          .home-stat-card p {
            margin: 0;
            color: var(--muted);
            font-weight: 750;
            font-size: 0.82rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
          }
          .home-stat-card h2 {
            margin: 0.3rem 0 0;
            color: var(--brand);
            font-size: 2rem;
            line-height: 1.1;
          }
          .home-hero {
            background:
              repeating-linear-gradient(115deg, rgba(255,255,255,0.10) 0 1px, transparent 1px 20px),
              linear-gradient(135deg, rgba(37,94,126,0.94), rgba(31,138,131,0.90)),
              linear-gradient(45deg, rgba(255,255,255,0.22), transparent);
            background-size: auto, 180% 180%, auto;
            color: #fff;
            border-radius: 8px;
            padding: 1.55rem 1.6rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255,255,255,0.25);
            animation: fadeUp 280ms ease-out both, headerSheen 10s ease-in-out infinite alternate;
            position: relative;
            overflow: hidden;
          }
          .home-hero::before {
            content: "";
            position: absolute;
            right: -7rem;
            top: -8rem;
            width: 18rem;
            height: 18rem;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(255,255,255,0.24), transparent 62%);
            animation: slowGlow 5s ease-in-out infinite;
          }
          .home-hero::after {
            content: "";
            position: absolute;
            inset: auto 1rem 0 1rem;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.72), transparent);
          }
          .home-hero h1 {
            color: #fff;
            margin: 0;
            font-size: clamp(1.65rem, 3vw, 2.35rem);
            line-height: 1.12;
          }
          .home-hero p {
            color: rgba(255,255,255,0.9);
            max-width: 54rem;
            margin: 0.55rem 0 0;
          }
          .home-launch-title {
            margin: 0.55rem 0 0.5rem;
            color: var(--brand);
            font-weight: 650;
            font-size: 0.98rem;
          }
          .summary-panel {
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: var(--soft-shadow);
            margin-top: 0.75rem;
            animation: fadeUp 220ms ease-out both;
          }
          .summary-panel h4 {
            color: var(--brand);
            font-size: 0.98rem;
            margin: 0.85rem 0 0.25rem;
          }
          .summary-panel h4:first-child { margin-top: 0; }
          .summary-panel ul { margin-top: 0.25rem; padding-left: 1.25rem; }
          .summary-panel li { margin-bottom: 0.28rem; }
          .summary-section {
            background: rgba(255,255,255,0.78);
            border: 1px solid #e0e9f2;
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            margin: 0.7rem 0;
            box-shadow: 0 4px 14px rgba(34,57,86,0.05);
          }
          .summary-section h4 {
            margin: 0 0 0.35rem;
            color: var(--brand);
            font-size: 0.96rem;
          }
          .summary-section p,
          .summary-section li {
            color: var(--ink);
          }

          div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.82);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            box-shadow: 0 3px 12px rgba(34,57,86,0.05);
            transition: transform 150ms ease, box-shadow 150ms ease;
          }
          div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(34,57,86,0.10);
          }
          div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-weight: 750;
          }
          div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 750;
          }

          input, textarea, div[data-baseweb="select"] > div {
            border-radius: 8px !important;
          }
          div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--soft-shadow);
          }

          .workflow-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.85rem 0 1rem;
          }
          .workflow-step {
            background: rgba(255,255,255,0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            box-shadow: var(--soft-shadow);
            transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
          }
          .workflow-step:hover {
            transform: translateY(-2px);
            background: rgba(255,255,255,0.96);
            border-color: rgba(255,255,255,0.72);
          }
          .workflow-step strong {
            display: block;
            color: var(--brand);
            font-size: 0.88rem;
            margin-bottom: 0.2rem;
          }
          .workflow-step span {
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.45;
          }

          @media (max-width: 800px) {
            .workflow-strip { grid-template-columns: 1fr; }
            .profile-hero h1 { font-size: 1.35rem; }
            .app-titlebar { margin-top: 0.2rem; }
          }

          .platform-footer {
            text-align: center;
            color: #8796aa;
            font-size: 0.75rem;
            padding: 1.5rem 0 0.5rem;
            border-top: 1px solid var(--line);
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
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    for title in SECTION_ORDER:
        if sections.get(title):
            st.markdown(f'<div class="summary-section"><h4>{title}</h4>', unsafe_allow_html=True)
            st.markdown(sections[title])
            st.markdown("</div>", unsafe_allow_html=True)
    for title in sections:
        if title not in SECTION_ORDER:
            st.markdown(f'<div class="summary-section"><h4>{title}</h4>', unsafe_allow_html=True)
            st.markdown(sections[title])
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
