"""Streamlit tumor board assistant — synthetic patients + local MedGemma via Ollama."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.summarizer import MODEL, summarize_patient

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "synthetic_patients" / "sample_patients.csv"

SECTION_ORDER = [
    "One-line case summary",
    "Key clinical facts",
    "Missing or unclear information",
    "MDT discussion questions",
    "Treatment considerations",
]


def load_patients() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def format_patient_data(row: pd.Series) -> str:
    return "\n".join(f"{col}: {row[col]}" for col in row.index)


def parse_summary_sections(text: str) -> dict[str, str]:
    """Split numbered model output into section title -> content."""
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
        content = sections.get(title)
        if content:
            st.subheader(title)
            st.markdown(content)

    extra = [t for t in sections if t not in SECTION_ORDER]
    for title in extra:
        st.subheader(title)
        st.markdown(sections[title])


@st.cache_data
def get_patients_df() -> pd.DataFrame:
    return load_patients()


def main() -> None:
    st.set_page_config(page_title="Tumor Board AI", layout="wide")
    st.title("Tumor Board AI")
    st.caption(
        "Synthetic oncology cases only. Summaries use local Ollama "
        f"(`{MODEL}`) and must not invent clinical facts beyond the record."
    )

    df = get_patients_df()
    labels = [
        f"{row['patient_id']} — {row['diagnosis']} (stage {row['stage']})"
        for _, row in df.iterrows()
    ]
    choice = st.selectbox("Select patient", labels, index=0)
    row = df.iloc[labels.index(choice)]

    st.subheader("Patient record")
    structured = row.to_dict()
    cols = st.columns(2)
    left_keys = ["patient_id", "age", "sex", "diagnosis", "stage", "ecog"]
    right_keys = ["biomarkers", "prior_treatment", "notes"]
    with cols[0]:
        for key in left_keys:
            st.markdown(f"**{key.replace('_', ' ').title()}:** {structured[key]}")
    with cols[1]:
        for key in right_keys:
            st.markdown(f"**{key.replace('_', ' ').title()}:** {structured[key]}")

    patient_data = format_patient_data(row)

    if st.button("Generate tumor board summary", type="primary"):
        with st.spinner(f"Calling Ollama ({MODEL})…"):
            try:
                summary = summarize_patient(patient_data)
            except Exception as exc:
                st.error(
                    "Could not reach Ollama. Ensure `ollama serve` is running and "
                    f"`ollama pull {MODEL}` has been completed.\n\n"
                    f"Details: {exc}"
                )
                return

        st.session_state["summary"] = summary
        st.session_state["summary_patient"] = row["patient_id"]

    if (
        st.session_state.get("summary")
        and st.session_state.get("summary_patient") == row["patient_id"]
    ):
        st.divider()
        st.subheader("Tumor board summary")
        display_summary(st.session_state["summary"])


if __name__ == "__main__":
    main()
