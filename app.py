"""Streamlit tumor board assistant — synthetic patients + local MedGemma via Ollama."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.summarizer import (
    MODEL,
    PROMPT_VERSION,
    check_ollama_reachable,
    stream_summarize_patient,
    summarize_patient,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "synthetic_patients" / "sample_patients.csv"

SECTION_ORDER = [
    "One-line case summary",
    "Key clinical facts",
    "Missing or unclear information",
    "MDT discussion questions",
    "Treatment considerations",
]

DISPLAY_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "diagnosis",
    "stage",
    "ecog",
    "biomarkers",
    "imaging",
    "pathology",
    "comorbidities",
    "medications",
    "pending_tests",
    "prior_treatment",
    "notes",
]


def load_patients() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def format_patient_data(row: pd.Series) -> str:
    return "\n".join(f"{col}: {row[col]}" for col in row.index)


def record_fingerprint(patient_data: str) -> str:
    return hashlib.sha256(patient_data.encode("utf-8")).hexdigest()[:16]


def parse_summary_sections(text: str) -> dict[str, str]:
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

    for title in sections:
        if title not in SECTION_ORDER:
            st.subheader(title)
            st.markdown(sections[title])


def summary_to_markdown(patient_id: str, summary: str) -> str:
    lines = [
        f"# Tumor board summary — {patient_id}",
        f"Model: {MODEL} | Prompt: {PROMPT_VERSION}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        summary,
    ]
    return "\n".join(lines)


def render_patient_record(row: pd.Series) -> None:
    structured = row.to_dict()
    keys = [k for k in DISPLAY_COLUMNS if k in structured]
    mid = (len(keys) + 1) // 2
    col_left, col_right = st.columns(2)
    with col_left:
        for key in keys[:mid]:
            label = key.replace("_", " ").title()
            value = structured[key]
            if pd.isna(value) or value == "":
                value = "—"
            st.markdown(f"**{label}:** {value}")
    with col_right:
        for key in keys[mid:]:
            label = key.replace("_", " ").title()
            value = structured[key]
            if pd.isna(value) or value == "":
                value = "—"
            st.markdown(f"**{label}:** {value}")


@st.cache_data
def get_patients_df() -> pd.DataFrame:
    return load_patients()


@st.cache_data(show_spinner=False)
def get_cached_summary(
    patient_id: str,
    patient_data: str,
    cache_bust: int,
    precomputed: str | None = None,
) -> str:
    """Cache summaries by patient id, record content, and bust token."""
    if precomputed is not None:
        return precomputed
    return summarize_patient(patient_data)


def _summary_cache_key(patient_id: str, fingerprint: str, cache_bust: int) -> str:
    return f"{patient_id}:{fingerprint}:{cache_bust}"


def _mark_summary_warmed(cache_key: str) -> None:
    warmed = set(st.session_state.get("summary_warmed_keys", set()))
    warmed.add(cache_key)
    st.session_state["summary_warmed_keys"] = warmed


def load_summary(
    patient_id: str,
    patient_data: str,
    fingerprint: str,
    cache_bust: int,
    *,
    force_refresh: bool,
) -> tuple[str, bool, float | None]:
    """
    Return (summary, from_cache, latency_sec).
    Cache hit: instant. Regenerate: stream. First generate: single Ollama call.
    """
    cache_key = _summary_cache_key(patient_id, fingerprint, cache_bust)
    warmed_keys = st.session_state.get("summary_warmed_keys", set())

    if force_refresh:
        start = time.perf_counter()
        summary = st.write_stream(stream_summarize_patient(patient_data))
        latency = time.perf_counter() - start
        get_cached_summary(patient_id, patient_data, cache_bust, precomputed=summary)
        _mark_summary_warmed(cache_key)
        return summary, False, latency

    from_cache = cache_key in warmed_keys
    summary = get_cached_summary(patient_id, patient_data, cache_bust)
    if not from_cache:
        _mark_summary_warmed(cache_key)
    return summary, from_cache, None


def main() -> None:
    st.set_page_config(page_title="Tumor Board AI", layout="wide")
    st.title("Tumor Board AI")

    with st.sidebar:
        st.caption("Build")
        st.markdown(f"- Model: `{MODEL}`")
        st.markdown(f"- Prompt: `{PROMPT_VERSION}`")
        st.markdown("- Data: synthetic only")

    st.caption(
        "Synthetic oncology cases only. Summaries use local Ollama and must not "
        "invent clinical facts beyond the record."
    )

    ollama_ok = check_ollama_reachable()
    if not ollama_ok:
        st.warning(
            "Ollama is not reachable. Start it with `ollama serve`, ensure "
            f"`ollama pull {MODEL}` is done, then refresh this page."
        )

    with st.expander("Running on Apple Silicon (M1 Mac)"):
        st.markdown(
            "MedGemma 1.5 4B is appropriate for an M1 MacBook Air: Ollama uses "
            "Apple GPU/Neural Engine and roughly **3–5 GB** of unified memory while "
            "the model is loaded. Close other heavy apps during generation."
        )

    df = get_patients_df()
    labels = [
        f"{row['patient_id']} — {row['diagnosis']} (stage {row['stage']})"
        for _, row in df.iterrows()
    ]
    choice = st.selectbox("Select patient", labels, index=0)
    row = df.iloc[labels.index(choice)]
    patient_id = str(row["patient_id"])
    patient_data = format_patient_data(row)
    fingerprint = record_fingerprint(patient_data)

    bust_key = f"cache_bust_{patient_id}"

    col_record, col_summary = st.columns(2, gap="large")

    with col_record:
        st.subheader("Patient record")
        render_patient_record(row)

    with col_summary:
        st.subheader("Tumor board summary")

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            generate = st.button(
                "Generate",
                type="primary",
                disabled=not ollama_ok,
                use_container_width=True,
            )
        with btn_col2:
            regenerate = st.button(
                "Regenerate",
                disabled=not ollama_ok,
                use_container_width=True,
            )
        with btn_col3:
            clear_cache = st.button("Clear cache", use_container_width=True)

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
            try:
                summary, from_cache, latency = load_summary(
                    patient_id,
                    patient_data,
                    fingerprint,
                    st.session_state.get(bust_key, 0),
                    force_refresh=regenerate,
                )
            except Exception as exc:
                st.error(
                    "Could not reach Ollama. Ensure `ollama serve` is running and "
                    f"`ollama pull {MODEL}` is completed.\n\nDetails: {exc}"
                )
            else:
                st.session_state["summary"] = summary
                st.session_state["summary_meta"] = {
                    "patient_id": patient_id,
                    "fingerprint": fingerprint,
                    "from_cache": from_cache,
                    "latency_sec": latency,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

        meta = st.session_state.get("summary_meta", {})
        summary = st.session_state.get("summary")
        show_summary = (
            summary
            and meta.get("patient_id") == patient_id
            and meta.get("fingerprint") == fingerprint
        )

        if show_summary:
            if meta.get("from_cache"):
                st.info("Loaded from cache (same patient record).")
            elif meta.get("latency_sec") is not None:
                st.caption(
                    f"Generated in {meta['latency_sec']:.1f}s · {meta.get('generated_at', '')}"
                )

            display_summary(summary)

            st.download_button(
                "Download Markdown",
                data=summary_to_markdown(patient_id, summary),
                file_name=f"{patient_id}_tumor_board_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.markdown(
                "_Select a patient and click **Generate** to create an MDT summary._"
            )


if __name__ == "__main__":
    main()
