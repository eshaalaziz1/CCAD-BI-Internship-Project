"""Tumor Board Assist — synthetic MDT workspace."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.constants import MODEL
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
    check_ollama_reachable,
    stream_summarize_patient,
    summarize_patient,
)
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
    precomputed: str | None = None,
) -> str:
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


def summary_to_markdown(patient_id: str, summary: str) -> str:
    return "\n".join(
        [
            f"# MDT summary — {patient_id}",
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
    choice = st.radio(
        "Patients",
        labels,
        label_visibility="collapsed",
        key="sidebar_patient_radio",
    )
    st.session_state.selected_patient_label = choice
    return filtered.iloc[labels.index(choice)]


def page_patient_chart(df: pd.DataFrame, ollama_ok: bool) -> None:
    row = get_selected_row(df)
    patient_id = str(row["patient_id"])
    patient_data = format_patient_data(row)
    fingerprint = record_fingerprint(patient_data)
    bust_key = f"cache_bust_{patient_id}"

    render_profile_hero(row)

    tab_profile, tab_summary = st.tabs(["Patient profile", "MDT summary"])

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

    with tab_summary:
        if not ollama_ok:
            st.warning("AI summaries unavailable — start Ollama locally to enable generation.")

        c1, c2, c3 = st.columns(3)
        with c1:
            generate = st.button("Generate MDT brief", type="primary", disabled=not ollama_ok)
        with c2:
            regenerate = st.button("Regenerate", disabled=not ollama_ok)
        with c3:
            clear_cache = st.button("Reset cache")

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
                st.session_state["summary"] = summary
                st.session_state["summary_meta"] = {
                    "patient_id": patient_id,
                    "fingerprint": fingerprint,
                    "from_cache": from_cache,
                    "latency_sec": latency,
                }
            except Exception as exc:
                st.error(f"Summary generation failed: {exc}")

        meta = st.session_state.get("summary_meta", {})
        summary = st.session_state.get("summary")
        if (
            summary
            and meta.get("patient_id") == patient_id
            and meta.get("fingerprint") == fingerprint
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
    st.subheader("Synthetic case generator")
    st.caption(
        "AI-generated fictional cases for training (Synthia-style workflow). "
        "Review and edit before saving."
    )

    if not ollama_ok:
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
        page_title="Tumor Board Assist",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    version = st.session_state.get("patients_version", 0)
    df = get_patients_df(version)
    ollama_ok = check_ollama_reachable()

    with st.sidebar:
        st.markdown("## Tumor Board Assist")
        st.caption("Multidisciplinary review workspace")
        if not ollama_ok:
            st.error("Ollama offline — AI features disabled")

        row = sidebar_patient_picker(df)
        st.divider()
        nav = st.radio(
            "Workspace",
            ["Patient chart", "Add patient", "Synthetic intake"],
            label_visibility="collapsed",
        )

    if nav == "Patient chart":
        page_patient_chart(df, ollama_ok)
    elif nav == "Add patient":
        page_add_patient(df)
    else:
        page_synthetic_intake(df, ollama_ok)

    render_footer()


if __name__ == "__main__":
    main()
