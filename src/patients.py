"""Load, save, and manage synthetic patient records."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "synthetic_patients" / "sample_patients.csv"
CUSTOM_PATH = PROJECT_ROOT / "data" / "synthetic_patients" / "custom_patients.json"

PATIENT_COLUMNS = [
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
    "intake_text",
    "source",
]

REQUIRED_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "diagnosis",
    "stage",
    "ecog",
]


def _empty_custom_file() -> None:
    CUSTOM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CUSTOM_PATH.exists():
        CUSTOM_PATH.write_text("[]", encoding="utf-8")


def load_custom_patients() -> list[dict]:
    _empty_custom_file()
    data = json.loads(CUSTOM_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return data


def save_custom_patients(records: list[dict]) -> None:
    _empty_custom_file()
    CUSTOM_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_builtin_patients() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["source"] = "reference"
    if "intake_text" not in df.columns:
        df["intake_text"] = ""
    return df


def load_all_patients() -> pd.DataFrame:
    builtin = load_builtin_patients()
    custom_rows = load_custom_patients()
    if not custom_rows:
        return builtin

    custom = pd.DataFrame(custom_rows)
    for col in PATIENT_COLUMNS:
        if col not in custom.columns:
            custom[col] = ""
    custom = custom[PATIENT_COLUMNS]
    return pd.concat([builtin, custom], ignore_index=True)


def _scalar_value(value) -> str | int | float | bool | None:
    """Coerce list/array fields from LLM JSON into a single scalar for normalization."""
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        parts = [_scalar_value(item) for item in value]
        return "; ".join(str(part) for part in parts if part not in (None, ""))
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            flat = value.flatten().tolist()
            return "; ".join(str(_scalar_value(item)) for item in flat if item is not None)
    except ImportError:
        pass
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return value.item()
    return value


def coerce_int(value, default: int = 0) -> int:
    value = _scalar_value(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return default
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else default


def _field_text(value) -> str:
    value = _scalar_value(value)
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def next_patient_id(df: pd.DataFrame) -> str:
    ids = df["patient_id"].astype(str)
    numbers = []
    for pid in ids:
        match = re.match(r"P(\d+)", pid, re.IGNORECASE)
        if match:
            numbers.append(int(match.group(1)))
    n = max(numbers, default=0) + 1
    return f"P{n:03d}"


def normalize_record(record: dict, *, source: str) -> dict:
    row = {col: "" for col in PATIENT_COLUMNS}
    row.update({k: record.get(k, "") for k in record})
    row["source"] = source
    row["patient_id"] = _field_text(row["patient_id"])
    row["age"] = coerce_int(row["age"], default=0)
    row["ecog"] = min(4, max(0, coerce_int(row["ecog"], default=0)))
    sex = _field_text(row["sex"]).upper()[:1]
    row["sex"] = sex if sex in "FMU" else "U"
    for col in PATIENT_COLUMNS:
        if col in ("patient_id", "age", "ecog", "source", "sex"):
            continue
        row[col] = _field_text(row[col])
    return row


def add_custom_patient(record: dict) -> pd.DataFrame:
    df = load_all_patients()
    normalized = normalize_record(record, source=record.get("source", "custom"))
    if not normalized["patient_id"]:
        normalized["patient_id"] = next_patient_id(df)

    if normalized["patient_id"] in df["patient_id"].astype(str).values:
        raise ValueError(f"Patient ID {normalized['patient_id']} already exists.")

    custom = load_custom_patients()
    custom.append(normalized)
    save_custom_patients(custom)
    return load_all_patients()


def delete_custom_patient(patient_id: str) -> pd.DataFrame:
    custom = [r for r in load_custom_patients() if r.get("patient_id") != patient_id]
    if len(custom) == len(load_custom_patients()):
        raise ValueError("Only custom or synthetic-added patients can be deleted.")
    save_custom_patients(custom)
    return load_all_patients()


def format_patient_data(row: pd.Series) -> str:
    lines: list[str] = []
    skip = {"intake_text", "source"}
    for col in PATIENT_COLUMNS:
        if col in skip:
            continue
        value = row.get(col, "")
        if pd.isna(value) or value == "":
            continue
        lines.append(f"{col}: {value}")
    intake = row.get("intake_text", "")
    if intake and not pd.isna(intake) and str(intake).strip():
        lines.append(f"additional_clinical_narrative:\n{intake}")
    return "\n".join(lines)


def patient_profile_label(row: pd.Series) -> str:
    return (
        f"{row['patient_id']} · {row['diagnosis']} "
        f"(stage {row['stage']}) · {row['age']}y {row['sex']}"
    )
