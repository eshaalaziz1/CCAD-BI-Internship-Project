#!/usr/bin/env python3
"""Run MedGemma summaries for all synthetic patients and save JSONL results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import MODEL, PROMPT_VERSION  # noqa: E402
from src.summarizer import check_ollama_reachable, summarize_patient  # noqa: E402

CSV_PATH = PROJECT_ROOT / "data" / "synthetic_patients" / "sample_patients.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def format_patient_data(row: pd.Series) -> str:
    return "\n".join(f"{col}: {row[col]}" for col in row.index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch tumor board summary eval")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path (default: outputs/eval_<timestamp>.jsonl)",
    )
    args = parser.parse_args()

    if not check_ollama_reachable():
        print("Ollama is not reachable. Run: ollama serve", file=sys.stderr)
        return 1

    df = pd.read_csv(CSV_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or OUTPUT_DIR / (
        f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )

    run_meta = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "patient_count": len(df),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps({"run": run_meta}))

    with out_path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            patient_id = str(row["patient_id"])
            patient_data = format_patient_data(row)
            start = time.perf_counter()
            try:
                summary = summarize_patient(patient_data)
                status = "ok"
                error = None
            except Exception as exc:
                summary = None
                status = "error"
                error = str(exc)
            latency = time.perf_counter() - start

            record = {
                "patient_id": patient_id,
                "status": status,
                "latency_sec": round(latency, 2),
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "patient_data": patient_data,
                "summary": summary,
                "error": error,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{patient_id}: {status} ({latency:.1f}s)")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
