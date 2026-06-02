"""Generate fictional patient records via Ollama (Synthia-style synthetic intake)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ollama

from src.constants import CHAT_OPTIONS, KEEP_ALIVE, MODEL
from src.patients import normalize_record

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "synthetic_patient.txt"


def _load_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def generate_synthetic_patient(
    cancer_hint: str,
    constraints: str = "None",
    model: str = MODEL,
) -> dict:
    """Ask MedGemma to produce one fictional patient JSON object."""
    prompt = _load_template().format(
        cancer_hint=cancer_hint or "General solid tumor",
        constraints=constraints or "None",
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={**CHAT_OPTIONS, "num_predict": 900},
        keep_alive=KEEP_ALIVE,
    )
    raw = response["message"]["content"]
    data = _extract_json(raw)
    data["source"] = "synthetic"
    return normalize_record(data, source="synthetic")
