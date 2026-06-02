"""Generate fictional patient records via Ollama (Synthia-style synthetic intake)."""

from __future__ import annotations

import json
import os
import re
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


def _extract_synthia_payload(payload: dict) -> dict:
    """Support common envelope formats returned by API-style providers."""
    if not isinstance(payload, dict):
        raise ValueError("Synthia response is not a JSON object.")
    for key in ("patient", "data", "result", "output"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return payload


def _generate_with_synthia_api(cancer_hint: str, constraints: str) -> dict:
    endpoint = os.getenv("SYNTHIA_API_URL", "").strip()
    api_key = os.getenv("SYNTHIA_API_KEY", "").strip()
    timeout_sec = int(os.getenv("SYNTHIA_TIMEOUT_SEC", "45"))
    model_name = os.getenv("SYNTHIA_MODEL", "synthia-med")

    if not endpoint or not api_key:
        raise RuntimeError(
            "Synthia is selected but not configured. Set SYNTHIA_API_URL and SYNTHIA_API_KEY."
        )

    payload = {
        "model": model_name,
        "task": "synthetic_oncology_patient",
        "response_format": "json",
        "input": {
            "cancer_hint": cancer_hint or "General solid tumor",
            "constraints": constraints or "None",
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            if response.status != HTTPStatus.OK:
                raise RuntimeError(f"Synthia API returned HTTP {response.status}")
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
        raise RuntimeError(f"Synthia API error ({exc.code}): {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Synthia API unreachable: {exc.reason}") from exc

    decoded = json.loads(raw)
    return _extract_synthia_payload(decoded)


def _generate_with_ollama(cancer_hint: str, constraints: str, model: str) -> dict:
    """Current local fallback until Synthia API credentials are configured."""
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
    return _extract_json(raw)


def generate_synthetic_patient(
    cancer_hint: str,
    constraints: str = "None",
    model: str = MODEL,
    provider: str | None = None,
) -> dict:
    """Produce one fictional patient JSON object via configured provider."""
    selected_provider = (provider or os.getenv("SYNTHETIC_GENERATOR_PROVIDER", "ollama")).lower()
    if selected_provider == "synthia":
        data = _generate_with_synthia_api(cancer_hint, constraints)
    else:
        data = _generate_with_ollama(cancer_hint, constraints, model)

    data["source"] = "synthetic"
    return normalize_record(data, source="synthetic")
