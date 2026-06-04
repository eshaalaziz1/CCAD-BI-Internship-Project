"""Generate fictional patient records via Ollama (Synthia-style synthetic intake)."""

from __future__ import annotations

import json
import logging
import os
import re
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import ollama

from src.constants import CHAT_OPTIONS, KEEP_ALIVE, MODEL
from src.patients import PATIENT_COLUMNS, normalize_record

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "synthetic_patient.txt"

JSON_KEYS = [col for col in PATIENT_COLUMNS if col != "source"]
SYNTHETIC_NUM_PREDICT = 1400


def _load_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _slice_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    if start >= 0:
        return text[start:]
    return text


def _escape_newlines_in_strings(blob: str) -> str:
    """Replace raw newlines inside double-quoted JSON strings with spaces."""
    out: list[str] = []
    in_string = False
    escape = False
    for ch in blob:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch in "\n\r":
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def _close_truncated_json(blob: str) -> str:
    """Best-effort close for cut-off model JSON."""
    blob = blob.strip()
    if blob.count('"') % 2 == 1:
        blob += '"'
    open_braces = blob.count("{") - blob.count("}")
    if open_braces > 0:
        blob = re.sub(r",\s*$", "", blob)
        blob += "}" * open_braces
    return blob


def _parse_loose_fields(text: str) -> dict:
    """Extract known patient fields from partially valid JSON text."""
    found: dict = {}
    for key in JSON_KEYS:
        str_match = re.search(
            rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if str_match:
            found[key] = str_match.group(1).replace("\\n", " ").replace('\\"', '"').strip()
            continue
        num_match = re.search(rf'"{re.escape(key)}"\s*:\s*(\d+)', text, flags=re.IGNORECASE)
        if num_match:
            found[key] = int(num_match.group(1))
            continue
        list_match = re.search(
            rf'"{re.escape(key)}"\s*:\s*(\[[^\]]*\])',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if list_match:
            try:
                found[key] = json.loads(list_match.group(1))
            except json.JSONDecodeError:
                found[key] = list_match.group(1)
    return found


def _extract_json(text: str) -> dict:
    raw = _strip_fences(text)
    candidates = [
        raw,
        _slice_json_object(raw),
        _escape_newlines_in_strings(_slice_json_object(raw)),
        _close_truncated_json(_escape_newlines_in_strings(_slice_json_object(raw))),
        _close_truncated_json(_slice_json_object(raw)),
    ]
    seen: set[str] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            last_error = exc

    loose = _parse_loose_fields(raw)
    if len(loose) >= 6:
        logger.warning("synthetic_json_recovered_via_loose_parse keys=%s", list(loose.keys()))
        return loose

    raise ValueError(
        f"Could not parse synthetic patient JSON ({last_error}). Try again or shorten constraints."
    ) from last_error


def _extract_synthia_payload(payload: dict) -> dict:
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
    prompt = _load_template().format(
        cancer_hint=cancer_hint or "General solid tumor",
        constraints=constraints or "None",
    )
    options = {**CHAT_OPTIONS, "num_predict": SYNTHETIC_NUM_PREDICT, "temperature": 0.15}
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive=KEEP_ALIVE,
                format="json",
            )
            raw = response["message"]["content"]
            return _extract_json(raw)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "synthetic_generation_attempt_failed attempt=%d error=%s",
                attempt + 1,
                exc,
            )
            options = {**options, "num_predict": SYNTHETIC_NUM_PREDICT + 400}

    raise ValueError(
        f"Synthetic patient generation failed after retries: {last_error}"
    ) from last_error


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
