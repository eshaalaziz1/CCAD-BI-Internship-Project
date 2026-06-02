"""Ollama + MedGemma tumor board summarization."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path

import ollama

from src.constants import (
    CHAT_OPTIONS,
    KEEP_ALIVE,
    MAX_RETRIES,
    MODEL,
    PROMPT_VERSION,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "tumor_board_summary.txt"

SUMMARY_FIELDS = {
    "one_line_case_summary": "One-line case summary",
    "key_clinical_facts": "Key clinical facts",
    "missing_or_unclear_information": "Missing or unclear information",
    "mdt_discussion_questions": "MDT discussion questions",
    "treatment_considerations": "Treatment considerations",
}

__all__ = [
    "MODEL",
    "PROMPT_VERSION",
    "check_ollama_reachable",
    "summarize_patient",
    "stream_summarize_patient",
]


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(patient_data: str) -> str:
    return load_prompt_template().format(patient_data=patient_data)


def build_json_prompt(patient_data: str) -> str:
    return f"""
You are assisting a multidisciplinary oncology tumor board.

Use ONLY the patient information provided.
Do not invent missing biomarkers, imaging findings, pathology, staging, treatment history,
or clinical facts not present in the patient data.
Do not include internal reasoning, chain-of-thought, analysis notes, confidence scores,
constraint checklists, prompt text, or any extra sections.

Return a single valid JSON object with exactly these keys:
- one_line_case_summary: concise string
- key_clinical_facts: array of short strings
- missing_or_unclear_information: array of short strings
- mdt_discussion_questions: array of short strings
- treatment_considerations: array of short strings, phrased as possibilities not final recommendations

Patient data:
{patient_data}
""".strip()


def check_ollama_reachable() -> bool:
    """Return True if the Ollama server responds."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def _chat(prompt: str, model: str, stream: bool, *, json_mode: bool = False):
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": CHAT_OPTIONS,
        "keep_alive": KEEP_ALIVE,
        "stream": stream,
    }
    if json_mode:
        kwargs["format"] = "json"
    return ollama.chat(**kwargs)


def _strip_model_noise(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<unused\d+>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>\s]+>", "", text)
    return text.strip()


def _format_value(value) -> str:
    if isinstance(value, list):
        items = [_strip_model_noise(str(item)) for item in value]
        items = [item for item in items if item]
        return "\n".join(f"- {item}" for item in items)
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            text = _strip_model_noise(str(item))
            if text:
                items.append(f"- {key}: {text}")
        return "\n".join(items)
    return _strip_model_noise(str(value))


def _extract_json(text: str) -> dict[str, str] | None:
    cleaned = _strip_model_noise(text)
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            normalized = {
                key: _format_value(data.get(key, ""))
                for key in SUMMARY_FIELDS
            }
            if any(normalized.values()):
                return normalized
    return None


def _parse_numbered_summary(text: str) -> dict[str, str] | None:
    cleaned = _strip_model_noise(text)
    title_lookup = {title.lower(): key for key, title in SUMMARY_FIELDS.items()}
    title_pattern = "|".join(re.escape(title) for title in SUMMARY_FIELDS.values())
    pattern = re.compile(
        rf"(?:^|\n)\s*(?:\d+\.\s*)?({title_pattern})\s*:?\s*(.*?)(?=(?:\n\s*(?:\d+\.\s*)?(?:{title_pattern})\s*:?)|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    sections: dict[str, str] = {}
    for match in pattern.finditer(cleaned):
        key = title_lookup[match.group(1).lower()]
        value = _strip_model_noise(match.group(2))
        if value:
            sections[key] = value

    if sections:
        return {key: sections.get(key, "") for key in SUMMARY_FIELDS}
    return None


def _canonical_markdown(sections: dict[str, str]) -> str:
    parts = []
    for idx, (key, title) in enumerate(SUMMARY_FIELDS.items(), start=1):
        value = _strip_model_noise(sections.get(key, ""))
        if not value:
            value = "Not specified in the provided patient data."
        parts.append(f"{idx}. {title}\n{value}")
    return "\n\n".join(parts)


def _summarize_once(patient_data: str, model: str, *, json_mode: bool) -> tuple[str, str]:
    prompt = build_json_prompt(patient_data) if json_mode else build_prompt(patient_data)
    response = _chat(prompt, model, stream=False, json_mode=json_mode)
    raw = response["message"]["content"]
    parsed = _extract_json(raw) or _parse_numbered_summary(raw)
    if not parsed:
        raise ValueError(f"MedGemma returned an unusable brief: {_strip_model_noise(raw)[:160]}")
    return _canonical_markdown(parsed), raw


def summarize_patient(patient_data: str, model: str = MODEL) -> str:
    """Generate a validated five-section MDT brief with MedGemma."""
    last_error: Exception | None = None
    attempts = [True, False] * (MAX_RETRIES + 1)
    for attempt, json_mode in enumerate(attempts, start=1):
        try:
            start = time.perf_counter()
            text, raw = _summarize_once(patient_data, model, json_mode=json_mode)
            elapsed = time.perf_counter() - start
            logger.info(
                "summary_ok model=%s prompt=%s json_mode=%s latency_sec=%.2f chars=%d raw_chars=%d",
                model,
                PROMPT_VERSION,
                json_mode,
                elapsed,
                len(text),
                len(raw),
            )
            return text
        except Exception as exc:
            last_error = exc
            logger.warning(
                "summary_attempt_failed model=%s attempt=%d json_mode=%s error=%s",
                model,
                attempt,
                json_mode,
                exc,
            )
    raise last_error  # type: ignore[misc]


def stream_summarize_patient(
    patient_data: str, model: str = MODEL
) -> Iterator[str]:
    """Yield summary text chunks as they arrive from Ollama."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            start = time.perf_counter()
            stream = _chat(build_prompt(patient_data), model, stream=True)
            for chunk in stream:
                part = chunk.get("message", {}).get("content") or ""
                if part:
                    yield part
            elapsed = time.perf_counter() - start
            logger.info(
                "summary_stream_ok model=%s prompt=%s latency_sec=%.2f",
                model,
                PROMPT_VERSION,
                elapsed,
            )
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "summary_stream_attempt_failed model=%s attempt=%d error=%s",
                model,
                attempt + 1,
                exc,
            )
    raise last_error  # type: ignore[misc]
