"""Ollama + MedGemma tumor board summarization."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

MODEL = "medgemma1.5"
PROMPT_VERSION = "v1"
TEMPERATURE = 0.2
NUM_PREDICT = 700
KEEP_ALIVE = "30m"
MAX_RETRIES = 1

CHAT_OPTIONS = {
    "temperature": TEMPERATURE,
    "num_predict": NUM_PREDICT,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "tumor_board_summary.txt"


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(patient_data: str) -> str:
    return load_prompt_template().format(patient_data=patient_data)


def check_ollama_reachable() -> bool:
    """Return True if the Ollama server responds."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def _chat(patient_data: str, model: str, stream: bool):
    return ollama.chat(
        model=model,
        messages=[{"role": "user", "content": build_prompt(patient_data)}],
        options=CHAT_OPTIONS,
        keep_alive=KEEP_ALIVE,
        stream=stream,
    )


def summarize_patient(patient_data: str, model: str = MODEL) -> str:
    """Generate a full summary with one retry on failure."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            start = time.perf_counter()
            response = _chat(patient_data, model, stream=False)
            elapsed = time.perf_counter() - start
            text = response["message"]["content"]
            logger.info(
                "summary_ok model=%s prompt=%s latency_sec=%.2f chars=%d",
                model,
                PROMPT_VERSION,
                elapsed,
                len(text),
            )
            return text
        except Exception as exc:
            last_error = exc
            logger.warning(
                "summary_attempt_failed model=%s attempt=%d error=%s",
                model,
                attempt + 1,
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
            stream = _chat(patient_data, model, stream=True)
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
