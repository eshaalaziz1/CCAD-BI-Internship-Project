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

REPORT_ANALYSIS_FIELDS = {
    "patient_snapshot": "Patient snapshot",
    "meeting_objective": "Meeting objective",
    "critical_facts": "Critical facts",
    "priority_problems": "Priority problems",
    "specialist_focus": "Specialist focus",
    "missing_data": "Missing data",
    "red_flags": "Red flags",
    "decision_points": "Decision points",
    "meeting_flow": "Suggested meeting flow",
}

__all__ = [
    "MODEL",
    "PROMPT_VERSION",
    "analyze_report",
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


def build_report_analysis_prompt(report_text: str) -> str:
    clipped_report = _clean_report_noise(report_text)[:18000]
    return f"""
You are assisting a multidisciplinary medical board preparing for a time-limited patient discussion.

Use ONLY the report text provided.
Do not invent labs, imaging, pathology, staging, treatment history, diagnoses, or recommendations
that are not supported by the report.
Do not include internal reasoning, chain-of-thought, confidence scores, prompt text, or extra sections.

Return exactly these numbered sections in Markdown:

1. Patient snapshot
Name/ID: ...
Age: ...
Sex: ...
Presenting problem: ...
Likely primary issue: ...

2. Meeting objective
One sentence.

3. Critical facts
- Up to 8 bullets.

4. Priority problems
- Problem: ... | Evidence: ... | Why it matters: ...

5. Specialist focus
- Specialist: ... | Review: ...

6. Missing data
- Up to 6 bullets.

7. Red flags
- Up to 5 bullets.

8. Decision points
- Up to 6 bullets.

9. Suggested meeting flow
- Up to 5 bullets.

Report text:
{clipped_report}
""".strip()


def check_ollama_reachable() -> bool:
    """Return True if the Ollama server responds."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def _chat(
    prompt: str,
    model: str,
    stream: bool,
    *,
    json_mode: bool = False,
    options: dict | None = None,
):
    chat_options = dict(CHAT_OPTIONS)
    if options:
        chat_options.update(options)
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": chat_options,
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


def _clean_report_noise(text: str) -> str:
    annotation_patterns = [
        r"\bDefine the\b.*$",
        r"\bspecifically as possible\b.*$",
        r"\bConvey the\b.*$",
        r"\bestablish a chronology\b.*$",
        r"\bcircumstances; exacerbating factors\b.*$",
        r"\bresolution; alleviating factors\b.*$",
        r"\bDescribe the natural history\b.*$",
        r"\bChange or new circumstances\b.*$",
        r"\bNew duration\b.*$",
        r"\bReason she come in\b.*$",
        r"\bWhat has patient tried\b.*$",
        r"\bRelevant positive\b.*$",
        r"\bReview of systems for the relevant\b.*$",
        r"\bRelevant risk factor\b.*$",
        r"\bThis highly relevant\b.*$",
        r"\btrivial detail\b.*$",
        r"\bAlways\b.*$",
        r"\bQuantity\b.*$",
        r"\bInclude over-the-counter\b.*$",
        r"\bComment specifically\b.*$",
        r"\bSeparate each ROS\b.*$",
        r"\bOK to refer\b.*$",
        r"\bList positive and negative\b.*$",
        r"\bCheck for orthostatic\b.*$",
        r"\bDescription may give\b.*$",
        r"\bComment on all organ systems\b.*$",
        r"\bList specific normal\b.*$",
        r"\bThis patient needs\b.*$",
        r"\bMore precise\b.*$",
        r"\bAlthough you can\b.*$",
        r"\bshown below\b.*$",
        r"\bto keep track\b.*$",
        r"\bThis list regroups\b.*$",
        r"\bsuspect are related\b.*$",
        r"\bYou should\b.*$",
    ]
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        for pattern in annotation_patterns:
            line = re.sub(pattern, "", line, flags=re.IGNORECASE).strip()
        line = re.sub(r"\s+(onset|character|location|radiation|duration)$", "", line, flags=re.IGNORECASE)
        if line.lower() == "history and physical examination comments":
            continue
        if line.lower() in {"onset", "character", "location", "radiation", "duration"}:
            continue
        if line:
            lines.append(line)
    return "\n".join(lines)


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


def _extract_report_json(text: str) -> dict | None:
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
        if isinstance(data, dict) and any(data.get(key) for key in REPORT_ANALYSIS_FIELDS):
            return data
    return None


def _bullet_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^(?:[-*]\s+|\d+[.)]\s+)(.+)$", stripped)
        lines.append(match.group(1).strip() if match else stripped)
    return lines


def _parse_key_value_lines(text: str) -> dict:
    data = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower().replace("/", "_or_").replace(" ", "_")
        data[normalized] = value.strip()
    return data


def _parse_report_markdown(text: str) -> dict | None:
    cleaned = _strip_model_noise(text)
    section_lookup = {title.lower(): key for key, title in REPORT_ANALYSIS_FIELDS.items()}
    title_pattern = "|".join(re.escape(title) for title in REPORT_ANALYSIS_FIELDS.values())
    pattern = re.compile(
        rf"(?:^|\n)\s*(?:\d+\.\s*)?({title_pattern})\s*:?\s*(.*?)(?=(?:\n\s*(?:\d+\.\s*)?(?:{title_pattern})\s*:?)|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    blocks: dict[str, str] = {}
    for match in pattern.finditer(cleaned):
        key = section_lookup[match.group(1).lower()]
        value = _strip_model_noise(match.group(2))
        if value:
            blocks[key] = value

    if not blocks:
        return None

    result = {
        "patient_snapshot": _parse_key_value_lines(blocks.get("patient_snapshot", "")),
        "meeting_objective": " ".join(_bullet_lines(blocks.get("meeting_objective", ""))),
        "critical_facts": _bullet_lines(blocks.get("critical_facts", "")),
        "priority_problems": _bullet_lines(blocks.get("priority_problems", "")),
        "specialist_focus": _bullet_lines(blocks.get("specialist_focus", "")),
        "missing_data": _bullet_lines(blocks.get("missing_data", "")),
        "red_flags": _bullet_lines(blocks.get("red_flags", "")),
        "decision_points": _bullet_lines(blocks.get("decision_points", "")),
        "meeting_flow": _bullet_lines(blocks.get("meeting_flow", "")),
    }
    return result if any(result.values()) else None


def _heuristic_report_analysis(report_text: str) -> dict:
    """Last-resort report extraction so upload never fails with a blank page."""
    text = _clean_report_noise(_strip_model_noise(report_text))
    name_match = re.search(r"Patient Name:\s*([^\n]+)", text, flags=re.IGNORECASE)
    age_sex_match = re.search(r"(\d{1,3})\s*y/o\s*([A-Z]+)", text, flags=re.IGNORECASE)
    chief_match = re.search(r"Chief Complaint.*?:\s*(.*?)(?:\n\s*\n|History of Present Illness)", text, flags=re.IGNORECASE | re.DOTALL)
    problem_section = re.search(r"Revised Problem List\s*(.*?)(?:Assessment and Differential Diagnosis|Plan:)", text, flags=re.IGNORECASE | re.DOTALL)
    plan_section = re.search(r"Plan:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)

    problems = _bullet_lines(problem_section.group(1))[:8] if problem_section else []
    plan_items = _bullet_lines(plan_section.group(1))[:6] if plan_section else []

    return {
        "patient_snapshot": {
            "name_or_id": name_match.group(1).strip() if name_match else "Uploaded report",
            "age": age_sex_match.group(1) if age_sex_match else "N/A",
            "sex": age_sex_match.group(2) if age_sex_match else "N/A",
            "presenting_problem": " ".join(chief_match.group(1).split()) if chief_match else "Not specified in the report.",
            "likely_primary_issue": problems[0] if problems else "Not specified in the report.",
        },
        "meeting_objective": "Review the key problems, risks, missing data, and immediate plan from the uploaded report.",
        "critical_facts": problems[:6] or ["No structured problem list was extracted."],
        "priority_problems": problems[:6],
        "specialist_focus": [
            "Primary team: confirm active problems and immediate management plan.",
            "Relevant specialists: review disease-specific risks, diagnostics, and treatment options.",
        ],
        "missing_data": ["Confirm labs, imaging, medication history, and pending tests if not included in the report."],
        "red_flags": [item for item in problems if re.search(r"chest pain|dyspnea|hypertension|murmur|bruit", item, re.IGNORECASE)][:5],
        "decision_points": plan_items[:6] or ["Clarify immediate diagnostic and therapeutic next steps."],
        "meeting_flow": [
            "Start with the presenting problem and current risk level.",
            "Review priority problems and supporting evidence.",
            "Identify missing information needed before final decisions.",
            "Assign specialty-specific follow-up questions.",
            "End with the agreed plan and next actions.",
        ],
    }


def _report_section_prompt(report_text: str, task: str, output_rule: str) -> str:
    clipped_report = _clean_report_noise(report_text)[:16000]
    return f"""
You are helping doctors prepare for a short multidisciplinary medical meeting.

Use ONLY the report text provided. Do not invent facts. Do not include hidden reasoning,
confidence scores, or prompt text.

Task:
{task}

Output rule:
{output_rule}

Report text:
{clipped_report}
""".strip()


def _run_report_section(report_text: str, task: str, output_rule: str, *, num_predict: int = 700) -> str:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 2):
        try:
            response = _chat(
                _report_section_prompt(report_text, task, output_rule),
                MODEL,
                stream=False,
                json_mode=False,
                options={"num_predict": num_predict, "temperature": 0.1, "num_ctx": 8192},
            )
            text = _strip_model_noise(response["message"]["content"])
            if text and text not in {"{", "}", "[]"}:
                return text
            raise ValueError(f"empty or unusable section output: {text!r}")
        except Exception as exc:
            last_error = exc
            logger.warning("report_section_failed attempt=%d task=%s error=%s", attempt + 1, task[:48], exc)
    raise last_error  # type: ignore[misc]


def _parse_problem_rows(lines: list[str]) -> list[dict] | list[str]:
    rows = []
    fallback = []
    for line in lines:
        parts = {}
        for chunk in line.split("|"):
            if ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            parts[key.strip().lower().replace(" ", "_")] = value.strip()
        if parts:
            rows.append(
                {
                    "problem": parts.get("problem", ""),
                    "evidence": parts.get("evidence", ""),
                    "why_it_matters": parts.get("why_it_matters", parts.get("why it matters", "")),
                }
            )
        else:
            fallback.append(line)
    return rows if rows else fallback


def _parse_focus_rows(lines: list[str]) -> list[dict] | list[str]:
    rows = []
    fallback = []
    for line in lines:
        parts = {}
        for chunk in line.split("|"):
            if ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            parts[key.strip().lower().replace(" ", "_")] = value.strip()
        if parts:
            rows.append(
                {
                    "specialist": parts.get("specialist", ""),
                    "what_they_need_to_review": parts.get("review", parts.get("what_they_need_to_review", "")),
                }
            )
        else:
            fallback.append(line)
    return rows if rows else fallback


def analyze_report(report_text: str, model: str = MODEL) -> dict:
    """Turn a long clinical report into a meeting-ready board briefing."""
    report_text = _clean_report_noise(report_text)
    start = time.perf_counter()
    try:
        snapshot_text = _run_report_section(
            report_text,
            "Extract the patient snapshot.",
            "Return exactly five lines: Name/ID: ... | Age: ... | Sex: ... | Presenting problem: ... | Likely primary issue: ...",
            num_predict=260,
        )
        objective_text = _run_report_section(
            report_text,
            "State the meeting objective.",
            "Return one concise sentence only.",
            num_predict=180,
        )
        facts_text = _run_report_section(
            report_text,
            "Extract the most decision-relevant clinical facts.",
            "Return 5 to 8 bullet points. Each bullet must be supported by the report.",
            num_predict=650,
        )
        problems_text = _run_report_section(
            report_text,
            "Identify the priority problems for discussion.",
            "Return up to 6 bullets using exactly: Problem: ... | Evidence: ... | Why it matters: ...",
            num_predict=850,
        )
        focus_text = _run_report_section(
            report_text,
            "Map the case to specialists who should contribute to the meeting.",
            "Return up to 5 bullets using exactly: Specialist: ... | Review: ...",
            num_predict=650,
        )
        missing_text = _run_report_section(
            report_text,
            "Identify missing information needed before a confident decision.",
            "Return up to 6 bullet points. If none, return '- Not specified in the report.'",
            num_predict=500,
        )
        red_flags_text = _run_report_section(
            report_text,
            "Identify urgent or high-risk red flags.",
            "Return up to 5 bullet points. If none, return '- No urgent red flags extracted from the report.'",
            num_predict=450,
        )
        decisions_text = _run_report_section(
            report_text,
            "List the decision points the team should answer.",
            "Return up to 6 bullet points.",
            num_predict=550,
        )
        flow_text = _run_report_section(
            report_text,
            "Create a practical meeting flow for discussing this patient quickly.",
            "Return 4 to 5 numbered or bulleted steps.",
            num_predict=450,
        )

        analysis = {
            "patient_snapshot": _parse_key_value_lines(snapshot_text.replace("|", "\n")),
            "meeting_objective": " ".join(_bullet_lines(objective_text)),
            "critical_facts": _bullet_lines(facts_text),
            "priority_problems": _parse_problem_rows(_bullet_lines(problems_text)),
            "specialist_focus": _parse_focus_rows(_bullet_lines(focus_text)),
            "missing_data": _bullet_lines(missing_text),
            "red_flags": _bullet_lines(red_flags_text),
            "decision_points": _bullet_lines(decisions_text),
            "meeting_flow": _bullet_lines(flow_text),
        }
        logger.info(
            "report_analysis_ok model=%s prompt=%s latency_sec=%.2f sectioned=true",
            model,
            PROMPT_VERSION,
            time.perf_counter() - start,
        )
        return analysis
    except Exception as exc:
        logger.warning("report_analysis_falling_back_to_heuristic error=%s", exc)
        return _heuristic_report_analysis(report_text)


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
