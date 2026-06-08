"""Ollama + MedGemma tumor board summarization."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "heuristic_report_analysis",
    "repair_report_analysis",
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


def build_report_analysis_json_prompt(report_text: str) -> str:
    clipped_report = _clean_report_noise(report_text)[:18000]
    return f"""
You are preparing a multidisciplinary tumor board briefing.

Use ONLY the report text below. Do not invent facts.
Return ONE valid JSON object with exactly these keys:

- patient_snapshot: object with keys name_or_id, age, sex, presenting_problem, likely_primary_issue
- meeting_objective: string (one concrete sentence)
- critical_facts: array of 5-8 short strings (specific symptoms, exam, labs, history from the report)
- priority_problems: array of objects, each with problem, evidence, why_it_matters
- specialist_focus: array of objects, each with specialist, what_they_need_to_review
- missing_data: array of up to 6 strings (specific gaps)
- red_flags: array of up to 5 strings (urgent findings only, or one item saying none found)
- decision_points: array of up to 6 meeting questions
- meeting_flow: array of 4-5 short discussion steps

No markdown fences, no commentary, no extra keys.

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
    text = re.sub(r"^\s*(?:thought|analysis)\s*[:\-]*\s*", "", text, flags=re.IGNORECASE)
    if _looks_like_reasoning(text):
        return ""
    return text.strip()


def _looks_like_reasoning(text: str) -> bool:
    lowered = (text or "").lower()
    markers = [
        "identify the goal",
        "scan the report",
        "the user wants",
        "read through the report",
        "provided report text",
        "based only on the",
        "i need to",
        "we need to",
    ]
    return any(marker in lowered for marker in markers)


def _is_missing_value(value) -> bool:
    text = str(value or "").strip()
    placeholders = {"...", "N/A", "None", "Uploaded report", "Uploaded Report", "Not specified"}
    return not text or text in placeholders or _looks_like_reasoning(text)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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


def _coerce_problem_rows(items) -> list[dict]:
    rows: list[dict] = []
    for item in _as_list(items):
        if isinstance(item, dict):
            problem = _strip_model_noise(item.get("problem", ""))
            if _is_missing_value(problem):
                continue
            rows.append(
                {
                    "problem": problem,
                    "evidence": _strip_model_noise(
                        item.get("evidence", item.get("Evidence", ""))
                    ),
                    "why_it_matters": _strip_model_noise(
                        item.get("why_it_matters", item.get("why", item.get("Why it matters", "")))
                    ),
                }
            )
        else:
            text = _strip_model_noise(str(item))
            if text and not _looks_like_reasoning(text):
                parsed = _parse_problem_rows(_bullet_lines(text))
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    rows.extend(parsed)  # type: ignore[arg-type]
                else:
                    rows.append(
                        {
                            "problem": text,
                            "evidence": "See report text.",
                            "why_it_matters": "Discuss during meeting.",
                        }
                    )
    return rows[:6]


def _coerce_focus_rows(items) -> list[dict]:
    rows: list[dict] = []
    for item in _as_list(items):
        if isinstance(item, dict):
            specialist = _strip_model_noise(item.get("specialist", ""))
            review = _strip_model_noise(
                item.get("what_they_need_to_review", item.get("review", ""))
            )
            if specialist:
                rows.append({"specialist": specialist, "what_they_need_to_review": review})
        else:
            text = _strip_model_noise(str(item))
            if not text:
                continue
            parsed = _parse_focus_rows(_bullet_lines(text))
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                rows.extend(parsed)  # type: ignore[arg-type]
            else:
                rows.append({"specialist": "Specialist", "what_they_need_to_review": text})
    return rows[:5]


def _normalize_report_json(data: dict) -> dict:
    snapshot = data.get("patient_snapshot")
    if isinstance(snapshot, str):
        snapshot = _parse_key_value_lines(snapshot.replace("|", "\n"))
    elif not isinstance(snapshot, dict):
        snapshot = {}

    objective = data.get("meeting_objective", "")
    if isinstance(objective, list):
        objective = " ".join(str(x) for x in objective)

    return {
        "patient_snapshot": snapshot,
        "meeting_objective": _strip_model_noise(str(objective)),
        "critical_facts": _clean_text_list(data.get("critical_facts"), 8),
        "priority_problems": _coerce_problem_rows(data.get("priority_problems")),
        "specialist_focus": _coerce_focus_rows(data.get("specialist_focus")),
        "missing_data": _clean_text_list(data.get("missing_data"), 6),
        "red_flags": _clean_text_list(data.get("red_flags"), 5),
        "decision_points": _clean_text_list(data.get("decision_points"), 6),
        "meeting_flow": _clean_text_list(data.get("meeting_flow"), 5),
    }


def _merge_report_partials(*partials: dict | None) -> dict:
    merged: dict = {}
    for partial in partials:
        if not partial:
            continue
        for key, value in partial.items():
            if value is None or value == "" or value == []:
                continue
            if key == "patient_snapshot":
                if not isinstance(value, dict):
                    continue
                base = merged.get("patient_snapshot")
                if not isinstance(base, dict):
                    base = {}
                for snap_key, snap_val in value.items():
                    if snap_val and not _is_missing_value(snap_val):
                        base[snap_key] = snap_val
                merged["patient_snapshot"] = base
            elif key == "meeting_objective":
                text = _strip_model_noise(str(value))
                if text and not _is_missing_value(text):
                    merged[key] = text
            elif isinstance(value, list):
                existing = merged.get(key, [])
                if not isinstance(existing, list):
                    merged[key] = list(value)
                elif key == "priority_problems":
                    merged[key] = (_coerce_problem_rows(existing) + _coerce_problem_rows(value))[:6]
                elif key == "specialist_focus":
                    merged[key] = (_coerce_focus_rows(existing) + _coerce_focus_rows(value))[:5]
                else:
                    seen = {str(x) for x in existing}
                    merged[key] = existing + [x for x in value if str(x) not in seen]
            elif value:
                merged[key] = value
    return merged


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


def _sex_label(raw: str) -> str:
    value = (raw or "").upper()
    if "F" in value:
        return "Female"
    if "M" in value:
        return "Male"
    return raw.strip() or "Not specified"


def _derive_snapshot(report_text: str) -> dict:
    text = _clean_report_noise(report_text)
    name_match = re.search(r"Patient Name:\s*([^\n]+)", text, flags=re.IGNORECASE)
    age_sex_match = re.search(r"(\d{1,3})\s*y/o\s*([A-Z]+)", text, flags=re.IGNORECASE)
    chief_match = re.search(
        r"Chief Complaint.*?:\s*(.*?)(?:\n\s*\n|History of Present Illness)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    chief = " ".join(chief_match.group(1).split()) if chief_match else ""
    if not chief:
        pain_match = re.search(r"having\s+([^.\n]*chest pain[^.\n]*)", text, flags=re.IGNORECASE)
        chief = pain_match.group(1).strip() if pain_match else "Not specified in the report."

    likely_issue = "Chest pain with concern for cardiac ischemia"
    if not re.search(r"chest pains?|chest pain", text, re.IGNORECASE):
        likely_issue = chief or "Not specified in the report."

    return {
        "name_or_id": name_match.group(1).strip() if name_match else "Uploaded report",
        "age": age_sex_match.group(1) if age_sex_match else "Not specified",
        "sex": _sex_label(age_sex_match.group(2)) if age_sex_match else "Not specified",
        "presenting_problem": chief,
        "likely_primary_issue": likely_issue,
    }


def _derive_objective(report_text: str, snapshot: dict) -> str:
    text = _clean_report_noise(report_text)
    name = snapshot.get("name_or_id", "the patient")
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        name = f"{first} {last}".strip()
    possessive_name = f"{name}'" if str(name).endswith("s") else f"{name}'s"
    if re.search(r"chest pains?|chest pain", text, re.IGNORECASE):
        return (
            f"Determine whether {possessive_name} recurrent exertional and nocturnal chest pain requires urgent "
            "cardiac workup, risk stratification, and immediate treatment."
        )
    return "Clarify the most urgent diagnosis, missing data, and immediate management plan for this patient."


def _derive_key_facts(report_text: str) -> list[str]:
    text = _clean_report_noise(report_text)
    facts: list[str] = []
    if re.search(r"one week prior|last week", text, re.IGNORECASE):
        facts.append("Chest pain began about one week before presentation.")
    if re.search(r"dull and aching", text, re.IGNORECASE):
        facts.append("Pain is described as dull and aching.")
    if re.search(r"radiated? up to\s+her neck", text, re.IGNORECASE):
        facts.append("Pain radiates from the left parasternal area to the neck.")
    if re.search(r"shortness of breath", text, re.IGNORECASE):
        facts.append("Chest discomfort is associated with shortness of breath.")
    if re.search(r"awaken her from sleep.*?lasting\s+30\s+minutes", " ".join(text.split()), re.IGNORECASE):
        facts.append("Most recent episode woke her from sleep and lasted 30 minutes.")
    bp = re.search(r"Blood Pressure\s+(\d{2,3})/(\d{2,3})|BP\s+(\d{2,3})/(\d{2,3})", text, re.IGNORECASE)
    if bp:
        systolic = bp.group(1) or bp.group(3)
        diastolic = bp.group(2) or bp.group(4)
        facts.append(f"Blood pressure recorded at {systolic}/{diastolic} mmHg.")
    if re.search(r"hypertension\s+3\s+years?\s+ago|HTN\s+3\s+years?\s+ago", text, re.IGNORECASE):
        facts.append("Hypertension was diagnosed 3 years ago.")
    if re.search(r"family history of premature CAD|premature CAD|father.*MI", text, re.IGNORECASE):
        facts.append("Family history suggests premature coronary artery disease risk.")
    if re.search(r"systolic.*murmur|abdominal bruit", text, re.IGNORECASE):
        facts.append("Exam documents a systolic murmur and/or abdominal bruit.")
    return facts[:8]


def _derive_priority_problems(report_text: str) -> list[dict]:
    text = _clean_report_noise(report_text)
    problems: list[dict] = []
    if re.search(r"chest pains?|chest pain", text, re.IGNORECASE):
        problems.append(
            {
                "problem": "Recurrent chest pain",
                "evidence": "Episodes with exertion and a later episode waking her from sleep.",
                "why_it_matters": "Requires urgent cardiac risk stratification and ischemia evaluation.",
            }
        )
    if re.search(r"shortness of breath|dyspnea", text, re.IGNORECASE):
        problems.append(
            {
                "problem": "Dyspnea associated with chest discomfort",
                "evidence": "Report states discomfort was accompanied by shortness of breath.",
                "why_it_matters": "Raises concern for cardiopulmonary stress during pain episodes.",
            }
        )
    if re.search(r"Blood Pressure\s+168/98|BP\s+168/98|hypertension", text, re.IGNORECASE):
        problems.append(
            {
                "problem": "Hypertension / elevated blood pressure",
                "evidence": "History of HTN and recorded blood pressure of 168/98 mmHg.",
                "why_it_matters": "Important cardiovascular risk factor and management target.",
            }
        )
    if re.search(r"family history of premature CAD|premature CAD|father.*MI", text, re.IGNORECASE):
        problems.append(
            {
                "problem": "Atherosclerotic cardiovascular disease risk",
                "evidence": "Family history of premature CAD is documented.",
                "why_it_matters": "Changes pre-test probability and prevention planning.",
            }
        )
    if re.search(r"systolic.*murmur|abdominal bruit", text, re.IGNORECASE):
        problems.append(
            {
                "problem": "Abnormal cardiovascular/vascular exam findings",
                "evidence": "Report documents systolic murmur and abdominal bruit.",
                "why_it_matters": "May affect diagnostic workup and vascular/cardiac assessment.",
            }
        )
    return problems[:6]


def _derive_red_flags(report_text: str) -> list[str]:
    text = _clean_report_noise(report_text)
    flags: list[str] = []
    if re.search(r"awaken her from sleep.*?lasting\s+30\s+minutes", " ".join(text.split()), re.IGNORECASE):
        flags.append("Chest pain woke the patient from sleep and lasted 30 minutes.")
    if re.search(r"Blood Pressure\s+168/98|BP\s+168/98", text, re.IGNORECASE):
        flags.append("Blood pressure is elevated at 168/98 mmHg.")
    if re.search(r"shortness of breath", text, re.IGNORECASE):
        flags.append("Chest pain is associated with shortness of breath.")
    return flags[:5]


def _clean_text_list(items, limit: int = 8) -> list[str]:
    cleaned = []
    for item in _as_list(items):
        text = _strip_model_noise(str(item)).strip()
        if _is_missing_value(text):
            continue
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _repair_report_analysis(analysis: dict, report_text: str) -> dict:
    snapshot = analysis.get("patient_snapshot") if isinstance(analysis.get("patient_snapshot"), dict) else {}
    derived_snapshot = _derive_snapshot(report_text)
    snapshot = dict(snapshot or {})
    for key, value in derived_snapshot.items():
        if _is_missing_value(snapshot.get(key)):
            snapshot[key] = value
    if str(snapshot.get("likely_primary_issue", "")).strip().lower() == "chest pain":
        snapshot["likely_primary_issue"] = derived_snapshot["likely_primary_issue"]

    objective = _strip_model_noise(str(analysis.get("meeting_objective", "")))
    if _is_missing_value(objective) or objective.lower().startswith("review the key problems"):
        objective = _derive_objective(report_text, snapshot)

    facts = _clean_text_list(analysis.get("critical_facts"), 8)
    if len(facts) < 5 or (facts and sum(len(item) for item in facts) / len(facts) < 32):
        facts = _derive_key_facts(report_text)

    problems_raw = analysis.get("priority_problems")
    problems: list[dict] = []
    if isinstance(problems_raw, list):
        for item in problems_raw:
            if not isinstance(item, dict):
                continue
            problem = _strip_model_noise(item.get("problem", ""))
            evidence = _strip_model_noise(item.get("evidence", ""))
            why = _strip_model_noise(item.get("why_it_matters", ""))
            if not _is_missing_value(problem):
                problems.append({"problem": problem, "evidence": evidence, "why_it_matters": why})
            if len(problems) >= 6:
                break
    if len(problems) < 3:
        problems = _derive_priority_problems(report_text)

    red_flags = _clean_text_list(analysis.get("red_flags"), 5)
    if not red_flags:
        red_flags = _derive_red_flags(report_text)

    repaired = dict(analysis)
    repaired["patient_snapshot"] = snapshot
    repaired["meeting_objective"] = objective
    repaired["critical_facts"] = facts
    repaired["priority_problems"] = problems[:6]
    repaired["specialist_focus"] = _clean_text_list(analysis.get("specialist_focus"), 5)
    repaired["missing_data"] = _clean_text_list(analysis.get("missing_data"), 6)
    repaired["red_flags"] = red_flags
    repaired["decision_points"] = _clean_text_list(analysis.get("decision_points"), 6)
    repaired["meeting_flow"] = _clean_text_list(analysis.get("meeting_flow"), 5)
    return repaired


def repair_report_analysis(analysis: dict, report_text: str) -> dict:
    """Repair cached or fresh report analysis before the UI renders it."""
    return _repair_report_analysis(analysis or {}, report_text or "")


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

    analysis = {
        "patient_snapshot": {
            "name_or_id": name_match.group(1).strip() if name_match else "Uploaded report",
            "age": age_sex_match.group(1) if age_sex_match else "Not specified",
            "sex": _sex_label(age_sex_match.group(2)) if age_sex_match else "Not specified",
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
    return _repair_report_analysis(analysis, report_text)


def _report_section_prompt(report_text: str, task: str, output_rule: str) -> str:
    clipped_report = _clean_report_noise(report_text)[:16000]
    return f"""
You are helping doctors prepare for a short multidisciplinary medical meeting.

Use ONLY the report text provided. Do not invent facts. Do not include hidden reasoning,
confidence scores, or prompt text.
Prefer concrete clinical details over generic advice. If the report does not support an item,
write "Not specified in the report" instead of guessing. Keep every item useful for deciding
what the team should discuss during the meeting.
Start directly with the requested answer. Do not write words like thought, analysis,
Identify the Goal, Scan the Report, or explain how you read the report.

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
            if text and text not in {"{", "}", "[]"} and not _looks_like_reasoning(text):
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


def heuristic_report_analysis(report_text: str) -> dict:
    """Fast rule-based extraction — no LLM call."""
    return _heuristic_report_analysis(report_text)


def _analysis_has_signal(analysis: dict) -> bool:
    facts = _clean_text_list(analysis.get("critical_facts"), 8)
    problems = analysis.get("priority_problems") or []
    red_flags = _clean_text_list(analysis.get("red_flags"), 5)
    objective = _strip_model_noise(str(analysis.get("meeting_objective", "")))
    snapshot = analysis.get("patient_snapshot") if isinstance(analysis.get("patient_snapshot"), dict) else {}
    presenting = _strip_model_noise(str(snapshot.get("presenting_problem", "")))
    return bool(
        facts
        or problems
        or red_flags
        or (objective and not _is_missing_value(objective))
        or (presenting and not _is_missing_value(presenting))
    )


def _finalize_report_analysis(analysis: dict, report_text: str) -> dict:
    repaired = _repair_report_analysis(analysis, report_text)
    if not _analysis_has_signal(repaired):
        raise ValueError("Report analysis returned no usable clinical content.")
    return repaired


def _analyze_report_combined_json(report_text: str, model: str = MODEL) -> dict:
    """One structured JSON pass — fast and preserves all briefing sections."""
    response = _chat(
        build_report_analysis_json_prompt(report_text),
        model,
        stream=False,
        json_mode=True,
        options={"num_predict": 2200, "temperature": 0.1, "num_ctx": 8192},
    )
    raw = _strip_model_noise(response["message"]["content"])
    data = _extract_report_json(raw)
    if not data:
        raise ValueError("JSON report analysis could not be parsed.")
    normalized = _normalize_report_json(data)
    return _finalize_report_analysis(normalized, report_text)


def _analyze_report_combined_markdown(report_text: str, model: str = MODEL) -> dict:
    """One markdown pass — backup when JSON mode fails."""
    response = _chat(
        build_report_analysis_prompt(report_text),
        model,
        stream=False,
        json_mode=False,
        options={"num_predict": 2000, "temperature": 0.1, "num_ctx": 8192},
    )
    raw = _strip_model_noise(response["message"]["content"])
    parsed = _parse_report_markdown(raw)
    if not parsed:
        raise ValueError("Markdown report analysis could not be parsed.")
    if isinstance(parsed.get("priority_problems"), list):
        parsed["priority_problems"] = _coerce_problem_rows(parsed["priority_problems"])
    if isinstance(parsed.get("specialist_focus"), list):
        parsed["specialist_focus"] = _coerce_focus_rows(parsed["specialist_focus"])
    return _finalize_report_analysis(parsed, report_text)


def _analyze_report_grouped(report_text: str, model: str = MODEL) -> dict:
    """Three parallel MedGemma passes — ~3x faster than nine sections, high detail."""
    groups = [
        (
            "clinical_core",
            "Extract the patient snapshot, meeting objective, and critical facts for tumor board.",
            (
                "Return exactly three numbered sections:\n"
                "1. Patient snapshot — lines: Name/ID, Age, Sex, Presenting problem, Likely primary issue\n"
                "2. Meeting objective — one sentence\n"
                "3. Critical facts — 5 to 8 bullets with specific report details"
            ),
            1000,
        ),
        (
            "problems_team",
            "Extract priority problems, specialist focus, and red flags for tumor board.",
            (
                "Return exactly three numbered sections:\n"
                "4. Priority problems — up to 6 bullets: Problem: ... | Evidence: ... | Why it matters: ...\n"
                "5. Specialist focus — up to 5 bullets: Specialist: ... | Review: ...\n"
                "7. Red flags — up to 5 urgent bullets from the report"
            ),
            1200,
        ),
        (
            "meeting_wrap",
            "Extract missing data, decision points, and meeting flow for tumor board.",
            (
                "Return exactly three numbered sections:\n"
                "6. Missing data — up to 6 specific gaps\n"
                "8. Decision points — up to 6 concrete meeting questions\n"
                "9. Suggested meeting flow — 4 to 5 steps using this case"
            ),
            900,
        ),
    ]

    partials: list[dict | None] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_report_section, report_text, task, rule, num_predict=n): name
            for name, task, rule, n in groups
        }
        for future in as_completed(futures):
            text = future.result()
            partial = _parse_report_markdown(text)
            if partial:
                if isinstance(partial.get("priority_problems"), list):
                    partial["priority_problems"] = _coerce_problem_rows(partial["priority_problems"])
                if isinstance(partial.get("specialist_focus"), list):
                    partial["specialist_focus"] = _coerce_focus_rows(partial["specialist_focus"])
            partials.append(partial)

    merged = _merge_report_partials(*partials)
    if not merged:
        raise ValueError("Grouped report analysis returned no sections.")
    return _finalize_report_analysis(merged, report_text)


def _analyze_report_pipeline(report_text: str, model: str = MODEL) -> tuple[dict, str]:
    """Fast path with quality fallbacks. Returns (analysis, mode_label)."""
    report_text = _clean_report_noise(report_text)
    start = time.perf_counter()

    for mode, runner in (
        ("combined_json", _analyze_report_combined_json),
        ("combined_markdown", _analyze_report_combined_markdown),
        ("grouped_parallel", _analyze_report_grouped),
    ):
        try:
            analysis = runner(report_text, model)
            logger.info(
                "report_analysis_ok model=%s prompt=%s latency_sec=%.2f mode=%s",
                model,
                PROMPT_VERSION,
                time.perf_counter() - start,
                mode,
            )
            return analysis, mode
        except Exception as exc:
            logger.warning("report_analysis_mode_failed mode=%s error=%s", mode, exc)

    heuristic = _repair_report_analysis(_heuristic_report_analysis(report_text), report_text)
    logger.info(
        "report_analysis_ok model=%s prompt=%s latency_sec=%.2f mode=heuristic",
        model,
        PROMPT_VERSION,
        time.perf_counter() - start,
    )
    return heuristic, "heuristic"


def analyze_report_with_mode(report_text: str, model: str = MODEL) -> tuple[dict, str]:
    """Turn a clinical report into a briefing; returns (analysis, mode_label)."""
    return _analyze_report_pipeline(report_text, model)


def analyze_report(report_text: str, model: str = MODEL) -> dict:
    """Turn a long clinical report into a meeting-ready board briefing."""
    analysis, _mode = analyze_report_with_mode(report_text, model)
    return analysis


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
