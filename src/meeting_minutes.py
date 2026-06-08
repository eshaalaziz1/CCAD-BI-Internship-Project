"""Export tumor board meeting minutes as Markdown or PDF."""

from __future__ import annotations

from datetime import datetime, timezone

from fpdf import FPDF

from src.board_store import MeetingState


def format_meeting_minutes_markdown(
    state: MeetingState,
    patient_labels: dict[str, str],
) -> str:
    """Build meeting minutes from persisted meeting state."""
    title = state.board_title.strip()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    if title:
        lines.append(f"# {title}")
    lines.extend([
        f"**Meeting date:** {state.meeting_date}",
        f"**Generated:** {generated}",
        "",
    ])

    if not state.cases:
        lines.append("_No cases were on the board for this meeting._")
        return "\n".join(lines)

    for idx, case in enumerate(state.cases, start=1):
        label = patient_labels.get(case.patient_id, case.patient_id)
        lines.append(f"## Case {idx}: {label}")
        lines.append(f"- **Patient ID:** {case.patient_id}")
        lines.append(f"- **Status:** {case.status}")
        if case.discussion_question.strip():
            lines.append(f"- **Discussion question:** {case.discussion_question.strip()}")
        lines.append("")

        lines.append("### Board decision")
        if case.recommendation.strip():
            lines.append(case.recommendation.strip())
        else:
            lines.append("_No recommendation recorded._")
        lines.append("")

        if case.rationale.strip():
            lines.append("### Rationale")
            lines.append(case.rationale.strip())
            lines.append("")

        follow_up = getattr(case, "follow_up_date", "")
        if follow_up.strip():
            lines.append(f"**Follow-up review:** {follow_up.strip()}")
            lines.append("")

        lines.append("### Action items")
        actions = [
            a
            for a in case.action_items
            if a.task.strip() or a.owner.strip() or a.due_date.strip()
        ]
        if actions:
            for action in actions:
                parts = [action.task.strip() or "Task not specified"]
                if action.owner.strip():
                    parts.append(f"Owner: {action.owner.strip()}")
                if action.due_date.strip():
                    parts.append(f"Due: {action.due_date.strip()}")
                lines.append(f"- {' · '.join(parts)}")
        else:
            lines.append("- _No action items recorded._")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _pdf_safe(text: str) -> str:
    """Keep PDF body text in Latin-1; replace unsupported characters."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


class _MinutesPDF(FPDF):
    """FPDF wrapper that keeps the cursor inside printable margins."""

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(left=22, top=22, right=22)

    @property
    def content_width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def write_block(
        self,
        text: str,
        *,
        style: str = "",
        size: int = 11,
        line_height: float = 6,
    ) -> None:
        self.set_font("Helvetica", style, size)
        self.set_x(self.l_margin)
        self.multi_cell(self.content_width, line_height, _pdf_safe(text))

    def write_heading(self, text: str, *, size: int = 12) -> None:
        self.ln(1)
        self.write_block(text, style="B", size=size, line_height=7)

    def ensure_vertical_space(self, height_mm: float) -> None:
        if self.get_y() + height_mm > self.page_break_trigger:
            self.add_page()


def format_meeting_minutes_pdf(
    state: MeetingState,
    patient_labels: dict[str, str],
) -> bytes:
    """Build printable meeting minutes as a PDF document."""
    title = _pdf_safe(state.board_title.strip())
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pdf = _MinutesPDF()
    pdf.add_page()

    if title:
        pdf.write_block(title, style="B", size=18, line_height=10)
        pdf.ln(2)

    pdf.write_block(f"Meeting date: {state.meeting_date}")
    pdf.write_block(f"Generated: {generated}")
    pdf.ln(4)

    if not state.cases:
        pdf.write_block("No cases were on the board for this meeting.", style="I")
        return bytes(pdf.output())

    for idx, case in enumerate(state.cases, start=1):
        label = _pdf_safe(patient_labels.get(case.patient_id, case.patient_id))
        pdf.ensure_vertical_space(28)
        pdf.write_heading(f"Case {idx}: {label}", size=14)

        pdf.write_block(f"Patient ID: {case.patient_id}")
        pdf.write_block(f"Status: {case.status}")
        if case.discussion_question.strip():
            pdf.write_block(f"Discussion question: {case.discussion_question.strip()}")
        pdf.ln(2)

        pdf.write_heading("Board decision", size=12)
        decision = case.recommendation.strip() or "No recommendation recorded."
        pdf.write_block(decision)
        pdf.ln(2)

        if case.rationale.strip():
            pdf.write_heading("Rationale", size=12)
            pdf.write_block(case.rationale.strip())
            pdf.ln(2)

        follow_up = getattr(case, "follow_up_date", "")
        if follow_up.strip():
            pdf.write_block(f"Follow-up review: {follow_up.strip()}")
            pdf.ln(2)

        pdf.write_heading("Action items", size=12)
        actions = [
            a
            for a in case.action_items
            if a.task.strip() or a.owner.strip() or a.due_date.strip()
        ]
        if actions:
            for action in actions:
                task = action.task.strip() or "Task not specified"
                line = task
                if action.owner.strip():
                    line += f" | Owner: {action.owner.strip()}"
                if action.due_date.strip():
                    line += f" | Due: {action.due_date.strip()}"
                pdf.write_block(f"- {line}")
        else:
            pdf.write_block("- No action items recorded.")
        pdf.ln(4)

    return bytes(pdf.output())
