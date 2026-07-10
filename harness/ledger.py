from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

from harness.agent_runner import RoundOutput
from harness.state import ResearchSession, utc_timestamp


@dataclass
class LedgerEntry:
    timestamp: str
    ledger_id: str
    round_id: int
    node_id: str
    parent_node_id: str | None
    phase: str
    hypothesis: str
    action: str
    result: str
    feedback: str
    failure_class: str | None
    next_action: str
    evidence_ids: list[str]
    selected_as_best: bool
    criteria_status: list[dict] = field(default_factory=list)
    critique: str = ""
    director_decision: str = ""


class ResearchLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append_round(
        self,
        session: ResearchSession,
        output: RoundOutput,
        verdict=None,
        evidence_ids: list[str] | None = None,
    ) -> LedgerEntry:
        round_id = session.round_id
        critique = getattr(verdict, "critique", "") if verdict else ""
        feedback = output.review_output
        if critique:
            feedback = f"{feedback}\n\nDirector critique:\n{critique}".strip()
        criteria_status = []
        if verdict:
            for item in getattr(verdict, "criteria_status", []) or []:
                criteria_status.append(asdict(item) if is_dataclass(item) else dict(item))
        entry = LedgerEntry(
            timestamp=utc_timestamp(),
            ledger_id=f"L-{round_id:03d}",
            round_id=round_id,
            node_id=f"N-{round_id:03d}",
            parent_node_id=session.last_trace_node_id,
            phase=session.phase,
            hypothesis=session.current_question or session.research_goal,
            action=output.subtask,
            result=output.sub_agent_output,
            feedback=feedback,
            failure_class=_failure_class(output),
            next_action=output.next_action,
            evidence_ids=(
                list(evidence_ids)
                if evidence_ids is not None
                else list(session.planning_scout.get("evidence_paper_ids") or [])
            ),
            selected_as_best=True,
            criteria_status=criteria_status,
            critique=critique,
            director_decision=getattr(verdict, "decision", "") if verdict else "",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
        session.last_trace_node_id = entry.node_id
        return entry

    def read_entries(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        entries: list[LedgerEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            data.setdefault("criteria_status", [])
            data.setdefault("critique", "")
            data.setdefault("director_decision", "")
            entries.append(LedgerEntry(**data))
        return entries


def _failure_class(output: RoundOutput) -> str | None:
    text = output.sub_agent_output
    lowered = text.lower()
    if "real sub-agent timeout" in lowered or "timeout" in lowered:
        return "timeout"
    if "real sub-agent failed" in lowered or "returncode=" in lowered:
        return "execution_failed"
    failure_section = _extract_failure_section(text)
    if failure_section:
        if _failure_section_is_empty(failure_section):
            return None
        if _looks_environment_related(failure_section):
            return "environment_issue"
        return "execution_failed"
    return None


def _extract_failure_section(text: str) -> str:
    lines = text.splitlines()
    capture = False
    captured: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _is_failure_heading(stripped):
            capture = True
            after_colon = re.split(r"[:：]", stripped, maxsplit=1)
            if len(after_colon) == 2 and after_colon[1].strip():
                captured.append(after_colon[1].strip())
            continue
        if capture and _is_next_section_heading(stripped):
            break
        if capture:
            captured.append(line)
    return "\n".join(captured).strip()


def _is_failure_heading(line: str) -> bool:
    normalized = line.strip().strip("*").strip("#").strip()
    return normalized == "失敗" or normalized.startswith("失敗:") or normalized.startswith("失敗：")


def _is_next_section_heading(line: str) -> bool:
    if not line:
        return False
    if line.startswith("#"):
        return True
    if line.startswith("**") and line.endswith("**"):
        return True
    if line.endswith((":", "：")) and not line.startswith(("-", "*")):
        return True
    return False


def _failure_section_is_empty(section: str) -> bool:
    meaningful = [
        line.strip().lstrip("-*").strip()
        for line in section.splitlines()
        if line.strip()
    ]
    if not meaningful:
        return True
    empty_markers = ("なし", "ありません", "無し", "特になし")
    return all(line.startswith(empty_markers) for line in meaningful)


def _looks_environment_related(section: str) -> bool:
    lowered = section.lower()
    environment_markers = [
        "pip",
        "gcloud",
        "credential",
        "credentials.db",
        "sandbox",
        "matplotlib",
        "cache",
        "書き込み不可",
        "未導入",
        "not found",
        "permission",
        "warning",
        "subcommand",
        "サブコマンド",
    ]
    return any(marker in lowered for marker in environment_markers)
