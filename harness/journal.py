from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harness.state import ResearchSession, utc_timestamp


SECRET_PATTERNS = [
    re.compile(r"(?i)(token|api[_-]?key|secret|password)\s*[:=]\s*([^\s]+)"),
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"),
]


JOURNAL_KEYS = [
    "timestamp",
    "event_type",
    "session_id",
    "version_label",
    "research_dir",
    "round_id",
    "mode",
    "project_name",
    "user_instruction",
    "research_goal",
    "research_brief_snapshot",
    "current_question",
    "planning_questions",
    "planning_answers",
    "main_agent_summary",
    "subtask",
    "conversation_sessions",
    "sub_agent_output",
    "review_output",
    "claude_consultation",
    "fresh_agent_output",
    "accepted_ideas",
    "rejected_ideas",
    "decision",
    "confidence",
    "next_action",
    "discord_report",
    "approval_requests",
    "approvals_received",
    "cost",
    "files_changed",
    "commands_run",
    "errors",
]


def _mask_string(value: str) -> str:
    value = SECRET_PATTERNS[0].sub(lambda match: f"{match.group(1)}=***", value)
    return SECRET_PATTERNS[1].sub("***", value)


def mask_secrets(value: Any) -> Any:
    if isinstance(value, str):
        return _mask_string(value)
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    if isinstance(value, dict):
        return {key: mask_secrets(item) for key, item in value.items()}
    return value


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, session: ResearchSession, event_type: str, **fields: Any) -> dict[str, Any]:
        entry = {key: None for key in JOURNAL_KEYS}
        entry.update(
            {
                "timestamp": utc_timestamp(),
                "event_type": event_type,
                "session_id": session.session_id,
                "version_label": session.version_label,
                "research_dir": session.research_dir,
                "round_id": session.round_id,
                "mode": session.mode.value,
                "project_name": session.project_name,
                "research_goal": session.research_goal,
                "current_question": session.current_question,
                "planning_questions": session.planning_questions,
                "planning_answers": session.planning_answers,
                "accepted_ideas": session.accepted_ideas,
                "rejected_ideas": session.rejected_ideas,
                "next_action": session.next_action,
                "approval_requests": {
                    key: value.__dict__ for key, value in session.approval_requests.items()
                },
                "approvals_received": session.approvals_received,
                "cost": session.cost.__dict__,
            }
        )
        entry.update(fields)
        entry = mask_secrets(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        return entry

    def read_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
