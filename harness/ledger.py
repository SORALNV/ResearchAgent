from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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


class ResearchLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append_round(self, session: ResearchSession, output: RoundOutput) -> LedgerEntry:
        round_id = session.round_id
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
            feedback=output.review_output,
            failure_class=_failure_class(output),
            next_action=output.next_action,
            evidence_ids=list(session.planning_scout.get("evidence_paper_ids") or []),
            selected_as_best=True,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
        session.last_trace_node_id = entry.node_id
        return entry

    def read_entries(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        return [
            LedgerEntry(**json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def _failure_class(output: RoundOutput) -> str | None:
    text = " ".join([output.sub_agent_output, output.review_output, output.decision]).lower()
    if "timeout" in text:
        return "timeout"
    if "failed" in text or "失敗" in text:
        return "execution_failed"
    if "未確認" in text:
        return "unverified"
    return None
