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
        existing = self.read_entries()
        same_round = [item for item in existing if item.round_id == round_id]
        node_id = f"N-{round_id:03d}"
        ledger_id = f"L-{round_id:03d}"
        if same_round:
            attempt = len(same_round) + 1
            node_id = f"N-{round_id:03d}-A{attempt}"
            ledger_id = f"L-{round_id:03d}-A{attempt}"

        round_status = str(
            getattr(output, "round_status", "continue") or "continue"
        )
        selected_as_best = round_status not in {"blocked", "failed"}
        entry = LedgerEntry(
            timestamp=utc_timestamp(),
            ledger_id=ledger_id,
            round_id=round_id,
            node_id=node_id,
            parent_node_id=session.last_trace_node_id,
            phase=session.phase,
            hypothesis=session.current_question or session.research_goal,
            action=output.subtask,
            result=output.sub_agent_output,
            feedback=output.review_output,
            failure_class=_failure_class(output),
            next_action=output.next_action,
            evidence_ids=list(
                dict.fromkeys(
                    list(
                        session.planning_scout.get("evidence_paper_ids") or []
                    )
                    + list(getattr(output, "new_evidence_ids", []) or [])
                )
            ),
            selected_as_best=selected_as_best,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
        session.last_trace_node_id = entry.node_id
        return entry

    def read_entries(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        result: list[LedgerEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                result.append(LedgerEntry(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return result


def _failure_class(output: RoundOutput) -> str | None:
    round_status = str(getattr(output, "round_status", "") or "").lower()
    if round_status == "blocked":
        return "blocked"
    if round_status == "failed":
        return "execution_failed"

    text = " ".join(
        [output.sub_agent_output, output.review_output, output.decision]
    ).lower()
    if "cancel" in text or "中断" in text:
        return "cancelled"
    if "timeout" in text:
        return "timeout"
    if "failed" in text or "失敗" in text:
        return "execution_failed"
    if "未確認" in text:
        return "unverified"
    return None
