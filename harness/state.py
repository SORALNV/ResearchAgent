from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.modes import Mode, require_transition


PLANNING_QUESTIONS = [
    "成功条件: 何をもって研究成功またはハーネスMVP成功としますか？",
    "自律度: 承認なしで自動実行してよい操作の境界線はどこですか？",
    "予算・時間上限: 最大ラウンド数、最大実行時間、API/トークン上限は？",
    "成果物形式: レポート、コード、実験ログ、図表、データなど何を最終成果物にしますか？",
    "外部アクセス範囲: 使ってよいデータ、API、Web、ネットワーク、秘匿情報の範囲は？",
    "Claude相談のトリガー: 常時相談か、重要な節目だけ相談か？",
]


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ApprovalRequest:
    approval_id: str
    operation: str
    reason: str
    impact: str
    dry_run_result: str
    status: str = "pending"


@dataclass
class PhaseGate:
    gate_id: str
    phase: str
    reason: str
    status: str = "pending"
    decision: str = ""
    created_at: str = field(default_factory=utc_timestamp)
    resolved_at: str | None = None


@dataclass
class CostState:
    api_calls: int = 0
    estimated_tokens: int = 0
    literature_searches: int = 0
    agent_calls: int = 0


@dataclass
class ResearchSession:
    session_id: str
    project_name: str
    mode: Mode
    research_goal: str
    created_at: str
    updated_at: str
    version_label: str = ""
    research_dir: str = ""
    round_id: int = 0
    current_question: str = ""
    next_action: str = ""
    planning_questions: list[str] = field(default_factory=lambda: list(PLANNING_QUESTIONS))
    planning_answers: dict[str, str] = field(default_factory=dict)
    planning_scout: dict[str, Any] = field(default_factory=dict)
    phase: str = "planning"
    phase_gates: dict[str, PhaseGate] = field(default_factory=dict)
    phase_decisions: list[dict[str, str]] = field(default_factory=list)
    last_trace_node_id: str | None = None
    accepted_ideas: list[str] = field(default_factory=list)
    rejected_ideas: list[str] = field(default_factory=list)
    redirects: list[str] = field(default_factory=list)
    approval_requests: dict[str, ApprovalRequest] = field(default_factory=dict)
    approvals_received: list[str] = field(default_factory=list)
    cost: CostState = field(default_factory=CostState)
    paused_from: Mode | None = None
    completed_reason: str | None = None

    @classmethod
    def new(cls, goal: str, project_name: str = "Research Harness MVP") -> "ResearchSession":
        now = utc_timestamp()
        return cls(
            session_id=f"RS-{uuid.uuid4().hex[:12]}",
            project_name=project_name,
            mode=Mode.PLANNING,
            research_goal=goal,
            created_at=now,
            updated_at=now,
            current_question="研究開始前の要件定義",
            next_action="Soraが /re start で承認するまでPLANNINGを継続する",
        )

    def transition_to(self, target: Mode) -> None:
        require_transition(self.mode, target)
        if target == Mode.PAUSED and self.mode != Mode.PAUSED:
            self.paused_from = self.mode
        self.mode = target
        self.updated_at = utc_timestamp()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["paused_from"] = self.paused_from.value if self.paused_from else None
        data["approval_requests"] = {
            key: asdict(value) for key, value in self.approval_requests.items()
        }
        data["phase_gates"] = {
            key: asdict(value) for key, value in self.phase_gates.items()
        }
        data["cost"] = asdict(self.cost)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchSession":
        approval_requests = {
            key: ApprovalRequest(**value)
            for key, value in data.get("approval_requests", {}).items()
        }
        phase_gates = {
            key: PhaseGate(**value)
            for key, value in data.get("phase_gates", {}).items()
        }
        return cls(
            session_id=data["session_id"],
            project_name=data["project_name"],
            mode=Mode(data["mode"]),
            research_goal=data["research_goal"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            version_label=data.get("version_label", ""),
            research_dir=data.get("research_dir", ""),
            round_id=data.get("round_id", 0),
            current_question=data.get("current_question", ""),
            next_action=data.get("next_action", ""),
            planning_questions=list(data.get("planning_questions", PLANNING_QUESTIONS)),
            planning_answers=dict(data.get("planning_answers", {})),
            planning_scout=dict(data.get("planning_scout", {})),
            phase=data.get("phase", "planning"),
            phase_gates=phase_gates,
            phase_decisions=list(data.get("phase_decisions", [])),
            last_trace_node_id=data.get("last_trace_node_id"),
            accepted_ideas=list(data.get("accepted_ideas", [])),
            rejected_ideas=list(data.get("rejected_ideas", [])),
            redirects=list(data.get("redirects", [])),
            approval_requests=approval_requests,
            approvals_received=list(data.get("approvals_received", [])),
            cost=CostState(**data.get("cost", {})),
            paused_from=Mode(data["paused_from"]) if data.get("paused_from") else None,
            completed_reason=data.get("completed_reason"),
        )


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> ResearchSession | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if "active_session_id" in data:
            active_session_path = data.get("active_session_path")
            session_path = (
                Path(active_session_path)
                if active_session_path
                else self.path.parent / "sessions" / data["active_session_id"] / "state.json"
            )
            if not session_path.exists():
                return self.restore_from_journal(data["active_session_id"])
            return ResearchSession.from_dict(json.loads(session_path.read_text(encoding="utf-8")))
        return ResearchSession.from_dict(data)

    def save(self, session: ResearchSession) -> None:
        if session.research_dir:
            session_path = Path(session.research_dir) / "state.json"
        else:
            session_path = self.path.parent / "sessions" / session.session_id / "state.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session.research_dir = str(session_path.parent)
        tmp_session_path = session_path.with_suffix(".json.tmp")
        tmp_session_path.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_session_path.replace(session_path)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_active_path = self.path.with_suffix(".json.tmp")
        tmp_active_path.write_text(
            json.dumps(
                {
                    "active_session_id": session.session_id,
                    "active_session_path": str(session_path),
                    "updated_at": session.updated_at,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        tmp_active_path.replace(self.path)

    def restore_from_journal(self, session_id: str) -> ResearchSession | None:
        active_data = {}
        if self.path.exists():
            active_data = json.loads(self.path.read_text(encoding="utf-8"))
        active_session_path = active_data.get("active_session_path")
        session_path = (
            Path(active_session_path)
            if active_session_path
            else self.path.parent / "sessions" / session_id / "state.json"
        )
        if session_path.exists():
            session = ResearchSession.from_dict(json.loads(session_path.read_text(encoding="utf-8")))
            self.save(session)
            return session
        journal_path = session_path.parent / "journal.jsonl"
        if not journal_path.exists():
            journal_path = self.path.parent / "sessions" / session_id / "journal.jsonl"
        if not journal_path.exists():
            return None
        session: ResearchSession | None = None
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if session is None:
                session = ResearchSession.new(event.get("research_goal") or "restored session")
                session.session_id = session_id
                session.research_dir = str(journal_path.parent)
                session.version_label = event.get("version_label") or session.version_label
                session.project_name = event.get("project_name") or session.project_name
            if event.get("mode"):
                session.mode = Mode(event["mode"])
            session.round_id = max(session.round_id, int(event.get("round_id") or 0))
            session.current_question = event.get("current_question") or session.current_question
            session.next_action = event.get("next_action") or session.next_action
            session.planning_answers.update(event.get("planning_answers") or {})
            if event.get("planning_scout"):
                session.planning_scout = event["planning_scout"]
            session.accepted_ideas = list(dict.fromkeys(session.accepted_ideas + (event.get("accepted_ideas") or [])))
            session.rejected_ideas = list(dict.fromkeys(session.rejected_ideas + (event.get("rejected_ideas") or [])))
            for key, value in (event.get("approval_requests") or {}).items():
                session.approval_requests[key] = ApprovalRequest(**value)
            session.approvals_received = list(
                dict.fromkeys(session.approvals_received + (event.get("approvals_received") or []))
            )
            cost = event.get("cost")
            if cost:
                session.cost = CostState(**cost)
        if session:
            self.save(session)
        return session
