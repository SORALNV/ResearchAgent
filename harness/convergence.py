from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.config import HarnessConfig
from harness.state import ResearchSession, utc_timestamp


@dataclass(frozen=True)
class ConvergenceDecision:
    action: str
    reason: str
    stagnation_rounds: int
    no_evidence_rounds: int
    signature: str
    round_status: str
    progress_score: float

    @property
    def should_complete(self) -> bool:
        return self.action == "complete"

    @property
    def needs_human_review(self) -> bool:
        return self.action == "human_review"


class ConvergenceTracker:
    SCHEMA_VERSION = 1

    def __init__(self, session: ResearchSession, config: HarnessConfig) -> None:
        self.session = session
        self.config = config
        self.path = Path(session.research_dir) / "artifacts" / "convergence.json"

    def evaluate(self, output: object) -> ConvergenceDecision:
        state = self.load()
        round_status = str(getattr(output, "round_status", "continue") or "continue").lower()
        confidence = str(getattr(output, "confidence", "mid") or "mid").lower()
        progress_score = _bounded_float(getattr(output, "progress_score", 0.5), 0.5)
        evidence = [str(item) for item in (getattr(output, "new_evidence_ids", []) or [])]
        signature = _signature(output)

        previous_signature = str(state.get("last_signature") or "")
        stagnation_rounds = int(state.get("stagnation_rounds") or 0)
        no_evidence_rounds = int(state.get("no_evidence_rounds") or 0)

        low_progress = progress_score < self.config.convergence_min_progress
        unchanged = bool(previous_signature) and signature == previous_signature
        if unchanged or (low_progress and not evidence):
            stagnation_rounds += 1
        else:
            stagnation_rounds = 0

        if evidence:
            no_evidence_rounds = 0
        else:
            no_evidence_rounds += 1

        action = "continue"
        reason = "continue"
        high_enough = (
            not self.config.convergence_require_high_confidence
            or confidence == "high"
        )
        if round_status == "completed" and high_enough:
            action = "complete"
            reason = "main integration marked the research complete"
        elif round_status in {"blocked", "failed"}:
            action = "blocked"
            reason = f"main integration reported round_status={round_status}"
        elif (
            self.config.convergence_patience > 0
            and stagnation_rounds >= self.config.convergence_patience
        ):
            action = "human_review"
            reason = f"research progress stagnated for {stagnation_rounds} rounds"
        elif (
            self.config.convergence_no_evidence_patience > 0
            and no_evidence_rounds >= self.config.convergence_no_evidence_patience
        ):
            action = "human_review"
            reason = f"no new evidence was added for {no_evidence_rounds} rounds"

        decision = ConvergenceDecision(
            action=action,
            reason=reason,
            stagnation_rounds=stagnation_rounds,
            no_evidence_rounds=no_evidence_rounds,
            signature=signature,
            round_status=round_status,
            progress_score=progress_score,
        )
        history = list(state.get("history") or [])
        history.append(
            {
                "timestamp": utc_timestamp(),
                "round_number": int(
                    getattr(output, "round_number", 0) or self.session.round_id
                ),
                "decision": asdict(decision),
                "confidence": confidence,
                "new_evidence_ids": evidence,
            }
        )
        self.save(
            {
                "schema_version": self.SCHEMA_VERSION,
                "session_id": self.session.session_id,
                "last_signature": signature,
                "stagnation_rounds": stagnation_rounds,
                "no_evidence_rounds": no_evidence_rounds,
                "last_decision": asdict(decision),
                "history": history[-100:],
                "updated_at": utc_timestamp(),
            }
        )
        return decision

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._new()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._new()
        if (
            data.get("schema_version") != self.SCHEMA_VERSION
            or data.get("session_id") != self.session.session_id
        ):
            return self._new()
        return data

    def snapshot(self) -> dict[str, Any]:
        state = self.load()
        return {
            "stagnation_rounds": int(state.get("stagnation_rounds") or 0),
            "no_evidence_rounds": int(state.get("no_evidence_rounds") or 0),
            "last_decision": dict(state.get("last_decision") or {}),
        }

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _new(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session.session_id,
            "last_signature": "",
            "stagnation_rounds": 0,
            "no_evidence_rounds": 0,
            "last_decision": {},
            "history": [],
            "updated_at": utc_timestamp(),
        }


def _signature(output: object) -> str:
    payload = {
        "decision": str(getattr(output, "decision", "") or ""),
        "next_action": str(getattr(output, "next_action", "") or ""),
        "accepted_ideas": sorted(
            str(item) for item in (getattr(output, "accepted_ideas", []) or [])
        ),
        "new_evidence_ids": sorted(
            str(item) for item in (getattr(output, "new_evidence_ids", []) or [])
        ),
        "unresolved_blockers": sorted(
            str(item) for item in (getattr(output, "unresolved_blockers", []) or [])
        ),
        "round_status": str(getattr(output, "round_status", "continue") or "continue"),
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _bounded_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, result))
