from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from harness.state import ResearchSession, utc_timestamp


class RoundCheckpointStore:
    """Atomic, stage-level checkpoint storage for one research round."""

    SCHEMA_VERSION = 1

    def __init__(self, session: ResearchSession, round_number: int, enabled: bool = True) -> None:
        self.session = session
        self.round_number = round_number
        self.enabled = enabled
        root = Path(session.research_dir)
        self.path = root / "artifacts" / "checkpoints" / f"R{round_number:03d}.json"
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.enabled or not self.path.exists():
                return self._new()
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return self._new()
            if data.get("schema_version") != self.SCHEMA_VERSION:
                return self._new()
            if data.get("session_id") != self.session.session_id:
                return self._new()
            if int(data.get("round_number") or 0) != self.round_number:
                return self._new()
            return data

    def save(self, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            data["schema_version"] = self.SCHEMA_VERSION
            data["session_id"] = self.session.session_id
            data["round_number"] = self.round_number
            data["updated_at"] = utc_timestamp()
            data["cost"] = dict(self.session.cost.__dict__)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def mark_status(
        self,
        data: dict[str, Any],
        status: str,
        *,
        stage: str | None = None,
        error: str | None = None,
    ) -> None:
        data["status"] = status
        if stage is not None:
            data["current_stage"] = stage
        if error is not None:
            data["last_error"] = error
        self.save(data)

    def _new(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session.session_id,
            "round_number": self.round_number,
            "status": "pending",
            "current_stage": "not_started",
            "plan_attempts": [],
            "tasks": [],
            "runs": {},
            "review_cycles": [],
            "fresh": None,
            "claude": None,
            "integration_attempts": [],
            "promotions": [],
            "operations": [],
            "protocol_errors": [],
            "final_output": None,
            "cost": dict(self.session.cost.__dict__),
            "updated_at": utc_timestamp(),
        }
