from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from harness.control_plane_json import json_copy, json_dict
from harness.control_plane_types import (
    SCHEMA_VERSION,
    EventLane,
    SteeringApplyPolicy,
    SteeringKind,
    SteeringStatus,
)

@dataclass(frozen=True)
class Event:
    event_id: str
    sequence: int
    event_type: str
    lane: EventLane
    project_id: str
    work_session_id: str
    created_at: str
    job_id: str | None = None
    actor: str = "system"
    payload: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "lane": self.lane.value,
            "project_id": self.project_id,
            "work_session_id": self.work_session_id,
            "job_id": self.job_id,
            "actor": self.actor,
            "payload": json_copy(self.payload),
            "idempotency_key": self.idempotency_key,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Event":
        return cls(
            event_id=str(data["event_id"]),
            sequence=int(data["sequence"]),
            event_type=str(data["event_type"]),
            lane=EventLane(str(data.get("lane") or EventLane.DATA.value)),
            project_id=str(data["project_id"]),
            work_session_id=str(data["work_session_id"]),
            job_id=(str(data["job_id"]) if data.get("job_id") else None),
            actor=str(data.get("actor") or "system"),
            payload=json_dict(data.get("payload")),
            idempotency_key=(
                str(data["idempotency_key"])
                if data.get("idempotency_key")
                else None
            ),
            created_at=str(data["created_at"]),
        )

@dataclass(frozen=True)
class Steering:
    steering_id: str
    project_id: str
    work_session_id: str
    kind: SteeringKind
    apply_policy: SteeringApplyPolicy
    status: SteeringStatus
    created_at: str
    updated_at: str
    text: str = ""
    job_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    source_event_id: str | None = None
    idempotency_key: str | None = None
    claimed_by: str | None = None
    claimed_at: str | None = None
    applied_checkpoint: str | None = None
    resolution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "steering_id": self.steering_id,
            "project_id": self.project_id,
            "work_session_id": self.work_session_id,
            "job_id": self.job_id,
            "kind": self.kind.value,
            "apply_policy": self.apply_policy.value,
            "status": self.status.value,
            "text": self.text,
            "payload": json_copy(self.payload),
            "source_event_id": self.source_event_id,
            "idempotency_key": self.idempotency_key,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at,
            "applied_checkpoint": self.applied_checkpoint,
            "resolution": self.resolution,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Steering":
        return cls(
            steering_id=str(data["steering_id"]),
            project_id=str(data["project_id"]),
            work_session_id=str(data["work_session_id"]),
            job_id=(str(data["job_id"]) if data.get("job_id") else None),
            kind=SteeringKind(str(data["kind"])),
            apply_policy=SteeringApplyPolicy(str(data["apply_policy"])),
            status=SteeringStatus(
                str(data.get("status") or SteeringStatus.PENDING.value)
            ),
            text=str(data.get("text") or ""),
            payload=json_dict(data.get("payload")),
            source_event_id=(
                str(data["source_event_id"]) if data.get("source_event_id") else None
            ),
            idempotency_key=(
                str(data["idempotency_key"])
                if data.get("idempotency_key")
                else None
            ),
            claimed_by=(str(data["claimed_by"]) if data.get("claimed_by") else None),
            claimed_at=(str(data["claimed_at"]) if data.get("claimed_at") else None),
            applied_checkpoint=(
                str(data["applied_checkpoint"])
                if data.get("applied_checkpoint")
                else None
            ),
            resolution=(str(data["resolution"]) if data.get("resolution") else None),
            created_at=str(data["created_at"]),
            updated_at=str(data.get("updated_at") or data["created_at"]),
        )
