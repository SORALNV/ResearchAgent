from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from harness.control_plane_json import json_copy, json_dict
from harness.control_plane_resources import ResourceRequirements
from harness.control_plane_types import SCHEMA_VERSION, Domain, JobStatus

@dataclass(frozen=True)
class JobSpec:
    project_id: str
    work_session_id: str
    domain: Domain
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    backend_preferences: tuple[str, ...] = ()
    max_runtime_seconds: int | None = None
    priority: int = 0
    parent_job_id: str | None = None
    experiment_id: str | None = None
    requires_approval: bool = False

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("job kind must be non-empty")
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "work_session_id": self.work_session_id,
            "domain": self.domain.value,
            "kind": self.kind,
            "payload": json_copy(self.payload),
            "resources": self.resources.to_dict(),
            "backend_preferences": list(self.backend_preferences),
            "max_runtime_seconds": self.max_runtime_seconds,
            "priority": self.priority,
            "parent_job_id": self.parent_job_id,
            "experiment_id": self.experiment_id,
            "requires_approval": self.requires_approval,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobSpec":
        return cls(
            project_id=str(data["project_id"]),
            work_session_id=str(data["work_session_id"]),
            domain=Domain(str(data["domain"])),
            kind=str(data["kind"]),
            payload=json_dict(data.get("payload")),
            resources=ResourceRequirements.from_dict(data.get("resources")),
            backend_preferences=tuple(
                str(item) for item in data.get("backend_preferences", [])
            ),
            max_runtime_seconds=(
                int(data["max_runtime_seconds"])
                if data.get("max_runtime_seconds") is not None
                else None
            ),
            priority=int(data.get("priority") or 0),
            parent_job_id=(str(data["parent_job_id"]) if data.get("parent_job_id") else None),
            experiment_id=(str(data["experiment_id"]) if data.get("experiment_id") else None),
            requires_approval=bool(data.get("requires_approval", False)),
        )

@dataclass(frozen=True)
class Job:
    job_id: str
    spec: JobSpec
    status: JobStatus
    created_at: str
    updated_at: str
    attempt: int = 0
    backend_id: str | None = None
    checkpoint_ref: str | None = None
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    artifact_refs: tuple[str, ...] = ()
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": self.job_id,
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "attempt": self.attempt,
            "backend_id": self.backend_id,
            "checkpoint_ref": self.checkpoint_ref,
            "lease_owner": self.lease_owner,
            "lease_expires_at": self.lease_expires_at,
            "artifact_refs": list(self.artifact_refs),
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Job":
        return cls(
            job_id=str(data["job_id"]),
            spec=JobSpec.from_dict(json_dict(data.get("spec"))),
            status=JobStatus(str(data.get("status") or JobStatus.QUEUED.value)),
            created_at=str(data["created_at"]),
            updated_at=str(data.get("updated_at") or data["created_at"]),
            attempt=int(data.get("attempt") or 0),
            backend_id=(str(data["backend_id"]) if data.get("backend_id") else None),
            checkpoint_ref=(
                str(data["checkpoint_ref"]) if data.get("checkpoint_ref") else None
            ),
            lease_owner=(str(data["lease_owner"]) if data.get("lease_owner") else None),
            lease_expires_at=(
                str(data["lease_expires_at"])
                if data.get("lease_expires_at")
                else None
            ),
            artifact_refs=tuple(str(item) for item in data.get("artifact_refs", [])),
            error=(str(data["error"]) if data.get("error") else None),
            started_at=(str(data["started_at"]) if data.get("started_at") else None),
            finished_at=(str(data["finished_at"]) if data.get("finished_at") else None),
            revision=int(data.get("revision") or 0),
        )
