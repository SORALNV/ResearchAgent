from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from harness.control_plane_json import json_copy, json_dict
from harness.control_plane_types import (
    SCHEMA_VERSION,
    Domain,
    ProjectStatus,
    WorkSessionStatus,
)

@dataclass(frozen=True)
class Project:
    project_id: str
    name: str
    domain: Domain
    status: ProjectStatus
    created_at: str
    updated_at: str
    root_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "project_id": self.project_id,
            "name": self.name,
            "domain": self.domain.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "root_ref": self.root_ref,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Project":
        return cls(
            project_id=str(data["project_id"]),
            name=str(data["name"]),
            domain=Domain(str(data["domain"])),
            status=ProjectStatus(str(data.get("status") or ProjectStatus.ACTIVE.value)),
            created_at=str(data["created_at"]),
            updated_at=str(data.get("updated_at") or data["created_at"]),
            root_ref=str(data.get("root_ref") or ""),
            metadata=json_dict(data.get("metadata")),
        )

@dataclass(frozen=True)
class WorkSession:
    work_session_id: str
    project_id: str
    title: str
    status: WorkSessionStatus
    created_at: str
    updated_at: str
    origin: str = "discord"
    external_ref: dict[str, str] = field(default_factory=dict)
    live_status_message_id: str | None = None
    current_job_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "work_session_id": self.work_session_id,
            "project_id": self.project_id,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "origin": self.origin,
            "external_ref": dict(self.external_ref),
            "live_status_message_id": self.live_status_message_id,
            "current_job_id": self.current_job_id,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkSession":
        return cls(
            work_session_id=str(data["work_session_id"]),
            project_id=str(data["project_id"]),
            title=str(data["title"]),
            status=WorkSessionStatus(
                str(data.get("status") or WorkSessionStatus.OPEN.value)
            ),
            created_at=str(data["created_at"]),
            updated_at=str(data.get("updated_at") or data["created_at"]),
            origin=str(data.get("origin") or "discord"),
            external_ref={
                str(key): str(value)
                for key, value in json_dict(data.get("external_ref")).items()
                if value is not None
            },
            live_status_message_id=(
                str(data["live_status_message_id"])
                if data.get("live_status_message_id")
                else None
            ),
            current_job_id=(str(data["current_job_id"]) if data.get("current_job_id") else None),
            metadata=json_dict(data.get("metadata")),
        )
