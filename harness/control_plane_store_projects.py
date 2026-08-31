from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from harness.control_plane_json import json_dict
from harness.control_plane_models import (
    ConflictError,
    Domain,
    InvalidTransitionError,
    Project,
    ProjectStatus,
    WorkSessionStatus,
)
from harness.control_plane_storage import atomic_write_json, new_id, validate_id
from harness.control_plane_store_recovery import ControlPlaneRecoveryStore
from harness.state import utc_timestamp

class ProjectStore(ControlPlaneRecoveryStore):
    """Long-lived project operations."""

    def create_project(
        self,
        name: str,
        domain: Domain | str,
        *,
        root_ref: str = "",
        metadata: Mapping[str, Any] | None = None,
        project_id: str | None = None,
    ) -> Project:
        if not name.strip():
            raise ValueError("project name must be non-empty")
        project_id = validate_id(project_id or new_id("PRJ"))
        now = utc_timestamp()
        project = Project(
            project_id=project_id,
            name=name.strip(),
            domain=Domain(domain),
            status=ProjectStatus.ACTIVE,
            root_ref=root_ref,
            metadata=json_dict(metadata),
            created_at=now,
            updated_at=now,
        )
        with self._mutation_lock():
            path = self._entity_path(self.projects_dir, project_id)
            if path.exists():
                raise ConflictError(f"project already exists: {project_id}")
            atomic_write_json(path, project.to_dict())
        return project

    def get_project(self, project_id: str) -> Project:
        return Project.from_dict(self._read_entity(self.projects_dir, project_id, "project"))

    def set_project_status(
        self,
        project_id: str,
        status: ProjectStatus | str,
    ) -> Project:
        target = ProjectStatus(status)
        with self._mutation_lock():
            current = self.get_project(project_id)
            if current.status == ProjectStatus.ARCHIVED and target != current.status:
                raise InvalidTransitionError("archived project cannot be reopened")
            if target == ProjectStatus.ARCHIVED and current.status != target:
                open_sessions = self.list_work_sessions(
                    project_id=project_id,
                    statuses=[
                        WorkSessionStatus.OPEN,
                        WorkSessionStatus.ACTIVE,
                        WorkSessionStatus.PAUSED,
                        WorkSessionStatus.BLOCKED,
                    ],
                )
                if open_sessions:
                    raise ConflictError(
                        "project cannot be archived while work sessions remain open: "
                        + ", ".join(
                            item.work_session_id for item in open_sessions[:5]
                        )
                    )
            updated = replace(current, status=target, updated_at=utc_timestamp())
            atomic_write_json(
                self._entity_path(self.projects_dir, project_id), updated.to_dict()
            )
        return updated

