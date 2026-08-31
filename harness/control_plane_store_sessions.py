from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from harness.control_plane_json import json_dict
from harness.control_plane_models import (
    ConflictError,
    InvalidTransitionError,
    JobStatus,
    ProjectStatus,
    WorkSession,
    WorkSessionStatus,
    _WORK_SESSION_TRANSITIONS,
)
from harness.control_plane_storage import (
    atomic_write_json,
    normalize_external_ref,
    new_id,
    read_json,
    validate_id,
)
from harness.control_plane_store_projects import ProjectStore
from harness.state import utc_timestamp

class WorkSessionStore(ProjectStore):
    """User-facing conversation and thread-binding operations."""

    def create_work_session(
        self,
        project_id: str,
        title: str,
        *,
        origin: str = "discord",
        external_ref: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        work_session_id: str | None = None,
    ) -> WorkSession:
        if not title.strip():
            raise ValueError("work session title must be non-empty")
        work_session_id = validate_id(work_session_id or new_id("WS"))
        with self._mutation_lock():
            project = self.get_project(project_id)
            if project.status == ProjectStatus.ARCHIVED:
                raise InvalidTransitionError("cannot create a session in an archived project")
            path = self._entity_path(self.sessions_dir, work_session_id)
            if path.exists():
                raise ConflictError(
                    f"work session already exists: {work_session_id}"
                )
            normalized_origin = origin.strip() or "unknown"
            normalized_ref = normalize_external_ref(external_ref)
            self._assert_external_ref_unique(
                normalized_ref,
                origin=normalized_origin,
            )
            now = utc_timestamp()
            session = WorkSession(
                work_session_id=work_session_id,
                project_id=project.project_id,
                title=title.strip(),
                status=WorkSessionStatus.OPEN,
                origin=normalized_origin,
                external_ref=normalized_ref,
                metadata=json_dict(metadata),
                created_at=now,
                updated_at=now,
            )
            atomic_write_json(path, session.to_dict())
        return session

    def find_work_session_by_external_ref(
        self,
        external_ref: Mapping[str, Any],
        *,
        origin: str | None = None,
    ) -> WorkSession | None:
        criteria = normalize_external_ref(external_ref)
        if not criteria:
            raise ValueError("external_ref lookup requires at least one field")
        normalized_origin = origin.strip() if origin else None
        matches: list[WorkSession] = []
        for path in sorted(self.sessions_dir.glob("*.json")):
            session = WorkSession.from_dict(read_json(path))
            if normalized_origin and session.origin != normalized_origin:
                continue
            if all(session.external_ref.get(key) == value for key, value in criteria.items()):
                matches.append(session)
        if len(matches) > 1:
            raise ConflictError(
                "external reference is ambiguous: "
                + ", ".join(item.work_session_id for item in matches)
            )
        return matches[0] if matches else None

    def get_work_session(self, work_session_id: str) -> WorkSession:
        return WorkSession.from_dict(
            self._read_entity(self.sessions_dir, work_session_id, "work session")
        )

    def list_work_sessions(
        self,
        *,
        project_id: str | None = None,
        statuses: Iterable[WorkSessionStatus | str] | None = None,
    ) -> list[WorkSession]:
        allowed = (
            {WorkSessionStatus(item) for item in statuses}
            if statuses is not None
            else None
        )
        result: list[WorkSession] = []
        for path in sorted(self.sessions_dir.glob("*.json")):
            session = WorkSession.from_dict(read_json(path))
            if project_id and session.project_id != project_id:
                continue
            if allowed is not None and session.status not in allowed:
                continue
            result.append(session)
        return sorted(result, key=lambda item: (item.created_at, item.work_session_id))

    def set_work_session_status(
        self,
        work_session_id: str,
        status: WorkSessionStatus | str,
    ) -> WorkSession:
        target = WorkSessionStatus(status)
        with self._mutation_lock():
            current = self.get_work_session(work_session_id)
            if target != current.status and target not in _WORK_SESSION_TRANSITIONS[current.status]:
                raise InvalidTransitionError(
                    f"invalid work session transition: {current.status.value} -> {target.value}"
                )
            if target == WorkSessionStatus.CLOSED and current.status != target:
                active_jobs = self.list_jobs(
                    work_session_id=work_session_id,
                    statuses=[
                        JobStatus.QUEUED,
                        JobStatus.RUNNING,
                        JobStatus.WAITING_APPROVAL,
                        JobStatus.PAUSED,
                    ],
                )
                if active_jobs:
                    raise ConflictError(
                        "work session cannot close while jobs are active: "
                        + ", ".join(item.job_id for item in active_jobs[:5])
                    )
            updated = replace(current, status=target, updated_at=utc_timestamp())
            atomic_write_json(
                self._entity_path(self.sessions_dir, work_session_id),
                updated.to_dict(),
            )
        return updated

    def bind_work_session(
        self,
        work_session_id: str,
        *,
        external_ref: Mapping[str, Any] | None = None,
        live_status_message_id: str | None = None,
    ) -> WorkSession:
        with self._mutation_lock():
            current = self.get_work_session(work_session_id)
            merged_ref = dict(current.external_ref)
            merged_ref.update(normalize_external_ref(external_ref))
            self._assert_external_ref_unique(
                merged_ref,
                origin=current.origin,
                exclude_work_session_id=current.work_session_id,
            )
            updated = replace(
                current,
                external_ref=merged_ref,
                live_status_message_id=(
                    live_status_message_id
                    if live_status_message_id is not None
                    else current.live_status_message_id
                ),
                updated_at=utc_timestamp(),
            )
            atomic_write_json(
                self._entity_path(self.sessions_dir, work_session_id),
                updated.to_dict(),
            )
        return updated

