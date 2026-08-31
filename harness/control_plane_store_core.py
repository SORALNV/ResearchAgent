from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from harness.control_plane_event_models import Event
from harness.control_plane_models import (
    SCHEMA_VERSION,
    ConflictError,
    JobStatus,
    NotFoundError,
    Project,
    WorkSession,
    _TERMINAL_JOB_STATUSES,
)
from harness.control_plane_storage import (
    atomic_write_json,
    cross_process_lock,
    external_identity,
    private_permissions,
    read_json,
    validate_id,
)
from harness.state import utc_timestamp


class ControlPlaneStoreCore(object):
    """Filesystem, locking, validation, and sequence recovery primitives."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.projects_dir = self.root / "projects"
        self.sessions_dir = self.root / "work_sessions"
        self.jobs_dir = self.root / "jobs"
        self.events_dir = self.root / "events"
        self.steering_dir = self.root / "steering"
        self.idempotency_dir = self.root / "idempotency"
        self.event_idempotency_dir = self.idempotency_dir / "events"
        self.steering_idempotency_dir = self.idempotency_dir / "steering"
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / ".control.lock"
        self._lock = threading.RLock()
        for directory in (
            self.root,
            self.projects_dir,
            self.sessions_dir,
            self.jobs_dir,
            self.events_dir,
            self.steering_dir,
            self.idempotency_dir,
            self.event_idempotency_dir,
            self.steering_idempotency_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            private_permissions(directory, directory=True)
        with self._mutation_lock():
            self._save_index(self._load_index(reconcile=True))

    def _assert_external_ref_unique(
        self,
        external_ref: Mapping[str, str],
        *,
        origin: str,
        exclude_work_session_id: str | None = None,
    ) -> None:
        identity = external_identity(external_ref)
        if not identity:
            return
        existing = self.find_work_session_by_external_ref(identity, origin=origin)
        if existing and existing.work_session_id != exclude_work_session_id:
            raise ConflictError(
                "external conversation is already bound to "
                f"{existing.work_session_id}"
            )

    def _next_active_job_id(
        self,
        work_session_id: str,
        *,
        exclude_job_id: str,
    ) -> str | None:
        candidates = [
            item
            for item in self.list_jobs(work_session_id=work_session_id)
            if item.job_id != exclude_job_id
            and item.status not in _TERMINAL_JOB_STATUSES
        ]
        status_order = {
            JobStatus.RUNNING: 0,
            JobStatus.WAITING_APPROVAL: 1,
            JobStatus.QUEUED: 2,
            JobStatus.PAUSED: 3,
        }
        candidates.sort(
            key=lambda item: (
                status_order.get(item.status, 9),
                -item.spec.priority,
                item.created_at,
                item.job_id,
            )
        )
        return candidates[0].job_id if candidates else None

    def _validate_scope(
        self,
        project_id: str,
        work_session_id: str,
        *,
        job_id: str | None,
    ) -> tuple[Project, WorkSession]:
        project = self.get_project(project_id)
        session = self.get_work_session(work_session_id)
        if session.project_id != project_id:
            raise ConflictError("work session does not belong to project")
        if job_id:
            job = self.get_job(job_id)
            if (
                job.spec.project_id != project_id
                or job.spec.work_session_id != work_session_id
            ):
                raise ConflictError("job does not belong to work session")
        return project, session

    def _read_entity(self, directory: Path, entity_id: str, label: str) -> dict[str, Any]:
        path = self._entity_path(directory, entity_id)
        if not path.exists():
            raise NotFoundError(f"{label} not found: {entity_id}")
        return read_json(path)

    def _entity_path(self, directory: Path, entity_id: str) -> Path:
        return directory / f"{validate_id(entity_id)}.json"

    def _find_event_by_id(self, event_id: str) -> Event:
        validate_id(event_id)
        matches = list(self.events_dir.glob(f"*-{event_id}.json"))
        if len(matches) != 1:
            raise NotFoundError(f"event not found: {event_id}")
        return Event.from_dict(read_json(matches[0]))

    def _load_index(self, *, reconcile: bool = False) -> dict[str, Any]:
        valid = False
        raw: dict[str, Any] = {}
        if self.index_path.exists():
            try:
                raw = read_json(self.index_path)
                valid = int(raw.get("schema_version") or 0) == SCHEMA_VERSION
                int(raw.get("next_event_sequence") or 1)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                raw = {}
                valid = False
        index: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "next_event_sequence": max(
                1, int(raw.get("next_event_sequence") or 1)
            ),
            "updated_at": str(raw.get("updated_at") or utc_timestamp()),
        }
        if reconcile or not valid:
            max_sequence = 0
            for path in self.events_dir.glob("*.json"):
                try:
                    sequence = int(path.name.split("-", 1)[0])
                except (TypeError, ValueError):
                    continue
                max_sequence = max(max_sequence, sequence)
            index["next_event_sequence"] = max(
                int(index["next_event_sequence"]),
                max_sequence + 1,
            )
        return index

    def _save_index(self, index: Mapping[str, Any]) -> None:
        payload = dict(index)
        payload["schema_version"] = SCHEMA_VERSION
        payload["updated_at"] = utc_timestamp()
        atomic_write_json(self.index_path, payload)

    @contextmanager
    def _mutation_lock(self) -> Iterator[None]:
        with self._lock:
            with cross_process_lock(self.lock_path):
                yield

