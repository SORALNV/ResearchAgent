from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from harness.control_plane_models import (
    ConflictError,
    Domain,
    InvalidTransitionError,
    Job,
    JobSpec,
    JobStatus,
    ProjectStatus,
    WorkSessionStatus,
    _JOB_TRANSITIONS,
    _TERMINAL_JOB_STATUSES,
)
from harness.control_plane_storage import (
    atomic_write_json,
    new_id,
    read_json,
    validate_id,
)
from harness.control_plane_store_sessions import WorkSessionStore
from harness.state import utc_timestamp

class JobStore(WorkSessionStore):
    """Backend-neutral job lifecycle and optimistic concurrency."""

    def create_job(
        self,
        spec: JobSpec,
        *,
        job_id: str | None = None,
    ) -> Job:
        job_id = validate_id(job_id or new_id("JOB"))
        with self._mutation_lock():
            project, session = self._validate_scope(
                spec.project_id,
                spec.work_session_id,
                job_id=None,
            )
            if project.status == ProjectStatus.ARCHIVED:
                raise InvalidTransitionError("cannot create a job in an archived project")
            if session.status == WorkSessionStatus.CLOSED:
                raise InvalidTransitionError("cannot create a job in a closed work session")
            if project.domain not in {Domain.HYBRID, spec.domain}:
                raise ConflictError(
                    f"job domain {spec.domain.value} does not match project domain "
                    f"{project.domain.value}"
                )
            if spec.parent_job_id:
                parent = self.get_job(spec.parent_job_id)
                if (
                    parent.spec.project_id != spec.project_id
                    or parent.spec.work_session_id != spec.work_session_id
                ):
                    raise ConflictError("parent job must belong to the same work session")
            path = self._entity_path(self.jobs_dir, job_id)
            if path.exists():
                raise ConflictError(f"job already exists: {job_id}")
            now = utc_timestamp()
            job = Job(
                job_id=job_id,
                spec=spec,
                status=JobStatus.QUEUED,
                created_at=now,
                updated_at=now,
            )
            atomic_write_json(path, job.to_dict())
            if session.current_job_id is None:
                session = replace(
                    session,
                    current_job_id=job_id,
                    status=(
                        WorkSessionStatus.ACTIVE
                        if session.status == WorkSessionStatus.OPEN
                        else session.status
                    ),
                    updated_at=now,
                )
                atomic_write_json(
                    self._entity_path(self.sessions_dir, session.work_session_id),
                    session.to_dict(),
                )
        return job

    def get_job(self, job_id: str) -> Job:
        return Job.from_dict(self._read_entity(self.jobs_dir, job_id, "job"))

    def transition_job(
        self,
        job_id: str,
        status: JobStatus | str,
        *,
        expected_revision: int | None = None,
        backend_id: str | None = None,
        checkpoint_ref: str | None = None,
        lease_owner: str | None = None,
        lease_expires_at: str | None = None,
        artifact_refs: Iterable[str] = (),
        error: str | None = None,
    ) -> Job:
        target = JobStatus(status)
        with self._mutation_lock():
            current = self.get_job(job_id)
            if expected_revision is not None and current.revision != expected_revision:
                raise ConflictError(
                    f"job revision conflict: expected {expected_revision}, "
                    f"found {current.revision}"
                )
            if target != current.status and target not in _JOB_TRANSITIONS[current.status]:
                raise InvalidTransitionError(
                    f"invalid job transition: {current.status.value} -> {target.value}"
                )
            if target == JobStatus.FAILED and not (error or current.error):
                raise ValueError("failed job requires an error message")
            now = utc_timestamp()
            entering_running = (
                target == JobStatus.RUNNING and current.status != JobStatus.RUNNING
            )
            terminal = target in _TERMINAL_JOB_STATUSES
            merged_artifacts = tuple(
                dict.fromkeys(
                    [*current.artifact_refs, *(str(item) for item in artifact_refs)]
                )
            )
            updated = replace(
                current,
                status=target,
                updated_at=now,
                attempt=current.attempt + (1 if entering_running else 0),
                backend_id=backend_id if backend_id is not None else current.backend_id,
                checkpoint_ref=(
                    checkpoint_ref
                    if checkpoint_ref is not None
                    else current.checkpoint_ref
                ),
                lease_owner=(
                    None if terminal else (lease_owner if lease_owner is not None else current.lease_owner)
                ),
                lease_expires_at=(
                    None
                    if terminal
                    else (
                        lease_expires_at
                        if lease_expires_at is not None
                        else current.lease_expires_at
                    )
                ),
                artifact_refs=merged_artifacts,
                error=error if error is not None else current.error,
                started_at=(current.started_at or now) if entering_running else current.started_at,
                finished_at=now if terminal else current.finished_at,
                revision=current.revision + 1,
            )
            atomic_write_json(
                self._entity_path(self.jobs_dir, job_id), updated.to_dict()
            )
            if terminal:
                session = self.get_work_session(updated.spec.work_session_id)
                if session.current_job_id == job_id:
                    session = replace(
                        session,
                        current_job_id=self._next_active_job_id(
                            session.work_session_id,
                            exclude_job_id=job_id,
                        ),
                        updated_at=now,
                    )
                    atomic_write_json(
                        self._entity_path(
                            self.sessions_dir, session.work_session_id
                        ),
                        session.to_dict(),
                    )
        return updated

    def list_jobs(
        self,
        *,
        work_session_id: str,
        statuses: Iterable[JobStatus | str] | None = None,
    ) -> list[Job]:
        allowed = (
            {JobStatus(item) for item in statuses} if statuses is not None else None
        )
        result: list[Job] = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            job = Job.from_dict(read_json(path))
            if job.spec.work_session_id != work_session_id:
                continue
            if allowed is not None and job.status not in allowed:
                continue
            result.append(job)
        return sorted(result, key=lambda item: (item.created_at, item.job_id))

