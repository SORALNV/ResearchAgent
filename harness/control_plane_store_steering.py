from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from harness.control_plane_json import json_dict
from harness.control_plane_models import (
    SCHEMA_VERSION,
    ConflictError,
    InvalidTransitionError,
    Steering,
    SteeringApplyPolicy,
    SteeringKind,
    SteeringStatus,
    _STEERING_DEFAULT_POLICY,
)
from harness.control_plane_storage import atomic_write_json, new_id, read_json
from harness.control_plane_store_events import EventStore
from harness.state import utc_timestamp


class ControlPlaneStore(EventStore):
    """Claimable steering and aggregate work-session snapshots."""

    def enqueue_steering(
        self,
        *,
        project_id: str,
        work_session_id: str,
        kind: SteeringKind | str,
        text: str = "",
        job_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        apply_policy: SteeringApplyPolicy | str | None = None,
        source_event_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> Steering:
        parsed_kind = SteeringKind(kind)
        policy = (
            SteeringApplyPolicy(apply_policy)
            if apply_policy is not None
            else _STEERING_DEFAULT_POLICY[parsed_kind]
        )
        if not text.strip() and not payload:
            raise ValueError("steering requires text or payload")
        with self._mutation_lock():
            self._validate_scope(project_id, work_session_id, job_id=job_id)
            if idempotency_key:
                existing = self._load_idempotent_steering(idempotency_key)
                if existing:
                    if (
                        existing.project_id != project_id
                        or existing.work_session_id != work_session_id
                        or existing.job_id != job_id
                    ):
                        raise ConflictError(
                            "steering idempotency key is already used in another scope"
                        )
                    return existing
            if source_event_id:
                source = self._find_event_by_id(source_event_id)
                if (
                    source.project_id != project_id
                    or source.work_session_id != work_session_id
                ):
                    raise ConflictError("source event is outside the steering scope")
            now = utc_timestamp()
            steering = Steering(
                steering_id=new_id("STR"),
                project_id=project_id,
                work_session_id=work_session_id,
                job_id=job_id,
                kind=parsed_kind,
                apply_policy=policy,
                status=SteeringStatus.PENDING,
                text=text.strip(),
                payload=json_dict(payload),
                source_event_id=source_event_id,
                idempotency_key=idempotency_key,
                created_at=now,
                updated_at=now,
            )
            if idempotency_key:
                self._save_idempotency_record(
                    self.steering_idempotency_dir,
                    kind="steering",
                    key=idempotency_key,
                    entity=steering.to_dict(),
                )
            atomic_write_json(
                self._entity_path(self.steering_dir, steering.steering_id),
                steering.to_dict(),
            )
        return steering

    def get_steering(self, steering_id: str) -> Steering:
        return Steering.from_dict(
            self._read_entity(self.steering_dir, steering_id, "steering")
        )

    def list_steering(
        self,
        *,
        work_session_id: str,
        job_id: str | None = None,
        statuses: Iterable[SteeringStatus | str] | None = None,
    ) -> list[Steering]:
        allowed = (
            {SteeringStatus(item) for item in statuses}
            if statuses is not None
            else None
        )
        result: list[Steering] = []
        for path in sorted(self.steering_dir.glob("*.json")):
            item = Steering.from_dict(read_json(path))
            if item.work_session_id != work_session_id:
                continue
            if job_id is not None and item.job_id not in {None, job_id}:
                continue
            if allowed is not None and item.status not in allowed:
                continue
            result.append(item)
        return sorted(result, key=lambda item: (item.created_at, item.steering_id))

    def claim_steering(
        self,
        *,
        work_session_id: str,
        consumer: str,
        job_id: str | None = None,
        limit: int = 20,
    ) -> list[Steering]:
        if not consumer.strip():
            raise ValueError("consumer must be non-empty")
        if limit <= 0:
            return []
        with self._mutation_lock():
            pending = self.list_steering(
                work_session_id=work_session_id,
                job_id=job_id,
                statuses=[SteeringStatus.PENDING],
            )[:limit]
            now = utc_timestamp()
            claimed: list[Steering] = []
            for item in pending:
                updated = replace(
                    item,
                    status=SteeringStatus.CLAIMED,
                    claimed_by=consumer.strip(),
                    claimed_at=now,
                    updated_at=now,
                )
                atomic_write_json(
                    self._entity_path(self.steering_dir, item.steering_id),
                    updated.to_dict(),
                )
                claimed.append(updated)
        return claimed

    def resolve_steering(
        self,
        steering_id: str,
        status: SteeringStatus | str,
        *,
        consumer: str | None = None,
        applied_checkpoint: str | None = None,
        resolution: str | None = None,
    ) -> Steering:
        target = SteeringStatus(status)
        if target not in {
            SteeringStatus.APPLIED,
            SteeringStatus.REJECTED,
            SteeringStatus.SUPERSEDED,
        }:
            raise ValueError("steering can only be resolved to a terminal status")
        with self._mutation_lock():
            current = self.get_steering(steering_id)
            if current.status not in {SteeringStatus.PENDING, SteeringStatus.CLAIMED}:
                if current.status == target:
                    return current
                raise InvalidTransitionError(
                    f"steering is already terminal: {current.status.value}"
                )
            if current.status == SteeringStatus.CLAIMED:
                if not consumer or current.claimed_by != consumer:
                    raise ConflictError(
                        f"steering is claimed by {current.claimed_by}, "
                        f"not {consumer or 'an unspecified consumer'}"
                    )
            updated = replace(
                current,
                status=target,
                applied_checkpoint=applied_checkpoint,
                resolution=resolution,
                updated_at=utc_timestamp(),
            )
            atomic_write_json(
                self._entity_path(self.steering_dir, steering_id),
                updated.to_dict(),
            )
        return updated

    def snapshot(
        self,
        work_session_id: str,
        *,
        event_limit: int = 50,
    ) -> dict[str, Any]:
        session = self.get_work_session(work_session_id)
        project = self.get_project(session.project_id)
        jobs = self.list_jobs(work_session_id=work_session_id)
        pending = self.list_steering(
            work_session_id=work_session_id,
            statuses=[SteeringStatus.PENDING, SteeringStatus.CLAIMED],
        )
        events = self.latest_events(
            work_session_id=work_session_id,
            limit=event_limit,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "project": project.to_dict(),
            "work_session": session.to_dict(),
            "jobs": [item.to_dict() for item in jobs],
            "pending_steering": [item.to_dict() for item in pending],
            "events": [item.to_dict() for item in events],
        }
