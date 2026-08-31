from __future__ import annotations

from typing import Any, Iterable, Mapping

from harness.control_plane_json import json_dict
from harness.control_plane_models import ConflictError, Event, EventLane
from harness.control_plane_storage import atomic_write_json, new_id, read_json
from harness.control_plane_store_jobs import JobStore
from harness.state import utc_timestamp

class EventStore(JobStore):
    """Immutable ordered event operations."""

    def append_event(
        self,
        *,
        event_type: str,
        lane: EventLane | str,
        project_id: str,
        work_session_id: str,
        job_id: str | None = None,
        actor: str = "system",
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Event:
        if not event_type.strip():
            raise ValueError("event_type must be non-empty")
        with self._mutation_lock():
            self._validate_scope(project_id, work_session_id, job_id=job_id)
            if idempotency_key:
                existing = self._load_idempotent_event(idempotency_key)
                if existing:
                    if (
                        existing.project_id != project_id
                        or existing.work_session_id != work_session_id
                        or existing.job_id != job_id
                    ):
                        raise ConflictError(
                            "event idempotency key is already used in another scope"
                        )
                    return existing
            index = self._load_index()
            sequence = int(index["next_event_sequence"])
            event = Event(
                event_id=new_id("EVT"),
                sequence=sequence,
                event_type=event_type.strip(),
                lane=EventLane(lane),
                project_id=project_id,
                work_session_id=work_session_id,
                job_id=job_id,
                actor=actor.strip() or "system",
                payload=json_dict(payload),
                idempotency_key=idempotency_key,
                created_at=utc_timestamp(),
            )
            # Reserve the sequence before publishing the event. A crash may leave
            # a harmless gap, but another process can never reuse the sequence.
            index["next_event_sequence"] = sequence + 1
            self._save_index(index)
            if idempotency_key:
                self._save_idempotency_record(
                    self.event_idempotency_dir,
                    kind="event",
                    key=idempotency_key,
                    entity=event.to_dict(),
                )
            path = self.events_dir / f"{sequence:020d}-{event.event_id}.json"
            atomic_write_json(path, event.to_dict())
        return event

    def list_events(
        self,
        *,
        work_session_id: str | None = None,
        job_id: str | None = None,
        after_sequence: int = 0,
        lanes: Iterable[EventLane | str] | None = None,
        limit: int = 100,
    ) -> list[Event]:
        if limit <= 0:
            return []
        allowed_lanes = (
            {EventLane(item) for item in lanes} if lanes is not None else None
        )
        result: list[Event] = []
        for path in sorted(self.events_dir.glob("*.json")):
            event = Event.from_dict(read_json(path))
            if event.sequence <= after_sequence:
                continue
            if work_session_id and event.work_session_id != work_session_id:
                continue
            if job_id and event.job_id != job_id:
                continue
            if allowed_lanes is not None and event.lane not in allowed_lanes:
                continue
            result.append(event)
            if len(result) >= limit:
                break
        return result

    def latest_events(
        self,
        *,
        work_session_id: str | None = None,
        job_id: str | None = None,
        lanes: Iterable[EventLane | str] | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Return the newest matching events in chronological order."""
        if limit <= 0:
            return []
        allowed_lanes = (
            {EventLane(item) for item in lanes} if lanes is not None else None
        )
        newest_first: list[Event] = []
        for path in sorted(self.events_dir.glob("*.json"), reverse=True):
            event = Event.from_dict(read_json(path))
            if work_session_id and event.work_session_id != work_session_id:
                continue
            if job_id and event.job_id != job_id:
                continue
            if allowed_lanes is not None and event.lane not in allowed_lanes:
                continue
            newest_first.append(event)
            if len(newest_first) >= limit:
                break
        return list(reversed(newest_first))

