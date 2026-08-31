from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from harness.control_plane_event_models import Event, Steering
from harness.control_plane_json import json_dict
from harness.control_plane_models import SCHEMA_VERSION, ConflictError
from harness.control_plane_storage import (
    atomic_write_json,
    idempotency_path,
    read_json,
)
from harness.control_plane_store_core import ControlPlaneStoreCore
from harness.state import utc_timestamp

class ControlPlaneRecoveryStore(ControlPlaneStoreCore):
    """Crash-recoverable idempotency records for events and steering."""

    def _load_idempotent_event(self, key: str) -> Event | None:
        record = self._load_idempotency_record(
            self.event_idempotency_dir,
            kind="event",
            key=key,
        )
        if record is None:
            return None
        event = Event.from_dict(json_dict(record.get("entity")))
        path = self.events_dir / f"{event.sequence:020d}-{event.event_id}.json"
        if not path.exists():
            atomic_write_json(path, event.to_dict())
        index = self._load_index()
        if int(index["next_event_sequence"]) <= event.sequence:
            index["next_event_sequence"] = event.sequence + 1
            self._save_index(index)
        return event

    def _load_idempotent_steering(self, key: str) -> Steering | None:
        record = self._load_idempotency_record(
            self.steering_idempotency_dir,
            kind="steering",
            key=key,
        )
        if record is None:
            return None
        steering = Steering.from_dict(json_dict(record.get("entity")))
        path = self._entity_path(self.steering_dir, steering.steering_id)
        if not path.exists():
            atomic_write_json(path, steering.to_dict())
        return steering

    def _load_idempotency_record(
        self,
        directory: Path,
        *,
        kind: str,
        key: str,
    ) -> dict[str, Any] | None:
        path = idempotency_path(directory, key)
        if not path.exists():
            return None
        record = read_json(path)
        if record.get("kind") != kind or record.get("key") != key:
            raise ConflictError(
                f"idempotency marker collision or corruption: {path.name}"
            )
        if not isinstance(record.get("entity"), Mapping):
            raise ConflictError(f"idempotency marker has no entity: {path.name}")
        return record

    def _save_idempotency_record(
        self,
        directory: Path,
        *,
        kind: str,
        key: str,
        entity: Mapping[str, Any],
    ) -> None:
        path = idempotency_path(directory, key)
        existing = self._load_idempotency_record(
            directory,
            kind=kind,
            key=key,
        )
        if existing is not None:
            if json_dict(existing.get("entity")) != json_dict(entity):
                raise ConflictError(f"idempotency key reused with different {kind}")
            return
        atomic_write_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "key": key,
                "entity": json_dict(entity),
                "created_at": utc_timestamp(),
            },
        )

