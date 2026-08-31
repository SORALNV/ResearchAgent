from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from harness.control_plane import (
    ConflictError,
    ControlPlaneStore,
    Domain,
    EventLane,
    InvalidTransitionError,
    JobSpec,
    JobStatus,
    ProjectStatus,
    ResourceRequirements,
    SteeringApplyPolicy,
    SteeringKind,
    SteeringStatus,
    WorkSessionStatus,
)


def _scope(tmp_path: Path, *, domain: Domain = Domain.HYBRID):
    store = ControlPlaneStore(tmp_path / "control")
    project = store.create_project("Agent project", domain)
    session = store.create_work_session(
        project.project_id,
        "Discord thread",
        external_ref={"channel_id": "10", "thread_id": "20"},
    )
    return store, project, session


def _job_spec(project_id: str, work_session_id: str, *, domain=Domain.RESEARCH):
    return JobSpec(
        project_id=project_id,
        work_session_id=work_session_id,
        domain=domain,
        kind="research_round",
        payload={"goal": "test"},
        resources=ResourceRequirements(
            cpu_cores=2,
            memory_mb=4096,
            gpu_count=1,
            gpu_memory_mb=16384,
        ),
        backend_preferences=("remote_gpu", "local_cpu"),
        max_runtime_seconds=3600,
    )


def test_idempotency_keys_cannot_cross_work_session_scope(tmp_path):
    store, project, first_session = _scope(tmp_path)
    second_session = store.create_work_session(project.project_id, "second")
    store.append_event(
        event_type="discord.message",
        lane=EventLane.CONTROL,
        project_id=project.project_id,
        work_session_id=first_session.work_session_id,
        idempotency_key="discord:message:shared",
    )
    with pytest.raises(ConflictError):
        store.append_event(
            event_type="discord.message",
            lane=EventLane.CONTROL,
            project_id=project.project_id,
            work_session_id=second_session.work_session_id,
            idempotency_key="discord:message:shared",
        )

    store.enqueue_steering(
        project_id=project.project_id,
        work_session_id=first_session.work_session_id,
        kind=SteeringKind.SUPPLEMENT,
        text="first",
        idempotency_key="discord:steering:shared",
    )
    with pytest.raises(ConflictError):
        store.enqueue_steering(
            project_id=project.project_id,
            work_session_id=second_session.work_session_id,
            kind=SteeringKind.SUPPLEMENT,
            text="second",
            idempotency_key="discord:steering:shared",
        )


def test_event_sequence_and_idempotency_survive_index_rebuild(tmp_path):
    store, project, session = _scope(tmp_path)
    first = store.append_event(
        event_type="session.created",
        lane=EventLane.AUDIT,
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        idempotency_key="discord:message:1",
    )
    duplicate = store.append_event(
        event_type="ignored.duplicate",
        lane=EventLane.DATA,
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        idempotency_key="discord:message:1",
    )
    assert duplicate == first

    store.index_path.unlink()
    recovered = ControlPlaneStore(store.root)
    duplicate_after_recovery = recovered.append_event(
        event_type="still.duplicate",
        lane=EventLane.DATA,
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        idempotency_key="discord:message:1",
    )
    second = recovered.append_event(
        event_type="session.updated",
        lane=EventLane.STATUS,
        project_id=project.project_id,
        work_session_id=session.work_session_id,
    )
    assert duplicate_after_recovery == first
    assert second.sequence == first.sequence + 1
    assert [item.sequence for item in recovered.list_events()] == [1, 2]


def test_idempotency_markers_restore_missing_entities(tmp_path):
    store, project, session = _scope(tmp_path)
    event = store.append_event(
        event_type="discord.message",
        lane=EventLane.CONTROL,
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        idempotency_key="discord:event:restore",
    )
    event_path = next(store.events_dir.glob(f"*-{event.event_id}.json"))
    event_path.unlink()
    restored_event = store.append_event(
        event_type="ignored",
        lane=EventLane.DATA,
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        idempotency_key="discord:event:restore",
    )
    assert restored_event == event
    assert next(store.events_dir.glob(f"*-{event.event_id}.json")).exists()

    steering = store.enqueue_steering(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        kind=SteeringKind.SUPPLEMENT,
        text="restore me",
        idempotency_key="discord:steering:restore",
    )
    steering_path = store.steering_dir / f"{steering.steering_id}.json"
    steering_path.unlink()
    restored_steering = store.enqueue_steering(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        kind=SteeringKind.CHANGE,
        text="ignored",
        idempotency_key="discord:steering:restore",
    )
    assert restored_steering == steering
    assert steering_path.exists()


def test_concurrent_event_writers_keep_unique_sequences(tmp_path):
    store, project, session = _scope(tmp_path)

    def append(index: int):
        return store.append_event(
            event_type="worker.tick",
            lane=EventLane.STATUS,
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            idempotency_key=f"tick:{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        events = list(pool.map(append, range(40)))

    sequences = sorted(item.sequence for item in events)
    assert sequences == list(range(1, 41))
    assert len(list(store.events_dir.glob("*.json"))) == 40


def test_snapshot_is_complete_and_json_serializable(tmp_path):
    store, project, session = _scope(tmp_path)
    job = store.create_job(_job_spec(project.project_id, session.work_session_id))
    store.append_event(
        event_type="job.queued",
        lane=EventLane.STATUS,
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        job_id=job.job_id,
    )
    store.enqueue_steering(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        job_id=job.job_id,
        kind=SteeringKind.NEW_HYPOTHESIS,
        text="try another model",
    )

    snapshot = store.snapshot(session.work_session_id)
    assert snapshot["project"]["project_id"] == project.project_id
    assert snapshot["work_session"]["work_session_id"] == session.work_session_id
    assert snapshot["jobs"][0]["job_id"] == job.job_id
    assert snapshot["pending_steering"][0]["apply_policy"] == "child_job"
    assert snapshot["events"][0]["event_type"] == "job.queued"
    json.dumps(snapshot)


def test_latest_events_and_snapshot_return_the_tail(tmp_path):
    store, project, session = _scope(tmp_path)
    for index in range(6):
        store.append_event(
            event_type=f"tick.{index}",
            lane=EventLane.STATUS,
            project_id=project.project_id,
            work_session_id=session.work_session_id,
        )

    latest = store.latest_events(work_session_id=session.work_session_id, limit=3)
    assert [item.event_type for item in latest] == ["tick.3", "tick.4", "tick.5"]
    snapshot = store.snapshot(session.work_session_id, event_limit=2)
    assert [item["event_type"] for item in snapshot["events"]] == [
        "tick.4",
        "tick.5",
    ]

