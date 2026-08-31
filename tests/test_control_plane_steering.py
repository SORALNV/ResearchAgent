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


def test_steering_defaults_claim_and_resolution(tmp_path):
    store, project, session = _scope(tmp_path)
    job = store.create_job(_job_spec(project.project_id, session.work_session_id))
    question = store.enqueue_steering(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        job_id=job.job_id,
        kind=SteeringKind.QUESTION,
        text="What is the current metric?",
        idempotency_key="message:question",
    )
    supplement = store.enqueue_steering(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        job_id=job.job_id,
        kind=SteeringKind.SUPPLEMENT,
        text="Add leakage check",
    )
    cancel = store.enqueue_steering(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        job_id=job.job_id,
        kind=SteeringKind.CANCEL,
        text="stop now",
    )
    assert question.apply_policy == SteeringApplyPolicy.READ_ONLY
    assert supplement.apply_policy == SteeringApplyPolicy.NEXT_CHECKPOINT
    assert cancel.apply_policy == SteeringApplyPolicy.IMMEDIATE

    claimed = store.claim_steering(
        work_session_id=session.work_session_id,
        job_id=job.job_id,
        consumer="worker-1",
        limit=2,
    )
    assert [item.status for item in claimed] == [
        SteeringStatus.CLAIMED,
        SteeringStatus.CLAIMED,
    ]
    applied = store.resolve_steering(
        claimed[0].steering_id,
        SteeringStatus.APPLIED,
        consumer="worker-1",
        applied_checkpoint="R1",
    )
    assert applied.applied_checkpoint == "R1"
    with pytest.raises(ConflictError):
        store.resolve_steering(
            claimed[1].steering_id,
            SteeringStatus.REJECTED,
            consumer="other-worker",
        )


def test_steering_idempotency_and_source_scope(tmp_path):
    store, project, session = _scope(tmp_path)
    event = store.append_event(
        event_type="discord.message",
        lane=EventLane.CONTROL,
        project_id=project.project_id,
        work_session_id=session.work_session_id,
    )
    first = store.enqueue_steering(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        kind=SteeringKind.CHANGE,
        text="change split",
        source_event_id=event.event_id,
        idempotency_key="discord:steer:1",
    )
    duplicate = store.enqueue_steering(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        kind=SteeringKind.CHANGE,
        text="different body is ignored",
        idempotency_key="discord:steer:1",
    )
    assert duplicate == first

    other_project = store.create_project("other", Domain.RESEARCH)
    other_session = store.create_work_session(other_project.project_id, "other")
    with pytest.raises(ConflictError):
        store.enqueue_steering(
            project_id=other_project.project_id,
            work_session_id=other_session.work_session_id,
            kind=SteeringKind.CHANGE,
            text="bad scope",
            source_event_id=event.event_id,
        )

