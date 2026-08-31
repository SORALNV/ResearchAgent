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


def test_round_trip_project_session_and_job(tmp_path):
    store, project, session = _scope(tmp_path)
    job = store.create_job(_job_spec(project.project_id, session.work_session_id))

    assert store.get_project(project.project_id) == project
    stored_session = store.get_work_session(session.work_session_id)
    assert stored_session.current_job_id == job.job_id
    assert stored_session.status == WorkSessionStatus.ACTIVE
    assert store.get_job(job.job_id).spec.resources.gpu_memory_mb == 16384

    running = store.transition_job(
        job.job_id,
        JobStatus.RUNNING,
        backend_id="remote-1",
        lease_owner="worker-a",
        lease_expires_at="2099-01-01T00:00:00Z",
    )
    assert running.attempt == 1
    assert running.revision == 1
    finished = store.transition_job(
        job.job_id,
        JobStatus.SUCCEEDED,
        expected_revision=running.revision,
        checkpoint_ref="checkpoints/R1.json",
        artifact_refs=["artifact://report", "artifact://report"],
    )
    assert finished.finished_at
    assert finished.lease_owner is None
    assert finished.artifact_refs == ("artifact://report",)
    assert store.get_work_session(session.work_session_id).current_job_id is None


def test_job_domain_and_parent_scope_are_enforced(tmp_path):
    store = ControlPlaneStore(tmp_path / "control")
    research = store.create_project("Research", Domain.RESEARCH)
    kaggle = store.create_project("Kaggle", Domain.KAGGLE)
    research_session = store.create_work_session(research.project_id, "r")
    kaggle_session = store.create_work_session(kaggle.project_id, "k")

    with pytest.raises(ConflictError):
        store.create_job(
            _job_spec(
                research.project_id,
                research_session.work_session_id,
                domain=Domain.KAGGLE,
            )
        )

    parent = store.create_job(
        _job_spec(kaggle.project_id, kaggle_session.work_session_id, domain=Domain.KAGGLE)
    )
    with pytest.raises(ConflictError):
        store.create_job(
            JobSpec(
                project_id=research.project_id,
                work_session_id=research_session.work_session_id,
                domain=Domain.RESEARCH,
                kind="child",
                parent_job_id=parent.job_id,
            )
        )


def test_transition_matrix_and_optimistic_revision(tmp_path):
    store, project, session = _scope(tmp_path)
    job = store.create_job(_job_spec(project.project_id, session.work_session_id))

    with pytest.raises(InvalidTransitionError):
        store.transition_job(job.job_id, JobStatus.SUCCEEDED)

    running = store.transition_job(job.job_id, JobStatus.RUNNING)
    with pytest.raises(ConflictError):
        store.transition_job(
            job.job_id,
            JobStatus.PAUSED,
            expected_revision=running.revision - 1,
        )
    failed = store.transition_job(job.job_id, JobStatus.FAILED, error="boom")
    with pytest.raises(InvalidTransitionError):
        store.transition_job(failed.job_id, JobStatus.RUNNING)


def test_work_session_and_project_terminal_states_are_not_reopened(tmp_path):
    store, project, session = _scope(tmp_path)
    closed = store.set_work_session_status(session.work_session_id, WorkSessionStatus.CLOSED)
    assert closed.status == WorkSessionStatus.CLOSED
    with pytest.raises(InvalidTransitionError):
        store.set_work_session_status(session.work_session_id, WorkSessionStatus.ACTIVE)

    archived = store.set_project_status(project.project_id, ProjectStatus.ARCHIVED)
    assert archived.status == ProjectStatus.ARCHIVED
    with pytest.raises(InvalidTransitionError):
        store.set_project_status(project.project_id, ProjectStatus.ACTIVE)


def test_active_jobs_block_session_close_and_project_archive(tmp_path):
    store, project, session = _scope(tmp_path)
    job = store.create_job(_job_spec(project.project_id, session.work_session_id))

    with pytest.raises(ConflictError):
        store.set_work_session_status(
            session.work_session_id,
            WorkSessionStatus.CLOSED,
        )
    with pytest.raises(ConflictError):
        store.set_project_status(project.project_id, ProjectStatus.ARCHIVED)

    store.transition_job(job.job_id, JobStatus.CANCELLED)
    store.set_work_session_status(session.work_session_id, WorkSessionStatus.CLOSED)
    archived = store.set_project_status(project.project_id, ProjectStatus.ARCHIVED)
    assert archived.status == ProjectStatus.ARCHIVED
    with pytest.raises(InvalidTransitionError):
        store.create_job(_job_spec(project.project_id, session.work_session_id))


def test_external_thread_binding_is_unique_and_lookupable(tmp_path):
    store = ControlPlaneStore(tmp_path / "control")
    project = store.create_project("Agent", Domain.HYBRID)
    first = store.create_work_session(
        project.project_id,
        "thread one",
        origin="discord",
        external_ref={"guild_id": "1", "thread_id": "99"},
    )

    assert (
        store.find_work_session_by_external_ref(
            {"thread_id": "99"},
            origin="discord",
        )
        == first
    )
    with pytest.raises(ConflictError):
        store.create_work_session(
            project.project_id,
            "duplicate thread",
            origin="discord",
            external_ref={"thread_id": "99"},
        )

    second = store.create_work_session(project.project_id, "unbound")
    with pytest.raises(ConflictError):
        store.bind_work_session(
            second.work_session_id,
            external_ref={"thread_id": "99"},
        )


def test_current_job_advances_to_another_active_job(tmp_path):
    store, project, session = _scope(tmp_path)
    first = store.create_job(_job_spec(project.project_id, session.work_session_id))
    second = store.create_job(
        JobSpec(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain=Domain.RESEARCH,
            kind="follow_up",
            priority=10,
        )
    )
    store.transition_job(first.job_id, JobStatus.RUNNING)
    store.transition_job(first.job_id, JobStatus.SUCCEEDED)

    assert store.get_work_session(session.work_session_id).current_job_id == second.job_id


def test_identifiers_cannot_escape_store_root(tmp_path):
    store = ControlPlaneStore(tmp_path / "control")
    with pytest.raises(ValueError):
        store.create_project("bad", Domain.RESEARCH, project_id="../escape")

