from __future__ import annotations

import time

from harness.compute.base import ComputeBroker
from harness.compute.fake import FakeComputeBackend
from harness.compute.scheduler import JobScheduler
from harness.platform.models import (
    Domain,
    JobSpec,
    JobStatus,
    Project,
    ResourceRequest,
    WorkSession,
    WorkSessionStatus,
)
from harness.platform.registry import PlatformRegistry


def _wait(registry, job_id, statuses, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = registry.get_job(job_id)
        if job and job.status in statuses:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job did not reach {statuses}: {registry.get_job(job_id)}")


def _make_job(registry, *, backend="fake", accelerator="gpu"):
    project = registry.create_project(
        Project.new(domain=Domain.KAGGLE, title="scheduler test")
    )
    session = registry.create_work_session(
        WorkSession.new(
            project_id=project.project_id,
            title="experiment",
            objective="run one durable job",
        )
    )
    spec = JobSpec.new(
        work_session_id=session.session_id,
        domain=Domain.KAGGLE,
        task_type="training",
        backend_preferences=(backend,),
        resources=ResourceRequest(
            accelerator=accelerator,
            min_vram_gb=8 if accelerator == "gpu" else 0,
            preferred_gpu_count=1 if accelerator == "gpu" else 0,
            cpu_cores=2,
            ram_gb=4,
            max_runtime_minutes=30,
            capabilities=("training",),
        ),
        outputs=("result.json",),
    )
    return session, registry.create_job(spec)


def test_scheduler_runs_job_offline_and_collects_result(tmp_path):
    registry = PlatformRegistry(tmp_path / "platform.sqlite3")
    session, record = _make_job(registry)
    backend = FakeComputeBackend()
    scheduler = JobScheduler(
        registry=registry,
        broker=ComputeBroker([backend]),
        root_dir=tmp_path / "runtime",
        max_concurrent_jobs=2,
        poll_interval_seconds=0.02,
    )
    scheduler.start()
    scheduler.enqueue(record.spec.job_id)
    completed = _wait(registry, record.spec.job_id, {JobStatus.COMPLETED})
    scheduler.stop(wait=True)
    assert completed.current_stage == "collected"
    assert completed.result["metric"] == "fake_score"
    session_after = registry.get_work_session(session.session_id)
    assert session_after is not None
    assert session_after.status == WorkSessionStatus.REVIEW
    events = registry.list_events(session.session_id)
    assert any(item.kind.value == "progress" for item in events)
    assert any("completed" in item.message.lower() for item in events)


def test_paid_backend_waits_for_explicit_approval(tmp_path):
    registry = PlatformRegistry(tmp_path / "platform.sqlite3")
    _, record = _make_job(registry)
    backend = FakeComputeBackend()
    scheduler = JobScheduler(
        registry=registry,
        broker=ComputeBroker([backend], paid_backends=("fake",)),
        root_dir=tmp_path / "runtime",
        poll_interval_seconds=0.02,
    )
    scheduler.start()
    scheduler.enqueue(record.spec.job_id)
    waiting = _wait(
        registry,
        record.spec.job_id,
        {JobStatus.WAITING_APPROVAL},
    )
    assert waiting.backend == "fake"
    assert not waiting.result.get("compute_approved")
    scheduler.approve_job(record.spec.job_id)
    completed = _wait(registry, record.spec.job_id, {JobStatus.COMPLETED})
    scheduler.stop(wait=True)
    assert completed.result["metric"] == "fake_score"


def test_broker_rejects_kaggle_backend_for_non_kaggle_research():
    backend = FakeComputeBackend()
    backend.name = "kaggle_notebook"
    broker = ComputeBroker([backend], allow_kaggle_for_research=False)
    spec = JobSpec.new(
        work_session_id="WS-test",
        domain=Domain.RESEARCH,
        task_type="training",
        backend_preferences=("kaggle_notebook",),
        resources=ResourceRequest(
            accelerator="gpu",
            min_vram_gb=1,
            preferred_gpu_count=1,
            capabilities=("training",),
        ),
    )
    decision = broker.decide(spec)
    assert decision.selected is None
    assert "kaggle_notebook" in decision.rejected
