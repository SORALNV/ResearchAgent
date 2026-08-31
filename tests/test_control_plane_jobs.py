import time

from harness.compute import ComputeBroker, FakeComputeBackend
from harness.control_plane import ControlPlaneRegistry, JobSpec
from harness.job_scheduler import JobScheduler
from harness.work_sessions import WorkSessionService, WorkSessionStore


def wait_for(registry, job_id, statuses, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = registry.get_job(job_id)
        if record and record.status in statuses:
            return record
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {statuses}")


def make_runtime(tmp_path):
    database = tmp_path / "control.sqlite3"
    registry = ControlPlaneRegistry(database)
    store = WorkSessionStore(database)
    scheduler = JobScheduler(
        registry,
        ComputeBroker([FakeComputeBackend()]),
        worker_count=2,
        queue_size=8,
        requeue_interrupted=False,
    )
    service = WorkSessionService(registry, store, scheduler)
    return registry, store, scheduler, service


def test_project_work_session_job_and_event_round_trip(tmp_path):
    registry, store, scheduler, service = make_runtime(tmp_path)
    try:
        project, session = service.create_session(
            domain="research",
            title="GPU comparison",
            project_root=tmp_path / "project",
        )
        spec = JobSpec.new(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain="research",
            task_type="analysis",
            payload={"fake_result": {"score": 0.9}},
            backend_preferences=("fake",),
            resources={"accelerator": "cpu"},
        )
        queued = service.queue_job(spec)
        assert queued.status == "queued"
        completed = wait_for(registry, spec.job_id, {"completed"})
        assert completed.backend == "fake"
        assert completed.result["score"] == 0.9
        events = registry.list_events(spec.job_id)
        assert events[0].event_type == "job_queued"
        assert any(event.event_type == "backend_selected" for event in events)
        assert any(event.event_type == "job_completed" for event in events)
        assert service.status(session.work_session_id)["jobs_total"] == 1
        assert store.list_messages(session.work_session_id)
    finally:
        scheduler.close(cancel_running=True)


def test_scheduler_cancel_is_propagated_to_backend(tmp_path):
    registry, _, scheduler, service = make_runtime(tmp_path)
    try:
        project, session = service.create_session(
            domain="research",
            title="cancel test",
            project_root=tmp_path / "cancel-project",
        )
        spec = JobSpec.new(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain="research",
            task_type="long",
            payload={
                "fake_delay_seconds": 0.2,
                "fake_steps": [
                    {"stage": "one"},
                    {"stage": "two"},
                    {"stage": "three"},
                ],
            },
            backend_preferences=("fake",),
            resources={"accelerator": "cpu"},
        )
        service.queue_job(spec)
        wait_for(registry, spec.job_id, {"running"})
        scheduler.cancel(spec.job_id, "test cancel")
        record = wait_for(registry, spec.job_id, {"cancelled"})
        assert record.cancel_requested
        assert "cancel" in (record.error or "")
    finally:
        scheduler.close(cancel_running=True)


def test_incomplete_job_recovery_is_fail_closed_by_default(tmp_path):
    database = tmp_path / "recover.sqlite3"
    registry = ControlPlaneRegistry(database)
    store = WorkSessionStore(database)
    scheduler = JobScheduler(
        registry,
        ComputeBroker([FakeComputeBackend()]),
        worker_count=1,
        requeue_interrupted=False,
    )
    service = WorkSessionService(registry, store, scheduler)
    project, session = service.create_session(
        domain="kaggle",
        title="recovery",
        project_root=tmp_path / "recovery-project",
    )
    spec = JobSpec.new(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        domain="kaggle",
        task_type="train",
        backend_preferences=("fake",),
        resources={"accelerator": "gpu"},
    )
    registry.create_job(spec)
    registry.claim_job(spec.job_id, "fake")
    registry.mark_running(spec.job_id, "REMOTE-ALREADY-STARTED")

    recovered = registry.recover_incomplete_jobs(requeue=False)
    assert recovered == [spec.job_id]
    record = registry.get_job(spec.job_id)
    assert record.status == "interrupted"
    assert record.backend_job_id is None
    assert "restarted" in (record.error or "")
