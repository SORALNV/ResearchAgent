import time

from harness.compute import ComputeBroker, FakeComputeBackend
from harness.config import HarnessConfig
from harness.control_plane import ControlPlaneRegistry, JobSpec
from harness.job_scheduler import JobScheduler
from harness.thread_bridge import FakeWorkThreadTransport, WorkSessionThreadBridge
from harness.work_session_dialogue import WorkSessionDialogueEngine
from harness.work_sessions import WorkSessionService, WorkSessionStore


def wait_for(registry, job_id, status="completed", timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = registry.get_job(job_id)
        if record and record.status == status:
            return record
        time.sleep(0.02)
    raise AssertionError(f"job did not reach {status}")


def test_thread_bridge_creates_one_thread_and_projects_job_milestones(tmp_path):
    database = tmp_path / "control.sqlite3"
    registry = ControlPlaneRegistry(database)
    store = WorkSessionStore(database)
    scheduler = JobScheduler(
        registry,
        ComputeBroker([FakeComputeBackend()]),
        worker_count=1,
    )
    service = WorkSessionService(registry, store, scheduler)
    transport = FakeWorkThreadTransport()
    bridge = WorkSessionThreadBridge(registry, store, service, transport)
    scheduler.subscribe(bridge.on_job_event)
    try:
        project, session = service.create_session(
            domain="research",
            title="Threaded experiment",
            project_root=tmp_path / "project",
        )
        bound = bridge.create_and_bind(
            session.work_session_id,
            initial_message="相談と実行はこのスレッドで続けます。",
        )
        assert bound.thread_id == "thread-1"
        assert len(transport.created) == 1
        again = bridge.create_and_bind(
            session.work_session_id,
            initial_message="duplicate",
        )
        assert again.thread_id == bound.thread_id
        assert len(transport.created) == 1

        job = service.queue_job(
            JobSpec.new(
                project_id=project.project_id,
                work_session_id=session.work_session_id,
                domain="research",
                task_type="analysis",
                payload={"fake_result": {"score": 0.75}},
                backend_preferences=("fake",),
                resources={"accelerator": "cpu"},
            )
        )
        wait_for(registry, job.spec.job_id)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            messages = transport.messages[bound.thread_id]
            if any("完了" in message for message in messages):
                break
            time.sleep(0.02)
        assert any("キュー" in message for message in transport.messages[bound.thread_id])
        assert any("完了" in message for message in transport.messages[bound.thread_id])
        assert bound.thread_id in transport.live
        assert "Latest job" in transport.live[bound.thread_id][1]
    finally:
        scheduler.close(cancel_running=True)


def test_thread_dialogue_answers_status_and_records_steering(tmp_path):
    database = tmp_path / "dialogue.sqlite3"
    registry = ControlPlaneRegistry(database)
    store = WorkSessionStore(database)
    scheduler = JobScheduler(
        registry,
        ComputeBroker([FakeComputeBackend()]),
        worker_count=1,
    )
    service = WorkSessionService(registry, store, scheduler)
    engine = WorkSessionDialogueEngine(
        HarnessConfig(project_root=tmp_path),
        service,
        store,
    )
    try:
        project, session = service.create_session(
            domain="research",
            title="Dialogue",
            project_root=tmp_path / "project",
        )
        status = engine.apply(
            session.work_session_id,
            "今どこまで進んでる？",
            actor="sora",
        )
        assert status.action == "status"
        assert session.work_session_id in status.response

        job = service.queue_job(
            JobSpec.new(
                project_id=project.project_id,
                work_session_id=session.work_session_id,
                domain="research",
                task_type="long",
                payload={
                    "fake_delay_seconds": 0.2,
                    "fake_steps": [{"stage": "one"}, {"stage": "two"}],
                },
                backend_preferences=("fake",),
                resources={"accelerator": "cpu"},
            )
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if registry.get_job(job.spec.job_id).status == "running":
                break
            time.sleep(0.02)
        steering = engine.apply(
            session.work_session_id,
            "実行時間は2時間以内にして",
            actor="sora",
        )
        assert steering.action == "steer"
        pending = store.list_steering(session.work_session_id, status="pending")
        assert len(pending) == 1
        assert "2時間" in pending[0].instruction
    finally:
        scheduler.close(cancel_running=True)
