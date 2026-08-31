from __future__ import annotations

import os
import time
from pathlib import Path

from harness.compute_backends import (
    FakeComputeBackend,
    LocalGpuInventory,
    LocalProcessBackend,
)
from harness.compute_feedback import ResultFeedbackEngine
from harness.compute_materializer import MaterializationResult
from harness.compute_scheduler import ComputeBroker, ComputeRuntimeStore, ComputeScheduler
from harness.control_plane import (
    ControlPlaneStore,
    Domain,
    JobSpec,
    JobStatus,
    ResourceRequirements,
)


class PassthroughMaterializer:
    def materialize(self, job, workspace: Path) -> MaterializationResult:
        workspace.mkdir(parents=True, exist_ok=True)
        return MaterializationResult(
            job=job,
            workspace=str(workspace),
            provider="test",
            smoke_command=(),
            smoke_stdout="",
            smoke_stderr="",
            generated_at="2026-01-01T00:00:00Z",
        )


def _setup(tmp_path: Path):
    store = ControlPlaneStore(tmp_path / "control")
    project = store.create_project("recovery", Domain.RESEARCH)
    session = store.create_work_session(project.project_id, "thread")
    job = store.create_job(
        JobSpec(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain=Domain.RESEARCH,
            kind="experiment",
            payload={"entrypoint": ["python", "run.py"]},
            resources=ResourceRequirements(accelerator="cpu"),
            backend_preferences=("fake",),
            max_runtime_seconds=60,
        )
    )
    return store, session, job


def _scheduler(tmp_path: Path, store, backend):
    root = tmp_path / "runtime"
    runtime = ComputeRuntimeStore(root / "state")
    feedback = ResultFeedbackEngine(store, root)
    broker = ComputeBroker(
        [backend],
        research_order=("fake",),
        kaggle_order=("fake",),
    )
    return ComputeScheduler(
        store=store,
        broker=broker,
        runtime_store=runtime,
        materializer=PassthroughMaterializer(),
        feedback=feedback,
        root_dir=root,
        max_concurrent_jobs=1,
        poll_interval_seconds=0.2,
    )


def test_running_job_recovers_after_core_scheduler_restart(tmp_path: Path):
    store, _, job = _setup(tmp_path)
    backend = FakeComputeBackend(complete_after_polls=5)
    first = _scheduler(tmp_path, store, backend)
    first.start(recover=False)
    first.enqueue(job.job_id)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = store.get_job(job.job_id)
        runtime = first.runtime_store.load(job.job_id)
        if current.status == JobStatus.RUNNING and runtime and runtime.handle:
            break
        time.sleep(0.02)
    assert store.get_job(job.job_id).status == JobStatus.RUNNING
    first.stop(wait=True, cancel_active=False)
    assert store.get_job(job.job_id).status == JobStatus.RUNNING

    second = _scheduler(tmp_path, store, backend)
    try:
        recovered = second.recover()
        assert job.job_id in recovered
        second.start(recover=False)
        second.run_until_idle(timeout_seconds=10)
        assert store.get_job(job.job_id).status == JobStatus.SUCCEEDED
    finally:
        second.stop(wait=True)


def test_experiment_environment_drops_all_control_plane_credentials(
    tmp_path: Path,
    monkeypatch,
):
    store, _, job = _setup(tmp_path)
    for name in (
        "DISCORD_BOT_TOKEN",
        "OPENAI_API_KEY",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "REMOTE_GPU_WORKER_TOKEN",
        "WORKER_TOKEN",
        "CODEX_HOME",
    ):
        monkeypatch.setenv(name, f"secret-{name}")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    environment = LocalProcessBackend._environment(job, tmp_path)
    for name in (
        "DISCORD_BOT_TOKEN",
        "OPENAI_API_KEY",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "REMOTE_GPU_WORKER_TOKEN",
        "WORKER_TOKEN",
        "CODEX_HOME",
    ):
        assert name not in environment
    assert environment["RESEARCH_AGENT_JOB_ID"] == job.job_id


def test_worker_compose_does_not_import_core_env_or_credentials():
    compose = Path("deploy/compose.worker.yaml").read_text(encoding="utf-8")
    assert "env_file:" not in compose
    assert "../.env" not in compose
    for forbidden in (
        "DISCORD_BOT_TOKEN",
        "OPENAI_API_KEY",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "CODEX_HOME",
    ):
        assert forbidden not in compose
    assert "WORKER_TOKEN" in compose

    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    assert ".env.*" in dockerignore
    assert "**/.env.*" in dockerignore
