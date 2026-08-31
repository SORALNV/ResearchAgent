from __future__ import annotations

from pathlib import Path

from harness.compute_backends import FakeComputeBackend
from harness.compute_feedback import ResultFeedbackEngine
from harness.compute_materializer import MaterializationResult
from harness.compute_models import BackendCapabilities, ComputeRuntimeRecord
from harness.compute_scheduler import ComputeBroker, ComputeRuntimeStore
from harness.compute_scheduler_safe import BackendBoundApprovalScheduler
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


def test_paid_compute_approval_is_bound_to_the_selected_backend(tmp_path: Path):
    store = ControlPlaneStore(tmp_path / "control")
    project = store.create_project("paid routing", Domain.RESEARCH)
    session = store.create_work_session(project.project_id, "thread")
    job = store.create_job(
        JobSpec(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain=Domain.RESEARCH,
            kind="experiment",
            payload={"entrypoint": ["python", "run.py"]},
            resources=ResourceRequirements(
                gpu_count=1,
                gpu_memory_mb=12000,
                accelerator="gpu",
            ),
            backend_preferences=("paid_a", "paid_b"),
            max_runtime_seconds=60,
        )
    )
    capabilities = BackendCapabilities(
        accelerators=("gpu",),
        domains=(Domain.RESEARCH,),
        gpu_count=1,
        gpu_memory_mb=24000,
    )
    paid_a = FakeComputeBackend(
        name="paid_a",
        capabilities=capabilities,
        approval_required=True,
    )
    paid_b = FakeComputeBackend(
        name="paid_b",
        capabilities=capabilities,
        approval_required=True,
    )
    broker = ComputeBroker(
        [paid_a, paid_b],
        research_order=("paid_a", "paid_b"),
        kaggle_order=("paid_a", "paid_b"),
    )
    runtime_store = ComputeRuntimeStore(tmp_path / "runtime" / "state")
    scheduler = BackendBoundApprovalScheduler(
        store=store,
        broker=broker,
        runtime_store=runtime_store,
        materializer=PassthroughMaterializer(),
        feedback=ResultFeedbackEngine(store, tmp_path / "runtime"),
        root_dir=tmp_path / "runtime",
        poll_interval_seconds=0.01,
    )

    waiting = store.transition_job(
        job.job_id,
        JobStatus.WAITING_APPROVAL,
        expected_revision=job.revision,
        backend_id="paid_a",
        checkpoint_ref="compute_approval",
    )
    runtime_store.save(
        ComputeRuntimeRecord(
            job_id=job.job_id,
            backend="paid_a",
            workspace=str(tmp_path / "runtime" / "workspaces" / job.job_id),
            artifacts_dir=str(tmp_path / "runtime" / "artifacts" / job.job_id),
            approval_required=True,
        )
    )

    approved = scheduler.approve_job(job.job_id, actor="discord-human:42")
    assert approved.status == JobStatus.QUEUED
    approved_runtime = runtime_store.load(job.job_id)
    assert approved_runtime is not None
    assert approved_runtime.approved is True
    assert approved_runtime.metadata["approved_backend"] == "paid_a"

    paid_a._available = False
    try:
        scheduler.start(recover=False)
        scheduler.run_until_idle(timeout_seconds=5)
        reselection = store.get_job(job.job_id)
        assert reselection.status == JobStatus.WAITING_APPROVAL
        assert reselection.backend_id == "paid_b"
        current_runtime = runtime_store.load(job.job_id)
        assert current_runtime is not None
        assert current_runtime.backend == "paid_b"
        assert current_runtime.approved is False
        assert "paid_a" in current_runtime.metadata["approval_invalidated_reason"]
        assert "paid_b" in current_runtime.metadata["approval_invalidated_reason"]
    finally:
        scheduler.stop(wait=True)
