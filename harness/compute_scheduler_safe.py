from __future__ import annotations

from dataclasses import replace

from harness.compute_models import ComputeRuntimeRecord
from harness.compute_scheduler import ComputeScheduler, ComputeStack
from harness.control_plane import EventLane, Job, JobStatus
from harness.state import utc_timestamp


class BackendBoundApprovalScheduler(ComputeScheduler):
    """ComputeScheduler whose operational approval is backend-specific.

    A paid-backend approval must not authorize a different paid Worker selected
    after availability or routing changes. The approved backend is persisted in
    ComputeRuntimeRecord.metadata and invalidated before the normal scheduler
    selection path runs when the next selected backend differs.
    """

    @classmethod
    def from_scheduler(
        cls,
        scheduler: ComputeScheduler,
    ) -> "BackendBoundApprovalScheduler":
        return cls(
            store=scheduler.store,
            broker=scheduler.broker,
            runtime_store=scheduler.runtime_store,
            materializer=scheduler.materializer,
            feedback=scheduler.feedback,
            root_dir=scheduler.root_dir,
            max_concurrent_jobs=scheduler.max_concurrent_jobs,
            poll_interval_seconds=scheduler.poll_interval_seconds,
            lease_seconds=scheduler.lease_seconds,
            max_unknown_polls=scheduler.max_unknown_polls,
            artifact_max_files=scheduler.artifact_max_files,
            artifact_max_bytes=scheduler.artifact_max_bytes,
            worker_id=scheduler.worker_id,
            event_callback=scheduler.event_callback,
        )

    def approve_job(self, job_id: str, *, actor: str) -> Job:
        job = self.store.get_job(job_id)
        if job.status != JobStatus.WAITING_APPROVAL:
            raise ValueError(f"job is not waiting for compute approval: {job_id}")
        runtime = self.runtime_store.load(job_id)
        if runtime is None or not runtime.backend:
            raise ValueError("compute selection state is missing")
        approved_backend = runtime.backend
        runtime = self.runtime_store.save(
            replace(
                runtime,
                approved=True,
                metadata={
                    **runtime.metadata,
                    "approved_backend": approved_backend,
                    "approved_by": actor,
                    "approved_at": utc_timestamp(),
                },
            )
        )
        updated = self.store.transition_job(
            job_id,
            JobStatus.QUEUED,
            expected_revision=job.revision,
            backend_id=approved_backend,
            error="",
        )
        self._emit(
            event_type="compute.approval.accepted",
            lane=EventLane.CONTROL,
            job=updated,
            actor=actor,
            payload={"backend": approved_backend},
            idempotency_key=(
                f"compute:{job_id}:approval:{approved_backend}:{actor}"
            ),
        )
        self.enqueue(job_id)
        return updated

    def _run_job(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        runtime = self.runtime_store.load(job_id)
        if (
            job.status == JobStatus.QUEUED
            and runtime is not None
            and runtime.approved
        ):
            selection = self.broker.decide(job.spec)
            approved_backend = str(
                runtime.metadata.get("approved_backend") or ""
            )
            if not approved_backend or selection.selected != approved_backend:
                self.runtime_store.save(
                    replace(
                        runtime,
                        approved=False,
                        metadata={
                            **runtime.metadata,
                            "approval_invalidated_at": utc_timestamp(),
                            "approval_invalidated_reason": (
                                "selected backend changed from "
                                f"{approved_backend or '<unbound>'} to "
                                f"{selection.selected or '<none>'}"
                            ),
                            "approved_backend": None,
                        },
                    )
                )
        super()._run_job(job_id)


def harden_compute_stack(stack: ComputeStack) -> ComputeStack:
    scheduler = BackendBoundApprovalScheduler.from_scheduler(stack.scheduler)
    return ComputeStack(
        broker=stack.broker,
        scheduler=scheduler,
        feedback=stack.feedback,
        runtime_store=stack.runtime_store,
    )
