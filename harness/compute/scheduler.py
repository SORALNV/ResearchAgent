from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from harness.compute.base import (
    BackendDecision,
    BackendStatus,
    ComputeBroker,
    ComputeHandle,
    backend_status_to_job_status,
)
from harness.platform.models import (
    EventKind,
    JobEvent,
    JobRecord,
    JobStatus,
    SteeringKind,
    WorkSessionStatus,
)
from harness.platform.registry import PlatformRegistry


EventCallback = Callable[[JobEvent], None]


class JobScheduler:
    """Durable scheduler for Kaggle, remote GPU, VM, and local jobs.

    A dispatcher reads job IDs from a bounded in-process queue. The source of
    truth remains SQLite, so queued/running jobs are recovered after restart.
    Each active job owns one scheduler worker; Discord and API threads never wait
    on training or Kaggle polling.
    """

    ACTIVE_STATUSES = (
        JobStatus.QUEUED,
        JobStatus.PREPARING,
        JobStatus.SUBMITTED,
        JobStatus.RUNNING,
        JobStatus.COLLECTING,
        JobStatus.CANCEL_REQUESTED,
    )

    def __init__(
        self,
        *,
        registry: PlatformRegistry,
        broker: ComputeBroker,
        root_dir: str | Path,
        max_concurrent_jobs: int = 2,
        poll_interval_seconds: float = 15.0,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.registry = registry
        self.broker = broker
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent_jobs = max(1, max_concurrent_jobs)
        self.poll_interval_seconds = max(0.2, poll_interval_seconds)
        self.event_callback = event_callback
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=10000)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_jobs,
            thread_name_prefix="compute-job",
        )
        self._dispatcher: threading.Thread | None = None
        self._stopping = threading.Event()
        self._lock = threading.RLock()
        self._active: set[str] = set()
        self._queued: set[str] = set()

    def start(self, *, recover: bool = True) -> None:
        with self._lock:
            if self._dispatcher and self._dispatcher.is_alive():
                return
            self._stopping.clear()
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="compute-scheduler",
                daemon=True,
            )
            self._dispatcher.start()
        if recover:
            for job in self.registry.list_jobs(statuses=self.ACTIVE_STATUSES, limit=5000):
                self.enqueue(job.spec.job_id, recover=True)

    def stop(self, *, wait: bool = True, cancel_active: bool = False) -> None:
        self._stopping.set()
        if cancel_active:
            for job_id in self.active_job_ids():
                try:
                    self.cancel(job_id)
                except Exception:
                    continue
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        dispatcher = self._dispatcher
        if wait and dispatcher:
            dispatcher.join(timeout=10)
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def enqueue(self, job_id: str, *, recover: bool = False) -> JobRecord:
        job = self._require_job(job_id)
        if job.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return job
        with self._lock:
            if job_id in self._queued or job_id in self._active:
                return job
            self._queued.add(job_id)
        if not recover or job.status == JobStatus.CREATED:
            job = self.registry.update_job(
                job_id,
                status=JobStatus.QUEUED,
                current_stage="scheduler_queue",
                progress=max(job.progress, 0.0),
            )
            self._update_session(job, WorkSessionStatus.QUEUED, "queued")
        self._queue.put_nowait(job_id)
        return job

    def approve_job(self, job_id: str) -> JobRecord:
        job = self._require_job(job_id)
        if job.status != JobStatus.WAITING_APPROVAL:
            raise ValueError(f"job is not waiting for approval: {job_id}")
        result = dict(job.result)
        result["compute_approved"] = True
        result["approved_at"] = time.time()
        updated = self.registry.update_job(
            job_id,
            status=JobStatus.QUEUED,
            current_stage="approved_for_compute",
            result=result,
            error="",
        )
        self.enqueue(job_id, recover=True)
        return updated

    def cancel(self, job_id: str) -> JobRecord:
        job = self._require_job(job_id)
        if job.status in {
            JobStatus.CREATED,
            JobStatus.QUEUED,
            JobStatus.WAITING_APPROVAL,
            JobStatus.BLOCKED,
        }:
            updated = self.registry.update_job(
                job_id,
                status=JobStatus.CANCELLED,
                current_stage="cancelled_before_submit",
                progress=job.progress,
            )
            self._update_session(updated, WorkSessionStatus.PAUSED, "cancelled")
            return updated
        if not job.backend:
            return self.registry.update_job(
                job_id,
                status=JobStatus.CANCEL_REQUESTED,
                current_stage="cancel_requested",
            )
        backend = self.broker.backend(job.backend)
        handle = backend.cancel(job)
        updated = self._apply_handle(job, handle)
        self._update_session(
            updated,
            WorkSessionStatus.PAUSED,
            handle.stage,
        )
        return updated

    def active_job_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._active))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "active": sorted(self._active),
                "queued": sorted(self._queued),
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "queue_size": self._queue.qsize(),
                "stopping": self._stopping.is_set(),
            }

    def _dispatch_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job_id is None:
                self._queue.task_done()
                return
            with self._lock:
                self._queued.discard(job_id)
                if job_id in self._active:
                    self._queue.task_done()
                    continue
                self._active.add(job_id)
            future = self._executor.submit(self._run_job, job_id)
            future.add_done_callback(
                lambda _future, value=job_id: self._job_finished(value)
            )
            self._queue.task_done()

    def _job_finished(self, job_id: str) -> None:
        with self._lock:
            self._active.discard(job_id)

    def _run_job(self, job_id: str) -> None:
        try:
            job = self._require_job(job_id)
            if job.status == JobStatus.CANCEL_REQUESTED:
                self.cancel(job_id)
                return
            decision = self.broker.decide(job.spec)
            self._emit_decision(job, decision)
            if decision.selected is None:
                updated = self.registry.update_job(
                    job_id,
                    status=JobStatus.BLOCKED,
                    current_stage="no_compute_backend",
                    error=decision.reason + ": " + json.dumps(decision.rejected),
                )
                self._update_session(
                    updated,
                    WorkSessionStatus.WAITING_INPUT,
                    "no_compute_backend",
                )
                return
            if decision.requires_approval and not job.result.get("compute_approved"):
                updated = self.registry.update_job(
                    job_id,
                    status=JobStatus.WAITING_APPROVAL,
                    backend=decision.selected,
                    current_stage="compute_approval",
                    result={
                        **job.result,
                        "backend_decision": {
                            "selected": decision.selected,
                            "reason": decision.reason,
                            "rejected": decision.rejected,
                        },
                    },
                )
                self._emit(
                    JobEvent.new(
                        work_session_id=job.spec.work_session_id,
                        job_id=job_id,
                        kind=EventKind.APPROVAL,
                        message=(
                            f"Compute approval required for backend {decision.selected}"
                        ),
                        payload={
                            "operation": "paid_compute",
                            "backend": decision.selected,
                            "job_id": job_id,
                            "resources": job.spec.resources.to_dict(),
                        },
                    )
                )
                self._update_session(
                    updated,
                    WorkSessionStatus.WAITING_APPROVAL,
                    "compute_approval",
                )
                return

            backend = self.broker.backend(decision.selected)
            workspace = self._workspace(job)
            if not job.backend_job_id:
                job = self.registry.update_job(
                    job_id,
                    status=JobStatus.PREPARING,
                    backend=backend.name,
                    current_stage="prepare",
                    progress=max(job.progress, 0.01),
                )
                self._update_session(job, WorkSessionStatus.RUNNING, "prepare")
                backend.prepare(job, workspace)
                self._apply_steering(job)
                if self._stopping.is_set():
                    return
                job = self._require_job(job_id)
                handle = backend.submit(job, workspace)
                job = self._apply_handle(job, handle)
            else:
                job = self.registry.update_job(
                    job_id,
                    status=job.status,
                    current_stage="resume_polling",
                    emit_event=False,
                )

            unknown_count = 0
            while not self._stopping.is_set():
                current = self._require_job(job_id)
                if current.status == JobStatus.CANCEL_REQUESTED:
                    self.cancel(job_id)
                    return
                if current.status in {
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }:
                    break
                self._apply_steering(current)
                handle = backend.poll(current)
                if handle.status == BackendStatus.UNKNOWN:
                    unknown_count += 1
                    if unknown_count >= 8:
                        updated = self.registry.update_job(
                            job_id,
                            status=JobStatus.BLOCKED,
                            current_stage="backend_status_unknown",
                            error=handle.error or handle.message,
                        )
                        self._update_session(
                            updated,
                            WorkSessionStatus.WAITING_INPUT,
                            "backend_status_unknown",
                        )
                        return
                else:
                    unknown_count = 0
                job = self._apply_handle(current, handle)
                if handle.terminal:
                    break
                time.sleep(self.poll_interval_seconds)

            job = self._require_job(job_id)
            if job.status == JobStatus.COMPLETED:
                job = self.registry.update_job(
                    job_id,
                    status=JobStatus.COLLECTING,
                    current_stage="collect",
                    progress=max(job.progress, 0.95),
                )
                destination = self.root_dir / "artifacts" / job_id
                handle = backend.collect(job, destination)
                job = self._apply_handle(job, handle)
                self._update_session(
                    job,
                    WorkSessionStatus.REVIEW,
                    "results_ready",
                )
            elif job.status == JobStatus.FAILED:
                self._update_session(job, WorkSessionStatus.FAILED, "job_failed")
            elif job.status == JobStatus.CANCELLED:
                self._update_session(job, WorkSessionStatus.PAUSED, "cancelled")
        except Exception as exc:
            try:
                job = self._require_job(job_id)
                updated = self.registry.update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    current_stage="scheduler_exception",
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._update_session(
                    updated,
                    WorkSessionStatus.FAILED,
                    "scheduler_exception",
                )
            except Exception:
                return

    def _apply_handle(self, job: JobRecord, handle: ComputeHandle) -> JobRecord:
        updated = self.registry.update_job(
            job.spec.job_id,
            status=backend_status_to_job_status(handle.status),
            backend=handle.backend,
            backend_job_id=handle.backend_job_id,
            current_stage=handle.stage,
            progress=handle.progress,
            result=handle.result if handle.result else job.result,
            error=handle.error or "",
        )
        kind = EventKind.ERROR if handle.error else (
            EventKind.MILESTONE if handle.terminal else EventKind.PROGRESS
        )
        self._emit(
            JobEvent.new(
                work_session_id=job.spec.work_session_id,
                job_id=job.spec.job_id,
                kind=kind,
                message=handle.message or f"{handle.status.value}: {handle.stage}",
                payload=handle.to_dict(),
            )
        )
        return updated

    def _apply_steering(self, job: JobRecord) -> None:
        events = self.registry.list_pending_steering(
            job.spec.work_session_id,
            job_id=job.spec.job_id,
        )
        for event in events:
            if event.kind == SteeringKind.CANCEL:
                self.registry.mark_steering_applied(event.steering_id)
                self.cancel(job.spec.job_id)
                return
            self._emit(
                JobEvent.new(
                    work_session_id=job.spec.work_session_id,
                    job_id=job.spec.job_id,
                    kind=EventKind.STEERING,
                    message=f"Steering acknowledged at checkpoint: {event.instruction}",
                    payload={
                        "steering_id": event.steering_id,
                        "kind": event.kind.value,
                        "apply_after": event.apply_after,
                    },
                )
            )
            self.registry.mark_steering_applied(event.steering_id)

    def _emit_decision(self, job: JobRecord, decision: BackendDecision) -> None:
        self._emit(
            JobEvent.new(
                work_session_id=job.spec.work_session_id,
                job_id=job.spec.job_id,
                kind=EventKind.STATUS,
                message=decision.reason,
                payload={
                    "selected": decision.selected,
                    "candidates": list(decision.ordered_candidates),
                    "rejected": {
                        key: list(value) for key, value in decision.rejected.items()
                    },
                    "requires_approval": decision.requires_approval,
                },
            )
        )

    def _emit(self, event: JobEvent) -> JobEvent:
        stored = self.registry.append_event(event)
        if self.event_callback:
            try:
                self.event_callback(stored)
            except Exception:
                pass
        return stored

    def _update_session(
        self,
        job: JobRecord,
        status: WorkSessionStatus,
        stage: str,
    ) -> None:
        try:
            self.registry.update_work_session(
                job.spec.work_session_id,
                status=status,
                current_stage=stage,
            )
        except Exception:
            pass

    def _workspace(self, job: JobRecord) -> Path:
        configured = job.spec.metadata.get("workspace")
        path = (
            Path(str(configured)).expanduser().resolve()
            if configured
            else self.root_dir / "jobs" / job.spec.job_id
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _require_job(self, job_id: str) -> JobRecord:
        job = self.registry.get_job(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id}")
        return job
