from __future__ import annotations

import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from harness.artifacts import ArtifactRecord, build_artifact_manifest
from harness.compute_backends import (
    KaggleNotebookBackend,
    LocalProcessBackend,
    RemoteGpuBackend,
    RemoteWorkerDescriptor,
    detect_local_gpu,
)
from harness.compute_feedback import ProviderFeedbackPlanner, ResultFeedbackEngine
from harness.compute_materializer import (
    ExperimentMaterializer,
    MaterializationResult,
    ProviderExperimentMaterializer,
)
from harness.compute_models import (
    BackendCapabilities,
    BackendHandle,
    BackendSelection,
    BackendState,
    ComputeBackend,
    ComputeRuntimeRecord,
    canonical_json_hash,
)
from harness.config import HarnessConfig
from harness.control_plane import (
    ConflictError,
    ControlPlaneStore,
    Domain,
    Event,
    EventLane,
    Job,
    JobSpec,
    JobStatus,
    SteeringKind,
    SteeringStatus,
)
from harness.state import utc_timestamp


EventCallback = Callable[[Event], None]


@dataclass(frozen=True)
class ComputeStack:
    broker: "ComputeBroker"
    scheduler: "ComputeScheduler"
    feedback: ResultFeedbackEngine
    runtime_store: "ComputeRuntimeStore"


class ComputeRuntimeStore:
    """Durable backend handles stored separately from immutable JobSpec."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def load(self, job_id: str) -> ComputeRuntimeRecord | None:
        path = self._path(job_id)
        if not path.is_file():
            return None
        with self._lock:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return (
            ComputeRuntimeRecord.from_dict(value)
            if isinstance(value, Mapping)
            else None
        )

    def save(self, record: ComputeRuntimeRecord) -> ComputeRuntimeRecord:
        path = self._path(record.job_id)
        payload = record.to_dict()
        payload["updated_at"] = utc_timestamp()
        with self._lock:
            _atomic_json(path, payload)
        return ComputeRuntimeRecord.from_dict(payload)

    def list(self) -> list[ComputeRuntimeRecord]:
        result: list[ComputeRuntimeRecord] = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, Mapping):
                    result.append(ComputeRuntimeRecord.from_dict(value))
            except Exception:
                continue
        return result

    def _path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{_safe_component(job_id)}.json"


class ComputeBroker:
    """Select the first available backend satisfying the common JobSpec."""

    def __init__(
        self,
        backends: Iterable[ComputeBackend] = (),
        *,
        research_order: Iterable[str] = (
            "remote_gpu",
            "local_gpu",
            "local_cpu",
        ),
        kaggle_order: Iterable[str] = (
            "kaggle_notebook",
            "remote_gpu",
            "local_gpu",
            "local_cpu",
        ),
    ) -> None:
        self.backends: dict[str, ComputeBackend] = {}
        self.research_order = tuple(dict.fromkeys(str(item) for item in research_order))
        self.kaggle_order = tuple(dict.fromkeys(str(item) for item in kaggle_order))
        for backend in backends:
            self.register(backend)

    def register(self, backend: ComputeBackend) -> None:
        if not backend.name.strip():
            raise ValueError("compute backend name must be non-empty")
        if backend.name in self.backends:
            raise ValueError(f"duplicate compute backend: {backend.name}")
        self.backends[backend.name] = backend

    def decide(self, spec: JobSpec) -> BackendSelection:
        ordered: list[str] = []

        def add(name: str) -> None:
            if name in self.backends and name not in ordered:
                ordered.append(name)

        for name in spec.backend_preferences:
            add(str(name))
        for name in (
            self.kaggle_order
            if spec.domain == Domain.KAGGLE
            else self.research_order
        ):
            add(name)
        for name in sorted(self.backends):
            add(name)

        rejected: dict[str, tuple[str, ...]] = {}
        for name in ordered:
            backend = self.backends[name]
            try:
                available, detail = backend.available()
            except Exception as exc:
                rejected[name] = (f"availability error: {type(exc).__name__}: {exc}",)
                continue
            if not available:
                rejected[name] = (f"unavailable: {detail}",)
                continue
            supported, reasons = backend.capabilities.satisfies(spec)
            if not supported:
                rejected[name] = reasons
                continue
            approval_required = bool(
                getattr(backend, "approval_required", False)
                or spec.requires_approval
            )
            return BackendSelection(
                selected=name,
                ordered_candidates=tuple(ordered),
                rejected=rejected,
                reason=(
                    f"{name} is the first configured backend that is available "
                    "and satisfies the JobSpec"
                ),
                approval_required=approval_required,
            )
        return BackendSelection(
            selected=None,
            ordered_candidates=tuple(ordered),
            rejected=rejected,
            reason="No configured compute backend satisfies the JobSpec",
        )

    def backend(self, name: str) -> ComputeBackend:
        try:
            return self.backends[name]
        except KeyError as exc:
            raise KeyError(f"unknown compute backend: {name}") from exc

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, backend in self.backends.items():
            try:
                available, detail = backend.available()
            except Exception as exc:
                available, detail = False, f"{type(exc).__name__}: {exc}"
            result[name] = {
                "available": available,
                "detail": detail,
                "approval_required": bool(
                    getattr(backend, "approval_required", False)
                ),
                "capabilities": backend.capabilities.to_dict(),
            }
        return result


class ComputeScheduler:
    """Asynchronous, restart-recoverable execution over pluggable backends."""

    RECOVERABLE_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING)
    TERMINAL_STATUSES = (
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    )

    def __init__(
        self,
        *,
        store: ControlPlaneStore,
        broker: ComputeBroker,
        runtime_store: ComputeRuntimeStore,
        materializer: ExperimentMaterializer,
        feedback: ResultFeedbackEngine,
        root_dir: str | Path,
        max_concurrent_jobs: int = 2,
        poll_interval_seconds: float = 15.0,
        lease_seconds: int = 300,
        max_unknown_polls: int = 8,
        artifact_max_files: int = 1000,
        artifact_max_bytes: int = 2 * 1024 * 1024 * 1024,
        worker_id: str | None = None,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.store = store
        self.broker = broker
        self.runtime_store = runtime_store
        self.materializer = materializer
        self.feedback = feedback
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent_jobs = max(1, int(max_concurrent_jobs))
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self.lease_seconds = max(30, int(lease_seconds))
        self.max_unknown_polls = max(1, int(max_unknown_polls))
        self.artifact_max_files = max(1, int(artifact_max_files))
        self.artifact_max_bytes = max(1, int(artifact_max_bytes))
        self.worker_id = worker_id or f"core-{os.getpid()}"
        self.event_callback = event_callback

        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=10000)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_jobs,
            thread_name_prefix="research-compute",
        )
        self._dispatcher: threading.Thread | None = None
        self._stopping = threading.Event()
        self._lock = threading.RLock()
        self._queued: set[str] = set()
        self._active: set[str] = set()

    def start(self, *, recover: bool = True) -> None:
        with self._lock:
            if self._dispatcher and self._dispatcher.is_alive():
                return
            self._stopping.clear()
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="research-compute-dispatcher",
                daemon=True,
            )
            self._dispatcher.start()
        if recover:
            self.recover()

    def stop(self, *, wait: bool = True, cancel_active: bool = False) -> None:
        self._stopping.set()
        if cancel_active:
            for job_id in self.active_job_ids():
                try:
                    self.cancel_job(job_id, actor="scheduler_shutdown")
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

    def recover(self) -> tuple[str, ...]:
        recovered: list[str] = []
        for session in self.store.list_work_sessions():
            for job in self.store.list_jobs(
                work_session_id=session.work_session_id,
                statuses=self.RECOVERABLE_STATUSES,
            ):
                self.enqueue(job.job_id)
                recovered.append(job.job_id)
        return tuple(recovered)

    def enqueue(self, job_id: str) -> Job:
        job = self.store.get_job(job_id)
        if job.status in self.TERMINAL_STATUSES:
            return job
        if job.status == JobStatus.WAITING_APPROVAL:
            return job
        if job.status == JobStatus.PAUSED:
            job = self.store.transition_job(
                job_id,
                JobStatus.QUEUED,
                expected_revision=job.revision,
                backend_id=job.backend_id,
                checkpoint_ref=job.checkpoint_ref,
                error="",
            )
        with self._lock:
            if job_id in self._queued or job_id in self._active:
                return job
            self._queued.add(job_id)
        self._queue.put_nowait(job_id)
        self._emit(
            event_type="compute.job.enqueued",
            lane=EventLane.STATUS,
            job=job,
            payload={"status": job.status.value},
            idempotency_key=f"compute:{job_id}:enqueued:{job.attempt}",
        )
        return job

    def approve_job(self, job_id: str, *, actor: str) -> Job:
        job = self.store.get_job(job_id)
        if job.status != JobStatus.WAITING_APPROVAL:
            raise ValueError(f"job is not waiting for compute approval: {job_id}")
        runtime = self.runtime_store.load(job_id)
        if runtime is None or not runtime.backend:
            raise ValueError("compute selection state is missing")
        runtime = self.runtime_store.save(replace(runtime, approved=True))
        updated = self.store.transition_job(
            job_id,
            JobStatus.QUEUED,
            expected_revision=job.revision,
            backend_id=runtime.backend,
            error="",
        )
        self._emit(
            event_type="compute.approval.accepted",
            lane=EventLane.CONTROL,
            job=updated,
            actor=actor,
            payload={"backend": runtime.backend},
            idempotency_key=(
                f"compute:{job_id}:approval:{runtime.backend}:{actor}"
            ),
        )
        self.enqueue(job_id)
        return updated

    def cancel_job(self, job_id: str, *, actor: str) -> Job:
        job = self.store.get_job(job_id)
        if job.status in self.TERMINAL_STATUSES:
            return job
        runtime = self.runtime_store.load(job_id)
        if job.status == JobStatus.RUNNING and runtime and runtime.backend:
            backend = self.broker.backend(runtime.backend)
            handle = backend.cancel(job, runtime.handle)
            self.runtime_store.save(replace(runtime, handle=handle))
        updated = self.store.transition_job(
            job_id,
            JobStatus.CANCELLED,
            expected_revision=job.revision,
            backend_id=(runtime.backend if runtime else job.backend_id),
            checkpoint_ref=(runtime.result_ref if runtime else job.checkpoint_ref),
            error="",
        )
        self._emit(
            event_type="compute.job.cancelled",
            lane=EventLane.CONTROL,
            job=updated,
            actor=actor,
            payload={"backend": runtime.backend if runtime else job.backend_id},
            idempotency_key=f"compute:{job_id}:cancelled",
        )
        return updated

    def run_until_idle(self, *, timeout_seconds: float = 30.0) -> None:
        self.start(recover=False)
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if not snapshot["active"] and not snapshot["queued"]:
                return
            time.sleep(0.02)
        raise TimeoutError("compute scheduler did not become idle")

    def active_job_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._active))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": sorted(self._active),
                "queued": sorted(self._queued),
                "queue_size": self._queue.qsize(),
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "poll_interval_seconds": self.poll_interval_seconds,
                "worker_id": self.worker_id,
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
            job = self.store.get_job(job_id)
            if job.status in self.TERMINAL_STATUSES or job.status == JobStatus.WAITING_APPROVAL:
                return
            runtime = self.runtime_store.load(job_id) or ComputeRuntimeRecord(
                job_id=job_id,
                workspace=str(self._workspace(job)),
                artifacts_dir=str(self._artifacts_dir(job)),
            )

            if job.status == JobStatus.QUEUED:
                selection = self.broker.decide(job.spec)
                runtime = self.runtime_store.save(
                    replace(
                        runtime,
                        backend=selection.selected,
                        approval_required=selection.approval_required,
                        selected_at=utc_timestamp(),
                        metadata={
                            **runtime.metadata,
                            "selection": selection.to_dict(),
                        },
                    )
                )
                self._emit_selection(job, selection)
                if selection.selected is None:
                    self._pause_job(
                        job,
                        stage="no_compute_backend",
                        error=(
                            selection.reason
                            + ": "
                            + json.dumps(selection.rejected, ensure_ascii=False)
                        ),
                    )
                    return
                if selection.approval_required and not runtime.approved:
                    waiting = self.store.transition_job(
                        job_id,
                        JobStatus.WAITING_APPROVAL,
                        expected_revision=job.revision,
                        backend_id=selection.selected,
                        checkpoint_ref="compute_approval",
                    )
                    self._emit(
                        event_type="compute.approval.required",
                        lane=EventLane.CONTROL,
                        job=waiting,
                        payload={
                            "backend": selection.selected,
                            "resources": waiting.spec.resources.to_dict(),
                            "reason": selection.reason,
                        },
                        idempotency_key=(
                            f"compute:{job_id}:approval-required:{selection.selected}"
                        ),
                    )
                    return
                job = self.store.transition_job(
                    job_id,
                    JobStatus.RUNNING,
                    expected_revision=job.revision,
                    backend_id=selection.selected,
                    lease_owner=self.worker_id,
                    lease_expires_at=self._lease_expiry(),
                    error="",
                )
            elif job.status != JobStatus.RUNNING:
                return

            runtime = self.runtime_store.load(job_id) or runtime
            if not runtime.backend:
                self._fail_job(job, "running job has no selected backend", "runtime_missing")
                return
            backend = self.broker.backend(runtime.backend)
            effective_job = self._effective_job(job, runtime)

            if runtime.handle is None:
                materialized = self.materializer.materialize(
                    job,
                    Path(runtime.workspace),
                )
                effective_job = materialized.job
                runtime = self.runtime_store.save(
                    replace(
                        runtime,
                        metadata={
                            **runtime.metadata,
                            "materialization": materialized.to_dict(),
                            "effective_job": effective_job.to_dict(),
                        },
                    )
                )
                self._emit(
                    event_type="compute.smoke.passed",
                    lane=EventLane.STATUS,
                    job=job,
                    payload={
                        "backend": backend.name,
                        "provider": materialized.provider,
                        "smoke_command": list(materialized.smoke_command),
                    },
                    idempotency_key=f"compute:{job_id}:smoke-passed",
                )
                handle = backend.submit(effective_job, Path(runtime.workspace))
                runtime = self.runtime_store.save(replace(runtime, handle=handle))
                self._emit(
                    event_type="compute.job.submitted",
                    lane=EventLane.STATUS,
                    job=job,
                    payload=handle.to_dict(),
                    idempotency_key=f"compute:{job_id}:submitted:{backend.name}",
                )

            self._poll_until_terminal(job_id, effective_job, backend)
        except ConflictError:
            return
        except Exception as exc:
            try:
                job = self.store.get_job(job_id)
                if job.status not in self.TERMINAL_STATUSES:
                    self._fail_job(
                        job,
                        f"{type(exc).__name__}: {exc}",
                        "scheduler_exception",
                    )
            except Exception:
                return

    def _poll_until_terminal(
        self,
        job_id: str,
        effective_job: Job,
        backend: ComputeBackend,
    ) -> None:
        while not self._stopping.is_set():
            current = self.store.get_job(job_id)
            if current.status != JobStatus.RUNNING:
                return
            runtime = self.runtime_store.load(job_id)
            if runtime is None or runtime.handle is None:
                self._fail_job(current, "backend handle is missing", "handle_missing")
                return
            if self._runtime_expired(current):
                try:
                    cancelled = backend.cancel(effective_job, runtime.handle)
                    self.runtime_store.save(replace(runtime, handle=cancelled))
                except Exception:
                    pass
                self._fail_job(
                    current,
                    "job exceeded max_runtime_seconds",
                    "runtime_limit_exceeded",
                )
                return
            if not self._apply_steering(current, effective_job, backend, runtime):
                return

            handle = backend.poll(effective_job, runtime.handle)
            unknown_polls = (
                runtime.unknown_polls + 1
                if handle.state == BackendState.UNKNOWN
                else 0
            )
            runtime = self.runtime_store.save(
                replace(
                    runtime,
                    handle=handle,
                    unknown_polls=unknown_polls,
                )
            )
            self._record_progress(current, runtime)
            self._renew_lease(current)

            if handle.state == BackendState.UNKNOWN:
                if unknown_polls >= self.max_unknown_polls:
                    self._pause_job(
                        self.store.get_job(job_id),
                        stage="backend_status_unknown",
                        error=handle.error or handle.message or "backend status unknown",
                    )
                    return
            elif handle.state == BackendState.FAILED:
                self._fail_job(
                    self.store.get_job(job_id),
                    handle.error or handle.message or "backend job failed",
                    handle.stage or "backend_failed",
                )
                return
            elif handle.state == BackendState.CANCELLED:
                self.cancel_job(job_id, actor=f"backend:{backend.name}")
                return
            elif handle.state == BackendState.SUCCEEDED:
                self._collect_and_complete(
                    self.store.get_job(job_id),
                    effective_job,
                    backend,
                    runtime,
                )
                return
            time.sleep(self.poll_interval_seconds)

    def _collect_and_complete(
        self,
        job: Job,
        effective_job: Job,
        backend: ComputeBackend,
        runtime: ComputeRuntimeRecord,
    ) -> None:
        if runtime.handle is None:
            raise ValueError("cannot collect without a backend handle")
        destination = Path(runtime.artifacts_dir).expanduser().resolve()
        collected = backend.collect(
            effective_job,
            runtime.handle,
            destination,
        )
        manifest, manifest_warnings = build_artifact_manifest(
            destination,
            max_files=self.artifact_max_files,
            max_bytes=self.artifact_max_bytes,
        )
        artifact_refs = tuple(
            _artifact_ref(job.job_id, record) for record in manifest
        )
        outcome = self.feedback.integrate(
            job=job,
            collected_result=collected.result,
            artifacts_dir=destination,
            artifact_refs=artifact_refs,
            backend=backend.name,
        )
        runtime = self.runtime_store.save(
            replace(
                runtime,
                collection_complete=True,
                result_ref=outcome.result_ref,
                metadata={
                    **runtime.metadata,
                    "collected": {
                        "artifact_paths": list(collected.artifact_paths),
                        "warnings": [
                            *collected.warnings,
                            *manifest_warnings,
                        ],
                        "metadata": collected.metadata,
                    },
                    "feedback_path": outcome.feedback_path,
                },
            )
        )
        completed = self.store.transition_job(
            job.job_id,
            JobStatus.SUCCEEDED,
            expected_revision=job.revision,
            backend_id=backend.name,
            checkpoint_ref=outcome.result_ref,
            artifact_refs=artifact_refs,
            error="",
        )
        self._emit(
            event_type="compute.job.completed",
            lane=EventLane.STATUS,
            job=completed,
            payload={
                "backend": backend.name,
                "result_ref": outcome.result_ref,
                "artifact_refs": list(artifact_refs),
                "next_hypotheses": [
                    item.to_dict() for item in outcome.proposals
                ],
                "next_required_human_action": "result_interpretation",
            },
            idempotency_key=f"compute:{job.job_id}:completed:{outcome.result_ref}",
        )

    def _apply_steering(
        self,
        job: Job,
        effective_job: Job,
        backend: ComputeBackend,
        runtime: ComputeRuntimeRecord,
    ) -> bool:
        claimed = self.store.claim_steering(
            work_session_id=job.spec.work_session_id,
            job_id=job.job_id,
            consumer=self.worker_id,
            limit=20,
        )
        if not claimed:
            return True
        workspace = Path(runtime.workspace).expanduser().resolve()
        steering_path = workspace / "steering.jsonl"
        steering_path.parent.mkdir(parents=True, exist_ok=True)
        with steering_path.open("a", encoding="utf-8") as handle_file:
            for item in claimed:
                handle_file.write(
                    json.dumps(item.to_dict(), ensure_ascii=False) + "\n"
                )
                handle_file.flush()
                if item.kind == SteeringKind.CANCEL:
                    cancelled_handle = backend.cancel(effective_job, runtime.handle)
                    self.runtime_store.save(
                        replace(runtime, handle=cancelled_handle)
                    )
                    self.store.resolve_steering(
                        item.steering_id,
                        SteeringStatus.APPLIED,
                        consumer=self.worker_id,
                        applied_checkpoint="compute_poll",
                        resolution="job cancelled",
                    )
                    self.cancel_job(job.job_id, actor="steering")
                    return False
                self.store.resolve_steering(
                    item.steering_id,
                    SteeringStatus.APPLIED,
                    consumer=self.worker_id,
                    applied_checkpoint="compute_poll",
                    resolution="written to steering.jsonl for experiment checkpoint",
                )
                self._emit(
                    event_type="compute.steering.applied",
                    lane=EventLane.CONTROL,
                    job=job,
                    payload={
                        "steering_id": item.steering_id,
                        "kind": item.kind.value,
                        "text": item.text,
                    },
                    idempotency_key=(
                        f"compute:{job.job_id}:steering:{item.steering_id}"
                    ),
                )
        return True

    def _effective_job(
        self,
        job: Job,
        runtime: ComputeRuntimeRecord,
    ) -> Job:
        value = runtime.metadata.get("effective_job")
        if isinstance(value, Mapping):
            try:
                return Job.from_dict(value)
            except Exception:
                pass
        materialized = runtime.metadata.get("materialization")
        if isinstance(materialized, Mapping):
            try:
                return MaterializationResult.from_dict(materialized).job
            except Exception:
                pass
        return job

    def _renew_lease(self, job: Job) -> None:
        try:
            current = self.store.get_job(job.job_id)
            if current.status != JobStatus.RUNNING:
                return
            self.store.transition_job(
                current.job_id,
                JobStatus.RUNNING,
                expected_revision=current.revision,
                backend_id=current.backend_id,
                checkpoint_ref=current.checkpoint_ref,
                lease_owner=self.worker_id,
                lease_expires_at=self._lease_expiry(),
            )
        except ConflictError:
            return

    def _record_progress(
        self,
        job: Job,
        runtime: ComputeRuntimeRecord,
    ) -> None:
        handle = runtime.handle
        if handle is None:
            return
        bucket = min(10, max(0, int(handle.progress * 10)))
        marker = f"{handle.stage}:{bucket}:{handle.state.value}"
        if runtime.metadata.get("last_progress_marker") == marker:
            return
        self.runtime_store.save(
            replace(
                runtime,
                metadata={
                    **runtime.metadata,
                    "last_progress_marker": marker,
                },
            )
        )
        self._emit(
            event_type="compute.job.progress",
            lane=EventLane.STATUS,
            job=job,
            payload=handle.to_dict(),
            idempotency_key=f"compute:{job.job_id}:progress:{marker}",
        )

    def _emit_selection(self, job: Job, selection: BackendSelection) -> None:
        digest = canonical_json_hash(selection.to_dict())
        self._emit(
            event_type="compute.backend.selected",
            lane=EventLane.STATUS,
            job=job,
            payload=selection.to_dict(),
            idempotency_key=f"compute:{job.job_id}:selection:{digest}",
        )

    def _pause_job(self, job: Job, *, stage: str, error: str) -> Job:
        paused = self.store.transition_job(
            job.job_id,
            JobStatus.PAUSED,
            expected_revision=job.revision,
            backend_id=job.backend_id,
            checkpoint_ref=stage,
            error=error,
        )
        self._emit(
            event_type="compute.job.paused",
            lane=EventLane.STATUS,
            job=paused,
            payload={"stage": stage, "error": error},
            idempotency_key=f"compute:{job.job_id}:paused:{stage}",
        )
        return paused

    def _fail_job(self, job: Job, error: str, stage: str) -> Job:
        if job.status in self.TERMINAL_STATUSES:
            return job
        failed = self.store.transition_job(
            job.job_id,
            JobStatus.FAILED,
            expected_revision=job.revision,
            backend_id=job.backend_id,
            checkpoint_ref=stage,
            error=error,
        )
        self._emit(
            event_type="compute.job.failed",
            lane=EventLane.STATUS,
            job=failed,
            payload={"stage": stage, "error": error},
            idempotency_key=f"compute:{job.job_id}:failed:{stage}",
        )
        return failed

    def _emit(
        self,
        *,
        event_type: str,
        lane: EventLane,
        job: Job,
        payload: Mapping[str, Any],
        actor: str = "compute:scheduler",
        idempotency_key: str | None = None,
    ) -> Event:
        event = self.store.append_event(
            event_type=event_type,
            lane=lane,
            project_id=job.spec.project_id,
            work_session_id=job.spec.work_session_id,
            job_id=job.job_id,
            actor=actor,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception:
                pass
        return event

    def _workspace(self, job: Job) -> Path:
        return self.root_dir / "workspaces" / _safe_component(job.job_id)

    def _artifacts_dir(self, job: Job) -> Path:
        return self.root_dir / "artifacts" / _safe_component(job.job_id)

    def _lease_expiry(self) -> str:
        return (
            datetime.now(timezone.utc) + timedelta(seconds=self.lease_seconds)
        ).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _runtime_expired(job: Job) -> bool:
        limit = job.spec.max_runtime_seconds
        if limit is None or not job.started_at:
            return False
        try:
            started = datetime.fromisoformat(job.started_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - started).total_seconds() > limit


def build_default_compute_stack(
    config: HarnessConfig,
    store: ControlPlaneStore,
    *,
    event_callback: EventCallback | None = None,
) -> ComputeStack:
    root = Path(os.getenv("COMPUTE_RUNTIME_DIR", "compute_runtime")).expanduser()
    if not root.is_absolute():
        root = config.project_root / root
    runtime_store = ComputeRuntimeStore(root / "state")

    inventory = detect_local_gpu()
    backends: list[ComputeBackend] = [
        KaggleNotebookBackend(
            command=os.getenv("KAGGLE_COMMAND", "kaggle"),
            api_token=os.getenv("KAGGLE_API_TOKEN") or None,
            username=os.getenv("KAGGLE_USERNAME") or None,
            timeout_seconds=_int_env("KAGGLE_COMMAND_TIMEOUT_SECONDS", 180),
        )
    ]
    remote_names: list[str] = []
    for descriptor in _remote_descriptors_from_env():
        backends.append(
            RemoteGpuBackend(
                descriptor,
                timeout_seconds=_int_env("REMOTE_WORKER_TIMEOUT_SECONDS", 45),
                max_bundle_files=_int_env("REMOTE_BUNDLE_MAX_FILES", 5000),
                max_bundle_bytes=_int_env(
                    "REMOTE_BUNDLE_MAX_BYTES", 64 * 1024 * 1024
                ),
            )
        )
        remote_names.append(descriptor.name)
    backends.extend(
        [
            LocalProcessBackend(
                name="local_gpu",
                gpu=True,
                inventory=inventory,
                max_cpu_cores=_optional_float_env("LOCAL_GPU_MAX_CPU_CORES"),
                max_memory_mb=_optional_int_env("LOCAL_GPU_MAX_MEMORY_MB"),
                max_storage_mb=_optional_int_env("LOCAL_GPU_MAX_STORAGE_MB"),
            ),
            LocalProcessBackend(
                name="local_cpu",
                gpu=False,
                inventory=inventory,
                max_cpu_cores=_optional_float_env("LOCAL_CPU_MAX_CORES"),
                max_memory_mb=_optional_int_env("LOCAL_CPU_MAX_MEMORY_MB"),
                max_storage_mb=_optional_int_env("LOCAL_CPU_MAX_STORAGE_MB"),
            ),
        ]
    )

    default_research = tuple([*remote_names, "local_gpu", "local_cpu"])
    default_kaggle = tuple(
        ["kaggle_notebook", *remote_names, "local_gpu", "local_cpu"]
    )
    broker = ComputeBroker(
        backends,
        research_order=_csv_env("COMPUTE_RESEARCH_BACKEND_ORDER")
        or default_research,
        kaggle_order=_csv_env("COMPUTE_KAGGLE_BACKEND_ORDER")
        or default_kaggle,
    )

    provider_planner = (
        ProviderFeedbackPlanner(config) if _provider_is_configured(config) else None
    )
    feedback = ResultFeedbackEngine(
        store,
        root,
        planner=provider_planner,
        max_proposals=_int_env("COMPUTE_MAX_HYPOTHESIS_PROPOSALS", 5),
    )
    materializer = ProviderExperimentMaterializer(config)
    scheduler = ComputeScheduler(
        store=store,
        broker=broker,
        runtime_store=runtime_store,
        materializer=materializer,
        feedback=feedback,
        root_dir=root,
        max_concurrent_jobs=_int_env("COMPUTE_MAX_CONCURRENT_JOBS", 2),
        poll_interval_seconds=_float_env("COMPUTE_POLL_INTERVAL_SECONDS", 15.0),
        lease_seconds=_int_env("COMPUTE_LEASE_SECONDS", 300),
        max_unknown_polls=_int_env("COMPUTE_MAX_UNKNOWN_POLLS", 8),
        artifact_max_files=_int_env(
            "COMPUTE_ARTIFACT_MAX_FILES", config.artifact_max_files
        ),
        artifact_max_bytes=_int_env(
            "COMPUTE_ARTIFACT_MAX_BYTES", config.artifact_max_bytes
        ),
        worker_id=os.getenv("COMPUTE_WORKER_ID") or None,
        event_callback=event_callback,
    )
    return ComputeStack(broker, scheduler, feedback, runtime_store)


def _remote_descriptors_from_env() -> list[RemoteWorkerDescriptor]:
    raw = os.getenv("COMPUTE_REMOTE_WORKERS_JSON", "").strip()
    entries: list[Mapping[str, Any]] = []
    if raw:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("COMPUTE_REMOTE_WORKERS_JSON must be valid JSON") from exc
        if not isinstance(value, list):
            raise ValueError("COMPUTE_REMOTE_WORKERS_JSON must be a JSON array")
        entries.extend(item for item in value if isinstance(item, Mapping))

    single_url = os.getenv("REMOTE_GPU_WORKER_URL", "").strip()
    if single_url:
        entries.append(
            {
                "name": os.getenv("REMOTE_GPU_WORKER_NAME", "remote_gpu"),
                "base_url": single_url,
                "token": os.getenv("REMOTE_GPU_WORKER_TOKEN", ""),
                "paid": _bool_env("REMOTE_GPU_WORKER_PAID", False),
                "capabilities": {
                    "accelerators": ["gpu"],
                    "domains": ["research", "kaggle"],
                    "gpu_count": _int_env("REMOTE_GPU_WORKER_GPU_COUNT", 1),
                    "gpu_memory_mb": _optional_int_env(
                        "REMOTE_GPU_WORKER_GPU_MEMORY_MB"
                    ),
                    "network_available": _bool_env(
                        "REMOTE_GPU_WORKER_NETWORK_AVAILABLE", True
                    ),
                    "labels": list(
                        _csv_env("REMOTE_GPU_WORKER_LABELS")
                        or ("training", "inference", "remote_worker")
                    ),
                    "recoverable": True,
                },
            }
        )

    descriptors: list[RemoteWorkerDescriptor] = []
    names: set[str] = set()
    for index, entry in enumerate(entries, 1):
        name = str(entry.get("name") or f"remote_gpu_{index}").strip()
        if not name or name in names:
            raise ValueError(f"duplicate or empty remote worker name: {name!r}")
        names.add(name)
        token = str(entry.get("token") or "").strip()
        token_env = str(entry.get("token_env") or "").strip()
        if token_env:
            token = os.getenv(token_env, "").strip()
        capabilities = BackendCapabilities.from_dict(
            entry.get("capabilities")
            if isinstance(entry.get("capabilities"), Mapping)
            else {
                "accelerators": ["gpu"],
                "domains": ["research", "kaggle"],
                "gpu_count": 1,
                "network_available": True,
                "labels": ["training", "inference", "remote_worker"],
                "recoverable": True,
            }
        )
        descriptors.append(
            RemoteWorkerDescriptor(
                name=name,
                base_url=str(entry.get("base_url") or "").strip(),
                token=token,
                capabilities=capabilities,
                paid=bool(entry.get("paid", False)),
            )
        )
    return descriptors


def _artifact_ref(job_id: str, record: ArtifactRecord) -> str:
    return (
        f"compute/{_safe_component(job_id)}/{record.path}"
        f"#sha256={record.sha256}"
    )


def _safe_component(value: str) -> str:
    cleaned = "".join(
        character
        if character.isalnum() or character in "-_"
        else "-"
        for character in str(value)
    )
    return cleaned.strip("-")[:120] or "job"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.01, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in {None, ""} else None
    except ValueError:
        return None


def _optional_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in {None, ""} else None
    except ValueError:
        return None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip()
            for item in os.getenv(name, "").split(",")
            if item.strip()
        )
    )


def _provider_is_configured(config: HarnessConfig) -> bool:
    return bool(
        config.main_agent_command
        or config.sub_agent_command
        or config.review_agent_command
        or os.getenv("OPENAI_API_KEY")
    )
