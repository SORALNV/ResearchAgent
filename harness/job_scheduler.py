from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from harness.compute import BackendRunResult, ComputeBroker
from harness.control_plane import (
    ControlPlaneRegistry,
    JobEvent,
    JobRecord,
    JobSpec,
    TERMINAL_JOB_STATUSES,
)


JobEventSubscriber = Callable[[JobEvent], None]


class JobScheduler:
    """Persistent background scheduler for local, Kaggle, remote, and cloud backends."""

    def __init__(
        self,
        registry: ControlPlaneRegistry,
        broker: ComputeBroker,
        *,
        worker_count: int = 2,
        queue_size: int = 256,
        requeue_interrupted: bool = True,
    ) -> None:
        self.registry = registry
        self.broker = broker
        self.worker_count = max(1, int(worker_count))
        self.queue_size = max(1, int(queue_size))
        self.requeue_interrupted = requeue_interrupted
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=self.queue_size)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._queued_ids: set[str] = set()
        self._cancel_events: dict[str, threading.Event] = {}
        self._subscribers: list[JobEventSubscriber] = []

    def start(self) -> None:
        with self._lock:
            if any(thread.is_alive() for thread in self._threads):
                return
            self._stop.clear()
            self.registry.recover_incomplete_jobs(
                requeue=self.requeue_interrupted,
            )
            self._threads = [
                threading.Thread(
                    target=self._worker_loop,
                    name=f"compute-scheduler-{index + 1}",
                    daemon=True,
                )
                for index in range(self.worker_count)
            ]
            for thread in self._threads:
                thread.start()
            for job_id in self.registry.queued_job_ids():
                self._enqueue(job_id)

    def close(self, *, cancel_running: bool = False) -> None:
        if cancel_running:
            with self._lock:
                for cancel_event in self._cancel_events.values():
                    cancel_event.set()
        self._stop.set()
        for _ in self._threads:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=10)
        with self._lock:
            self._threads.clear()
            self._queued_ids.clear()

    def submit(self, spec: JobSpec) -> JobRecord:
        self.start()
        record = self.registry.create_job(spec)
        self._enqueue(spec.job_id)
        return record

    def cancel(self, job_id: str, reason: str = "user requested") -> JobRecord:
        record = self.registry.request_cancel(job_id, reason)
        with self._lock:
            cancel_event = self._cancel_events.get(job_id)
            if cancel_event is not None:
                cancel_event.set()
        if record.status == "queued":
            record = self.registry.finish_job(
                job_id,
                status="cancelled",
                error=reason,
            )
        return record

    def retry(self, job_id: str) -> JobRecord:
        record = self.registry.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.status not in {"failed", "cancelled", "interrupted", "blocked"}:
            raise ValueError(f"job cannot be retried from {record.status}")
        new_spec = JobSpec.new(
            project_id=record.spec.project_id,
            work_session_id=record.spec.work_session_id,
            domain=record.spec.domain,
            task_type=record.spec.task_type,
            payload={**record.spec.payload, "retry_of": job_id},
            resources=record.spec.resources,
            backend_preferences=record.spec.backend_preferences,
            outputs=record.spec.outputs,
            max_runtime_seconds=record.spec.max_runtime_seconds,
        )
        return self.submit(new_spec)

    def subscribe(self, callback: JobEventSubscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            alive = sum(1 for thread in self._threads if thread.is_alive())
            active = sorted(self._cancel_events)
            queued = sorted(self._queued_ids)
        return {
            "running": bool(alive),
            "worker_count": alive,
            "queue_depth": self._queue.qsize(),
            "queued_job_ids": queued,
            "active_job_ids": active,
            "backends": self.broker.status(),
        }

    def _enqueue(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._queued_ids or job_id in self._cancel_events:
                return
            self._queued_ids.add(job_id)
        try:
            self._queue.put_nowait(job_id)
        except queue.Full:
            with self._lock:
                self._queued_ids.discard(job_id)
            self.registry.finish_job(
                job_id,
                status="failed",
                error=f"scheduler queue full ({self.queue_size})",
            )
            raise RuntimeError(f"scheduler queue full ({self.queue_size})")

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job_id is None:
                self._queue.task_done()
                return
            with self._lock:
                self._queued_ids.discard(job_id)
            try:
                self._execute(job_id)
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        record = self.registry.get_job(job_id)
        if record is None or record.status in TERMINAL_JOB_STATUSES:
            return
        if record.cancel_requested:
            self.registry.finish_job(
                job_id,
                status="cancelled",
                error="cancel requested before execution",
            )
            return

        try:
            selection = self.broker.select(record.spec)
        except Exception as exc:
            self.registry.finish_job(
                job_id,
                status="blocked",
                error=str(exc),
                result={"backend_selection_error": str(exc)},
            )
            return

        cancel_event = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = cancel_event
        try:
            self.registry.claim_job(job_id, selection.backend.capabilities.name)
            self.registry.mark_running(job_id)
            self._publish(
                self.registry.append_event(
                    job_id,
                    "backend_selected",
                    {
                        "backend": selection.backend.capabilities.name,
                        "reason": selection.reason,
                    },
                )
            )

            def emit(event_type: str, payload: dict[str, Any]) -> None:
                event_payload = dict(payload)
                event = self.registry.append_event(job_id, event_type, event_payload)
                if event_type == "backend_started" and event_payload.get("backend_job_id"):
                    self.registry.mark_running(
                        job_id,
                        str(event_payload["backend_job_id"]),
                    )
                if event_type in {"backend_progress", "fold_end", "epoch_end", "stage"}:
                    self.registry.update_progress(job_id, event_payload)
                self._publish(event)

            result = selection.backend.run(
                record.spec,
                emit=emit,
                cancel_event=cancel_event,
            )
            self._finish_from_backend_result(job_id, result)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.registry.finish_job(
                job_id,
                status="failed",
                error=error,
                result={"traceback": traceback.format_exc(limit=30)},
            )
        finally:
            with self._lock:
                self._cancel_events.pop(job_id, None)

    def _finish_from_backend_result(
        self,
        job_id: str,
        result: BackendRunResult,
    ) -> None:
        if result.status == "completed":
            self.registry.mark_collecting(job_id)
            record = self.registry.finish_job(
                job_id,
                status="completed",
                result={
                    **result.result,
                    "backend_job_id": result.backend_job_id,
                },
            )
        elif result.status == "cancelled":
            record = self.registry.finish_job(
                job_id,
                status="cancelled",
                result=result.result,
                error=result.error or "cancelled",
            )
        elif result.status == "blocked":
            record = self.registry.finish_job(
                job_id,
                status="blocked",
                result=result.result,
                error=result.error or "blocked",
            )
        else:
            record = self.registry.finish_job(
                job_id,
                status="failed",
                result=result.result,
                error=result.error or f"backend returned {result.status}",
            )
        final_events = self.registry.list_events(job_id, limit=1)
        if final_events:
            self._publish(final_events[-1])
        self._publish_synthetic_summary(record)

    def _publish(self, event: JobEvent) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                continue

    def _publish_synthetic_summary(self, record: JobRecord) -> None:
        event = JobEvent(
            sequence=-1,
            job_id=record.spec.job_id,
            work_session_id=record.spec.work_session_id,
            event_type="job_summary",
            payload={
                "status": record.status,
                "backend": record.backend,
                "attempt": record.attempt,
                "result": record.result,
                "error": record.error,
                "spec": asdict(record.spec),
            },
            created_at=record.updated_at,
        )
        self._publish(event)
