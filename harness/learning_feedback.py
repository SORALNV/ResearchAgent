from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from harness.compute_feedback import FeedbackOutcome, ResultFeedbackEngine
from harness.control_plane import EventLane, Job
from harness.iteration_memo import IterationMemo, IterationMemoEngine


class LearningResultFeedbackAdapter:
    """Decorate the existing feedback engine without changing ComputeScheduler APIs."""

    def __init__(
        self,
        base: ResultFeedbackEngine,
        memo_engine: IterationMemoEngine,
    ) -> None:
        self.base = base
        self.memo_engine = memo_engine
        self.method_store = memo_engine.method_store
        self.store = base.store
        self._last_memo_by_job: dict[str, IterationMemo] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def integrate(
        self,
        *,
        job: Job,
        collected_result: Mapping[str, Any] | None,
        artifacts_dir: str | Path,
        artifact_refs: tuple[str, ...] | list[str] = (),
        backend: str,
    ) -> FeedbackOutcome:
        outcome = self.base.integrate(
            job=job,
            collected_result=collected_result,
            artifacts_dir=artifacts_dir,
            artifact_refs=artifact_refs,
            backend=backend,
        )
        try:
            memo = self.memo_engine.integrate(
                job=job,
                result=outcome.result,
                result_ref=outcome.result_ref,
                proposals=[item.to_dict() for item in outcome.proposals],
                artifact_refs=artifact_refs,
                backend=backend,
            )
            if memo is not None:
                self._last_memo_by_job[job.job_id] = memo
        except Exception as exc:
            digest = hashlib.sha256(
                f"{type(exc).__name__}:{exc}".encode("utf-8")
            ).hexdigest()[:16]
            self.store.append_event(
                event_type=IterationMemoEngine.MEMO_FAILED_EVENT,
                lane=EventLane.STATUS,
                project_id=job.spec.project_id,
                work_session_id=job.spec.work_session_id,
                job_id=job.job_id,
                actor="agent:iteration-memo",
                payload={
                    "result_ref": outcome.result_ref,
                    "error": f"{type(exc).__name__}: {exc}",
                    "experiment_completion_preserved": True,
                },
                idempotency_key=f"kaggle-memo:{job.job_id}:failed:{digest}",
            )
        return outcome

    def last_memo(self, job_id: str) -> IterationMemo | None:
        return self._last_memo_by_job.get(str(job_id))
