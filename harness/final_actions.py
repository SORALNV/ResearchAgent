from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from harness.compute_discord import (
    AutonomousRoutedDiscordService,
    build_autonomous_routed_service,
)
from harness.config import HarnessConfig
from harness.control_plane import ControlPlaneStore, Domain, Event, EventLane
from harness.discord_channel_map import ChannelResolution, DiscordLocation
from harness.discord_thread_router import (
    DiscordThreadRoute,
    DiscordThreadRouter,
)
from harness.human_decision_policy import (
    HumanDecisionKind,
    HumanDecisionVerdict,
)
from harness.kaggle_submission import (
    KaggleSubmissionPipeline,
    SubmissionBlockedError,
    SubmissionUncertainError,
    build_kaggle_submission_pipeline,
)
from harness.paper_pipeline import (
    PaperGenerationPipeline,
    PaperPipelineBlockedError,
    build_paper_generation_pipeline,
)
from harness.routed_discord_adapter import RoutedDecisionReply
from harness.state import utc_timestamp


class FinalActionKind(str, Enum):
    KAGGLE_SUBMISSION = "kaggle_submission"
    PAPER_GENERATION = "paper_generation"


class FinalActionState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class FinalActionRecord:
    action_id: str
    kind: FinalActionKind
    project_id: str
    work_session_id: str
    subject_ref: str
    decision_event_id: str
    state: FinalActionState = FinalActionState.QUEUED
    attempts: int = 0
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    retryable: bool = True
    next_retry_at: float = 0.0
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in {
            FinalActionState.SUCCEEDED,
            FinalActionState.FAILED,
            FinalActionState.CANCELLED,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinalActionRecord":
        return cls(
            action_id=str(data["action_id"]),
            kind=FinalActionKind(str(data["kind"])),
            project_id=str(data["project_id"]),
            work_session_id=str(data["work_session_id"]),
            subject_ref=str(data["subject_ref"]),
            decision_event_id=str(data.get("decision_event_id") or ""),
            state=FinalActionState(
                str(data.get("state") or FinalActionState.QUEUED.value)
            ),
            attempts=max(0, int(data.get("attempts") or 0)),
            result=_json_dict(data.get("result")),
            error=(str(data["error"]) if data.get("error") else None),
            retryable=bool(data.get("retryable", True)),
            next_retry_at=float(data.get("next_retry_at") or 0.0),
            created_at=str(data.get("created_at") or utc_timestamp()),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
            started_at=(str(data["started_at"]) if data.get("started_at") else None),
            finished_at=(str(data["finished_at"]) if data.get("finished_at") else None),
        )


class FinalActionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.actions_dir = self.root / "actions"
        self.actions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def save(self, record: FinalActionRecord) -> FinalActionRecord:
        updated = replace(record, updated_at=utc_timestamp())
        with self._lock:
            _atomic_json(self._path(updated.action_id), updated.to_dict())
        return updated

    def get(self, action_id: str) -> FinalActionRecord | None:
        path = self._path(action_id)
        if not path.is_file():
            return None
        with self._lock:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return FinalActionRecord.from_dict(value) if isinstance(value, Mapping) else None

    def list(
        self,
        *,
        work_session_id: str | None = None,
    ) -> list[FinalActionRecord]:
        result: list[FinalActionRecord] = []
        for path in sorted(self.actions_dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, Mapping):
                    continue
                record = FinalActionRecord.from_dict(value)
            except Exception:
                continue
            if work_session_id and record.work_session_id != work_session_id:
                continue
            result.append(record)
        return sorted(result, key=lambda item: (item.created_at, item.action_id))

    def _path(self, action_id: str) -> Path:
        return self.actions_dir / f"{_safe_component(action_id)}.json"


class FinalActionCoordinator:
    """Durable final-action queue driven only by immutable human decisions."""

    ACTION_EVENT = "final_action.state_changed"

    def __init__(
        self,
        *,
        router: DiscordThreadRouter,
        submission: KaggleSubmissionPipeline,
        paper: PaperGenerationPipeline,
        root_dir: str | Path,
        scan_interval_seconds: float = 10.0,
        retry_interval_seconds: float = 30.0,
        max_failure_attempts: int = 3,
        max_concurrent_actions: int = 2,
        kaggle_submission_enabled: bool = True,
        paper_pipeline_enabled: bool = True,
    ) -> None:
        self.router = router
        self.store: ControlPlaneStore = router.store
        self.submission = submission
        self.paper = paper
        self.actions = FinalActionStore(root_dir)
        self.scan_interval_seconds = max(0.2, float(scan_interval_seconds))
        self.retry_interval_seconds = max(0.2, float(retry_interval_seconds))
        self.max_failure_attempts = max(1, int(max_failure_attempts))
        self.max_concurrent_actions = max(1, int(max_concurrent_actions))
        self.kaggle_submission_enabled = bool(kaggle_submission_enabled)
        self.paper_pipeline_enabled = bool(paper_pipeline_enabled)

        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=10000)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_actions,
            thread_name_prefix="research-final-action",
        )
        self._dispatcher: threading.Thread | None = None
        self._scanner: threading.Thread | None = None
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
                name="final-action-dispatcher",
                daemon=True,
            )
            self._scanner = threading.Thread(
                target=self._scan_loop,
                name="final-action-scanner",
                daemon=True,
            )
            self._dispatcher.start()
            self._scanner.start()
        if recover:
            self.recover()

    def stop(self, *, wait: bool = False) -> None:
        self._stopping.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if wait:
            if self._scanner:
                self._scanner.join(timeout=10)
            if self._dispatcher:
                self._dispatcher.join(timeout=10)
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def recover(self) -> tuple[str, ...]:
        recovered: list[str] = []
        for record in self.actions.list():
            if record.state == FinalActionState.RUNNING:
                record = self.actions.save(
                    replace(
                        record,
                        state=FinalActionState.QUEUED,
                        error="Core restarted while final action was running; reconciling",
                        retryable=True,
                        next_retry_at=0.0,
                    )
                )
            if not record.terminal:
                self.enqueue(record.action_id)
                recovered.append(record.action_id)
        self.scan_once()
        return tuple(dict.fromkeys(recovered))

    def observe_decision(self, decision: RoutedDecisionReply) -> FinalActionRecord | None:
        kind = _action_kind(decision.kind)
        if kind is None:
            return None
        session = self.store.get_work_session(decision.work_session_id)
        project = self.store.get_project(session.project_id)
        action_id = _action_id(decision.work_session_id, kind, decision.subject_ref)
        existing = self.actions.get(action_id)
        if decision.verdict != HumanDecisionVerdict.ACCEPT:
            if existing is None:
                return None
            cancelled = self.actions.save(
                replace(
                    existing,
                    decision_event_id=decision.event_id,
                    state=FinalActionState.CANCELLED,
                    error=f"latest human decision is {decision.verdict.value}",
                    retryable=False,
                    finished_at=utc_timestamp(),
                )
            )
            self._emit(cancelled)
            return cancelled
        if existing and existing.decision_event_id == decision.event_id:
            if existing.state == FinalActionState.QUEUED:
                self.enqueue(existing.action_id)
            elif (
                existing.state == FinalActionState.BLOCKED
                and existing.next_retry_at <= time.time()
            ):
                self.enqueue(existing.action_id)
            return existing
        if existing and existing.state == FinalActionState.SUCCEEDED:
            return existing
        record = FinalActionRecord(
            action_id=action_id,
            kind=kind,
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            subject_ref=decision.subject_ref,
            decision_event_id=decision.event_id,
            state=FinalActionState.QUEUED,
            attempts=existing.attempts if existing else 0,
            result=existing.result if existing else {},
            created_at=existing.created_at if existing else utc_timestamp(),
        )
        record = self.actions.save(record)
        self._emit(record)
        self.enqueue(record.action_id)
        return record

    def enqueue(self, action_id: str) -> FinalActionRecord:
        record = self.actions.get(action_id)
        if record is None:
            raise KeyError(action_id)
        if record.terminal:
            return record
        with self._lock:
            if action_id in self._queued or action_id in self._active:
                return record
            self._queued.add(action_id)
        self._queue.put_nowait(action_id)
        return record

    def scan_once(self) -> None:
        now = time.time()
        for session in self.store.list_work_sessions():
            try:
                project = self.store.get_project(session.project_id)
            except Exception:
                continue
            if project.domain == Domain.KAGGLE:
                try:
                    self.submission.discover_work_session(session.work_session_id)
                    self.submission.refresh_history(
                        work_session_id=session.work_session_id
                    )
                except Exception:
                    pass
            self._scan_session_decisions(session.work_session_id)
        for record in self.actions.list():
            if record.terminal:
                continue
            if record.state == FinalActionState.BLOCKED and record.next_retry_at > now:
                continue
            self.enqueue(record.action_id)

    def run_until_idle(self, *, timeout_seconds: float = 30.0) -> None:
        self.start(recover=True)
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if not snapshot["active"] and not snapshot["queued"]:
                return
            time.sleep(0.02)
        raise TimeoutError("final-action coordinator did not become idle")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": sorted(self._active),
                "queued": sorted(self._queued),
                "queue_size": self._queue.qsize(),
                "scan_interval_seconds": self.scan_interval_seconds,
                "retry_interval_seconds": self.retry_interval_seconds,
                "max_concurrent_actions": self.max_concurrent_actions,
                "stopping": self._stopping.is_set(),
            }

    def status_lines(self, work_session_id: str) -> tuple[str, ...]:
        records = self.actions.list(work_session_id=work_session_id)
        if not records:
            return ("- なし",)
        return tuple(
            (
                f"- {item.action_id}: {item.kind.value}/{item.state.value}; "
                f"subject={item.subject_ref}; attempts={item.attempts}; "
                f"error={item.error or '-'}"
            )
            for item in records[-20:]
        )

    def _scan_session_decisions(self, work_session_id: str) -> None:
        events = self.store.latest_events(
            work_session_id=work_session_id,
            lanes=[EventLane.CONTROL],
            limit=5000,
        )
        latest: dict[tuple[HumanDecisionKind, str], Event] = {}
        prefix = "human.decision."
        for event in events:
            if not event.event_type.startswith(prefix):
                continue
            try:
                kind = HumanDecisionKind(
                    str(event.payload.get("kind") or event.event_type[len(prefix) :])
                )
            except ValueError:
                continue
            if kind not in {
                HumanDecisionKind.KAGGLE_SUBMISSION,
                HumanDecisionKind.RESEARCH_PAPER,
            }:
                continue
            subject = str(event.payload.get("subject_ref") or "").strip()
            if subject:
                latest[(kind, subject)] = event
        for (kind, subject), event in latest.items():
            try:
                verdict = HumanDecisionVerdict(
                    str(event.payload.get("verdict") or "")
                )
            except ValueError:
                continue
            self.observe_decision(
                RoutedDecisionReply(
                    domain=self.store.get_project(event.project_id).domain,
                    work_session_id=event.work_session_id,
                    kind=kind,
                    verdict=verdict,
                    subject_ref=subject,
                    event_id=event.event_id,
                )
            )

    def _scan_loop(self) -> None:
        while not self._stopping.wait(self.scan_interval_seconds):
            try:
                self.scan_once()
            except Exception:
                continue

    def _dispatch_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                action_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if action_id is None:
                self._queue.task_done()
                return
            with self._lock:
                self._queued.discard(action_id)
                if action_id in self._active:
                    self._queue.task_done()
                    continue
                self._active.add(action_id)
            future = self._executor.submit(self._run_action, action_id)
            future.add_done_callback(
                lambda _future, value=action_id: self._action_finished(value)
            )
            self._queue.task_done()

    def _action_finished(self, action_id: str) -> None:
        with self._lock:
            self._active.discard(action_id)

    def _run_action(self, action_id: str) -> None:
        record = self.actions.get(action_id)
        if record is None or record.terminal:
            return
        route = self._route(record.work_session_id)
        if not self._decision_still_accepted(record):
            cancelled = self.actions.save(
                replace(
                    record,
                    state=FinalActionState.CANCELLED,
                    error="latest human decision no longer accepts this action",
                    retryable=False,
                    finished_at=utc_timestamp(),
                )
            )
            self._emit(cancelled)
            return
        running = self.actions.save(
            replace(
                record,
                state=FinalActionState.RUNNING,
                attempts=record.attempts + 1,
                error=None,
                started_at=record.started_at or utc_timestamp(),
                next_retry_at=0.0,
            )
        )
        self._emit(running)
        try:
            if running.kind == FinalActionKind.KAGGLE_SUBMISSION:
                if not self.kaggle_submission_enabled:
                    raise SubmissionBlockedError(
                        "KAGGLE_SUBMISSION_ENABLED is false"
                    )
                result = self.submission.execute(
                    route,
                    subject_ref=running.subject_ref,
                ).to_dict()
            elif running.kind == FinalActionKind.PAPER_GENERATION:
                if not self.paper_pipeline_enabled:
                    raise PaperPipelineBlockedError(
                        "PAPER_PIPELINE_ENABLED is false"
                    )
                result = self.paper.execute(
                    route,
                    subject_ref=running.subject_ref,
                ).to_dict()
            else:
                raise ValueError(f"unsupported final action: {running.kind.value}")
            completed = self.actions.save(
                replace(
                    running,
                    state=FinalActionState.SUCCEEDED,
                    result=result,
                    error=None,
                    retryable=False,
                    finished_at=utc_timestamp(),
                )
            )
            self._emit(completed)
        except (SubmissionBlockedError, PaperPipelineBlockedError) as exc:
            blocked = self.actions.save(
                replace(
                    running,
                    state=FinalActionState.BLOCKED,
                    error=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                    next_retry_at=time.time() + self.retry_interval_seconds,
                )
            )
            self._emit(blocked)
        except SubmissionUncertainError as exc:
            uncertain = self.actions.save(
                replace(
                    running,
                    state=FinalActionState.BLOCKED,
                    error=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                    next_retry_at=time.time() + self.retry_interval_seconds,
                )
            )
            self._emit(uncertain)
        except PermissionError as exc:
            cancelled = self.actions.save(
                replace(
                    running,
                    state=FinalActionState.CANCELLED,
                    error=f"PermissionError: {exc}",
                    retryable=False,
                    finished_at=utc_timestamp(),
                )
            )
            self._emit(cancelled)
        except Exception as exc:
            retryable = running.attempts < self.max_failure_attempts
            failed = self.actions.save(
                replace(
                    running,
                    state=(
                        FinalActionState.BLOCKED
                        if retryable
                        else FinalActionState.FAILED
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                    retryable=retryable,
                    next_retry_at=(
                        time.time() + self.retry_interval_seconds
                        if retryable
                        else 0.0
                    ),
                    finished_at=None if retryable else utc_timestamp(),
                )
            )
            self._emit(failed)

    def _decision_still_accepted(self, record: FinalActionRecord) -> bool:
        decision_kind = (
            HumanDecisionKind.KAGGLE_SUBMISSION
            if record.kind == FinalActionKind.KAGGLE_SUBMISSION
            else HumanDecisionKind.RESEARCH_PAPER
        )
        event_type = "human.decision." + decision_kind.value
        latest: Event | None = None
        for event in self.store.latest_events(
            work_session_id=record.work_session_id,
            lanes=[EventLane.CONTROL],
            limit=5000,
        ):
            if event.event_type != event_type:
                continue
            if str(event.payload.get("subject_ref") or "") != record.subject_ref:
                continue
            latest = event
        if latest is None:
            return False
        try:
            return HumanDecisionVerdict(
                str(latest.payload.get("verdict") or "")
            ) == HumanDecisionVerdict.ACCEPT
        except ValueError:
            return False

    def _route(self, work_session_id: str) -> DiscordThreadRoute:
        session = self.store.get_work_session(work_session_id)
        project = self.store.get_project(session.project_id)
        route_channel_id = str(
            session.metadata.get("route_channel_id")
            or session.external_ref.get("parent_channel_id")
            or session.external_ref.get("channel_id")
            or session.external_ref.get("conversation_id")
            or "0"
        )
        return DiscordThreadRoute(
            resolution=ChannelResolution(
                domain=project.domain,
                route_channel_id=route_channel_id,
                inherited_from_parent=False,
            ),
            project=project,
            work_session=session,
        )

    def _emit(self, record: FinalActionRecord) -> Event:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "state": record.state.value,
                    "attempts": record.attempts,
                    "error": record.error,
                    "result": record.result,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:20]
        return self.store.append_event(
            event_type=self.ACTION_EVENT,
            lane=EventLane.STATUS,
            project_id=record.project_id,
            work_session_id=record.work_session_id,
            actor="core:final-actions",
            payload=record.to_dict(),
            idempotency_key=(
                f"final-action:{record.action_id}:{record.state.value}:{digest}"
            ),
        )


class CompleteRoutedDiscordService(AutonomousRoutedDiscordService):
    """Compute, Kaggle submission, and paper generation under human-only gates."""

    def __init__(
        self,
        router: DiscordThreadRouter,
        dispatcher,
        compute,
        final_actions: FinalActionCoordinator,
    ) -> None:
        super().__init__(router, dispatcher, compute)
        self.final_actions = final_actions

    def start(self) -> None:
        super().start()
        self.final_actions.start(recover=True)

    def stop(self, *, wait: bool = False) -> None:
        self.final_actions.stop(wait=wait)
        super().stop(wait=wait)

    def record_decision(
        self,
        location: DiscordLocation,
        *,
        title: str,
        kind: HumanDecisionKind | str,
        verdict: HumanDecisionVerdict | str,
        subject_ref: str,
        note: str,
        actor_id: str,
        message_id: str,
        actor_is_human: bool,
        project_id: str | None = None,
    ) -> RoutedDecisionReply:
        decision = super().record_decision(
            location,
            title=title,
            kind=kind,
            verdict=verdict,
            subject_ref=subject_ref,
            note=note,
            actor_id=actor_id,
            message_id=message_id,
            actor_is_human=actor_is_human,
            project_id=project_id,
        )
        self.final_actions.observe_decision(decision)
        return decision

    def status(
        self,
        location: DiscordLocation,
        *,
        title: str,
        project_id: str | None = None,
    ) -> str:
        base = super().status(
            location,
            title=title,
            project_id=project_id,
        )
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        if route.domain == Domain.KAGGLE:
            candidate_lines = self.final_actions.submission.status_lines(
                route.work_session.work_session_id
            )
            domain_lines = ["Kaggle submission candidates:", *candidate_lines]
        else:
            paper_lines = self.final_actions.paper.status_lines(
                route.work_session.work_session_id
            )
            domain_lines = ["Paper artifacts:", *paper_lines]
        action_lines = self.final_actions.status_lines(
            route.work_session.work_session_id
        )
        return "\n".join(
            [
                base,
                *domain_lines,
                "Final actions:",
                *action_lines,
                "Final-action runtime:",
                json.dumps(self.final_actions.snapshot(), ensure_ascii=False),
            ]
        )


def build_complete_routed_service(
    config: HarnessConfig,
    router: DiscordThreadRouter,
    *,
    submission: KaggleSubmissionPipeline | None = None,
    paper: PaperGenerationPipeline | None = None,
) -> CompleteRoutedDiscordService:
    base = build_autonomous_routed_service(config, router)
    root = Path(
        os.getenv("FINAL_ACTION_RUNTIME_DIR", "final_actions")
    ).expanduser()
    if not root.is_absolute():
        root = config.project_root / root
    submission_pipeline = submission or build_kaggle_submission_pipeline(
        router=router,
        project_root=config.project_root,
    )
    paper_pipeline = paper or build_paper_generation_pipeline(
        config=config,
        router=router,
    )
    coordinator = FinalActionCoordinator(
        router=router,
        submission=submission_pipeline,
        paper=paper_pipeline,
        root_dir=root,
        scan_interval_seconds=_float_env("FINAL_ACTION_SCAN_SECONDS", 10.0),
        retry_interval_seconds=_float_env("FINAL_ACTION_RETRY_SECONDS", 30.0),
        max_failure_attempts=_int_env("FINAL_ACTION_MAX_FAILURE_ATTEMPTS", 3),
        max_concurrent_actions=_int_env("FINAL_ACTION_MAX_CONCURRENT", 2),
        kaggle_submission_enabled=_bool_env(
            "KAGGLE_SUBMISSION_ENABLED", True
        ),
        paper_pipeline_enabled=_bool_env("PAPER_PIPELINE_ENABLED", True),
    )
    return CompleteRoutedDiscordService(
        router,
        base.dispatcher,
        base.compute,
        coordinator,
    )


def _action_kind(kind: HumanDecisionKind) -> FinalActionKind | None:
    if kind == HumanDecisionKind.KAGGLE_SUBMISSION:
        return FinalActionKind.KAGGLE_SUBMISSION
    if kind == HumanDecisionKind.RESEARCH_PAPER:
        return FinalActionKind.PAPER_GENERATION
    return None


def _action_id(
    work_session_id: str,
    kind: FinalActionKind,
    subject_ref: str,
) -> str:
    digest = hashlib.sha256(
        (work_session_id + "\0" + kind.value + "\0" + subject_ref).encode("utf-8")
    ).hexdigest()[:24]
    return f"FINAL-{kind.value.upper().replace('_', '-')}-{digest}"


def _json_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    encoded = json.dumps(dict(value), ensure_ascii=False, allow_nan=False, default=str)
    decoded = json.loads(encoded)
    return dict(decoded) if isinstance(decoded, dict) else {}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_component(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value)
    )
    return cleaned.strip("-")[:160] or "action"


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.2, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in {None, ""}:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
