from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from harness.control_plane import ControlPlaneRegistry, JobEvent, JobRecord, WorkSessionRecord
from harness.work_sessions import WorkSessionService, WorkSessionStore


@dataclass(frozen=True)
class CreatedThread:
    guild_id: str | None
    channel_id: str
    thread_id: str
    starter_message_id: str | None = None


class WorkThreadTransport(Protocol):
    def create_thread(
        self,
        *,
        title: str,
        initial_message: str,
        tags: tuple[str, ...] = (),
    ) -> CreatedThread: ...

    def send(self, thread_id: str, content: str) -> str | None: ...

    def upsert_live_status(
        self,
        thread_id: str,
        content: str,
        message_id: str | None = None,
    ) -> str: ...

    def set_tags(self, thread_id: str, tags: tuple[str, ...]) -> None: ...


MILESTONE_EVENTS = {
    "job_queued",
    "backend_selected",
    "backend_started",
    "kaggle_package_ready",
    "kaggle_push_finished",
    "kaggle_output_finished",
    "remote_cancel_sent",
    "job_completed",
    "job_failed",
    "job_cancelled",
    "job_blocked",
    "job_recovered",
    "job_summary",
}

STATUS_TAGS = {
    "planning": ("Planning",),
    "waiting_input": ("Waiting Input",),
    "queued": ("Queued",),
    "running": ("Running",),
    "review": ("Review",),
    "waiting_approval": ("Waiting Approval",),
    "paused": ("Paused",),
    "completed": ("Completed",),
    "failed": ("Failed",),
    "cancelled": ("Cancelled",),
}


class WorkSessionThreadBridge:
    """Project scheduler events into one Discord thread per WorkSession."""

    def __init__(
        self,
        registry: ControlPlaneRegistry,
        store: WorkSessionStore,
        service: WorkSessionService,
        transport: WorkThreadTransport,
    ) -> None:
        self.registry = registry
        self.store = store
        self.service = service
        self.transport = transport
        self._lock = threading.RLock()

    def create_and_bind(
        self,
        work_session_id: str,
        *,
        initial_message: str,
    ) -> WorkSessionRecord:
        session = self._require_session(work_session_id)
        if session.thread_id:
            return session
        created = self.transport.create_thread(
            title=session.title,
            initial_message=initial_message,
            tags=STATUS_TAGS.get(session.status, ()),
        )
        bound = self.registry.bind_thread(
            work_session_id,
            guild_id=created.guild_id,
            channel_id=created.channel_id,
            thread_id=created.thread_id,
        )
        self.store.update_view_state(
            work_session_id,
            live_message_id=created.starter_message_id,
        )
        self.refresh(work_session_id)
        return bound

    def on_job_event(self, event: JobEvent) -> None:
        session = self.registry.get_work_session(event.work_session_id)
        if session is None or not session.thread_id:
            return
        with self._lock:
            if event.sequence > 0:
                view = self.store.get_view_state(session.work_session_id)
                if event.sequence <= view.event_cursor:
                    return
                self.store.update_view_state(
                    session.work_session_id,
                    event_cursor=event.sequence,
                )
            if event.event_type in MILESTONE_EVENTS:
                content = render_milestone(event)
                if content:
                    self.transport.send(session.thread_id, content)
                    self.store.append_message(
                        work_session_id=session.work_session_id,
                        actor="ResearchAgent",
                        source="job-event",
                        kind="event",
                        content=content,
                        metadata={
                            "job_id": event.job_id,
                            "event_type": event.event_type,
                            "sequence": event.sequence,
                        },
                    )
            self.refresh(session.work_session_id)

    def refresh(self, work_session_id: str) -> str:
        session = self._require_session(work_session_id)
        if not session.thread_id:
            return ""
        content = render_live_status(self.service.status(work_session_id))
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        view = self.store.get_view_state(work_session_id)
        if digest == view.last_status_hash:
            return content
        message_id = self.transport.upsert_live_status(
            session.thread_id,
            content,
            view.live_message_id,
        )
        self.store.update_view_state(
            work_session_id,
            live_message_id=message_id,
            last_status_hash=digest,
        )
        current = self.registry.get_work_session(work_session_id)
        if current:
            self.transport.set_tags(
                session.thread_id,
                STATUS_TAGS.get(current.status, ()),
            )
        return content

    def post_assistant(self, work_session_id: str, content: str) -> None:
        session = self._require_session(work_session_id)
        if not session.thread_id:
            raise ValueError("work session is not bound to a thread")
        self.transport.send(session.thread_id, content)
        self.service.record_assistant_message(work_session_id, content)

    def _require_session(self, work_session_id: str) -> WorkSessionRecord:
        session = self.registry.get_work_session(work_session_id)
        if session is None:
            raise KeyError(work_session_id)
        return session


def render_live_status(status: dict[str, Any]) -> str:
    latest = status.get("latest_job") or {}
    progress = latest.get("progress") or {}
    result = latest.get("result") or {}
    stage = (
        progress.get("stage")
        or progress.get("kernel_status")
        or progress.get("remote_status")
        or latest.get("status")
        or "idle"
    )
    active = status.get("active_jobs") or []
    score = _find_score(result)
    lines = [
        f"**{status.get('work_session_id')} / {status.get('status')}**",
        f"Project: `{status.get('project_id')}` ({status.get('domain')})",
        f"Title: {status.get('title')}",
        f"Stage: `{stage}`",
        f"Active jobs: {', '.join(f'`{item}`' for item in active) if active else 'なし'}",
    ]
    if latest:
        lines.extend(
            [
                f"Latest job: `{latest.get('job_id')}`",
                f"Backend: `{latest.get('backend') or '未選択'}`",
                f"Attempt: {latest.get('attempt', 0)}",
            ]
        )
    if score is not None:
        lines.append(f"Metric: `{score}`")
    if latest.get("error"):
        lines.append(f"Last error: {str(latest['error'])[:500]}")
    lines.append(f"Pending steering: {status.get('pending_steering', 0)}")
    return "\n".join(lines)


def render_milestone(event: JobEvent) -> str:
    payload = event.payload
    event_type = event.event_type
    if event_type == "job_queued":
        return f"📥 `{event.job_id}` をキューへ登録しました。"
    if event_type == "backend_selected":
        return (
            f"🧭 `{event.job_id}` の実行先を `{payload.get('backend')}` に決定しました。\n"
            f"理由: {payload.get('reason', '条件一致')}"
        )
    if event_type == "backend_started":
        return (
            f"▶️ `{event.job_id}` を開始しました。\n"
            f"Backend job: `{payload.get('backend_job_id', 'unknown')}`"
        )
    if event_type == "kaggle_package_ready":
        return (
            "📦 Kaggle Notebookパッケージを生成しました。\n"
            f"Kernel: `{payload.get('kernel_ref')}`"
        )
    if event_type == "kaggle_push_finished":
        success = int(payload.get("returncode") or 0) == 0
        return "☁️ Kaggleへ投入しました。" if success else "⚠️ Kaggleへの投入に失敗しました。"
    if event_type == "kaggle_output_finished":
        return f"📤 Kaggle outputを回収しました: `{payload.get('output_dir')}`"
    if event_type == "remote_cancel_sent":
        return "⏹️ Remote Workerへ停止要求を送信しました。"
    if event_type == "job_recovered":
        return (
            f"♻️ 再起動後に `{event.job_id}` を復旧しました。\n"
            f"状態: `{payload.get('new_status')}`"
        )
    if event_type in {"job_completed", "job_failed", "job_cancelled", "job_blocked"}:
        label = {
            "job_completed": "✅ 完了",
            "job_failed": "❌ 失敗",
            "job_cancelled": "⏹️ 中止",
            "job_blocked": "⏸️ 承認・設定待ち",
        }[event_type]
        error = payload.get("error")
        return f"{label}: `{event.job_id}`" + (f"\n理由: {str(error)[:1000]}" if error else "")
    if event_type == "job_summary":
        status = payload.get("status")
        result = payload.get("result") or {}
        score = _find_score(result)
        lines = [f"**Job summary** `{event.job_id}` status=`{status}`"]
        if score is not None:
            lines.append(f"Metric: `{score}`")
        if payload.get("error"):
            lines.append(f"Error: {str(payload['error'])[:1000]}")
        return "\n".join(lines)
    return ""


def _find_score(result: dict[str, Any]) -> Any:
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        for key in ("cv_mean", "score", "metric", "value"):
            if key in metrics:
                return metrics[key]
    for key in ("cv_mean", "score", "metric"):
        if key in result and not isinstance(result[key], (dict, list)):
            return result[key]
    return None


class FakeWorkThreadTransport:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.messages: dict[str, list[str]] = {}
        self.live: dict[str, tuple[str, str]] = {}
        self.tags: dict[str, tuple[str, ...]] = {}

    def create_thread(
        self,
        *,
        title: str,
        initial_message: str,
        tags: tuple[str, ...] = (),
    ) -> CreatedThread:
        thread_id = f"thread-{len(self.created) + 1}"
        message_id = f"live-{len(self.created) + 1}"
        self.created.append(
            {"title": title, "initial_message": initial_message, "tags": tags}
        )
        self.messages[thread_id] = [initial_message]
        self.tags[thread_id] = tags
        return CreatedThread(None, "forum-1", thread_id, message_id)

    def send(self, thread_id: str, content: str) -> str:
        self.messages.setdefault(thread_id, []).append(content)
        return f"message-{len(self.messages[thread_id])}"

    def upsert_live_status(
        self,
        thread_id: str,
        content: str,
        message_id: str | None = None,
    ) -> str:
        identifier = message_id or f"live-{thread_id}"
        self.live[thread_id] = (identifier, content)
        return identifier

    def set_tags(self, thread_id: str, tags: tuple[str, ...]) -> None:
        self.tags[thread_id] = tags
