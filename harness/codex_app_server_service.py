from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from harness.codex_app_server import (
    CodexAppServerBusy,
    CodexAppServerRuntime,
    CodexRuntimeEvent,
    PendingCodexApproval,
    get_shared_codex_app_server,
)
from harness.config import HarnessConfig
from harness.control_plane import Event, EventLane, NotFoundError
from harness.discord_channel_map import DiscordLocation
from harness.discord_thread_router import DiscordIngressResult, DiscordThreadRoute, DiscordThreadRouter
from harness.final_actions import build_complete_routed_service


CodexDiscordEventSink = Callable[[str, str], None]


@dataclass(frozen=True)
class CodexSteerResult:
    work_session_id: str
    thread_id: str
    turn_id: str
    source_event_id: str
    cached: bool = False


@dataclass(frozen=True)
class CodexInterruptResult:
    work_session_id: str
    thread_id: str
    turn_id: str
    cached: bool = False


@dataclass(frozen=True)
class CodexApprovalResult:
    work_session_id: str
    approval_ref: str
    decision: str
    thread_id: str
    turn_id: str


class CodexAppServerRoutedService:
    """Add Codex App Server thread control to the existing routed service.

    The wrapped service remains responsible for Research/Kaggle domain routing,
    Compute jobs, final actions, and human research-direction gates. This layer
    only owns the Codex App Server transport and its Discord-facing control
    surface.
    """

    def __init__(
        self,
        base_service: Any,
        runtime: CodexAppServerRuntime,
    ) -> None:
        self.base_service = base_service
        self.router: DiscordThreadRouter = base_service.router
        self.dispatcher = base_service.dispatcher
        self.compute = getattr(base_service, "compute", None)
        self.final_actions = getattr(base_service, "final_actions", None)
        self.codex_app_server = runtime
        self._event_sink: CodexDiscordEventSink | None = None
        self._sink_lock = threading.RLock()
        self._listener_token = runtime.add_listener(self._on_codex_event)
        self._started = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_service, name)

    def start(self) -> None:
        if self._started:
            return
        self.codex_app_server.start()
        try:
            self.base_service.start()
        except Exception:
            self.codex_app_server.stop()
            raise
        self._started = True

    def stop(self, *, wait: bool = False) -> None:
        try:
            self.base_service.stop(wait=wait)
        finally:
            self.codex_app_server.remove_listener(self._listener_token)
            self.codex_app_server.stop()
            self._started = False

    def set_codex_event_sink(
        self,
        sink: CodexDiscordEventSink | None,
    ) -> None:
        with self._sink_lock:
            self._event_sink = sink

    def handle_message(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_service.handle_message(*args, **kwargs)

    def record_decision(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_service.record_decision(*args, **kwargs)

    def check_gate(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_service.check_gate(*args, **kwargs)

    def status(
        self,
        location: DiscordLocation,
        *,
        title: str,
        project_id: str | None = None,
    ) -> str:
        base = self.base_service.status(
            location,
            title=title,
            project_id=project_id,
        )
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        state = self.codex_status_for_route(route)
        active = state.get("active_turn")
        threads = state.get("threads") or []
        approvals = state.get("pending_approvals") or []
        lines = [
            base,
            "Codex App Server:",
            f"- running: {bool(state.get('running'))}",
            f"- bound_threads: {len(threads)}",
            f"- active_turn: {json.dumps(active, ensure_ascii=False) if active else '-'}",
            f"- pending_approvals: {len(approvals)}",
        ]
        for item in approvals[:10]:
            lines.append(
                "  - "
                + str(item.get("approval_ref") or "?")
                + ": "
                + str(item.get("kind") or item.get("method") or "approval")
            )
        return "\n".join(lines)

    def codex_status(
        self,
        location: DiscordLocation,
        *,
        title: str,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        return self.codex_status_for_route(route)

    def codex_status_for_route(self, route: DiscordThreadRoute) -> dict[str, Any]:
        session_id = route.work_session.work_session_id
        state = self.codex_app_server.status(session_id=session_id)
        active = self.codex_app_server.client.active_for_binding(
            f"discord:{session_id}"
        )
        state["active_turn"] = (
            {
                "thread_id": active.thread_id,
                "turn_id": active.turn_id,
                "status": active.status,
            }
            if active is not None
            else None
        )
        state["pending_approvals"] = [
            item.to_dict()
            for item in self._active_pending_approvals(session_id, active)
        ]
        # stderr can contain provider diagnostics and is intentionally excluded
        # from the Discord-facing status payload.
        state.pop("stderr_tail", None)
        return state

    def try_steer_codex(
        self,
        location: DiscordLocation,
        *,
        message_id: str,
        actor_id: str,
        text: str,
        title: str,
        project_id: str | None = None,
    ) -> CodexSteerResult | None:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        session_id = route.work_session.work_session_id
        active = self.codex_app_server.client.active_for_binding(
            f"discord:{session_id}"
        )
        if active is None:
            return None

        ingress = self.router.ingest_message(
            location,
            message_id=message_id,
            actor_id=actor_id,
            text=text,
            title=title,
            project_id=project_id,
        )
        cached = self._find_control_event(
            session_id,
            "codex.turn.steer.sent",
            "source_event_id",
            ingress.event.event_id,
        )
        if cached is not None:
            return CodexSteerResult(
                work_session_id=session_id,
                thread_id=str(cached.payload.get("thread_id") or active.thread_id),
                turn_id=str(cached.payload.get("turn_id") or active.turn_id),
                source_event_id=ingress.event.event_id,
                cached=True,
            )

        requested = self.router.store.append_event(
            event_type="codex.turn.steer.requested",
            lane=EventLane.CONTROL,
            project_id=route.project.project_id,
            work_session_id=session_id,
            actor=f"discord:{actor_id}",
            payload={
                "source_event_id": ingress.event.event_id,
                "thread_id": active.thread_id,
                "turn_id": active.turn_id,
                "message_id": str(message_id),
            },
            idempotency_key=f"codex:steer-request:{ingress.event.event_id}",
        )
        try:
            result = self.codex_app_server.steer(
                session_id=session_id,
                text=text,
                client_user_message_id=f"discord:{message_id}",
            )
        except CodexAppServerBusy:
            # The active turn may have completed between the preflight check and
            # the official turn/steer request. The idempotent Discord ingress can
            # safely be processed as a new turn by the caller.
            return None
        sent = self.router.store.append_event(
            event_type="codex.turn.steer.sent",
            lane=EventLane.CONTROL,
            project_id=route.project.project_id,
            work_session_id=session_id,
            actor="core:codex-app-server",
            payload={
                "source_event_id": ingress.event.event_id,
                "request_event_id": requested.event_id,
                "thread_id": str(result["thread_id"]),
                "turn_id": str(result["turn_id"]),
                "message_id": str(message_id),
            },
            idempotency_key=f"codex:steer-sent:{ingress.event.event_id}",
        )
        return CodexSteerResult(
            work_session_id=session_id,
            thread_id=str(sent.payload["thread_id"]),
            turn_id=str(sent.payload["turn_id"]),
            source_event_id=ingress.event.event_id,
        )

    def steer_codex(
        self,
        location: DiscordLocation,
        *,
        message_id: str,
        actor_id: str,
        text: str,
        title: str,
        project_id: str | None = None,
    ) -> CodexSteerResult:
        result = self.try_steer_codex(
            location,
            message_id=message_id,
            actor_id=actor_id,
            text=text,
            title=title,
            project_id=project_id,
        )
        if result is None:
            raise CodexAppServerBusy(
                "this WorkSession has no steerable Discord Codex turn"
            )
        return result

    def interrupt_codex(
        self,
        location: DiscordLocation,
        *,
        title: str,
        actor_id: str,
        request_id: str,
        project_id: str | None = None,
    ) -> CodexInterruptResult:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        session_id = route.work_session.work_session_id
        cached = self._find_control_event(
            session_id,
            "codex.turn.interrupt.sent",
            "request_id",
            str(request_id),
        )
        if cached is not None:
            return CodexInterruptResult(
                work_session_id=session_id,
                thread_id=str(cached.payload.get("thread_id") or ""),
                turn_id=str(cached.payload.get("turn_id") or ""),
                cached=True,
            )
        result = self.codex_app_server.interrupt(session_id=session_id)
        event = self.router.store.append_event(
            event_type="codex.turn.interrupt.sent",
            lane=EventLane.CONTROL,
            project_id=route.project.project_id,
            work_session_id=session_id,
            actor=f"discord-human:{actor_id}",
            payload={
                "request_id": str(request_id),
                "thread_id": str(result["thread_id"]),
                "turn_id": str(result["turn_id"]),
            },
            idempotency_key=f"codex:interrupt:{session_id}:{request_id}",
        )
        return CodexInterruptResult(
            work_session_id=session_id,
            thread_id=str(event.payload["thread_id"]),
            turn_id=str(event.payload["turn_id"]),
        )

    def pending_codex_approvals(
        self,
        location: DiscordLocation,
        *,
        title: str,
        project_id: str | None = None,
    ) -> tuple[PendingCodexApproval, ...]:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        session_id = route.work_session.work_session_id
        active = self.codex_app_server.client.active_for_binding(
            f"discord:{session_id}"
        )
        return self._active_pending_approvals(session_id, active)

    def resolve_codex_approval(
        self,
        location: DiscordLocation,
        *,
        title: str,
        approval_ref: str,
        decision: str,
        actor_id: str,
        request_id: str,
        project_id: str | None = None,
    ) -> CodexApprovalResult:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        session_id = route.work_session.work_session_id
        pending = {
            item.approval_ref: item
            for item in self.pending_codex_approvals(
                location,
                title=title,
                project_id=project_id,
            )
        }
        if approval_ref not in pending:
            raise KeyError(
                "unknown or no-longer-active Codex approval in this WorkSession: "
                + approval_ref
            )
        approval = self.codex_app_server.resolve_approval(
            session_id=session_id,
            approval_ref=approval_ref,
            decision=decision,
        )
        event = self.router.store.append_event(
            event_type="codex.approval.discord_decision",
            lane=EventLane.CONTROL,
            project_id=route.project.project_id,
            work_session_id=session_id,
            actor=f"discord-human:{actor_id}",
            payload={
                "request_id": str(request_id),
                "approval_ref": approval.approval_ref,
                "decision": decision,
                "thread_id": approval.thread_id,
                "turn_id": approval.turn_id,
                "item_id": approval.item_id,
                "request_method": approval.method,
            },
            idempotency_key=(
                f"codex:approval:{session_id}:{approval.approval_ref}:{request_id}"
            ),
        )
        return CodexApprovalResult(
            work_session_id=session_id,
            approval_ref=approval.approval_ref,
            decision=str(event.payload["decision"]),
            thread_id=approval.thread_id,
            turn_id=approval.turn_id,
        )

    def _active_pending_approvals(
        self,
        session_id: str,
        active: Any | None,
    ) -> tuple[PendingCodexApproval, ...]:
        if active is None:
            return ()
        return tuple(
            item
            for item in self.codex_app_server.pending_approvals(
                session_id=session_id
            )
            if item.thread_id == active.thread_id and item.turn_id == active.turn_id
        )

    def _find_control_event(
        self,
        work_session_id: str,
        event_type: str,
        field: str,
        value: str,
    ) -> Event | None:
        for event in reversed(
            self.router.store.latest_events(
                work_session_id=work_session_id,
                lanes=[EventLane.CONTROL],
                limit=2000,
            )
        ):
            if event.event_type != event_type:
                continue
            if str(event.payload.get(field) or "") == str(value):
                return event
        return None

    def _on_codex_event(self, event: CodexRuntimeEvent) -> None:
        session_id = event.session_id
        if not session_id:
            return
        try:
            session = self.router.store.get_work_session(session_id)
            project = self.router.store.get_project(session.project_id)
        except NotFoundError:
            return
        payload = _safe_event_payload(event)
        lane = (
            EventLane.CONTROL
            if event.method in {
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
                "approval/resolved",
                "approval/expired",
                "serverRequest/resolved",
            }
            else EventLane.STATUS
        )
        self.router.store.append_event(
            event_type="codex.app_server." + event.method.replace("/", "."),
            lane=lane,
            project_id=project.project_id,
            work_session_id=session_id,
            actor="codex-app-server",
            payload=payload,
            idempotency_key=f"codex:runtime-event:{event.event_id}",
        )
        if event.thread_id:
            self.router.store.append_event(
                event_type="codex.thread.bound",
                lane=EventLane.CONTROL,
                project_id=project.project_id,
                work_session_id=session_id,
                actor="core:codex-app-server",
                payload={
                    "thread_id": event.thread_id,
                    "binding_key": f"discord:{session_id}",
                    "source_method": event.method,
                },
                idempotency_key=f"codex:thread-binding:{session_id}:{event.thread_id}",
            )
        message = _format_event_for_discord(event, payload)
        if not message:
            return
        with self._sink_lock:
            sink = self._event_sink
        if sink is not None:
            try:
                sink(session_id, message)
            except Exception:
                return


def build_codex_app_server_routed_service(
    config: HarnessConfig,
    router: DiscordThreadRouter,
) -> CodexAppServerRoutedService:
    """Build the existing complete service with one shared App Server runtime."""

    base = build_complete_routed_service(config, router)
    runtime = get_shared_codex_app_server(config)
    return CodexAppServerRoutedService(base, runtime)


def _safe_event_payload(event: CodexRuntimeEvent) -> dict[str, Any]:
    params = event.params
    payload: dict[str, Any] = {
        "method": event.method,
        "event_id": event.event_id,
        "thread_id": event.thread_id,
        "turn_id": event.turn_id,
        "approval_ref": event.approval_ref,
    }
    if event.method in {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }:
        payload.update(
            {
                "item_id": str(params.get("itemId") or ""),
                "reason": str(params.get("reason") or "")[:2000],
                "command": str(params.get("command") or "")[:4000],
                "cwd": str(params.get("cwd") or "")[:1000],
                "grant_root": str(params.get("grantRoot") or "")[:1000],
                "environment_id": str(params.get("environmentId") or "")[:200],
            }
        )
        return payload
    if event.method in {"turn/started", "turn/completed"}:
        turn = params.get("turn")
        if isinstance(turn, Mapping):
            error = turn.get("error")
            payload.update(
                {
                    "status": str(turn.get("status") or ""),
                    "duration_ms": turn.get("durationMs"),
                    "error": _error_text(error),
                }
            )
        return payload
    if event.method in {"item/started", "item/completed"}:
        item = params.get("item")
        if isinstance(item, Mapping):
            item_type = str(item.get("type") or "")
            payload.update(
                {
                    "item_id": str(item.get("id") or ""),
                    "item_type": item_type,
                    "status": str(item.get("status") or ""),
                }
            )
            if item_type == "commandExecution":
                payload.update(
                    {
                        "command": str(item.get("command") or "")[:4000],
                        "cwd": str(item.get("cwd") or "")[:1000],
                        "exit_code": item.get("exitCode"),
                        "duration_ms": item.get("durationMs"),
                    }
                )
            elif item_type == "mcpToolCall":
                payload.update(
                    {
                        "server": str(item.get("server") or "")[:300],
                        "tool": str(item.get("tool") or "")[:300],
                        "duration_ms": item.get("durationMs"),
                    }
                )
            elif item_type == "collabAgentToolCall":
                payload.update(
                    {
                        "tool": str(item.get("tool") or "")[:200],
                        "receiver_thread_ids": [
                            str(value)
                            for value in (item.get("receiverThreadIds") or [])[:20]
                        ],
                        "model": str(item.get("model") or "")[:200],
                    }
                )
            elif item_type == "subAgentActivity":
                payload.update(
                    {
                        "kind": str(item.get("kind") or "")[:200],
                        "agent_thread_id": str(item.get("agentThreadId") or "")[:300],
                        "agent_path": str(item.get("agentPath") or "")[:1000],
                    }
                )
        return payload
    if event.method == "turn/plan/updated":
        plan = params.get("plan")
        payload["explanation"] = str(params.get("explanation") or "")[:2000]
        if isinstance(plan, list):
            payload["plan"] = [
                {
                    "step": str(item.get("step") or "")[:1000],
                    "status": str(item.get("status") or "")[:100],
                }
                for item in plan[:30]
                if isinstance(item, Mapping)
            ]
        return payload
    if event.method == "thread/started":
        thread = params.get("thread")
        if isinstance(thread, Mapping):
            payload.update(
                {
                    "thread_id": str(thread.get("id") or event.thread_id or ""),
                    "parent_thread_id": str(thread.get("parentThreadId") or ""),
                }
            )
        return payload
    if event.method == "thread/tokenUsage/updated":
        usage = params.get("tokenUsage")
        if isinstance(usage, Mapping):
            payload["token_usage"] = _numeric_tree(usage)
        return payload
    if event.method in {"approval/resolved", "approval/expired", "serverRequest/resolved"}:
        payload.update(
            {
                "request_id": str(params.get("requestId") or ""),
                "decision": str(params.get("decision") or "")[:100],
                "reason": str(params.get("reason") or "")[:1000],
            }
        )
        return payload
    if event.method == "error":
        payload["error"] = _error_text(params.get("error") or params.get("message"))
    return payload


def _format_event_for_discord(
    event: CodexRuntimeEvent,
    payload: Mapping[str, Any],
) -> str | None:
    method = event.method
    if method == "turn/started":
        return f"Codex turn started: `{event.turn_id or '?'}`"
    if method == "turn/completed":
        return (
            "Codex turn completed: status=`"
            + str(payload.get("status") or "unknown")
            + "` turn=`"
            + str(event.turn_id or "?")
            + "`"
        )
    if method == "turn/plan/updated":
        lines = ["Codex plan updated"]
        explanation = str(payload.get("explanation") or "").strip()
        if explanation:
            lines.append(explanation)
        for item in payload.get("plan") or []:
            if isinstance(item, Mapping):
                lines.append(
                    "- ["
                    + str(item.get("status") or "pending")
                    + "] "
                    + str(item.get("step") or "")
                )
        return "\n".join(lines)[:1900]
    if method in {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }:
        lines = [
            "Codex approval required",
            f"- approval: `{event.approval_ref or '?'}`",
            f"- type: `{method}`",
        ]
        command = str(payload.get("command") or "").strip()
        if command:
            lines.append(f"- command: `{command[:1200]}`")
        reason = str(payload.get("reason") or "").strip()
        if reason:
            lines.append(f"- reason: {reason[:1200]}")
        lines.append(
            "Use `/agent codex_approval` with `accept`, "
            "`acceptForSession`, `decline`, or `cancel`."
        )
        return "\n".join(lines)[:1900]
    if method == "item/started":
        item_type = str(payload.get("item_type") or "")
        if item_type == "commandExecution":
            return "Codex command started: `" + str(payload.get("command") or "")[:1500] + "`"
        if item_type == "fileChange":
            return "Codex file change started."
        if item_type == "mcpToolCall":
            return (
                "Codex MCP tool started: `"
                + str(payload.get("server") or "?")
                + "/"
                + str(payload.get("tool") or "?")
                + "`"
            )
        if item_type in {"collabAgentToolCall", "subAgentActivity"}:
            return f"Codex Harness multi-agent event: `{item_type}`"
    if method == "item/completed":
        item_type = str(payload.get("item_type") or "")
        if item_type == "commandExecution":
            return (
                "Codex command completed: status=`"
                + str(payload.get("status") or "unknown")
                + "` exit=`"
                + str(payload.get("exit_code"))
                + "`"
            )
        if item_type == "fileChange":
            return "Codex file change completed: status=`" + str(payload.get("status") or "unknown") + "`"
        if item_type in {"mcpToolCall", "collabAgentToolCall", "subAgentActivity"}:
            return f"Codex item completed: `{item_type}`"
    if method == "thread/started" and payload.get("parent_thread_id"):
        return (
            "Codex Harness subagent thread started: `"
            + str(payload.get("thread_id") or "?")
            + "`"
        )
    if method == "error":
        return "Codex App Server error: " + str(payload.get("error") or "unknown error")[:1500]
    return None


def _error_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(
            value.get("message")
            or value.get("additionalDetails")
            or value.get("codexErrorInfo")
            or value
        )[:4000]
    return str(value or "")[:4000]


def _numeric_tree(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[str(key)] = _numeric_tree(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[str(key)] = item
        elif item is None:
            result[str(key)] = None
    return result
