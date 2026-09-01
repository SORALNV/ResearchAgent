from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from harness.codex_app_server import (
    CodexAppServerRuntime,
    CodexAppServerSettings,
    CodexRuntimeEvent,
)
from harness.codex_app_server_service import CodexAppServerRoutedService
from harness.control_plane import ControlPlaneStore, Domain
from harness.discord_channel_map import DiscordLocation
from harness.discord_thread_router import ChannelDomainMap, DiscordThreadRouter


class _QueueReader:
    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()

    def readline(self) -> str:
        return self._queue.get()

    def push(self, value: Mapping[str, Any]) -> None:
        self._queue.put(json.dumps(dict(value), separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._queue.put("")


class _FakeStdin:
    def __init__(self, server: "_FakeAppServer") -> None:
        self.server = server
        self._buffer = ""

    def write(self, value: str | bytes) -> int:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.server.receive(json.loads(line))
        return len(value)

    def flush(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, server: "_FakeAppServer") -> None:
        self.server = server
        self.stdin = _FakeStdin(server)
        self.stdout = server.stdout
        self.stderr = server.stderr
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.close()
        self.stderr.close()

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()

    def wait(self, timeout: float | None = None) -> int:
        return int(self.returncode or 0)


class _FakeAppServer:
    def __init__(self) -> None:
        self.stdout = _QueueReader()
        self.stderr = _QueueReader()
        self.process = _FakeProcess(self)
        self.messages: list[dict[str, Any]] = []
        self.client_responses: list[dict[str, Any]] = []
        self.thread_id = "thr-root"
        self.turn_number = 0
        self.active_turn_id: str | None = None
        self.turn_started = threading.Event()
        self._lock = threading.RLock()

    def factory(
        self,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> _FakeProcess:
        assert list(command[:2]) == ["codex", "app-server"]
        assert cwd.is_dir()
        assert "DISCORD_BOT_TOKEN" not in environment
        return self.process

    def receive(self, message: Mapping[str, Any]) -> None:
        value = dict(message)
        with self._lock:
            self.messages.append(value)
        method = value.get("method")
        request_id = value.get("id")
        if method is None and request_id is not None:
            with self._lock:
                self.client_responses.append(value)
            return
        if request_id is None:
            return
        if method == "initialize":
            self.respond(request_id, {"serverInfo": {"name": "fake"}})
            return
        if method == "thread/start":
            self.respond(request_id, {"thread": {"id": self.thread_id}})
            self.notify(
                "thread/started",
                {"thread": {"id": self.thread_id, "parentThreadId": None}},
            )
            return
        if method == "thread/resume":
            self.respond(request_id, {"thread": {"id": self.thread_id}})
            return
        if method == "turn/start":
            self.turn_number += 1
            self.active_turn_id = f"turn-{self.turn_number}"
            turn = {
                "id": self.active_turn_id,
                "items": [],
                "itemsView": "full",
                "status": "inProgress",
                "error": None,
                "startedAt": 1,
                "completedAt": None,
                "durationMs": None,
            }
            self.respond(request_id, {"turn": turn})
            self.notify(
                "turn/started",
                {"threadId": self.thread_id, "turn": turn},
            )
            self.turn_started.set()
            return
        if method == "turn/steer":
            self.respond(request_id, {"turnId": value["params"]["expectedTurnId"]})
            return
        if method == "turn/interrupt":
            self.respond(request_id, {})
            self.complete(status="interrupted", text="interrupted")
            return
        raise AssertionError(f"unexpected request: {value}")

    def respond(self, request_id: Any, result: Mapping[str, Any]) -> None:
        self.stdout.push({"id": request_id, "result": dict(result)})

    def notify(self, method: str, params: Mapping[str, Any]) -> None:
        self.stdout.push({"method": method, "params": dict(params)})

    def complete(self, *, status: str = "completed", text: str = "done") -> None:
        assert self.active_turn_id is not None
        turn_id = self.active_turn_id
        item = {
            "type": "agentMessage",
            "id": f"msg-{turn_id}",
            "text": text,
            "phase": "final_answer",
            "memoryCitation": None,
            "delivery": None,
        }
        self.notify(
            "item/completed",
            {
                "threadId": self.thread_id,
                "turnId": turn_id,
                "item": item,
                "completedAtMs": 1,
            },
        )
        self.notify(
            "turn/completed",
            {
                "threadId": self.thread_id,
                "turn": {
                    "id": turn_id,
                    "items": [item],
                    "itemsView": "full",
                    "status": status,
                    "error": None,
                    "startedAt": 1,
                    "completedAt": 2,
                    "durationMs": 10,
                },
            },
        )
        self.active_turn_id = None
        self.turn_started.clear()

    def request_command_approval(self, request_id: int = 900) -> None:
        assert self.active_turn_id is not None
        self.stdout.push(
            {
                "method": "item/commandExecution/requestApproval",
                "id": request_id,
                "params": {
                    "kind": "command",
                    "threadId": self.thread_id,
                    "turnId": self.active_turn_id,
                    "itemId": "item-command",
                    "startedAtMs": 1,
                    "approvalId": None,
                    "environmentId": "local",
                    "reason": "test approval",
                    "command": "python train.py",
                    "cwd": "/workspace",
                    "commandActions": [],
                    "proposedExecpolicyAmendment": None,
                    "proposedNetworkPolicyAmendments": None,
                    "networkApprovalContext": None,
                },
            }
        )

    def start_subagent(self) -> None:
        self.notify(
            "thread/started",
            {
                "thread": {
                    "id": "thr-child",
                    "parentThreadId": self.thread_id,
                }
            },
        )
        self.notify(
            "item/started",
            {
                "threadId": self.thread_id,
                "turnId": self.active_turn_id,
                "item": {
                    "type": "collabAgentToolCall",
                    "id": "collab-1",
                    "tool": "spawnAgent",
                    "status": "inProgress",
                    "senderThreadId": self.thread_id,
                    "receiverThreadIds": ["thr-child"],
                    "prompt": "inspect tests",
                    "model": None,
                    "reasoningEffort": None,
                    "agentsStates": {},
                },
            },
        )

    def requests(self, method: str) -> list[dict[str, Any]]:
        with self._lock:
            return [item for item in self.messages if item.get("method") == method]


def _settings(tmp_path: Path) -> CodexAppServerSettings:
    return CodexAppServerSettings(
        command=("codex", "app-server", "--listen", "stdio://"),
        state_dir=tmp_path / "app-server",
        request_timeout_seconds=2.0,
        turn_timeout_seconds=5.0,
        approval_policy="on-request",
        approvals_reviewer="user",
    )


def _wait(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def _run_in_thread(runtime: CodexAppServerRuntime, tmp_path: Path, session_id: str):
    result: dict[str, Any] = {}

    def target() -> None:
        result["value"] = runtime.run_turn(
            session_id=session_id,
            role="planning",
            stage="discord_research_consultation",
            task_id=None,
            prompt="inspect the repository",
            cwd=tmp_path / "workspace",
            sandbox="workspace-write",
            client_user_message_id="discord:100",
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, result


def test_settings_reject_codex_exec_and_require_user_approval_routing(tmp_path: Path):
    with pytest.raises(ValueError, match="official.*codex app-server"):
        CodexAppServerSettings.from_environment(
            tmp_path,
            {"CODEX_APP_SERVER_COMMAND": "codex exec --json"},
        )
    with pytest.raises(ValueError, match="requires.*user"):
        CodexAppServerSettings.from_environment(
            tmp_path,
            {
                "CODEX_APP_SERVER_COMMAND": "codex app-server --listen stdio://",
                "CODEX_APP_SERVER_APPROVALS_REVIEWER": "auto_review",
            },
        )


def test_thread_is_persisted_and_resumed_for_the_same_discord_work_session(
    tmp_path: Path,
):
    fake = _FakeAppServer()
    runtime = CodexAppServerRuntime(_settings(tmp_path), process_factory=fake.factory)

    first_thread, first_result = _run_in_thread(runtime, tmp_path, "WS-1")
    assert fake.turn_started.wait(2)
    fake.complete(text="first")
    first_thread.join(2)
    assert first_result["value"].output == "first"

    second_thread, second_result = _run_in_thread(runtime, tmp_path, "WS-1")
    assert fake.turn_started.wait(2)
    fake.complete(text="second")
    second_thread.join(2)
    assert second_result["value"].output == "second"

    assert len(fake.requests("thread/start")) == 1
    assert len(fake.requests("thread/resume")) == 1
    assert len(fake.requests("turn/start")) == 2
    turn_params = fake.requests("turn/start")[0]["params"]
    assert turn_params["threadId"] == "thr-root"
    assert turn_params["input"] == [
        {
            "type": "text",
            "text": "inspect the repository",
            "text_elements": [],
        }
    ]
    assert turn_params["approvalPolicy"] == "on-request"
    assert turn_params["approvalsReviewer"] == "user"
    assert "multiAgentMode" not in turn_params
    binding = runtime.bindings.get("discord:WS-1")
    assert binding is not None
    assert binding.thread_id == "thr-root"
    runtime.stop()


def test_active_turn_supports_official_steer_interrupt_and_approval(tmp_path: Path):
    fake = _FakeAppServer()
    runtime = CodexAppServerRuntime(_settings(tmp_path), process_factory=fake.factory)
    events: list[CodexRuntimeEvent] = []
    runtime.add_listener(events.append)

    thread, result = _run_in_thread(runtime, tmp_path, "WS-2")
    assert fake.turn_started.wait(2)
    active_turn = str(fake.active_turn_id)

    response = runtime.steer(
        session_id="WS-2",
        text="also run the focused test",
        client_user_message_id="discord:101",
    )
    assert response["turn_id"] == active_turn
    steer = fake.requests("turn/steer")[-1]
    assert steer["params"]["expectedTurnId"] == active_turn
    assert steer["params"]["input"][0]["text_elements"] == []

    fake.request_command_approval()
    _wait(lambda: bool(runtime.pending_approvals(session_id="WS-2")))
    approval = runtime.pending_approvals(session_id="WS-2")[0]
    assert approval.thread_id == "thr-root"
    runtime.resolve_approval(
        session_id="WS-2",
        approval_ref=approval.approval_ref,
        decision="accept",
    )
    _wait(lambda: bool(fake.client_responses))
    assert fake.client_responses[-1] == {
        "id": 900,
        "result": {"decision": "accept"},
    }

    fake.start_subagent()
    _wait(
        lambda: any(
            item.thread_id == "thr-child" and item.session_id == "WS-2"
            for item in events
        )
    )

    interrupted = runtime.interrupt(session_id="WS-2")
    assert interrupted["turn_id"] == active_turn
    request = fake.requests("turn/interrupt")[-1]
    assert request["params"] == {
        "threadId": "thr-root",
        "turnId": active_turn,
    }
    thread.join(2)
    assert result["value"].status == "interrupted"
    assert result["value"].cancelled is True
    runtime.stop()


class _BaseService:
    def __init__(self, router: DiscordThreadRouter) -> None:
        self.router = router
        self.dispatcher = SimpleNamespace(router=router)
        self.compute = None
        self.final_actions = None
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self, *, wait: bool = False) -> None:
        self.started = False

    def status(self, *args: Any, **kwargs: Any) -> str:
        return "base status"


def test_routed_service_persists_safe_events_and_steers_active_turn(tmp_path: Path):
    fake = _FakeAppServer()
    runtime = CodexAppServerRuntime(_settings(tmp_path), process_factory=fake.factory)
    router = DiscordThreadRouter(
        ControlPlaneStore(tmp_path / "control-plane"),
        ChannelDomainMap({"100": Domain.RESEARCH}),
    )
    service = CodexAppServerRoutedService(_BaseService(router), runtime)
    location = DiscordLocation(guild_id="1", channel_id="100")
    route = router.resolve_work_session(location, title="Research")
    session_id = route.work_session.work_session_id
    delivered: list[tuple[str, str]] = []
    service.set_codex_event_sink(lambda sid, text: delivered.append((sid, text)))

    result: dict[str, Any] = {}

    def target() -> None:
        result["value"] = runtime.run_turn(
            session_id=session_id,
            role="planning",
            stage="discord_research_consultation",
            task_id=None,
            prompt="first message must not be persisted in runtime status events",
            cwd=tmp_path / "workspace-service",
            sandbox="read-only",
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert fake.turn_started.wait(2)
    steer = service.steer_codex(
        location,
        message_id="200",
        actor_id="300",
        text="steer from Discord",
        title="Research",
    )
    assert steer.work_session_id == session_id
    assert fake.requests("turn/steer")[-1]["params"]["expectedTurnId"] == steer.turn_id

    fake.request_command_approval()
    _wait(lambda: bool(service.pending_codex_approvals(location, title="Research")))
    approval = service.pending_codex_approvals(location, title="Research")[0]
    service.resolve_codex_approval(
        location,
        title="Research",
        approval_ref=approval.approval_ref,
        decision="decline",
        actor_id="300",
        request_id="201",
    )
    fake.complete(text="completed after decline")
    thread.join(2)

    events = router.store.latest_events(
        work_session_id=session_id,
        limit=500,
    )
    assert any(item.event_type == "codex.thread.bound" for item in events)
    assert any(item.event_type == "codex.turn.steer.sent" for item in events)
    assert any(
        item.event_type
        == "codex.app_server.item.commandExecution.requestApproval"
        for item in events
    )
    serialized = json.dumps([item.to_dict() for item in events], ensure_ascii=False)
    assert "first message must not be persisted" not in serialized
    assert any(sid == session_id and "approval required" in text for sid, text in delivered)
    service.stop()


def test_discord_adapter_exposes_app_server_controls_and_main_uses_it():
    root = Path(__file__).resolve().parents[1]
    adapter = (root / "harness" / "codex_discord_adapter.py").read_text(
        encoding="utf-8"
    )
    main = (root / "main.py").read_text(encoding="utf-8")
    for command in (
        'name="steer"',
        'name="interrupt"',
        'name="codex_status"',
        'name="codex_approvals"',
        'name="codex_approval"',
    ):
        assert command in adapter
    assert "CodexAppServerDiscordBotAdapter" in main
    assert "DomainRoutedDiscordBotAdapter(" not in main
