from __future__ import annotations

import atexit
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from harness.artifacts import ArtifactRecord, build_artifact_manifest
from harness.cost import estimate_tokens
from harness.state import utc_timestamp

# Stable v2 server-initiated approvals currently exposed to Discord. Other
# server requests are rejected with a JSON-RPC error rather than receiving an
# invented response payload.
APPROVAL_METHODS = {
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
}
APPROVAL_DECISIONS = {"accept", "acceptForSession", "decline", "cancel"}
TERMINAL_TURN_STATUSES = {"completed", "interrupted", "failed"}


class CodexAppServerError(RuntimeError):
    """Base error for the official Codex App Server transport."""


class CodexAppServerUnavailable(CodexAppServerError):
    pass


class CodexAppServerTimeout(CodexAppServerError):
    pass


class CodexAppServerBusy(CodexAppServerError):
    pass


class CodexAppServerProtocolError(CodexAppServerError):
    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


@dataclass(frozen=True)
class CodexAppServerSettings:
    command: tuple[str, ...]
    state_dir: Path
    request_timeout_seconds: float = 30.0
    turn_timeout_seconds: float | None = None
    approval_policy: str = "on-request"
    approvals_reviewer: str = "user"
    model: str | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    personality: str | None = None
    network_access: bool = False
    client_name: str = "research_agent_discord"
    client_title: str = "ResearchAgent Discord"
    client_version: str = "0.3.0"

    @classmethod
    def from_environment(
        cls,
        project_root: str | Path,
        environ: Mapping[str, str] | None = None,
    ) -> "CodexAppServerSettings":
        source = dict(os.environ if environ is None else environ)
        raw_command = source.get(
            "CODEX_APP_SERVER_COMMAND",
            "codex app-server --listen stdio://",
        )
        try:
            command = tuple(shlex.split(raw_command))
        except ValueError as exc:
            raise ValueError("CODEX_APP_SERVER_COMMAND is not valid argv") from exc
        if not command:
            raise ValueError("CODEX_APP_SERVER_COMMAND must not be empty")
        executable = Path(command[0]).name.lower()
        if executable not in {"codex", "codex.exe"} or "app-server" not in command[1:]:
            raise ValueError(
                "CODEX_APP_SERVER_COMMAND must invoke the official `codex app-server`"
            )
        if "exec" in command[1:]:
            raise ValueError("codex exec is disabled; use codex app-server")

        root = Path(project_root).expanduser().resolve()
        state_dir = Path(
            source.get("CODEX_APP_SERVER_STATE_DIR", "codex_app_server")
        ).expanduser()
        if not state_dir.is_absolute():
            state_dir = root / state_dir

        approval_policy = source.get(
            "CODEX_APP_SERVER_APPROVAL_POLICY", "on-request"
        ).strip()
        if approval_policy not in {"untrusted", "on-request", "never"}:
            raise ValueError(
                "CODEX_APP_SERVER_APPROVAL_POLICY must be untrusted, on-request, or never"
            )
        approvals_reviewer = source.get(
            "CODEX_APP_SERVER_APPROVALS_REVIEWER", "user"
        ).strip()
        if approvals_reviewer != "user":
            raise ValueError(
                "Discord approval routing requires CODEX_APP_SERVER_APPROVALS_REVIEWER=user"
            )

        return cls(
            command=command,
            state_dir=state_dir,
            request_timeout_seconds=max(
                1.0,
                _float_value(
                    source.get("CODEX_APP_SERVER_REQUEST_TIMEOUT_SECONDS"), 30.0
                ),
            ),
            turn_timeout_seconds=_optional_positive_float(
                source.get("CODEX_APP_SERVER_TURN_TIMEOUT_SECONDS")
            ),
            approval_policy=approval_policy,
            approvals_reviewer=approvals_reviewer,
            model=_optional_text(source.get("CODEX_APP_SERVER_MODEL")),
            reasoning_effort=_optional_text(
                source.get("CODEX_APP_SERVER_REASONING_EFFORT")
            ),
            reasoning_summary=_optional_text(
                source.get("CODEX_APP_SERVER_REASONING_SUMMARY")
            ),
            personality=_optional_text(source.get("CODEX_APP_SERVER_PERSONALITY")),
            network_access=_bool_value(
                source.get("CODEX_APP_SERVER_NETWORK_ACCESS"), False
            ),
            client_name=(
                source.get("CODEX_APP_SERVER_CLIENT_NAME")
                or "research_agent_discord"
            ).strip(),
            client_title=(
                source.get("CODEX_APP_SERVER_CLIENT_TITLE")
                or "ResearchAgent Discord"
            ).strip(),
            client_version=(
                source.get("CODEX_APP_SERVER_CLIENT_VERSION") or "0.3.0"
            ).strip(),
        )


@dataclass(frozen=True)
class CodexThreadBinding:
    binding_key: str
    session_id: str
    thread_id: str
    cwd: str
    role: str
    task_id: str | None
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CodexThreadBinding":
        return cls(
            binding_key=str(data["binding_key"]),
            session_id=str(data["session_id"]),
            thread_id=str(data["thread_id"]),
            cwd=str(data["cwd"]),
            role=str(data.get("role") or "main"),
            task_id=(str(data["task_id"]) if data.get("task_id") else None),
            created_at=str(data.get("created_at") or utc_timestamp()),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


class CodexThreadBindingStore:
    """Durably map ResearchAgent WorkSessions/scopes to Codex thread IDs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / "thread_bindings.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def get(self, binding_key: str) -> CodexThreadBinding | None:
        with self._lock:
            return self._read_all().get(binding_key)

    def put(self, binding: CodexThreadBinding) -> CodexThreadBinding:
        with self._lock:
            values = self._read_all()
            existing = values.get(binding.binding_key)
            stored = replace(
                binding,
                created_at=existing.created_at if existing else binding.created_at,
                updated_at=utc_timestamp(),
            )
            values[stored.binding_key] = stored
            self._write_all(values)
            return stored

    def list_for_session(self, session_id: str) -> tuple[CodexThreadBinding, ...]:
        with self._lock:
            values = [
                item for item in self._read_all().values() if item.session_id == session_id
            ]
        return tuple(sorted(values, key=lambda item: item.binding_key))

    def _read_all(self) -> dict[str, CodexThreadBinding]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, Mapping):
            return {}
        result: dict[str, CodexThreadBinding] = {}
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                continue
            try:
                result[str(key)] = CodexThreadBinding.from_dict(value)
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def _write_all(self, values: Mapping[str, CodexThreadBinding]) -> None:
        payload = {
            key: value.to_dict()
            for key, value in sorted(values.items(), key=lambda pair: pair[0])
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        temporary.replace(self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


@dataclass(frozen=True)
class PendingCodexApproval:
    approval_ref: str
    request_id: str | int
    method: str
    session_id: str | None
    thread_id: str
    turn_id: str
    item_id: str
    params: dict[str, Any]
    created_at: str = field(default_factory=utc_timestamp)

    @property
    def kind(self) -> str:
        if self.method == "item/commandExecution/requestApproval":
            return "command"
        return "file_change"

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_ref": self.approval_ref,
            "method": self.method,
            "kind": self.kind,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "item_id": self.item_id,
            "params": _json_dict(self.params),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CodexRuntimeEvent:
    event_id: str
    method: str
    session_id: str | None
    thread_id: str | None
    turn_id: str | None
    params: dict[str, Any]
    approval_ref: str | None = None
    created_at: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "method": self.method,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "params": _json_dict(self.params),
            "approval_ref": self.approval_ref,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CodexTurnResult:
    session_id: str
    binding_key: str
    thread_id: str
    turn_id: str
    status: str
    output: str
    error: str | None
    duration_seconds: float
    cancelled: bool = False
    timed_out: bool = False


@dataclass
class _PendingResponse:
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Mapping[str, Any] | None = None


@dataclass
class _TurnRun:
    session_id: str
    binding_key: str
    thread_id: str
    turn_id: str
    started_monotonic: float
    done: threading.Event = field(default_factory=threading.Event)
    deltas: list[str] = field(default_factory=list)
    final_messages: list[str] = field(default_factory=list)
    status: str = "inProgress"
    error: str | None = None


class ProcessLike(Protocol):
    stdin: Any
    stdout: Any
    stderr: Any

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


ProcessFactory = Callable[[Sequence[str], Path, Mapping[str, str]], ProcessLike]
RuntimeListener = Callable[[CodexRuntimeEvent], None]


class CodexAppServerClient:
    """Thin bidirectional JSONL client for the official App Server v2 API.

    App Server intentionally omits the ``jsonrpc`` field on the wire. Each line
    is one request, response, notification, or server-initiated request.
    """

    def __init__(
        self,
        settings: CodexAppServerSettings,
        *,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self.settings = settings
        self.process_factory = process_factory or _default_process_factory
        self._process: ProcessLike | None = None
        self._start_lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._next_request_id = 1
        self._pending: dict[str | int, _PendingResponse] = {}
        self._listeners: dict[str, RuntimeListener] = {}
        self._approvals: dict[str, PendingCodexApproval] = {}
        self._thread_sessions: dict[str, str] = {}
        self._owned_threads: set[str] = set()
        self._active_by_thread: dict[str, _TurnRun] = {}
        self._runs: dict[tuple[str, str], _TurnRun] = {}
        self._notification_backlog: dict[
            tuple[str, str], list[tuple[str, dict[str, Any]]]
        ] = {}
        self._stderr_tail: list[str] = []
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._closed = False

    @property
    def running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None and not self._closed

    @property
    def stderr_tail(self) -> str:
        with self._state_lock:
            return "\n".join(self._stderr_tail[-50:])[-8000:]

    def add_listener(self, listener: RuntimeListener) -> str:
        token = uuid.uuid4().hex
        with self._state_lock:
            self._listeners[token] = listener
        return token

    def remove_listener(self, token: str) -> None:
        with self._state_lock:
            self._listeners.pop(token, None)

    def start(self) -> None:
        with self._start_lock:
            if self.running:
                return
            if self._closed:
                raise CodexAppServerUnavailable("Codex App Server client is closed")
            executable = self.settings.command[0]
            if (
                self.process_factory is _default_process_factory
                and not Path(executable).is_absolute()
                and shutil.which(executable) is None
            ):
                raise CodexAppServerUnavailable(
                    f"Codex executable is unavailable: {executable}"
                )
            self.settings.state_dir.mkdir(parents=True, exist_ok=True)
            try:
                process = self.process_factory(
                    self.settings.command,
                    self.settings.state_dir,
                    _app_server_environment(),
                )
            except Exception as exc:
                raise CodexAppServerUnavailable(
                    f"unable to start Codex App Server: {type(exc).__name__}: {exc}"
                ) from exc
            self._process = process
            self._reader = threading.Thread(
                target=self._read_stdout,
                name="codex-app-server-stdout",
                daemon=True,
            )
            self._stderr_reader = threading.Thread(
                target=self._read_stderr,
                name="codex-app-server-stderr",
                daemon=True,
            )
            self._reader.start()
            self._stderr_reader.start()
            try:
                self._request_started(
                    "initialize",
                    {
                        "clientInfo": {
                            "name": self.settings.client_name,
                            "title": self.settings.client_title,
                            "version": self.settings.client_version,
                        },
                        "capabilities": {
                            "experimentalApi": False,
                            "requestAttestation": False,
                        },
                    },
                    timeout=self.settings.request_timeout_seconds,
                )
                # ClientNotification is exactly {"method":"initialized"}.
                self.notify("initialized")
            except Exception:
                self._terminate_process()
                raise

    def stop(self) -> None:
        with self._start_lock:
            self._closed = True
            self._fail_pending("Codex App Server stopped")
            self._expire_approvals("Codex App Server stopped")
            self._terminate_process()

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        self.start()
        return self._request_started(
            method,
            params or {},
            timeout=(
                self.settings.request_timeout_seconds if timeout is None else timeout
            ),
        )

    def notify(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.running:
            self.start()
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = _json_dict(params)
        self._send(payload)

    def bind_thread(self, thread_id: str, session_id: str) -> None:
        with self._state_lock:
            normalized = str(thread_id)
            self._thread_sessions[normalized] = str(session_id)
            self._owned_threads.add(normalized)

    def register_turn(self, run: _TurnRun) -> None:
        key = (run.thread_id, run.turn_id)
        with self._state_lock:
            existing = self._active_by_thread.get(run.thread_id)
            if existing is not None and not existing.done.is_set():
                raise CodexAppServerBusy(
                    f"Codex thread already has an active turn: {run.thread_id}"
                )
            self._runs[key] = run
            self._active_by_thread[run.thread_id] = run
            backlog = self._notification_backlog.pop(key, [])
        for method, params in backlog:
            self._apply_turn_notification(method, params, run)

    def active_for_session(self, session_id: str) -> _TurnRun | None:
        with self._state_lock:
            for run in self._active_by_thread.values():
                if run.session_id == session_id and not run.done.is_set():
                    return run
        return None

    def active_for_thread(self, thread_id: str) -> _TurnRun | None:
        with self._state_lock:
            run = self._active_by_thread.get(thread_id)
            if run is not None and not run.done.is_set():
                return run
        return None

    def active_for_binding(self, binding_key: str) -> _TurnRun | None:
        with self._state_lock:
            for run in self._active_by_thread.values():
                if run.binding_key == binding_key and not run.done.is_set():
                    return run
        return None

    def pending_approvals(
        self,
        *,
        session_id: str | None = None,
    ) -> tuple[PendingCodexApproval, ...]:
        with self._state_lock:
            values = list(self._approvals.values())
        if session_id is not None:
            values = [item for item in values if item.session_id == session_id]
        return tuple(sorted(values, key=lambda item: item.created_at))

    def resolve_approval(
        self,
        approval_ref: str,
        decision: str,
        *,
        session_id: str | None = None,
    ) -> PendingCodexApproval:
        normalized = _normalize_approval_decision(decision)
        with self._state_lock:
            approval = self._approvals.get(approval_ref)
            if approval is None:
                raise KeyError(f"unknown Codex approval: {approval_ref}")
            if session_id is not None and approval.session_id != session_id:
                raise PermissionError(
                    "Codex approval does not belong to this WorkSession"
                )
        self._send(
            {
                "id": approval.request_id,
                "result": {"decision": normalized},
            }
        )
        with self._state_lock:
            self._approvals.pop(approval_ref, None)
        self._emit_event(
            method="approval/resolved",
            params={
                "approvalRef": approval_ref,
                "decision": normalized,
                "requestMethod": approval.method,
            },
            session_id=approval.session_id,
            thread_id=approval.thread_id,
            turn_id=approval.turn_id,
            approval_ref=approval_ref,
        )
        return approval

    def _request_started(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Any:
        with self._state_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            pending = _PendingResponse()
            self._pending[request_id] = pending
        self._send(
            {"method": method, "id": request_id, "params": _json_dict(params)}
        )
        if not pending.event.wait(timeout=max(0.01, timeout)):
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise CodexAppServerTimeout(
                f"Codex App Server request timed out: {method}"
            )
        if pending.error is not None:
            code_value = pending.error.get("code")
            code = int(code_value) if isinstance(code_value, int) else None
            raise CodexAppServerProtocolError(
                str(
                    pending.error.get("message")
                    or f"Codex App Server request failed: {method}"
                ),
                code=code,
                data=pending.error.get("data"),
            )
        return pending.result

    def _send(self, payload: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            raise CodexAppServerUnavailable(
                "Codex App Server process is not running"
            )
        line = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            try:
                process.stdin.write(line + "\n")
            except TypeError:
                process.stdin.write((line + "\n").encode("utf-8"))
            process.stdin.flush()

    def _read_stdout(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            while True:
                raw = process.stdout.readline()
                if raw in {b"", ""}:
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                line = str(raw).strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._append_stderr("non-JSON stdout: " + line[-1000:])
                    continue
                if isinstance(message, Mapping):
                    self._dispatch_message(message)
        finally:
            returncode = process.poll()
            detail = (
                f"Codex App Server exited: returncode={returncode}; "
                + self.stderr_tail
            )
            self._fail_pending(detail)
            self._expire_approvals(detail)
            with self._state_lock:
                runs = list(self._active_by_thread.values())
            for run in runs:
                if not run.done.is_set():
                    run.status = "failed"
                    run.error = "Codex App Server process exited"
                    run.done.set()

    def _read_stderr(self) -> None:
        process = self._process
        if process is None:
            return
        while True:
            raw = process.stderr.readline()
            if raw in {b"", ""}:
                return
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            self._append_stderr(str(raw).rstrip())

    def _append_stderr(self, line: str) -> None:
        if not line:
            return
        with self._state_lock:
            self._stderr_tail.append(line)
            if len(self._stderr_tail) > 200:
                del self._stderr_tail[:-200]

    def _dispatch_message(self, message: Mapping[str, Any]) -> None:
        if "method" in message and "id" in message:
            self._handle_server_request(message)
            return
        if "method" in message:
            self._handle_notification(message)
            return
        if "id" not in message:
            return
        request_id = message.get("id")
        with self._state_lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        if isinstance(message.get("error"), Mapping):
            pending.error = dict(message["error"])
        else:
            pending.result = message.get("result")
        pending.event.set()

    def _handle_server_request(self, message: Mapping[str, Any]) -> None:
        method = str(message.get("method") or "")
        request_id = message.get("id")
        raw_params = message.get("params")
        params = dict(raw_params) if isinstance(raw_params, Mapping) else {}
        if method not in APPROVAL_METHODS:
            self._send(
                {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unsupported server request: {method}",
                    },
                }
            )
            self._emit_event(
                method="server/requestUnsupported",
                params={"requestMethod": method, "params": params},
                session_id=self._session_for_params(params),
                thread_id=_optional_text(params.get("threadId")),
                turn_id=_optional_text(params.get("turnId")),
            )
            return

        thread_id = str(params.get("threadId") or "")
        turn_id = str(params.get("turnId") or "")
        item_id = str(params.get("itemId") or "")
        approval_ref = _approval_ref(method, request_id, params)
        approval = PendingCodexApproval(
            approval_ref=approval_ref,
            request_id=request_id,
            method=method,
            session_id=self._session_for_params(params),
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=item_id,
            params=params,
        )
        with self._state_lock:
            self._approvals[approval_ref] = approval
        self._emit_event(
            method=method,
            params=params,
            session_id=approval.session_id,
            thread_id=thread_id,
            turn_id=turn_id,
            approval_ref=approval_ref,
        )

    def _handle_notification(self, message: Mapping[str, Any]) -> None:
        method = str(message.get("method") or "")
        raw_params = message.get("params")
        params = dict(raw_params) if isinstance(raw_params, Mapping) else {}
        thread_id = _optional_text(params.get("threadId"))
        if method == "thread/started":
            thread = params.get("thread")
            if isinstance(thread, Mapping):
                thread_id = _optional_text(thread.get("id"))
                parent_id = _optional_text(thread.get("parentThreadId"))
                if thread_id and parent_id:
                    with self._state_lock:
                        parent_session = self._thread_sessions.get(parent_id)
                        if parent_session:
                            self._thread_sessions[thread_id] = parent_session
        turn_id = _notification_turn_id(params)
        session_id = self._session_for_thread(thread_id)
        self._learn_subagent_threads(params, session_id)

        if method == "serverRequest/resolved":
            self._resolve_server_request_notification(params)

        if thread_id and turn_id:
            with self._state_lock:
                run = self._runs.get((thread_id, turn_id))
                if run is None and thread_id in self._owned_threads:
                    self._notification_backlog.setdefault(
                        (thread_id, turn_id), []
                    ).append((method, params))
            if run is not None:
                self._apply_turn_notification(method, params, run)

        self._emit_event(
            method=method,
            params=params,
            session_id=session_id,
            thread_id=thread_id,
            turn_id=turn_id,
        )

    def _apply_turn_notification(
        self,
        method: str,
        params: Mapping[str, Any],
        run: _TurnRun,
    ) -> None:
        if method == "item/agentMessage/delta":
            delta = params.get("delta")
            if isinstance(delta, str):
                run.deltas.append(delta)
            return
        if method == "item/completed":
            item = params.get("item")
            if isinstance(item, Mapping) and item.get("type") == "agentMessage":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    run.final_messages.append(text)
            return
        if method != "turn/completed":
            return

        turn = params.get("turn")
        if isinstance(turn, Mapping):
            run.status = str(turn.get("status") or "failed")
            items = turn.get("items")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, Mapping) or item.get("type") != "agentMessage":
                        continue
                    text = item.get("text")
                    if (
                        isinstance(text, str)
                        and text.strip()
                        and text not in run.final_messages
                    ):
                        run.final_messages.append(text)
            error = turn.get("error")
            if isinstance(error, Mapping):
                run.error = str(
                    error.get("message")
                    or error.get("additionalDetails")
                    or error
                )
            elif error:
                run.error = str(error)
        else:
            run.status = "failed"
            run.error = "turn/completed did not contain a turn object"

        self._expire_approvals_for_turn(
            run.thread_id,
            run.turn_id,
            "turn completed before approval was resolved",
        )
        run.done.set()
        with self._state_lock:
            current = self._active_by_thread.get(run.thread_id)
            if current is run:
                self._active_by_thread.pop(run.thread_id, None)
            self._runs.pop((run.thread_id, run.turn_id), None)

    def _resolve_server_request_notification(
        self,
        params: Mapping[str, Any],
    ) -> None:
        request_id = params.get("requestId")
        if request_id is None:
            return
        with self._state_lock:
            refs = [
                ref
                for ref, approval in self._approvals.items()
                if str(approval.request_id) == str(request_id)
            ]
            for ref in refs:
                self._approvals.pop(ref, None)

    def _session_for_params(self, params: Mapping[str, Any]) -> str | None:
        return self._session_for_thread(_optional_text(params.get("threadId")))

    def _session_for_thread(self, thread_id: str | None) -> str | None:
        if not thread_id:
            return None
        with self._state_lock:
            return self._thread_sessions.get(thread_id)

    def _learn_subagent_threads(
        self,
        params: Mapping[str, Any],
        session_id: str | None,
    ) -> None:
        if not session_id:
            return
        item = params.get("item")
        if not isinstance(item, Mapping):
            return
        children: list[str] = []
        if item.get("type") == "subAgentActivity":
            child = _optional_text(item.get("agentThreadId"))
            if child:
                children.append(child)
        elif item.get("type") == "collabAgentToolCall":
            raw_children = item.get("receiverThreadIds")
            if isinstance(raw_children, list):
                children.extend(
                    str(child).strip()
                    for child in raw_children
                    if str(child).strip()
                )
        if children:
            with self._state_lock:
                for child in children:
                    self._thread_sessions.setdefault(child, session_id)

    def _emit_event(
        self,
        *,
        method: str,
        params: Mapping[str, Any],
        session_id: str | None,
        thread_id: str | None,
        turn_id: str | None,
        approval_ref: str | None = None,
    ) -> None:
        event = CodexRuntimeEvent(
            event_id=f"CODEX-{uuid.uuid4().hex}",
            method=method,
            session_id=session_id,
            thread_id=thread_id,
            turn_id=turn_id,
            params=_json_dict(params),
            approval_ref=approval_ref,
        )
        with self._state_lock:
            listeners = list(self._listeners.values())
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                continue

    def _expire_approvals_for_turn(
        self,
        thread_id: str,
        turn_id: str,
        reason: str,
    ) -> None:
        with self._state_lock:
            approvals = [
                item
                for item in self._approvals.values()
                if item.thread_id == thread_id and item.turn_id == turn_id
            ]
            for item in approvals:
                self._approvals.pop(item.approval_ref, None)
        for approval in approvals:
            self._emit_event(
                method="approval/expired",
                params={
                    "approvalRef": approval.approval_ref,
                    "requestMethod": approval.method,
                    "reason": reason,
                },
                session_id=approval.session_id,
                thread_id=approval.thread_id,
                turn_id=approval.turn_id,
                approval_ref=approval.approval_ref,
            )

    def _expire_approvals(self, reason: str) -> None:
        with self._state_lock:
            approvals = list(self._approvals.values())
            self._approvals.clear()
        for approval in approvals:
            self._emit_event(
                method="approval/expired",
                params={
                    "approvalRef": approval.approval_ref,
                    "requestMethod": approval.method,
                    "reason": reason,
                },
                session_id=approval.session_id,
                thread_id=approval.thread_id,
                turn_id=approval.turn_id,
                approval_ref=approval.approval_ref,
            )

    def _fail_pending(self, message: str) -> None:
        with self._state_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.error = {"code": -32000, "message": message}
            item.event.set()

    def _terminate_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


class CodexAppServerRuntime:
    """Thread/turn facade used by Discord and the existing Harness providers."""

    def __init__(
        self,
        settings: CodexAppServerSettings,
        *,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self.settings = settings
        self.client = CodexAppServerClient(
            settings,
            process_factory=process_factory,
        )
        self.bindings = CodexThreadBindingStore(settings.state_dir)
        self._binding_locks: dict[str, threading.RLock] = {}
        self._binding_locks_guard = threading.RLock()

    def start(self) -> None:
        self.client.start()

    def stop(self) -> None:
        self.client.stop()

    def add_listener(self, listener: RuntimeListener) -> str:
        return self.client.add_listener(listener)

    def remove_listener(self, token: str) -> None:
        self.client.remove_listener(token)

    def run_turn(
        self,
        *,
        session_id: str,
        role: str,
        stage: str,
        task_id: str | None,
        prompt: str,
        cwd: str | Path,
        sandbox: str,
        cancellation: Any | None = None,
        client_user_message_id: str | None = None,
    ) -> CodexTurnResult:
        workspace = Path(cwd).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        binding_key = _binding_key(
            session_id=session_id,
            role=role,
            stage=stage,
            task_id=task_id,
            cwd=workspace,
        )
        lock = self._lock_for_binding(binding_key)
        with lock:
            binding = self._ensure_thread(
                binding_key=binding_key,
                session_id=session_id,
                role=role,
                task_id=task_id,
                cwd=workspace,
                sandbox=sandbox,
            )
            active = self.client.active_for_thread(binding.thread_id)
            if active is not None:
                raise CodexAppServerBusy(
                    f"Codex thread already has active turn {active.turn_id}"
                )
            params: dict[str, Any] = {
                "threadId": binding.thread_id,
                "clientUserMessageId": (
                    client_user_message_id
                    or f"researchagent:{session_id}:{uuid.uuid4().hex}"
                ),
                "input": [_text_input(prompt)],
                "cwd": str(workspace),
                "approvalPolicy": self.settings.approval_policy,
                "approvalsReviewer": self.settings.approvals_reviewer,
                "sandboxPolicy": _sandbox_policy(
                    sandbox,
                    workspace,
                    network_access=self.settings.network_access,
                ),
            }
            if self.settings.model:
                params["model"] = self.settings.model
            if self.settings.reasoning_effort:
                params["effort"] = self.settings.reasoning_effort
            if self.settings.reasoning_summary:
                params["summary"] = self.settings.reasoning_summary
            if self.settings.personality:
                params["personality"] = self.settings.personality

            response = self.client.request("turn/start", params)
            if not isinstance(response, Mapping) or not isinstance(
                response.get("turn"), Mapping
            ):
                raise CodexAppServerProtocolError(
                    "turn/start response did not contain a turn object"
                )
            turn_id = str(response["turn"].get("id") or "")
            if not turn_id:
                raise CodexAppServerProtocolError(
                    "turn/start response did not contain turn.id"
                )
            run = _TurnRun(
                session_id=session_id,
                binding_key=binding_key,
                thread_id=binding.thread_id,
                turn_id=turn_id,
                started_monotonic=time.monotonic(),
            )
            self.client.register_turn(run)

        interrupted = False
        timed_out = False
        deadline = (
            run.started_monotonic + self.settings.turn_timeout_seconds
            if self.settings.turn_timeout_seconds is not None
            else None
        )
        while not run.done.wait(0.2):
            if cancellation is not None and bool(
                getattr(cancellation, "cancelled", False)
            ):
                if not interrupted:
                    self._interrupt_run(run)
                    interrupted = True
            if deadline is not None and time.monotonic() >= deadline:
                if not interrupted:
                    self._interrupt_run(run)
                    interrupted = True
                timed_out = True
                if not run.done.wait(self.settings.request_timeout_seconds):
                    run.status = "failed"
                    run.error = "turn did not complete after timeout interrupt"
                    run.done.set()
                break
            if not self.client.running:
                run.status = "failed"
                run.error = "Codex App Server process stopped"
                run.done.set()
                break

        output = (
            run.final_messages[-1].strip()
            if run.final_messages
            else "".join(run.deltas).strip()
        )
        status = run.status if run.status in TERMINAL_TURN_STATUSES else "failed"
        return CodexTurnResult(
            session_id=session_id,
            binding_key=binding_key,
            thread_id=binding.thread_id,
            turn_id=turn_id,
            status=status,
            output=output,
            error=run.error,
            duration_seconds=max(0.0, time.monotonic() - run.started_monotonic),
            cancelled=status == "interrupted" or interrupted,
            timed_out=timed_out,
        )

    def steer(
        self,
        *,
        session_id: str,
        text: str,
        client_user_message_id: str | None = None,
    ) -> dict[str, Any]:
        run = self._discord_run(session_id)
        if run is None:
            raise CodexAppServerBusy(
                "this WorkSession has no active Discord Codex turn"
            )
        params: dict[str, Any] = {
            "threadId": run.thread_id,
            "input": [_text_input(text)],
            "expectedTurnId": run.turn_id,
        }
        if client_user_message_id:
            params["clientUserMessageId"] = client_user_message_id
        response = self.client.request("turn/steer", params)
        return {
            "thread_id": run.thread_id,
            "turn_id": run.turn_id,
            "response": response,
        }

    def interrupt(self, *, session_id: str) -> dict[str, Any]:
        run = self._discord_run(session_id)
        if run is None:
            raise CodexAppServerBusy(
                "this WorkSession has no active Discord Codex turn"
            )
        response = self.client.request(
            "turn/interrupt",
            {"threadId": run.thread_id, "turnId": run.turn_id},
        )
        return {
            "thread_id": run.thread_id,
            "turn_id": run.turn_id,
            "response": response,
        }

    def resolve_approval(
        self,
        *,
        session_id: str,
        approval_ref: str,
        decision: str,
    ) -> PendingCodexApproval:
        return self.client.resolve_approval(
            approval_ref,
            decision,
            session_id=session_id,
        )

    def pending_approvals(
        self,
        *,
        session_id: str,
    ) -> tuple[PendingCodexApproval, ...]:
        return self.client.pending_approvals(session_id=session_id)

    def status(self, *, session_id: str) -> dict[str, Any]:
        active = self.client.active_for_session(session_id)
        return {
            "running": self.client.running,
            "threads": [
                item.to_dict() for item in self.bindings.list_for_session(session_id)
            ],
            "active_turn": (
                {
                    "thread_id": active.thread_id,
                    "turn_id": active.turn_id,
                    "status": active.status,
                }
                if active is not None
                else None
            ),
            "pending_approvals": [
                item.to_dict()
                for item in self.pending_approvals(session_id=session_id)
            ],
            "stderr_tail": self.client.stderr_tail[-2000:],
        }

    def _interrupt_run(self, run: _TurnRun) -> None:
        self.client.request(
            "turn/interrupt",
            {"threadId": run.thread_id, "turnId": run.turn_id},
        )

    def _discord_run(self, session_id: str) -> _TurnRun | None:
        return self.client.active_for_binding(f"discord:{session_id}")

    def _ensure_thread(
        self,
        *,
        binding_key: str,
        session_id: str,
        role: str,
        task_id: str | None,
        cwd: Path,
        sandbox: str,
    ) -> CodexThreadBinding:
        binding = self.bindings.get(binding_key)
        if binding is not None:
            try:
                response = self.client.request(
                    "thread/resume",
                    {
                        "threadId": binding.thread_id,
                        "cwd": str(cwd),
                        "approvalPolicy": self.settings.approval_policy,
                        "approvalsReviewer": self.settings.approvals_reviewer,
                        "sandbox": _sandbox_mode(sandbox),
                        "excludeTurns": True,
                    },
                )
                thread_id = _thread_id_from_response(response)
                if thread_id != binding.thread_id:
                    raise CodexAppServerProtocolError(
                        "thread/resume returned a different thread id"
                    )
                binding = self.bindings.put(replace(binding, cwd=str(cwd)))
                self.client.bind_thread(thread_id, session_id)
                return binding
            except CodexAppServerProtocolError as exc:
                if not _thread_missing_error(exc):
                    raise

        params: dict[str, Any] = {
            "cwd": str(cwd),
            "approvalPolicy": self.settings.approval_policy,
            "approvalsReviewer": self.settings.approvals_reviewer,
            "sandbox": _sandbox_mode(sandbox),
            "ephemeral": False,
        }
        if self.settings.model:
            params["model"] = self.settings.model
        if self.settings.personality:
            params["personality"] = self.settings.personality
        response = self.client.request("thread/start", params)
        thread_id = _thread_id_from_response(response)
        binding = self.bindings.put(
            CodexThreadBinding(
                binding_key=binding_key,
                session_id=session_id,
                thread_id=thread_id,
                cwd=str(cwd),
                role=role,
                task_id=task_id,
            )
        )
        self.client.bind_thread(thread_id, session_id)
        return binding

    def _lock_for_binding(self, binding_key: str) -> threading.RLock:
        with self._binding_locks_guard:
            return self._binding_locks.setdefault(binding_key, threading.RLock())


class CodexAppServerAgentExecutor:
    """Adapter from the existing Harness executor contract to App Server turns."""

    def __init__(
        self,
        config: Any,
        lock: threading.RLock,
        cancellation: Any,
        invocation_cls: type,
        *,
        runtime: CodexAppServerRuntime | None = None,
    ) -> None:
        self.config = config
        self.lock = lock
        self.cancellation = cancellation
        self.invocation_cls = invocation_cls
        self.runtime = runtime or get_shared_codex_app_server(config)

    def run(
        self,
        *,
        session: Any,
        role: str,
        stage: str,
        prompt: str,
        command_text: str | None,
        sandbox: str,
        task_id: str | None = None,
        working_dir: Path | None = None,
    ) -> Any:
        workspace = Path(
            working_dir
            or getattr(session, "research_dir", "")
            or self.config.project_root
        ).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        input_tokens = estimate_tokens(prompt)
        if bool(getattr(self.cancellation, "cancelled", False)):
            return self.invocation_cls(
                role=role,
                stage=stage,
                task_id=task_id,
                command=("provider:codex_app_server",),
                output="Codex turn cancelled before start.",
                stderr=str(getattr(self.cancellation, "reason", "cancelled")),
                returncode=130,
                duration_seconds=0.0,
                cancelled=True,
                workspace=str(workspace),
                estimated_input_tokens=input_tokens,
            )
        with self.lock:
            max_calls = int(getattr(self.config, "max_agent_calls", 0) or 0)
            if max_calls > 0 and session.cost.agent_calls >= max_calls:
                return self.invocation_cls(
                    role=role,
                    stage=stage,
                    task_id=task_id,
                    command=("provider:codex_app_server",),
                    output="Codex App Server skipped: MAX_AGENT_CALLS reached.",
                    stderr="agent call budget exhausted",
                    returncode=125,
                    duration_seconds=0.0,
                    skipped=True,
                    workspace=str(workspace),
                    estimated_input_tokens=input_tokens,
                )
            session.cost.agent_calls += 1

        started = time.monotonic()
        try:
            result = self.runtime.run_turn(
                session_id=str(session.session_id),
                role=role,
                stage=stage,
                task_id=task_id,
                prompt=prompt,
                cwd=workspace,
                sandbox=sandbox,
                cancellation=self.cancellation,
            )
        except CodexAppServerUnavailable as exc:
            return self.invocation_cls(
                role=role,
                stage=stage,
                task_id=task_id,
                command=("provider:codex_app_server",),
                output="Codex App Server is unavailable.",
                stderr=str(exc),
                returncode=127,
                duration_seconds=time.monotonic() - started,
                skipped=True,
                workspace=str(workspace),
                estimated_input_tokens=input_tokens,
            )
        except Exception as exc:
            return self.invocation_cls(
                role=role,
                stage=stage,
                task_id=task_id,
                command=("provider:codex_app_server",),
                output=f"Codex App Server failed: {type(exc).__name__}",
                stderr=str(exc),
                returncode=1,
                duration_seconds=time.monotonic() - started,
                workspace=str(workspace),
                estimated_input_tokens=input_tokens,
            )

        artifacts: tuple[ArtifactRecord, ...] = ()
        warnings: tuple[str, ...] = ()
        if sandbox == "workspace-write" and bool(
            getattr(self.config, "artifact_promotion_enabled", True)
        ):
            records, raw_warnings = build_artifact_manifest(
                workspace,
                max_files=int(getattr(self.config, "artifact_max_files", 500)),
                max_bytes=int(
                    getattr(self.config, "artifact_max_bytes", 100 * 1024 * 1024)
                ),
            )
            artifacts = tuple(records)
            warnings = tuple(raw_warnings)

        output_tokens = estimate_tokens(result.output)
        with self.lock:
            session.cost.estimated_tokens += input_tokens + output_tokens
        returncode = (
            0
            if result.status == "completed"
            else 130
            if result.status == "interrupted"
            else 1
        )
        return self.invocation_cls(
            role=role,
            stage=stage,
            task_id=task_id,
            command=(
                "provider:codex_app_server",
                f"thread_id:{result.thread_id}",
                f"turn_id:{result.turn_id}",
            ),
            output=result.output,
            stderr=result.error or "",
            returncode=returncode,
            duration_seconds=result.duration_seconds,
            timed_out=result.timed_out,
            cancelled=result.cancelled,
            workspace=str(workspace),
            artifacts=artifacts,
            artifact_warnings=warnings,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
        )


_SHARED_LOCK = threading.RLock()
_SHARED_RUNTIMES: dict[str, CodexAppServerRuntime] = {}


def get_shared_codex_app_server(config: Any) -> CodexAppServerRuntime:
    settings = CodexAppServerSettings.from_environment(config.project_root)
    key = str(settings.state_dir) + "\0" + "\0".join(settings.command)
    with _SHARED_LOCK:
        runtime = _SHARED_RUNTIMES.get(key)
        if runtime is None:
            runtime = CodexAppServerRuntime(settings)
            _SHARED_RUNTIMES[key] = runtime
        return runtime


def reset_shared_codex_app_servers() -> None:
    with _SHARED_LOCK:
        values = list(_SHARED_RUNTIMES.values())
        _SHARED_RUNTIMES.clear()
    for runtime in values:
        try:
            runtime.stop()
        except Exception:
            continue


atexit.register(reset_shared_codex_app_servers)


def format_codex_event_for_discord(event: CodexRuntimeEvent) -> str | None:
    params = event.params
    method = event.method
    if method == "thread/started":
        thread = params.get("thread")
        if isinstance(thread, Mapping) and thread.get("parentThreadId"):
            return (
                "Codex Harness subagent thread started: `"
                + str(thread.get("id") or "?")
                + "`"
            )
        return None
    if method == "turn/started":
        return f"Codex turn started: `{event.turn_id or '?'}`"
    if method == "turn/completed":
        turn = params.get("turn")
        status = turn.get("status") if isinstance(turn, Mapping) else "unknown"
        return (
            f"Codex turn completed: status=`{status}` "
            f"turn=`{event.turn_id or '?'}`"
        )
    if method in APPROVAL_METHODS:
        reason = str(params.get("reason") or "no reason supplied")
        command = str(params.get("command") or "")
        lines = [
            "Codex approval required",
            f"- approval: `{event.approval_ref}`",
            f"- reason: {reason[:1200]}",
        ]
        if command:
            lines.insert(2, f"- command: `{command[:1200]}`")
        lines.append(
            "Use `/agent codex_approval` with accept, acceptForSession, "
            "decline, or cancel."
        )
        return "\n".join(lines)[:1900]
    if method == "turn/plan/updated":
        plan = params.get("plan")
        lines = ["Codex plan updated"]
        if isinstance(plan, list):
            for item in plan[:12]:
                if isinstance(item, Mapping):
                    lines.append(
                        "- ["
                        + str(item.get("status") or "pending")
                        + "] "
                        + str(item.get("step") or "")[:500]
                    )
        return "\n".join(lines)[:1900]
    if method == "item/started":
        item = params.get("item")
        if not isinstance(item, Mapping):
            return None
        item_type = str(item.get("type") or "item")
        if item_type == "commandExecution":
            return (
                "Codex command started: `"
                + str(item.get("command") or "")[:1200]
                + "`"
            )
        if item_type in {"fileChange", "mcpToolCall", "collabAgentToolCall", "subAgentActivity"}:
            return f"Codex item started: `{item_type}`"
    if method == "item/completed":
        item = params.get("item")
        if isinstance(item, Mapping):
            return f"Codex item completed: `{item.get('type') or 'item'}`"
    if method == "error":
        return "Codex App Server error: " + str(
            params.get("message") or params.get("error") or params
        )[:1500]
    return None


def _binding_key(
    *,
    session_id: str,
    role: str,
    stage: str,
    task_id: str | None,
    cwd: Path,
) -> str:
    if stage.startswith("discord_"):
        return f"discord:{session_id}"
    digest = hashlib.sha256(str(cwd).encode("utf-8")).hexdigest()[:16]
    return f"harness:{session_id}:{role}:{task_id or '-'}:{digest}"


def _thread_id_from_response(response: Any) -> str:
    if not isinstance(response, Mapping) or not isinstance(
        response.get("thread"), Mapping
    ):
        raise CodexAppServerProtocolError(
            "thread response did not contain a thread object"
        )
    thread_id = str(response["thread"].get("id") or "")
    if not thread_id:
        raise CodexAppServerProtocolError(
            "thread response did not contain thread.id"
        )
    return thread_id


def _text_input(text: str) -> dict[str, Any]:
    # Official v2 UserInput is camelCase on the wire.
    return {"type": "text", "text": str(text), "textElements": []}


def _sandbox_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"read-only", "workspace-write", "danger-full-access"}:
        return normalized
    raise ValueError(f"unsupported Codex sandbox mode: {value}")


def _sandbox_policy(
    value: str,
    cwd: Path,
    *,
    network_access: bool,
) -> dict[str, Any]:
    mode = _sandbox_mode(value)
    if mode == "read-only":
        return {"type": "readOnly", "networkAccess": bool(network_access)}
    if mode == "workspace-write":
        return {
            "type": "workspaceWrite",
            "writableRoots": [str(cwd)],
            "networkAccess": bool(network_access),
            "excludeTmpdirEnvVar": False,
            "excludeSlashTmp": False,
        }
    return {"type": "dangerFullAccess"}


def _app_server_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "APPDATA",
        "LOCALAPPDATA",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "CODEX_HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "RUST_LOG",
        "LOG_FORMAT",
        "TMP",
        "TEMP",
        "TMPDIR",
    }
    environment = {
        key: value for key, value in os.environ.items() if key in allowed
    }
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def _default_process_factory(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
) -> ProcessLike:
    return subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=os.name != "nt",
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )


def _notification_turn_id(params: Mapping[str, Any]) -> str | None:
    direct = _optional_text(params.get("turnId"))
    if direct:
        return direct
    turn = params.get("turn")
    if isinstance(turn, Mapping):
        return _optional_text(turn.get("id"))
    return None


def _approval_ref(
    method: str,
    request_id: str | int,
    params: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256(
        (
            method
            + "\0"
            + str(request_id)
            + "\0"
            + str(params.get("threadId") or "")
            + "\0"
            + str(params.get("turnId") or "")
            + "\0"
            + str(params.get("itemId") or "")
            + "\0"
            + str(params.get("approvalId") or "")
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"CAP-{digest.upper()}"


def _normalize_approval_decision(value: str) -> str:
    normalized = str(value).strip()
    aliases = {
        "approve": "accept",
        "approved": "accept",
        "yes": "accept",
        "session": "acceptForSession",
        "accept_for_session": "acceptForSession",
        "reject": "decline",
        "deny": "decline",
        "no": "decline",
        "interrupt": "cancel",
    }
    normalized = aliases.get(normalized, aliases.get(normalized.lower(), normalized))
    if normalized not in APPROVAL_DECISIONS:
        raise ValueError(
            "decision must be accept, acceptForSession, decline, or cancel"
        )
    return normalized


def _thread_missing_error(error: CodexAppServerProtocolError) -> bool:
    text = str(error).lower()
    return "thread" in text and any(
        token in text for token in ("not found", "unknown", "does not exist")
    )


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    encoded = json.dumps(
        dict(value), ensure_ascii=False, allow_nan=False, default=str
    )
    decoded = json.loads(encoded)
    return dict(decoded) if isinstance(decoded, dict) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value) if value not in {None, ""} else default
    except (TypeError, ValueError):
        return default


def _optional_positive_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("turn timeout must be a positive number") from exc
    if parsed <= 0:
        raise ValueError("turn timeout must be a positive number")
    return parsed


def _bool_value(value: Any, default: bool) -> bool:
    if value in {None, ""}:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "APPROVAL_DECISIONS",
    "APPROVAL_METHODS",
    "CodexAppServerAgentExecutor",
    "CodexAppServerBusy",
    "CodexAppServerClient",
    "CodexAppServerError",
    "CodexAppServerProtocolError",
    "CodexAppServerRuntime",
    "CodexAppServerSettings",
    "CodexAppServerTimeout",
    "CodexAppServerUnavailable",
    "CodexRuntimeEvent",
    "CodexThreadBinding",
    "CodexThreadBindingStore",
    "CodexTurnResult",
    "PendingCodexApproval",
    "ProcessFactory",
    "ProcessLike",
    "RuntimeListener",
    "format_codex_event_for_discord",
    "get_shared_codex_app_server",
    "reset_shared_codex_app_servers",
]
