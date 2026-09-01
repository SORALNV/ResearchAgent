"""Compatibility facade for the official Codex App Server v2 runtime.

The protocol implementation lives in :mod:`harness.codex_app_server_v2`.  This
module preserves the repository's existing imports and adds the small amount of
coordination needed when a Discord steer arrives while ``turn/start`` is still
being acknowledged by App Server.
"""

from __future__ import annotations

import atexit
import os
import shlex
import threading
from pathlib import Path
from typing import Any, Mapping

from harness import codex_app_server_v2 as _impl
from harness.codex_app_server_v2 import *  # noqa: F401,F403


class CodexAppServerSettings(_impl.CodexAppServerSettings):
    """Settings with an explicit diagnostic for the removed ``codex exec`` path."""

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
        except ValueError:
            command = ()
        if (
            command
            and Path(command[0]).name.lower() in {"codex", "codex.exe"}
            and "exec" in command[1:]
        ):
            raise ValueError("codex exec is disabled; use codex app-server")
        base = _impl.CodexAppServerSettings.from_environment(
            project_root,
            source,
        )
        return cls(**base.__dict__)


class CodexAppServerRuntime(_impl.CodexAppServerRuntime):
    """Runtime that makes the ``turn/start``/``turn/steer`` boundary race-safe."""

    def __init__(
        self,
        settings: CodexAppServerSettings,
        *,
        process_factory: _impl.ProcessFactory | None = None,
    ) -> None:
        super().__init__(settings, process_factory=process_factory)
        self._starting_guard = threading.RLock()
        self._starting_by_binding: dict[str, threading.Event] = {}

        original_register = self.client.register_turn

        def register_and_signal(run: Any) -> None:
            original_register(run)
            with self._starting_guard:
                event = self._starting_by_binding.pop(run.binding_key, None)
            if event is not None:
                event.set()

        # The client is private to this runtime. Wrapping registration keeps the
        # official wire protocol untouched while allowing an immediately
        # following Discord message to wait for the returned turn ID and steer it.
        self.client.register_turn = register_and_signal  # type: ignore[method-assign]

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
    ) -> _impl.CodexTurnResult:
        workspace = Path(cwd).expanduser().resolve()
        binding_key = _impl._binding_key(
            session_id=session_id,
            role=role,
            stage=stage,
            task_id=task_id,
            cwd=workspace,
        )
        event = threading.Event()
        with self._starting_guard:
            existing = self._starting_by_binding.get(binding_key)
            owner = existing is None
            if owner:
                self._starting_by_binding[binding_key] = event
            else:
                event = existing
        try:
            return super().run_turn(
                session_id=session_id,
                role=role,
                stage=stage,
                task_id=task_id,
                prompt=prompt,
                cwd=workspace,
                sandbox=sandbox,
                cancellation=cancellation,
                client_user_message_id=client_user_message_id,
            )
        finally:
            if owner:
                with self._starting_guard:
                    current = self._starting_by_binding.get(binding_key)
                    if current is event:
                        self._starting_by_binding.pop(binding_key, None)
                event.set()

    def _discord_run(self, session_id: str) -> Any | None:
        active = super()._discord_run(session_id)
        if active is not None:
            return active
        binding_key = f"discord:{session_id}"
        with self._starting_guard:
            event = self._starting_by_binding.get(binding_key)
        if event is not None:
            event.wait(
                timeout=min(
                    max(float(self.settings.request_timeout_seconds), 0.1),
                    2.0,
                )
            )
        return super()._discord_run(session_id)


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


class CodexAppServerAgentExecutor(_impl.CodexAppServerAgentExecutor):
    """Existing Harness executor contract backed by the shared race-safe runtime."""

    def __init__(
        self,
        config: Any,
        lock: threading.RLock,
        cancellation: Any,
        invocation_cls: type,
        *,
        runtime: CodexAppServerRuntime | None = None,
    ) -> None:
        super().__init__(
            config,
            lock,
            cancellation,
            invocation_cls,
            runtime=runtime or get_shared_codex_app_server(config),
        )


atexit.register(reset_shared_codex_app_servers)

__all__ = list(_impl.__all__)
