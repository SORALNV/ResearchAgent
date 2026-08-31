from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Mapping

from harness.config import HarnessConfig
from harness.state import ResearchSession


class AgentCancelledError(RuntimeError):
    pass


_SAFE_BASE_ENV = {
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SYSTEMROOT",
    "WINDIR",
    # Codex authentication/config can be mounted independently of the
    # invocation-specific HOME. The value is a path, not a credential.
    "CODEX_HOME",
}

_HARD_DENY_PREFIXES = ("DISCORD_",)
_HARD_DENY_EXACT = {
    "DISCORD_BOT_TOKEN",
}


class ProcessCancellationController:
    """Track agent process groups and terminate them from a control-plane thread."""

    def __init__(self, grace_seconds: float = 3.0) -> None:
        self.grace_seconds = max(0.1, grace_seconds)
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._reason = ""
        self._active: dict[int, subprocess.Popen[str]] = {}

    def reset(self) -> None:
        with self._lock:
            if any(process.poll() is None for process in self._active.values()):
                raise RuntimeError("cannot reset cancellation while agent processes are active")
            self._active.clear()
            self._reason = ""
            self._cancelled.clear()

    def register(self, process: subprocess.Popen[str]) -> bool:
        with self._lock:
            if self._cancelled.is_set():
                return False
            self._active[process.pid] = process
            return True

    def unregister(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._active.pop(process.pid, None)

    def cancel(self, reason: str = "cancel requested") -> int:
        with self._lock:
            self._reason = reason
            self._cancelled.set()
            processes = [process for process in self._active.values() if process.poll() is None]
        target_count = len(processes)
        for process in processes:
            _signal_terminate(process)
        deadline = time.monotonic() + self.grace_seconds
        while processes and time.monotonic() < deadline:
            processes = [process for process in processes if process.poll() is None]
            if processes:
                time.sleep(0.05)
        for process in processes:
            _signal_kill(process)
        return target_count

    def terminate_one(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        _signal_terminate(process)
        try:
            process.wait(timeout=self.grace_seconds)
        except subprocess.TimeoutExpired:
            _signal_kill(process)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def reason(self) -> str:
        return self._reason or "cancel requested"

    def active_processes(self) -> tuple[subprocess.Popen[str], ...]:
        with self._lock:
            return tuple(self._active.values())

    @property
    def active_count(self) -> int:
        return sum(1 for process in self.active_processes() if process.poll() is None)


def build_agent_environment(
    config: HarnessConfig,
    session: ResearchSession,
    *,
    role: str,
    stage: str,
    task_id: str | None,
    working_dir: Path,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an allowlisted environment instead of inheriting the Bot process environment."""

    parent = dict(source or os.environ)
    explicit = set(config.agent_env_allowlist)
    allowed = set(_SAFE_BASE_ENV) | explicit
    environment: dict[str, str] = {}

    for key in sorted(allowed):
        if _hard_denied(key):
            continue
        if key not in parent:
            continue
        if _looks_sensitive(key) and key not in explicit:
            continue
        environment[key] = parent[key]

    if config.agent_home_mode == "preserve":
        for key in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
            if key in parent and not _hard_denied(key):
                environment[key] = parent[key]
    else:
        research_root = Path(session.research_dir or config.project_root)
        invocation_name = "-".join(
            part for part in (role, stage, task_id or "global") if part
        )
        isolated_home = (
            research_root
            / "artifacts"
            / "agent_home"
            / _safe_component(invocation_name)
        )
        isolated_home.mkdir(parents=True, exist_ok=True)
        environment["HOME"] = str(isolated_home)
        environment["USERPROFILE"] = str(isolated_home)
        environment["XDG_CONFIG_HOME"] = str(isolated_home / ".config")
        environment["XDG_CACHE_HOME"] = str(isolated_home / ".cache")
        environment["XDG_DATA_HOME"] = str(isolated_home / ".local" / "share")
        for directory in (
            Path(environment["XDG_CONFIG_HOME"]),
            Path(environment["XDG_CACHE_HOME"]),
            Path(environment["XDG_DATA_HOME"]),
        ):
            directory.mkdir(parents=True, exist_ok=True)

    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RESEARCH_AGENT_SESSION_ID": session.session_id,
            "RESEARCH_AGENT_ROLE": role,
            "RESEARCH_AGENT_STAGE": stage,
            "RESEARCH_AGENT_TASK_ID": task_id or "",
            "RESEARCH_AGENT_WORKSPACE": str(working_dir),
        }
    )
    return environment


def _hard_denied(name: str) -> bool:
    upper = name.upper()
    return upper in _HARD_DENY_EXACT or any(
        upper.startswith(prefix) for prefix in _HARD_DENY_PREFIXES
    )


def _looks_sensitive(name: str) -> bool:
    upper = name.upper()
    return any(fragment in upper for fragment in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "APIKEY"))


def _signal_terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
        except OSError:
            pass


def _signal_kill(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except OSError:
            pass


def _safe_component(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:96] or "agent"
