from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from harness.compute import BackendCapabilities, BackendRunResult, EventSink
from harness.control_plane import JobSpec


class LocalProcessBackend:
    """Run bounded CPU smoke/analysis jobs without a shell or inherited secrets."""

    SAFE_ENV = {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PYTHONPATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SYSTEMROOT",
        "WINDIR",
        "TMP",
        "TEMP",
        "TMPDIR",
    }

    def __init__(
        self,
        *,
        name: str = "local_cpu",
        workspace_root: str | Path,
        max_ram_gb: float = 0.0,
        environment: Mapping[str, str] | None = None,
        cancel_grace_seconds: float = 3.0,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        source = dict(os.environ if environment is None else environment)
        self.environment = {
            key: value for key, value in source.items() if key in self.SAFE_ENV and value
        }
        self.cancel_grace_seconds = max(0.1, float(cancel_grace_seconds))
        self._capabilities = BackendCapabilities(
            name=name,
            domains=("research", "kaggle"),
            accelerators=("cpu",),
            max_vram_gb=0.0,
            max_ram_gb=max(0.0, float(max_ram_gb)),
            supports_cancel=True,
            supports_live_events=True,
            supports_network=False,
            metadata={"local": True, "shell": False},
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def run(
        self,
        spec: JobSpec,
        *,
        emit: EventSink,
        cancel_event: threading.Event,
    ) -> BackendRunResult:
        try:
            workspace, command = self._prepare(spec)
        except Exception as exc:
            return BackendRunResult.failed(f"local job preparation failed: {exc}")
        backend_job_id = f"LOCAL-{spec.job_id}"
        emit(
            "backend_started",
            {
                "backend_job_id": backend_job_id,
                "workspace": str(workspace),
                "command": _redact(command),
            },
        )
        process: subprocess.Popen[str] | None = None
        started = time.monotonic()
        timeout = spec.max_runtime_seconds or int(
            spec.payload.get("timeout_seconds") or 900
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=workspace,
                env={
                    **self.environment,
                    "PYTHONUNBUFFERED": "1",
                    "RESEARCH_AGENT_JOB_ID": spec.job_id,
                    "RESEARCH_AGENT_WORKSPACE": str(workspace),
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=os.name != "nt",
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            stdout_thread = threading.Thread(
                target=_drain,
                args=(process.stdout, stdout_lines, emit, "stdout"),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_drain,
                args=(process.stderr, stderr_lines, emit, "stderr"),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            while process.poll() is None:
                if cancel_event.is_set():
                    _terminate(process, self.cancel_grace_seconds)
                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)
                    return BackendRunResult.cancelled(
                        "local job cancelled",
                        result={
                            "workspace": str(workspace),
                            "stdout": "".join(stdout_lines)[-20000:],
                            "stderr": "".join(stderr_lines)[-20000:],
                        },
                        backend_job_id=backend_job_id,
                    )
                if time.monotonic() - started > timeout:
                    _terminate(process, self.cancel_grace_seconds)
                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)
                    return BackendRunResult.failed(
                        f"local job timed out after {timeout}s",
                        result={
                            "workspace": str(workspace),
                            "stdout": "".join(stdout_lines)[-20000:],
                            "stderr": "".join(stderr_lines)[-20000:],
                        },
                        backend_job_id=backend_job_id,
                    )
                cancel_event.wait(0.1)
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            result = self._collect_result(spec, workspace, stdout, stderr)
            if process.returncode != 0:
                return BackendRunResult.failed(
                    f"local job returned {process.returncode}",
                    result=result,
                    backend_job_id=backend_job_id,
                )
            return BackendRunResult.completed(
                result,
                backend_job_id=backend_job_id,
            )
        except OSError as exc:
            return BackendRunResult.failed(
                f"local process failed to start: {exc}",
                backend_job_id=backend_job_id,
            )
        finally:
            if process is not None and process.poll() is None:
                _terminate(process, self.cancel_grace_seconds)

    def _prepare(self, spec: JobSpec) -> tuple[Path, list[str]]:
        source_raw = spec.payload.get("source_dir")
        if not source_raw:
            raise ValueError("payload.source_dir is required")
        source = Path(str(source_raw)).expanduser().resolve()
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"unsafe source_dir: {source}")
        entrypoint = _safe_relative(str(spec.payload.get("entrypoint") or "run.py"))
        workspace = self.workspace_root / spec.job_id / f"attempt-{int(spec.payload.get('attempt') or 1):02d}"
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(
            source,
            workspace,
            symlinks=False,
            ignore=shutil.ignore_patterns(
                ".git",
                ".env",
                ".venv",
                "__pycache__",
                "kaggle.json",
                "credentials.json",
                "runtime",
                "research_runs",
            ),
        )
        script = (workspace / entrypoint).resolve()
        try:
            script.relative_to(workspace.resolve())
        except ValueError as exc:
            raise ValueError("entrypoint escapes workspace") from exc
        if not script.is_file() or script.is_symlink():
            raise ValueError(f"entrypoint not found: {entrypoint}")
        arguments = spec.payload.get("args") or []
        if not isinstance(arguments, list):
            raise ValueError("payload.args must be a list")
        if script.suffix.lower() == ".py":
            command = [sys.executable, str(script), *[str(item) for item in arguments]]
        else:
            executable = spec.payload.get("executable")
            if not executable:
                raise ValueError("non-Python entrypoint requires payload.executable")
            parts = shlex.split(str(executable))
            if not parts:
                raise ValueError("empty executable")
            command = [*parts, str(script), *[str(item) for item in arguments]]
        return workspace, command

    def _collect_result(
        self,
        spec: JobSpec,
        workspace: Path,
        stdout: str,
        stderr: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "workspace": str(workspace),
            "stdout": stdout[-20000:],
            "stderr": stderr[-20000:],
            "files": [
                path.relative_to(workspace).as_posix()
                for path in sorted(workspace.rglob("*"))
                if path.is_file() and not path.is_symlink()
            ],
        }
        missing: list[str] = []
        for relative in spec.outputs:
            path = workspace / _safe_relative(relative)
            if not path.exists():
                missing.append(relative)
        result["missing_outputs"] = missing
        for name in ("result.json", "metrics.json"):
            path = workspace / name
            if not path.is_file():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
            else:
                result[name.removesuffix(".json")] = value
        if missing:
            result["collection_error"] = "missing required outputs"
        return result


def _drain(stream, lines: list[str], emit: EventSink, name: str) -> None:
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        lines.append(line)
        stripped = line.strip()
        if not stripped:
            continue
        event = _structured_event(stripped)
        if event is not None:
            emit(str(event.pop("event_type", "worker_event")), event)
        elif name == "stderr":
            emit("worker_stderr", {"message": stripped[-1000:]})
    stream.close()


def _structured_event(line: str) -> dict[str, Any] | None:
    if not line.startswith("{") or not line.endswith("}"):
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if "event_type" not in value and "event" in value:
        value["event_type"] = value.pop("event")
    return value if value.get("event_type") else None


def _terminate(process: subprocess.Popen[str], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
    except OSError:
        try:
            process.terminate()
        except OSError:
            return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass


def _safe_relative(value: str) -> str:
    path = Path(value.strip())
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def _redact(command: list[str]) -> list[str]:
    result: list[str] = []
    secret_next = False
    for part in command:
        lower = part.lower()
        if secret_next:
            result.append("***")
            secret_next = False
        elif any(fragment in lower for fragment in ("token=", "password=", "api_key=")):
            result.append(part.split("=", 1)[0] + "=***")
        else:
            result.append(part)
            secret_next = lower in {"--token", "--password", "--api-key"}
    return result
