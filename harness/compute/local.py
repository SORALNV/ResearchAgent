from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.compute.base import (
    BackendCapabilities,
    BackendStatus,
    ComputeHandle,
)
from harness.platform.models import Domain, JobRecord


@dataclass
class _LocalProcess:
    process: subprocess.Popen[bytes]
    stdout_path: Path
    stderr_path: Path
    workspace: Path
    stdout_handle: Any
    stderr_handle: Any


class LocalProcessBackend:
    """Small CPU backend for smoke tests, preprocessing, and validation.

    The Control Plane is not a GPU worker. This backend advertises CPU only and
    rejects jobs that request a GPU. It executes argv directly without a shell.
    """

    name = "local_process"

    def __init__(
        self,
        *,
        max_cpu_cores: int | None = None,
        max_ram_gb: float | None = None,
    ) -> None:
        self.capabilities = BackendCapabilities(
            accelerators=("cpu",),
            max_cpu_cores=max_cpu_cores or max(1, os.cpu_count() or 1),
            max_ram_gb=max_ram_gb,
            detailed_progress=True,
            supports_cancel=True,
            supports_kaggle_data=False,
            domains=(Domain.RESEARCH, Domain.KAGGLE),
            tags=("smoke_test", "preprocessing", "validation", "inference"),
        )
        self._lock = threading.RLock()
        self._processes: dict[str, _LocalProcess] = {}

    def available(self) -> tuple[bool, str]:
        return True, f"local CPU cores={self.capabilities.max_cpu_cores}"

    def prepare(self, job: JobRecord, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        if not job.spec.entrypoint.strip():
            raise ValueError("local job entrypoint must not be empty")
        command = shlex.split(job.spec.entrypoint)
        if not command:
            raise ValueError("local job entrypoint is invalid")
        if command[0].startswith("."):
            executable = (workspace / command[0]).resolve()
            try:
                executable.relative_to(workspace.resolve())
            except ValueError as exc:
                raise PermissionError("local entrypoint escapes workspace") from exc
        manifest = {
            "job_id": job.spec.job_id,
            "command": command,
            "workspace": str(workspace.resolve()),
            "resources": job.spec.resources.to_dict(),
        }
        (workspace / "local_job.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    def submit(self, job: JobRecord, workspace: Path) -> ComputeHandle:
        manifest = self.prepare(job, workspace)
        command = list(manifest["command"])
        stdout_path = workspace / "stdout.log"
        stderr_path = workspace / "stderr.log"
        stdout_handle = stdout_path.open("ab", buffering=0)
        stderr_handle = stderr_path.open("ab", buffering=0)
        environment = self._environment(job, workspace)
        try:
            process = subprocess.Popen(
                command,
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=os.name != "nt",
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            raise
        with self._lock:
            self._processes[job.spec.job_id] = _LocalProcess(
                process=process,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                workspace=workspace,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
            )
        return ComputeHandle(
            backend=self.name,
            backend_job_id=str(process.pid),
            status=BackendStatus.RUNNING,
            stage="running",
            progress=0.1,
            message=f"Local process started: pid={process.pid}",
            metadata={"workspace": str(workspace), "command": command},
        )

    def poll(self, job: JobRecord) -> ComputeHandle:
        state = self._state(job.spec.job_id)
        if state is None:
            return ComputeHandle(
                backend=self.name,
                backend_job_id=job.backend_job_id or "unknown",
                status=BackendStatus.UNKNOWN,
                stage="process_not_recoverable",
                progress=job.progress,
                message="Local process handle is unavailable after restart",
                error="Local jobs cannot be reattached after Core restart",
            )
        returncode = state.process.poll()
        if returncode is None:
            progress = _read_progress(state.workspace, job.progress)
            return ComputeHandle(
                backend=self.name,
                backend_job_id=str(state.process.pid),
                status=BackendStatus.RUNNING,
                stage=_read_stage(state.workspace, "running"),
                progress=progress,
                message=_tail(state.stdout_path, 1000) or "Local process running",
            )
        self._close_state(job.spec.job_id)
        if returncode == 0:
            return ComputeHandle(
                backend=self.name,
                backend_job_id=str(state.process.pid),
                status=BackendStatus.COMPLETED,
                stage="completed",
                progress=1.0,
                message=_tail(state.stdout_path, 2000) or "Local process completed",
                result=_read_result(state.workspace),
            )
        return ComputeHandle(
            backend=self.name,
            backend_job_id=str(state.process.pid),
            status=BackendStatus.FAILED,
            stage="failed",
            progress=1.0,
            message="Local process failed",
            error=_tail(state.stderr_path, 4000) or f"returncode={returncode}",
        )

    def cancel(self, job: JobRecord) -> ComputeHandle:
        state = self._state(job.spec.job_id)
        if state is None or state.process.poll() is not None:
            return ComputeHandle(
                backend=self.name,
                backend_job_id=job.backend_job_id or "unknown",
                status=BackendStatus.CANCELLED,
                stage="cancelled",
                progress=job.progress,
                message="Local process is not running",
            )
        _terminate_group(state.process)
        self._close_state(job.spec.job_id)
        return ComputeHandle(
            backend=self.name,
            backend_job_id=str(state.process.pid),
            status=BackendStatus.CANCELLED,
            stage="cancelled",
            progress=job.progress,
            message="Local process group terminated",
        )

    def collect(self, job: JobRecord, destination: Path) -> ComputeHandle:
        workspace = Path(str(job.spec.metadata.get("workspace") or "")).expanduser()
        if not workspace.is_absolute():
            workspace = Path.cwd() / workspace if str(workspace) else Path.cwd()
        workspace = workspace.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        collected: list[str] = []
        for relative in job.spec.outputs:
            source = (workspace / relative).resolve()
            try:
                source.relative_to(workspace)
            except ValueError:
                continue
            if not source.is_file():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            collected.append(relative)
        return ComputeHandle(
            backend=self.name,
            backend_job_id=job.backend_job_id or "local",
            status=BackendStatus.COMPLETED,
            stage="collected",
            progress=1.0,
            message=f"Collected {len(collected)} local outputs",
            result=_read_result(workspace),
            metadata={"collected": collected, "destination": str(destination)},
        )

    def _state(self, job_id: str) -> _LocalProcess | None:
        with self._lock:
            return self._processes.get(job_id)

    def _close_state(self, job_id: str) -> None:
        with self._lock:
            state = self._processes.pop(job_id, None)
        if state:
            try:
                state.stdout_handle.close()
            except Exception:
                pass
            try:
                state.stderr_handle.close()
            except Exception:
                pass

    @staticmethod
    def _environment(job: JobRecord, workspace: Path) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "HOME",
                "USERPROFILE",
                "LANG",
                "LC_ALL",
                "PYTHONPATH",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "TMP",
                "TEMP",
                "TMPDIR",
            }
        }
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "RESEARCH_AGENT_JOB_ID": job.spec.job_id,
                "RESEARCH_AGENT_WORK_SESSION_ID": job.spec.work_session_id,
                "RESEARCH_AGENT_WORKSPACE": str(workspace),
            }
        )
        return environment


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass


def _read_result(workspace: Path) -> dict[str, Any]:
    path = workspace / "result.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _read_progress(workspace: Path, default: float) -> float:
    path = workspace / "progress.json"
    if not path.is_file():
        return min(0.9, max(0.1, default))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return min(0.99, max(0.0, float(value.get("progress") or default)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return min(0.9, max(0.1, default))


def _read_stage(workspace: Path, default: str) -> str:
    path = workspace / "progress.json"
    if not path.is_file():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return str(value.get("stage") or default)
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def _tail(path: Path, limit: int) -> str:
    if not path.is_file():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", errors="replace").strip()
