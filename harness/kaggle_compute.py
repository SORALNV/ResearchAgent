from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from harness.compute import BackendCapabilities, BackendRunResult, EventSink
from harness.control_plane import JobSpec
from harness.state import utc_timestamp


@dataclass(frozen=True)
class KaggleCommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class KaggleTransport(Protocol):
    def push(self, package_dir: Path) -> KaggleCommandResult: ...

    def status(self, kernel_ref: str) -> KaggleCommandResult: ...

    def output(self, kernel_ref: str, output_dir: Path) -> KaggleCommandResult: ...


class KaggleCliTransport:
    """Credential-isolated adapter around the official Kaggle CLI."""

    SAFE_ENV = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "KAGGLE_CONFIG_DIR",
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
        executable: str = "kaggle",
        timeout_seconds: int = 120,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.executable = executable
        self.timeout_seconds = max(1, int(timeout_seconds))
        source = dict(environment or os.environ)
        self.environment = {
            key: value
            for key, value in source.items()
            if key in self.SAFE_ENV and value
        }

    def push(self, package_dir: Path) -> KaggleCommandResult:
        return self._run(["kernels", "push", "-p", str(package_dir)])

    def status(self, kernel_ref: str) -> KaggleCommandResult:
        return self._run(["kernels", "status", kernel_ref])

    def output(self, kernel_ref: str, output_dir: Path) -> KaggleCommandResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._run(
            ["kernels", "output", kernel_ref, "-p", str(output_dir), "--quiet"]
        )

    def _run(self, arguments: list[str]) -> KaggleCommandResult:
        command = [self.executable, *arguments]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                env=self.environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return KaggleCommandResult(
                command=tuple(command),
                returncode=124,
                stdout=_text(exc.stdout),
                stderr=_text(exc.stderr) or f"timeout after {self.timeout_seconds}s",
            )
        except OSError as exc:
            return KaggleCommandResult(
                command=tuple(command),
                returncode=127,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
            )
        return KaggleCommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
        )


class KaggleNotebookPackageBuilder:
    """Create a deterministic Kaggle kernel package from a validated JobSpec."""

    EXCLUDED_NAMES = {
        ".git",
        ".env",
        ".venv",
        "__pycache__",
        "kaggle.json",
        "credentials.json",
    }

    def __init__(self, package_root: str | Path) -> None:
        self.package_root = Path(package_root)

    def build(self, spec: JobSpec) -> tuple[Path, str]:
        payload = spec.payload
        kernel_ref = str(payload.get("kernel_ref") or "").strip()
        if "/" not in kernel_ref or kernel_ref.startswith("/"):
            raise ValueError("payload.kernel_ref must be owner/slug")
        code_file = _safe_relative(str(payload.get("code_file") or "run.py"))
        source_dir_raw = payload.get("source_dir")
        if not source_dir_raw:
            raise ValueError("payload.source_dir is required")
        source_dir = Path(str(source_dir_raw)).expanduser().resolve()
        if not source_dir.is_dir():
            raise ValueError(f"source_dir not found: {source_dir}")

        package_dir = self.package_root / spec.job_id
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True)
        self._copy_source(source_dir, package_dir)
        if not (package_dir / code_file).is_file():
            raise ValueError(f"code_file not found in package: {code_file}")

        accelerator = str(spec.resources.get("accelerator") or "cpu").lower()
        metadata = {
            "id": kernel_ref,
            "title": str(payload.get("kernel_title") or spec.job_id)[:80],
            "code_file": code_file,
            "language": str(payload.get("language") or "python"),
            "kernel_type": str(payload.get("kernel_type") or _kernel_type(code_file)),
            "is_private": bool(payload.get("is_private", True)),
            "enable_gpu": accelerator == "gpu",
            "enable_internet": bool(payload.get("enable_internet", False)),
            "dataset_sources": _string_list(payload.get("dataset_sources")),
            "competition_sources": _string_list(payload.get("competition_sources")),
            "kernel_sources": _string_list(payload.get("kernel_sources")),
            "model_sources": _string_list(payload.get("model_sources")),
        }
        (package_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (package_dir / "research-agent-job.json").write_text(
            json.dumps(
                {
                    "generated_at": utc_timestamp(),
                    "job": spec.to_dict(),
                    "package_sha256": _tree_hash(package_dir),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return package_dir, kernel_ref

    def _copy_source(self, source: Path, destination: Path) -> None:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if any(part in self.EXCLUDED_NAMES for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"symlink not allowed in Kaggle package: {relative}")
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)


class KaggleNotebookBackend:
    """Run a JobSpec through Kaggle Kernels push/status/output."""

    TERMINAL_SUCCESS = {"complete", "completed", "success", "succeeded"}
    TERMINAL_FAILURE = {"error", "failed", "failure", "cancelled", "canceled"}

    def __init__(
        self,
        transport: KaggleTransport,
        *,
        package_root: str | Path,
        output_root: str | Path,
        poll_interval_seconds: float = 15.0,
        max_poll_seconds: int = 6 * 60 * 60,
        max_vram_gb: float = 16.0,
    ) -> None:
        self.transport = transport
        self.builder = KaggleNotebookPackageBuilder(package_root)
        self.output_root = Path(output_root)
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self.max_poll_seconds = max(1, int(max_poll_seconds))
        self._capabilities = BackendCapabilities(
            name="kaggle_notebook",
            domains=("kaggle",),
            accelerators=("cpu", "gpu"),
            max_vram_gb=max(0.0, float(max_vram_gb)),
            max_ram_gb=32.0,
            supports_cancel=False,
            supports_live_events=False,
            supports_network=True,
            metadata={
                "remote": True,
                "progress_granularity": "kernel_status",
                "cancel_semantics": "local polling only; remote run may continue",
            },
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
            package_dir, kernel_ref = self.builder.build(spec)
        except Exception as exc:
            return BackendRunResult.failed(f"package build failed: {exc}")

        emit(
            "kaggle_package_ready",
            {
                "kernel_ref": kernel_ref,
                "package_dir": str(package_dir),
                "package_sha256": _tree_hash(package_dir),
            },
        )
        if cancel_event.is_set():
            return BackendRunResult.cancelled("cancelled before Kaggle push")

        pushed = self.transport.push(package_dir)
        emit(
            "kaggle_push_finished",
            {
                "kernel_ref": kernel_ref,
                "returncode": pushed.returncode,
                "stdout": pushed.stdout[-2000:],
                "stderr": pushed.stderr[-2000:],
            },
        )
        if not pushed.ok:
            return BackendRunResult.failed(
                f"kaggle kernels push failed: {pushed.stderr or pushed.stdout}",
                backend_job_id=kernel_ref,
            )

        emit("backend_started", {"backend_job_id": kernel_ref})
        deadline = time.monotonic() + min(
            self.max_poll_seconds,
            spec.max_runtime_seconds or self.max_poll_seconds,
        )
        last_status = "unknown"
        while time.monotonic() < deadline:
            if cancel_event.is_set():
                emit(
                    "kaggle_polling_cancelled",
                    {
                        "kernel_ref": kernel_ref,
                        "warning": "remote Kaggle run may continue",
                    },
                )
                return BackendRunResult.cancelled(
                    "local polling cancelled; remote Kaggle run may continue",
                    backend_job_id=kernel_ref,
                )
            status_result = self.transport.status(kernel_ref)
            if not status_result.ok:
                emit(
                    "kaggle_status_error",
                    {
                        "returncode": status_result.returncode,
                        "stderr": status_result.stderr[-2000:],
                    },
                )
                cancel_event.wait(self.poll_interval_seconds)
                continue
            last_status = _parse_kernel_status(status_result.stdout)
            emit(
                "backend_progress",
                {
                    "stage": "kaggle_kernel",
                    "kernel_ref": kernel_ref,
                    "kernel_status": last_status,
                },
            )
            if last_status in self.TERMINAL_SUCCESS:
                return self._collect(spec, kernel_ref, emit)
            if last_status in self.TERMINAL_FAILURE:
                return BackendRunResult.failed(
                    f"Kaggle kernel ended with status {last_status}",
                    backend_job_id=kernel_ref,
                )
            cancel_event.wait(self.poll_interval_seconds)

        return BackendRunResult.failed(
            f"Kaggle kernel polling timed out; last status={last_status}",
            backend_job_id=kernel_ref,
        )

    def _collect(
        self,
        spec: JobSpec,
        kernel_ref: str,
        emit: EventSink,
    ) -> BackendRunResult:
        output_dir = self.output_root / spec.job_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        collected = self.transport.output(kernel_ref, output_dir)
        emit(
            "kaggle_output_finished",
            {
                "kernel_ref": kernel_ref,
                "returncode": collected.returncode,
                "output_dir": str(output_dir),
                "stderr": collected.stderr[-2000:],
            },
        )
        if not collected.ok:
            return BackendRunResult.failed(
                f"kaggle kernels output failed: {collected.stderr or collected.stdout}",
                backend_job_id=kernel_ref,
            )
        missing = [
            relative
            for relative in spec.outputs
            if not (output_dir / _safe_relative(relative)).exists()
        ]
        result = {
            "kernel_ref": kernel_ref,
            "output_dir": str(output_dir),
            "output_sha256": _tree_hash(output_dir),
            "files": [
                path.relative_to(output_dir).as_posix()
                for path in sorted(output_dir.rglob("*"))
                if path.is_file() and not path.is_symlink()
            ],
            "missing_outputs": missing,
        }
        for name in ("result.json", "metrics.json"):
            path = output_dir / name
            if not path.is_file():
                continue
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
            else:
                result[name.removesuffix(".json")] = parsed
        if missing:
            return BackendRunResult.failed(
                "required Kaggle outputs are missing: " + ", ".join(missing),
                result=result,
                backend_job_id=kernel_ref,
            )
        return BackendRunResult.completed(
            result,
            backend_job_id=kernel_ref,
        )


class FakeKaggleTransport:
    """Scriptable transport for credential-free CI tests."""

    def __init__(
        self,
        statuses: list[str] | tuple[str, ...] = ("queued", "running", "complete"),
        *,
        outputs: Mapping[str, str | bytes] | None = None,
        fail_push: bool = False,
        fail_output: bool = False,
    ) -> None:
        self.statuses = list(statuses)
        self.outputs = dict(outputs or {})
        self.fail_push = fail_push
        self.fail_output = fail_output
        self.pushed_packages: list[Path] = []
        self.status_calls = 0

    def push(self, package_dir: Path) -> KaggleCommandResult:
        self.pushed_packages.append(package_dir)
        return KaggleCommandResult(
            command=("fake", "push"),
            returncode=1 if self.fail_push else 0,
            stdout="",
            stderr="push failed" if self.fail_push else "",
        )

    def status(self, kernel_ref: str) -> KaggleCommandResult:
        index = min(self.status_calls, max(0, len(self.statuses) - 1))
        status = self.statuses[index] if self.statuses else "complete"
        self.status_calls += 1
        return KaggleCommandResult(
            command=("fake", "status", kernel_ref),
            returncode=0,
            stdout=f'Kernel {kernel_ref} has status "{status}"',
            stderr="",
        )

    def output(self, kernel_ref: str, output_dir: Path) -> KaggleCommandResult:
        if self.fail_output:
            return KaggleCommandResult(
                command=("fake", "output", kernel_ref),
                returncode=1,
                stdout="",
                stderr="output failed",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for relative, content in self.outputs.items():
            path = output_dir / _safe_relative(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")
        return KaggleCommandResult(
            command=("fake", "output", kernel_ref),
            returncode=0,
            stdout="output collected",
            stderr="",
        )


def _parse_kernel_status(text: str) -> str:
    normalized = text.strip().lower()
    for candidate in (
        "complete",
        "completed",
        "success",
        "succeeded",
        "running",
        "queued",
        "pending",
        "error",
        "failed",
        "failure",
        "cancelled",
        "canceled",
    ):
        if f'"{candidate}"' in normalized or f"'{candidate}'" in normalized:
            return candidate
    for candidate in (
        "complete",
        "completed",
        "running",
        "queued",
        "pending",
        "error",
        "failed",
        "cancelled",
    ):
        if candidate in normalized:
            return candidate
    return normalized[:80] or "unknown"


def _safe_relative(value: str) -> str:
    path = Path(value.strip())
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def _kernel_type(code_file: str) -> str:
    return "notebook" if code_file.lower().endswith(".ipynb") else "script"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
