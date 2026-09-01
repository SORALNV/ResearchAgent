from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from harness.codex_app_server import (
    CodexAppServerRuntime,
    CodexAppServerSettings,
    normalize_official_codex_payload,
)
from harness.compute_backends import RemoteGpuBackend, RemoteWorkerDescriptor
from harness.compute_models import (
    BackendCapabilities,
    BackendHandle,
    BackendState,
)
from harness.compute_scheduler_safe import BackendBoundApprovalScheduler
from harness.config import HarnessConfig
from harness.control_plane import (
    ControlPlaneStore,
    Domain,
    JobSpec,
    ResourceRequirements,
)
from harness.discord_channel_map import DiscordLocation
from harness.production_hardening import (
    DiscordAccessPolicy,
    HardenedRemoteGpuBackend,
    SameOriginBoundedUrllibTransport,
    _SameOriginRedirectHandler,
    apply_production_hardening,
)
from main import build_routed_discord_service


ROOT = Path(__file__).resolve().parents[1]


def _location(channel_id: str = "100") -> DiscordLocation:
    return DiscordLocation(
        guild_id="1",
        channel_id=channel_id,
        parent_channel_id=None,
        thread_id=None,
    )


def test_environment_app_server_uses_current_text_elements_wire_field(
    tmp_path: Path,
) -> None:
    source = {
        "method": "turn/start",
        "params": {
            "input": [
                {"type": "text", "text": "hello", "textElements": []}
            ]
        },
    }
    normalized = normalize_official_codex_payload(source)
    assert normalized["params"]["input"] == [
        {"type": "text", "text": "hello", "text_elements": []}
    ]
    assert "textElements" not in str(normalized)

    settings = CodexAppServerSettings.from_environment(
        tmp_path,
        {"CODEX_APP_SERVER_COMMAND": "codex app-server --listen stdio://"},
    )
    runtime = CodexAppServerRuntime(settings)
    prepared = runtime._prepare_outbound_payload(source)
    assert prepared["params"]["input"][0]["text_elements"] == []
    assert "textElements" not in str(prepared)


def test_discord_acl_is_fail_closed_and_honors_channel_owner() -> None:
    location = _location()
    records: dict[str, Any] = {
        location.conversation_id: SimpleNamespace(created_by="42")
    }

    class Registry:
        def get(self, value: DiscordLocation):
            return records.get(value.conversation_id)

    calls: list[tuple[str, str]] = []
    service = SimpleNamespace(
        registry=Registry(),
        compute=None,
        setup_channel=lambda *args, **kwargs: calls.append(("setup", kwargs["actor_id"])),
        handle_message=lambda *args, **kwargs: calls.append(("message", kwargs["actor_id"])),
        finish_channel=lambda *args, **kwargs: calls.append(("finish", kwargs["actor_id"])),
        resolve_codex_approval=lambda *args, **kwargs: calls.append(("approval", kwargs["actor_id"])),
        approve_compute=lambda *args, **kwargs: calls.append(("compute", kwargs["actor_id"])),
    )
    apply_production_hardening(
        service,
        environ={
            "DISCORD_ACCESS_CONTROL_REQUIRED": "true",
            "DISCORD_ALLOWED_USER_IDS": "42",
        },
    )

    service.handle_message(location, actor_id="42")
    service.resolve_codex_approval(location, actor_id="42")
    with pytest.raises(PermissionError):
        service.handle_message(location, actor_id="99")
    with pytest.raises(PermissionError):
        service.approve_compute(location, actor_id="99")
    assert calls == [("message", "42"), ("approval", "42")]

    policy = DiscordAccessPolicy.from_environment(
        {"DISCORD_ACCESS_CONTROL_REQUIRED": "true"}
    )
    with pytest.raises(PermissionError, match="DISCORD_ALLOWED_USER_IDS"):
        policy.require_setup("42")


def test_active_service_uses_backend_bound_compute_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCORD_RESEARCH_CHANNEL_IDS", "100")
    monkeypatch.setenv("DISCORD_ALLOWED_USER_IDS", "42")
    monkeypatch.setenv("CONTROL_PLANE_DIR", str(tmp_path / "control"))
    monkeypatch.delenv("REMOTE_GPU_WORKER_URL", raising=False)
    monkeypatch.delenv("COMPUTE_REMOTE_WORKERS_JSON", raising=False)

    service = build_routed_discord_service(HarnessConfig(project_root=tmp_path))
    assert isinstance(service.compute.scheduler, BackendBoundApprovalScheduler)
    assert service.scheduler is service.compute.scheduler
    assert bool(getattr(service, "_production_discord_acl_installed", False))


def _remote_job(tmp_path: Path):
    store = ControlPlaneStore(tmp_path / "control")
    project = store.create_project("remote", Domain.RESEARCH)
    session = store.create_work_session(project.project_id, "remote")
    job = store.create_job(
        JobSpec(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain=Domain.RESEARCH,
            kind="experiment",
            payload={"entrypoint": ["python", "run.py"]},
            resources=ResourceRequirements(accelerator="cpu"),
        )
    )
    handle = BackendHandle(
        backend="remote",
        backend_job_id="remote-job",
        state=BackendState.SUCCEEDED,
        stage="completed",
        result={"summary": "done"},
    )
    return job, handle


class _ArtifactTransport:
    def __init__(self, response: Mapping[str, Any], payload: bytes) -> None:
        self.response = dict(response)
        self.payload = payload
        self.downloads: list[str] = []

    def request(self, method, url, *, payload, headers, timeout_seconds):
        return dict(self.response)

    def download(self, url, destination, *, headers, timeout_seconds):
        self.downloads.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payload)


def _descriptor() -> RemoteWorkerDescriptor:
    return RemoteWorkerDescriptor(
        name="remote",
        base_url="https://worker.example/api",
        token="secret",
        capabilities=BackendCapabilities(
            accelerators=("cpu",),
            domains=(Domain.RESEARCH,),
            cpu_cores=4,
        ),
    )


def _hardened_backend(transport: Any) -> HardenedRemoteGpuBackend:
    return HardenedRemoteGpuBackend(
        _descriptor(),
        transport=transport,
        max_artifact_file_bytes=16,
        max_artifact_total_bytes=32,
        max_artifact_files=4,
        max_response_bytes=4096,
    )


def test_remote_artifact_requires_same_origin_size_and_hash(tmp_path: Path) -> None:
    payload = b"artifact"
    digest = hashlib.sha256(payload).hexdigest()
    job, handle = _remote_job(tmp_path)

    valid_transport = _ArtifactTransport(
        {
            "artifacts": [
                {
                    "path": "metrics.json",
                    "download_path": "/v1/jobs/remote-job/artifacts/metrics.json",
                    "size_bytes": len(payload),
                    "sha256": digest,
                }
            ],
            "result": {"summary": "done"},
        },
        payload,
    )
    result = _hardened_backend(valid_transport).collect(
        job,
        handle,
        tmp_path / "valid",
    )
    assert "metrics.json" in result.artifact_paths
    assert (tmp_path / "valid" / "metrics.json").read_bytes() == payload

    cross_origin = _ArtifactTransport(
        {
            "artifacts": [
                {
                    "path": "metrics.json",
                    "url": "https://evil.example/steal",
                    "size_bytes": len(payload),
                    "sha256": digest,
                }
            ]
        },
        payload,
    )
    with pytest.raises(PermissionError, match="changed origin"):
        _hardened_backend(cross_origin).collect(
            job,
            handle,
            tmp_path / "cross-origin",
        )
    assert cross_origin.downloads == []

    missing_hash = _ArtifactTransport(
        {
            "artifacts": [
                {
                    "path": "metrics.json",
                    "download_path": "/artifact",
                    "size_bytes": len(payload),
                }
            ]
        },
        payload,
    )
    with pytest.raises(ValueError, match="valid SHA-256"):
        _hardened_backend(missing_hash).collect(
            job,
            handle,
            tmp_path / "missing-hash",
        )

    oversized = _ArtifactTransport(
        {
            "artifacts": [
                {
                    "path": "large.bin",
                    "download_path": "/large.bin",
                    "size_bytes": 17,
                    "sha256": digest,
                }
            ]
        },
        payload,
    )
    with pytest.raises(ValueError, match="too large"):
        _hardened_backend(oversized).collect(
            job,
            handle,
            tmp_path / "oversized",
        )


def test_default_remote_transport_is_bounded_and_redirects_stay_same_origin() -> None:
    plain = RemoteGpuBackend(_descriptor())
    hardened = HardenedRemoteGpuBackend.from_backend(
        plain,
        max_artifact_file_bytes=1024,
        max_artifact_total_bytes=2048,
        max_artifact_files=10,
        max_response_bytes=4096,
    )
    assert isinstance(hardened.transport, SameOriginBoundedUrllibTransport)

    handler = _SameOriginRedirectHandler("https://worker.example/api")
    request = __import__("urllib.request").request.Request(
        "https://worker.example/api/artifact",
        headers={"Authorization": "Bearer secret"},
    )
    with pytest.raises(PermissionError, match="changed origin"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://evil.example/artifact",
        )


def test_codex_state_is_excluded_and_non_root_home_is_writable() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "codex-home/" in gitignore.splitlines()
    assert "**/.codex/" in gitignore.splitlines()
    assert "codex-home" in dockerignore.splitlines()
    assert "**/.codex" in dockerignore.splitlines()
    assert (
        "/home/researchagent:rw,nosuid,nodev,size=256m,"
        "uid=10001,gid=10001,mode=0700"
    ) in compose
    assert "seccomp=unconfined" not in compose
