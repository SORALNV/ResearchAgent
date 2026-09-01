from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from harness.compute_backends import (
    RemoteGpuBackend,
    UrllibRemoteTransport,
)
from harness.compute_models import CollectedResult, safe_relative_path
from harness.compute_scheduler import ComputeStack
from harness.compute_scheduler_safe import (
    BackendBoundApprovalScheduler,
    harden_compute_stack,
)
from harness.discord_channel_map import DiscordLocation


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DiscordAccessPolicy:
    """Fail-closed authorization for Discord state-changing operations.

    Interactive setup is restricted to the configured global user allowlist.
    The user that creates a channel session is persisted as its owner and keeps
    access even if the allowlist is later narrowed. Read-only status operations
    remain available to users who can already see the Discord channel.
    """

    allowed_user_ids: frozenset[str]
    admin_user_ids: frozenset[str]
    required: bool = True

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "DiscordAccessPolicy":
        source = dict(os.environ if environ is None else environ)
        return cls(
            allowed_user_ids=_discord_ids(source.get("DISCORD_ALLOWED_USER_IDS")),
            admin_user_ids=_discord_ids(source.get("DISCORD_ADMIN_USER_IDS")),
            required=_bool_value(
                source.get("DISCORD_ACCESS_CONTROL_REQUIRED"),
                True,
            ),
        )

    @property
    def global_user_ids(self) -> frozenset[str]:
        return self.allowed_user_ids | self.admin_user_ids

    def require_setup(self, actor_id: str) -> None:
        actor = _discord_id(actor_id, "actor_id")
        if actor in self.global_user_ids:
            return
        if self.required:
            raise PermissionError(
                "Discord setup is restricted. Add the user to "
                "DISCORD_ALLOWED_USER_IDS or DISCORD_ADMIN_USER_IDS."
            )

    def require_channel_action(
        self,
        actor_id: str,
        *,
        owner_id: str,
        action: str,
    ) -> None:
        actor = _discord_id(actor_id, "actor_id")
        owner = str(owner_id or "").strip()
        if actor in self.global_user_ids or (owner.isdigit() and actor == owner):
            return
        if self.required:
            raise PermissionError(
                f"Discord user {actor} is not authorized to {action} in this channel. "
                "Only the channel-session owner or an explicitly allowed user may do so."
            )


def apply_production_hardening(
    service: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Install active-path ACL, backend-bound approvals, and safe Worker I/O."""

    source = dict(os.environ if environ is None else environ)
    _activate_compute_hardening(service, source)
    _install_discord_acl(service, DiscordAccessPolicy.from_environment(source))
    return service


def _install_discord_acl(service: Any, policy: DiscordAccessPolicy) -> None:
    if bool(getattr(service, "_production_discord_acl_installed", False)):
        return
    registry = getattr(service, "registry", None)
    if registry is None:
        return

    def channel_owner(location: DiscordLocation) -> str:
        record = registry.get(location)
        if record is None:
            raise PermissionError(
                "This Discord channel is not configured. An authorized user must run /agent setup."
            )
        return str(getattr(record, "created_by", "") or "")

    original_setup = service.setup_channel

    def setup_channel(
        location: DiscordLocation,
        *,
        mode: str,
        subject: str,
        target_ref: str,
        actor_id: str,
    ) -> Any:
        existing = registry.get(location)
        if existing is None:
            policy.require_setup(actor_id)
        else:
            policy.require_channel_action(
                actor_id,
                owner_id=str(getattr(existing, "created_by", "") or ""),
                action="configure the channel session",
            )
        return original_setup(
            location,
            mode=mode,
            subject=subject,
            target_ref=target_ref,
            actor_id=actor_id,
        )

    service.setup_channel = setup_channel

    _guard_method(
        service,
        "handle_message",
        policy,
        channel_owner,
        "send an Agent instruction",
    )
    _guard_method(
        service,
        "finish_channel",
        policy,
        channel_owner,
        "archive the channel session",
    )
    _guard_method(
        service,
        "try_steer_codex",
        policy,
        channel_owner,
        "steer the active Codex turn",
    )
    _guard_method(
        service,
        "steer_codex",
        policy,
        channel_owner,
        "steer the active Codex turn",
    )
    _guard_method(
        service,
        "interrupt_codex",
        policy,
        channel_owner,
        "interrupt the active Codex turn",
    )
    _guard_method(
        service,
        "resolve_codex_approval",
        policy,
        channel_owner,
        "resolve a Codex approval",
    )
    _guard_method(
        service,
        "approve_compute",
        policy,
        channel_owner,
        "approve a Compute backend",
    )
    _guard_method(
        service,
        "cancel_compute",
        policy,
        channel_owner,
        "cancel a Compute job",
    )
    _guard_method(
        service,
        "record_decision",
        policy,
        channel_owner,
        "record a human decision",
    )
    service.discord_access_policy = policy
    service._production_discord_acl_installed = True


def _guard_method(
    service: Any,
    name: str,
    policy: DiscordAccessPolicy,
    channel_owner: Any,
    action: str,
) -> None:
    original = getattr(service, name, None)
    if not callable(original):
        return

    def guarded(*args: Any, **kwargs: Any) -> Any:
        location = kwargs.get("location")
        if location is None and args:
            location = args[0]
        actor_id = kwargs.get("actor_id")
        if actor_id is None:
            raise PermissionError(f"actor_id is required to {action}")
        if not isinstance(location, DiscordLocation):
            raise PermissionError(f"Discord location is required to {action}")
        policy.require_channel_action(
            str(actor_id),
            owner_id=channel_owner(location),
            action=action,
        )
        return original(*args, **kwargs)

    setattr(service, name, guarded)


def _activate_compute_hardening(
    service: Any,
    source: Mapping[str, str],
) -> None:
    current = getattr(service, "compute", None)
    if not isinstance(current, ComputeStack):
        return
    old_scheduler = current.scheduler
    hardened = (
        current
        if isinstance(old_scheduler, BackendBoundApprovalScheduler)
        else harden_compute_stack(current)
    )
    _harden_remote_backends(hardened, source)

    for node in _service_graph(service):
        try:
            if getattr(node, "compute", None) is current:
                setattr(node, "compute", hardened)
        except Exception:
            pass
        try:
            if getattr(node, "scheduler", None) is old_scheduler:
                setattr(node, "scheduler", hardened.scheduler)
        except Exception:
            pass


def _service_graph(service: Any):
    seen: set[int] = set()
    current = service
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = vars(current).get("base_service")


def _harden_remote_backends(
    stack: ComputeStack,
    source: Mapping[str, str],
) -> None:
    per_file = _positive_int(
        source.get("REMOTE_ARTIFACT_MAX_FILE_BYTES"),
        2 * 1024 * 1024 * 1024,
    )
    total = _positive_int(
        source.get("REMOTE_ARTIFACT_MAX_TOTAL_BYTES"),
        4 * 1024 * 1024 * 1024,
    )
    max_files = _positive_int(source.get("REMOTE_ARTIFACT_MAX_FILES"), 2000)
    max_response = _positive_int(
        source.get("REMOTE_WORKER_MAX_RESPONSE_BYTES"),
        8 * 1024 * 1024,
    )
    for name, backend in list(stack.broker.backends.items()):
        if isinstance(backend, HardenedRemoteGpuBackend):
            continue
        if isinstance(backend, RemoteGpuBackend):
            stack.broker.backends[name] = HardenedRemoteGpuBackend.from_backend(
                backend,
                max_artifact_file_bytes=per_file,
                max_artifact_total_bytes=total,
                max_artifact_files=max_files,
                max_response_bytes=max_response,
            )


class SameOriginBoundedUrllibTransport(UrllibRemoteTransport):
    """Urllib transport that never forwards Worker credentials cross-origin."""

    def __init__(
        self,
        base_url: str,
        *,
        max_response_bytes: int,
        max_download_bytes: int,
    ) -> None:
        self.base_url = _validated_http_url(base_url)
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.max_download_bytes = max(1, int(max_download_bytes))
        self._opener = urllib.request.build_opener(
            _SameOriginRedirectHandler(self.base_url)
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        target = _require_same_origin(self.base_url, url)
        data = (
            json.dumps(dict(payload), ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            if payload is not None
            else None
        )
        request = urllib.request.Request(
            target,
            data=data,
            method=method,
            headers=dict(headers),
        )
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                _require_same_origin(self.base_url, response.geturl())
                body = _read_bounded(response, self.max_response_bytes)
        except urllib.error.HTTPError as exc:
            detail = _read_bounded(exc, self.max_response_bytes).decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(
                f"remote worker HTTP {exc.code}: {detail[-4000:]}"
            ) from exc
        value = json.loads(body.decode("utf-8") or "{}")
        if not isinstance(value, dict):
            raise RuntimeError("remote worker returned a non-object response")
        return value

    def download(
        self,
        url: str,
        destination: Path,
        *,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> None:
        target_url = _require_same_origin(self.base_url, url)
        request = urllib.request.Request(target_url, headers=dict(headers))
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            destination.name + f".download-{uuid.uuid4().hex}.tmp"
        )
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                _require_same_origin(self.base_url, response.geturl())
                advertised = response.headers.get("Content-Length")
                if advertised is not None and int(advertised) > self.max_download_bytes:
                    raise ValueError("remote artifact exceeds the configured byte limit")
                written = 0
                with temporary.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > self.max_download_bytes:
                            raise ValueError(
                                "remote artifact exceeded the configured byte limit while streaming"
                            )
                        output.write(chunk)
                temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        target = urllib.parse.urljoin(req.full_url, newurl)
        _require_same_origin(self.base_url, target)
        return super().redirect_request(req, fp, code, msg, headers, target)


class HardenedRemoteGpuBackend(RemoteGpuBackend):
    """Remote Worker backend with mandatory artifact size/hash provenance."""

    def __init__(
        self,
        *args: Any,
        max_artifact_file_bytes: int,
        max_artifact_total_bytes: int,
        max_artifact_files: int,
        max_response_bytes: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_artifact_file_bytes = max(1, int(max_artifact_file_bytes))
        self.max_artifact_total_bytes = max(1, int(max_artifact_total_bytes))
        self.max_artifact_files = max(1, int(max_artifact_files))
        if type(self.transport) is UrllibRemoteTransport:
            self.transport = SameOriginBoundedUrllibTransport(
                self.descriptor.base_url,
                max_response_bytes=max_response_bytes,
                max_download_bytes=self.max_artifact_file_bytes,
            )

    @classmethod
    def from_backend(
        cls,
        backend: RemoteGpuBackend,
        *,
        max_artifact_file_bytes: int,
        max_artifact_total_bytes: int,
        max_artifact_files: int,
        max_response_bytes: int,
    ) -> "HardenedRemoteGpuBackend":
        return cls(
            backend.descriptor,
            transport=backend.transport,
            timeout_seconds=backend.timeout_seconds,
            max_bundle_files=backend.max_bundle_files,
            max_bundle_bytes=backend.max_bundle_bytes,
            max_artifact_file_bytes=max_artifact_file_bytes,
            max_artifact_total_bytes=max_artifact_total_bytes,
            max_artifact_files=max_artifact_files,
            max_response_bytes=max_response_bytes,
        )

    def collect(
        self,
        job: Any,
        handle: Any,
        destination: Path,
    ) -> CollectedResult:
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        response = self._request(
            "GET",
            "/v1/jobs/"
            + urllib.parse.quote(handle.backend_job_id, safe="")
            + "/artifacts",
        )
        raw_artifacts = response.get("artifacts")
        if raw_artifacts is None:
            raw_artifacts = []
        if not isinstance(raw_artifacts, list):
            raise ValueError("remote artifact manifest must be a list")
        if len(raw_artifacts) > self.max_artifact_files:
            raise ValueError("remote artifact manifest exceeds the file-count limit")

        manifest: list[tuple[str, str, int, str]] = []
        seen_paths: set[str] = set()
        total_bytes = 0
        for item in raw_artifacts:
            if not isinstance(item, Mapping):
                raise ValueError("remote artifact manifest entry must be an object")
            relative = safe_relative_path(str(item.get("path") or ""))
            if not relative or relative in seen_paths:
                raise ValueError("remote artifact path is unsafe or duplicated")
            seen_paths.add(relative)
            raw_url = str(item.get("url") or item.get("download_path") or "").strip()
            if not raw_url:
                raise ValueError(f"remote artifact has no download URL: {relative}")
            url = urllib.parse.urljoin(
                self.descriptor.base_url.rstrip("/") + "/",
                raw_url,
            )
            url = _require_same_origin(self.descriptor.base_url, url)
            size = _manifest_size(item.get("size_bytes"), relative)
            digest = str(item.get("sha256") or "").strip().lower()
            if not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"remote artifact has no valid SHA-256: {relative}")
            if size > self.max_artifact_file_bytes:
                raise ValueError(f"remote artifact is too large: {relative}")
            total_bytes += size
            if total_bytes > self.max_artifact_total_bytes:
                raise ValueError("remote artifacts exceed the total byte limit")
            manifest.append((relative, url, size, digest))

        staging = destination / f".incoming-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        collected: list[str] = []
        try:
            for relative, url, expected_size, expected_hash in manifest:
                target = (staging / relative).resolve()
                try:
                    target.relative_to(staging)
                except ValueError as exc:
                    raise ValueError(
                        f"remote artifact escapes the staging directory: {relative}"
                    ) from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                self.transport.download(
                    url,
                    target,
                    headers=self._headers(),
                    timeout_seconds=max(60, self.timeout_seconds),
                )
                if target.is_symlink() or not target.is_file():
                    raise ValueError(f"remote artifact was not a regular file: {relative}")
                if target.stat().st_size != expected_size:
                    raise ValueError(f"remote artifact size mismatch: {relative}")
                if _sha256(target) != expected_hash:
                    raise ValueError(f"remote artifact hash mismatch: {relative}")

            for relative, _, _, _ in manifest:
                source = (staging / relative).resolve()
                target = (destination / relative).resolve()
                try:
                    target.relative_to(destination)
                except ValueError as exc:
                    raise ValueError(
                        f"remote artifact escapes the destination: {relative}"
                    ) from exc
                if target.exists() and target.is_symlink():
                    raise ValueError(f"remote artifact target is a symlink: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
                collected.append(relative)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        result = (
            dict(response.get("result"))
            if isinstance(response.get("result"), Mapping)
            else dict(handle.result)
        )
        if result and not (destination / "result.json").is_file():
            _atomic_json(destination / "result.json", result)
            collected.append("result.json")
        return CollectedResult(
            result=result,
            artifact_paths=tuple(dict.fromkeys(collected)),
            metadata={"response": response, "destination": str(destination)},
        )


def _validated_http_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(str(value).strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("remote Worker URL must be absolute http(s)")
    if parsed.username or parsed.password:
        raise ValueError("remote Worker URL must not contain user information")
    return urllib.parse.urlunsplit(parsed)


def _origin(value: str) -> tuple[str, str, int]:
    parsed = urllib.parse.urlsplit(_validated_http_url(value))
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    return scheme, str(parsed.hostname).lower(), port


def _require_same_origin(base_url: str, candidate_url: str) -> str:
    candidate = _validated_http_url(candidate_url)
    if _origin(base_url) != _origin(candidate):
        raise PermissionError("remote Worker redirect/artifact URL changed origin")
    return candidate


def _read_bounded(stream: Any, limit: int) -> bytes:
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("remote Worker response exceeds the configured byte limit")
    return data


def _manifest_size(value: Any, relative: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"remote artifact has no valid size: {relative}")
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"remote artifact has no valid size: {relative}") from exc
    if size < 0:
        raise ValueError(f"remote artifact has a negative size: {relative}")
    return size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _discord_ids(value: Any) -> frozenset[str]:
    text = str(value or "").strip()
    if not text:
        return frozenset()
    raw: list[Any]
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Discord user ID allowlist must be valid JSON or CSV") from exc
        if not isinstance(decoded, list):
            raise ValueError("Discord user ID allowlist JSON must be an array")
        raw = decoded
    else:
        raw = re.split(r"[\s,]+", text)
    return frozenset(
        _discord_id(item, "Discord user ID")
        for item in raw
        if str(item).strip()
    )


def _discord_id(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text.isdigit():
        raise ValueError(f"{name} must be a Discord snowflake")
    return text


def _positive_int(value: Any, default: int) -> int:
    if value in {None, ""}:
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("production hardening byte/file limits must be integers") from exc
    if parsed <= 0:
        raise ValueError("production hardening byte/file limits must be positive")
    return parsed


def _bool_value(value: Any, default: bool) -> bool:
    if value in {None, ""}:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
