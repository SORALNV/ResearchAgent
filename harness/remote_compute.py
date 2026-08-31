from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from harness.compute import BackendCapabilities, BackendRunResult, EventSink
from harness.control_plane import JobSpec


class RemoteWorkerClient(Protocol):
    def submit(self, spec: JobSpec) -> dict[str, Any]: ...

    def status(self, remote_job_id: str) -> dict[str, Any]: ...

    def events(self, remote_job_id: str, after_sequence: int) -> list[dict[str, Any]]: ...

    def cancel(self, remote_job_id: str, reason: str) -> dict[str, Any]: ...


class HttpRemoteWorkerClient:
    """Typed HTTP client intended for Tailscale/VPN-connected GPU workers."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        timeout_seconds: int = 30,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("remote worker base_url must be http(s)")
        if not token:
            raise ValueError("remote worker token is required")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = max(1, int(timeout_seconds))

    def submit(self, spec: JobSpec) -> dict[str, Any]:
        return self._request("POST", "/v1/jobs", {"job": spec.to_dict()})

    def status(self, remote_job_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            "/v1/jobs/" + urllib.parse.quote(remote_job_id, safe=""),
        )

    def events(self, remote_job_id: str, after_sequence: int) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/v1/jobs/"
            + urllib.parse.quote(remote_job_id, safe="")
            + "/events?after="
            + str(max(0, int(after_sequence))),
        )
        values = payload.get("events")
        return [dict(item) for item in values] if isinstance(values, list) else []

    def cancel(self, remote_job_id: str, reason: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/jobs/" + urllib.parse.quote(remote_job_id, safe="") + "/cancel",
            {"reason": reason},
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "ResearchAgent-RemoteCompute/1",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            method=method,
            data=body,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"remote worker HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"remote worker connection failed: {exc.reason}") from exc
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("remote worker returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("remote worker response must be a JSON object")
        return value


class RemoteComputeBackend:
    """Poll a remote GPU/VM worker while streaming structured events."""

    TERMINAL = {"completed", "failed", "cancelled", "blocked"}

    def __init__(
        self,
        name: str,
        client: RemoteWorkerClient,
        capabilities: BackendCapabilities,
        *,
        poll_interval_seconds: float = 2.0,
        max_poll_seconds: int = 24 * 60 * 60,
    ) -> None:
        if capabilities.name != name:
            capabilities = BackendCapabilities(
                name=name,
                domains=capabilities.domains,
                accelerators=capabilities.accelerators,
                max_vram_gb=capabilities.max_vram_gb,
                max_ram_gb=capabilities.max_ram_gb,
                supports_cancel=capabilities.supports_cancel,
                supports_live_events=capabilities.supports_live_events,
                supports_network=capabilities.supports_network,
                metadata=capabilities.metadata,
            )
        self._capabilities = capabilities
        self.client = client
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.max_poll_seconds = max(1, int(max_poll_seconds))

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
        submitted = self.client.submit(spec)
        remote_job_id = str(
            submitted.get("remote_job_id")
            or submitted.get("job_id")
            or ""
        )
        if not remote_job_id:
            return BackendRunResult.failed("remote worker did not return a job id")
        emit(
            "backend_started",
            {
                "backend_job_id": remote_job_id,
                "worker": self.capabilities.name,
            },
        )

        deadline = time.monotonic() + min(
            self.max_poll_seconds,
            spec.max_runtime_seconds or self.max_poll_seconds,
        )
        sequence = 0
        cancel_sent = False
        last_status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if cancel_event.is_set() and not cancel_sent:
                cancel_sent = True
                response = self.client.cancel(remote_job_id, "control plane cancel requested")
                emit("remote_cancel_sent", response)
            for event in self.client.events(remote_job_id, sequence):
                try:
                    event_sequence = int(event.get("sequence") or sequence)
                except (TypeError, ValueError):
                    event_sequence = sequence
                sequence = max(sequence, event_sequence)
                event_type = str(event.get("event_type") or "remote_event")
                payload = event.get("payload")
                emit(
                    event_type,
                    dict(payload) if isinstance(payload, dict) else {"value": payload},
                )
            last_status = self.client.status(remote_job_id)
            status = str(last_status.get("status") or "unknown").lower()
            emit(
                "backend_progress",
                {
                    "stage": "remote_worker",
                    "worker": self.capabilities.name,
                    "remote_job_id": remote_job_id,
                    "remote_status": status,
                    "progress": last_status.get("progress") or {},
                },
            )
            if status in self.TERMINAL:
                result = last_status.get("result")
                result_payload = dict(result) if isinstance(result, dict) else {}
                error = str(last_status.get("error") or "") or None
                if status == "completed":
                    return BackendRunResult.completed(
                        result_payload,
                        backend_job_id=remote_job_id,
                    )
                if status == "cancelled":
                    return BackendRunResult.cancelled(
                        error or "remote job cancelled",
                        result=result_payload,
                        backend_job_id=remote_job_id,
                    )
                return BackendRunResult(
                    status=status,
                    result=result_payload,
                    error=error or f"remote worker returned {status}",
                    backend_job_id=remote_job_id,
                )
            cancel_event.wait(self.poll_interval_seconds)

        if not cancel_sent and self.capabilities.supports_cancel:
            try:
                self.client.cancel(remote_job_id, "control plane timeout")
            except Exception:
                pass
        return BackendRunResult.failed(
            "remote worker polling timed out; "
            f"last_status={last_status.get('status', 'unknown')}",
            result={"last_status": last_status},
            backend_job_id=remote_job_id,
        )


@dataclass
class FakeRemoteWorkerClient:
    statuses: list[dict[str, Any]]
    events_by_call: list[list[dict[str, Any]]] | None = None
    remote_job_id: str = "REMOTE-1"

    def __post_init__(self) -> None:
        self._status_index = 0
        self._event_index = 0
        self.cancel_requests: list[str] = []
        self.submitted: list[JobSpec] = []

    def submit(self, spec: JobSpec) -> dict[str, Any]:
        self.submitted.append(spec)
        return {"remote_job_id": self.remote_job_id, "status": "queued"}

    def status(self, remote_job_id: str) -> dict[str, Any]:
        if not self.statuses:
            return {"status": "completed", "result": {}}
        index = min(self._status_index, len(self.statuses) - 1)
        self._status_index += 1
        return dict(self.statuses[index])

    def events(self, remote_job_id: str, after_sequence: int) -> list[dict[str, Any]]:
        if not self.events_by_call:
            return []
        index = min(self._event_index, len(self.events_by_call) - 1)
        self._event_index += 1
        return [dict(item) for item in self.events_by_call[index]]

    def cancel(self, remote_job_id: str, reason: str) -> dict[str, Any]:
        self.cancel_requests.append(reason)
        return {"status": "cancel_requested", "remote_job_id": remote_job_id}
