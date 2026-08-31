from __future__ import annotations

from typing import Mapping

from harness.compute_backends import RemoteGpuBackend
from harness.compute_models import BackendCapabilities
from harness.compute_scheduler import ComputeStack


class HealthAwareRemoteGpuBackend(RemoteGpuBackend):
    """Remote backend that treats the Worker's live inventory as authoritative."""

    def available(self) -> tuple[bool, str]:
        if not self.descriptor.base_url or not self.descriptor.token:
            return False, "remote worker URL or token is missing"
        try:
            response = self._request("GET", "/health")
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        raw_capabilities = response.get("capabilities")
        if isinstance(raw_capabilities, Mapping):
            self.capabilities = BackendCapabilities.from_dict(raw_capabilities)
        return bool(response.get("ok", True)), str(
            response.get("detail") or self.descriptor.base_url
        )


def refresh_remote_backends(stack: ComputeStack) -> ComputeStack:
    """Replace configured RemoteGpuBackend objects with health-aware clients."""

    for name, backend in tuple(stack.broker.backends.items()):
        if not isinstance(backend, RemoteGpuBackend):
            continue
        if isinstance(backend, HealthAwareRemoteGpuBackend):
            continue
        stack.broker.backends[name] = HealthAwareRemoteGpuBackend(
            backend.descriptor,
            transport=backend.transport,
            timeout_seconds=backend.timeout_seconds,
            max_bundle_files=backend.max_bundle_files,
            max_bundle_bytes=backend.max_bundle_bytes,
        )
    return stack
