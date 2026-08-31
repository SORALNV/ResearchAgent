from __future__ import annotations

from pathlib import Path


def build_service(project_root: str | Path | None = None):
    """Build PlatformService with portable hardened implementations.

    The initial PlatformService module intentionally keeps imports simple. This
    bootstrap replaces the local backend and Codex adapter before constructing
    the service, avoiding host-specific code paths while preserving one service
    implementation for Windows containers and Jetson.
    """

    import harness.platform.service as service_module
    from harness.compute.portable_local import PortableLocalProcessBackend
    from harness.runtime.verified_codex import VerifiedCodexCliRuntime

    service_module.LocalProcessBackend = PortableLocalProcessBackend
    service_module.CodexCliRuntime = VerifiedCodexCliRuntime
    return service_module.PlatformService.from_env(project_root)
