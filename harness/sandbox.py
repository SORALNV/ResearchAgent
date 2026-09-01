from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import Mapping

from harness.config import HarnessConfig


class SandboxUnavailableError(RuntimeError):
    pass


def build_agent_command(
    config: HarnessConfig,
    *,
    command_text: str,
    sandbox_mode: str,
    working_dir: Path,
    research_root: Path,
    environment: Mapping[str, str],
) -> list[str]:
    """Build an argv-only command with an explicit sandbox policy.

    Codex commands are deliberately rejected here. Codex is a protocol-backed
    provider owned by ``CodexAppServerRuntime``; generic subprocess execution is
    retained only for explicitly configured non-Codex tools.
    """

    parts = shlex.split(command_text)
    if not parts:
        raise ValueError("agent command is empty")

    executable = Path(parts[0]).name.lower()
    if executable in {"codex", "codex.exe"}:
        raise SandboxUnavailableError(
            "direct Codex CLI execution is disabled; use the codex_app_server provider"
        )

    backend = config.agent_sandbox_backend
    if backend == "auto":
        backend = "bwrap" if os.name != "nt" and shutil.which("bwrap") else "none"

    if backend == "bwrap":
        return _build_bwrap_command(
            config,
            parts=parts,
            sandbox_mode=sandbox_mode,
            working_dir=working_dir,
            research_root=research_root,
            environment=environment,
        )

    if backend != "none":
        raise SandboxUnavailableError(f"unknown sandbox backend: {backend}")
    if not config.agent_allow_unsandboxed_generic:
        raise SandboxUnavailableError(
            "generic agent command requires bubblewrap or "
            "AGENT_ALLOW_UNSANDBOXED_GENERIC=true"
        )
    return parts


def sandbox_capability(config: HarnessConfig) -> tuple[bool, str]:
    backend = config.agent_sandbox_backend
    if backend == "auto":
        if os.name != "nt" and shutil.which("bwrap"):
            return True, "auto -> bwrap"
        if config.agent_allow_unsandboxed_generic:
            return True, "auto -> unsandboxed generic explicitly allowed"
        return False, "auto: bubblewrap not found and unsandboxed generic denied"
    if backend == "bwrap":
        found = os.name != "nt" and bool(shutil.which("bwrap"))
        return found, "bwrap found" if found else "bwrap not found"
    if backend == "none":
        return (
            config.agent_allow_unsandboxed_generic,
            "unsandboxed generic explicitly allowed"
            if config.agent_allow_unsandboxed_generic
            else "unsandboxed generic denied",
        )
    return False, f"unknown backend: {backend}"


def _build_bwrap_command(
    config: HarnessConfig,
    *,
    parts: list[str],
    sandbox_mode: str,
    working_dir: Path,
    research_root: Path,
    environment: Mapping[str, str],
) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxUnavailableError("bubblewrap (bwrap) is not installed")
    if os.name == "nt":
        raise SandboxUnavailableError("bubblewrap backend is not supported on Windows")

    working_dir = working_dir.resolve()
    research_root = research_root.resolve()
    project_root = config.project_root.resolve()

    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
    ]
    if config.agent_network_policy == "deny":
        command.append("--unshare-net")

    for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt", "/nix/store"):
        candidate = Path(path)
        if candidate.exists():
            command.extend(["--ro-bind", path, path])

    for path in (
        "/etc/ssl",
        "/etc/alternatives",
        "/etc/ld.so.cache",
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
        "/etc/hosts",
    ):
        candidate = Path(path)
        if candidate.exists():
            command.extend(["--ro-bind", path, path])

    command.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])

    bound_roots: set[Path] = set()
    for root in (project_root, research_root):
        if root in bound_roots or not root.exists():
            continue
        command.extend(["--ro-bind", str(root), str(root)])
        bound_roots.add(root)

    home = Path(environment.get("HOME", str(working_dir))).resolve()
    home.mkdir(parents=True, exist_ok=True)
    if sandbox_mode == "workspace-write":
        working_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--bind", str(working_dir), str(working_dir)])
    else:
        command.extend(["--ro-bind", str(working_dir), str(working_dir)])
    command.extend(["--bind", str(home), str(home)])

    command.extend(["--chdir", str(working_dir), "--"])
    command.extend(parts)
    return command
