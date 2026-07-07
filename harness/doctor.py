from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from harness.config import HarnessConfig


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


def run_doctor(config: HarnessConfig) -> list[DoctorCheck]:
    checks = [
        DoctorCheck("project_root", config.project_root.exists(), str(config.project_root)),
        DoctorCheck("research_archive_dir", _ensure_dir(config.research_archive_path), str(config.research_archive_path)),
        DoctorCheck("important_channel", bool(config.discord_important_channel_id), config.discord_important_channel_id or "未設定"),
        DoctorCheck("log_channel", bool(config.discord_log_channel_id), config.discord_log_channel_id or "未設定"),
        DoctorCheck("paper_provider", config.paper_provider in {"fake", "arxiv"}, config.paper_provider),
        _command_check("codex", ["codex", "--version"]),
    ]
    if config.sub_agent_command:
        executable = config.sub_agent_command.split()[0]
        checks.append(_command_check(f"sub_agent_command:{executable}", [executable, "--version"]))
    else:
        checks.append(DoctorCheck("sub_agent_command", False, "未設定"))
    return checks


def render_doctor(checks: list[DoctorCheck]) -> str:
    lines = ["ResearchAgent doctor"]
    for check in checks:
        mark = "OK" if check.ok else "NG"
        lines.append(f"- {mark} {check.name}: {check.detail}")
    return "\n".join(lines)


def _ensure_dir(path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    return path.exists() and path.is_dir()


def _command_check(name: str, command: list[str]) -> DoctorCheck:
    if not shutil.which(command[0]):
        return DoctorCheck(name, False, "not found")
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return DoctorCheck(name, False, f"{type(exc).__name__}: {exc}")
    output = (completed.stdout or completed.stderr).strip().splitlines()
    detail = output[0] if output else f"returncode={completed.returncode}"
    return DoctorCheck(name, completed.returncode == 0, detail)

