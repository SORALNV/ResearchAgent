from __future__ import annotations

import shlex
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
        DoctorCheck("agent_home_mode", config.agent_home_mode == "isolated", config.agent_home_mode),
        DoctorCheck(
            "agent_env_allowlist",
            not any(item.upper().startswith("DISCORD_") for item in config.agent_env_allowlist),
            ", ".join(config.agent_env_allowlist) or "empty (secure default)",
        ),
        DoctorCheck("checkpoint_enabled", config.checkpoint_enabled, str(config.checkpoint_enabled)),
        DoctorCheck("artifact_promotion_enabled", config.artifact_promotion_enabled, str(config.artifact_promotion_enabled)),
        DoctorCheck(
            "multi_agent_limits",
            config.sub_agent_count >= 1 and config.agent_parallelism >= 1,
            (
                f"subs={config.sub_agent_count}, parallelism={config.agent_parallelism}, "
                f"review_retries={config.max_review_retries}, protocol_retries={config.max_protocol_retries}"
            ),
        ),
        _command_check("codex", ["codex", "--version"]),
    ]
    for role, command_text in (
        ("main_agent_command", config.main_agent_command),
        ("sub_agent_command", config.sub_agent_command),
        ("review_agent_command", config.review_agent_command),
        ("fresh_agent_command", config.fresh_agent_command),
        ("claude_agent_command", config.claude_agent_command),
    ):
        if command_text:
            executable = shlex.split(command_text)[0]
            checks.append(_command_check(f"{role}:{executable}", [executable, "--version"]))
        else:
            checks.append(DoctorCheck(role, False, "未設定 (role fallback may apply)"))
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
