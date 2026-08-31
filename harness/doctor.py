from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.config import HarnessConfig
from harness.provider_executor import RuntimeSettings
from harness.sandbox import sandbox_capability


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


def run_doctor(config: HarnessConfig) -> list[DoctorCheck]:
    sandbox_ok, sandbox_detail = sandbox_capability(config)
    runtime = RuntimeSettings.from_environment()
    selected = set(runtime.global_order)
    for order in runtime.role_orders.values():
        selected.update(order)
    if not selected:
        for command_text in (
            config.main_agent_command,
            config.sub_agent_command,
            config.review_agent_command,
            config.fresh_agent_command,
            config.claude_agent_command,
        ):
            if not command_text:
                continue
            executable = Path(shlex.split(command_text)[0]).name
            selected.add("codex_cli" if executable == "codex" else "cli")

    openai_selected = bool({"openai_responses", "openai_computer"} & selected)
    computer_ready = (
        not runtime.computer_enabled
        or (
            bool(runtime.computer_model)
            and bool(runtime.computer_bridge_url)
            and bool(runtime.computer_allowed_stages)
            and runtime.computer_require_approval
        )
    )
    codex_home = os.getenv("CODEX_HOME")
    codex_home_detail = codex_home or "未設定（Codex既定値）"

    checks = [
        DoctorCheck("project_root", config.project_root.exists(), str(config.project_root)),
        DoctorCheck(
            "research_archive_dir",
            _ensure_dir(config.research_archive_path),
            str(config.research_archive_path),
        ),
        DoctorCheck(
            "runtime_architecture",
            platform.machine().lower() in {"x86_64", "amd64", "aarch64", "arm64"},
            f"machine={platform.machine()}, container={Path('/.dockerenv').exists()}",
        ),
        DoctorCheck(
            "important_channel",
            bool(config.discord_important_channel_id),
            config.discord_important_channel_id or "未設定",
        ),
        DoctorCheck(
            "log_channel",
            bool(config.discord_log_channel_id),
            config.discord_log_channel_id or "未設定",
        ),
        DoctorCheck(
            "paper_provider",
            config.paper_provider in {"fake", "arxiv"},
            config.paper_provider,
        ),
        DoctorCheck(
            "runtime_provider_order",
            bool(selected),
            _format_provider_orders(runtime, selected),
        ),
        DoctorCheck(
            "openai_responses",
            (not openai_selected)
            or (bool(runtime.openai_api_key) and bool(runtime.openai_model)),
            (
                "not selected"
                if not openai_selected
                else (
                    f"api_key={'configured' if runtime.openai_api_key else 'missing'}, "
                    f"model={runtime.openai_model or 'missing'}, "
                    f"base_url={'custom' if runtime.openai_base_url else 'default'}"
                )
            ),
        ),
        DoctorCheck(
            "openai_computer",
            computer_ready,
            (
                "disabled"
                if not runtime.computer_enabled
                else (
                    f"model={runtime.computer_model or 'missing'}, "
                    f"bridge={'configured' if runtime.computer_bridge_url else 'missing'}, "
                    f"stages={','.join(runtime.computer_allowed_stages) or 'missing'}, "
                    f"approval_required={runtime.computer_require_approval}"
                )
            ),
        ),
        DoctorCheck(
            "codex_home",
            ("codex_cli" not in selected) or bool(codex_home),
            codex_home_detail,
        ),
        DoctorCheck(
            "agent_home_mode",
            config.agent_home_mode == "isolated",
            config.agent_home_mode,
        ),
        DoctorCheck(
            "agent_env_allowlist",
            not any(
                item.upper().startswith("DISCORD_")
                for item in config.agent_env_allowlist
            ),
            ", ".join(config.agent_env_allowlist) or "empty (secure default)",
        ),
        DoctorCheck(
            "agent_os_sandbox",
            sandbox_ok,
            (
                f"backend={config.agent_sandbox_backend}, "
                f"network={config.agent_network_policy}, {sandbox_detail}"
            ),
        ),
        DoctorCheck(
            "unsandboxed_generic",
            not (
                config.agent_sandbox_backend == "none"
                and config.agent_allow_unsandboxed_generic
            ),
            str(config.agent_allow_unsandboxed_generic),
        ),
        DoctorCheck(
            "checkpoint_enabled",
            config.checkpoint_enabled,
            str(config.checkpoint_enabled),
        ),
        DoctorCheck(
            "artifact_promotion_enabled",
            config.artifact_promotion_enabled,
            str(config.artifact_promotion_enabled),
        ),
        DoctorCheck(
            "multi_agent_limits",
            config.sub_agent_count >= 1 and config.agent_parallelism >= 1,
            (
                f"subs={config.sub_agent_count}, "
                f"parallelism={config.agent_parallelism}, "
                f"review_retries={config.max_review_retries}, "
                f"protocol_retries={config.max_protocol_retries}"
            ),
        ),
        DoctorCheck(
            "convergence_policy",
            config.convergence_patience >= 0
            and config.convergence_no_evidence_patience >= 0
            and 0.0 <= config.convergence_min_progress <= 1.0,
            (
                f"stagnation_patience={config.convergence_patience}, "
                f"no_evidence_patience={config.convergence_no_evidence_patience}, "
                f"min_progress={config.convergence_min_progress}"
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
            checks.append(
                _command_check(f"{role}:{executable}", [executable, "--version"])
            )
        else:
            checks.append(
                DoctorCheck(role, False, "未設定 (provider/role fallback may apply)")
            )
    return checks


def render_doctor(checks: list[DoctorCheck]) -> str:
    lines = ["ResearchAgent doctor"]
    for check in checks:
        mark = "OK" if check.ok else "NG"
        lines.append(f"- {mark} {check.name}: {check.detail}")
    return "\n".join(lines)


def _format_provider_orders(runtime: RuntimeSettings, selected: set[str]) -> str:
    parts = [
        "global=" + (",".join(runtime.global_order) if runtime.global_order else "inferred")
    ]
    for role in sorted(runtime.role_orders):
        parts.append(f"{role}={','.join(runtime.role_orders[role])}")
    parts.append("selected=" + (",".join(sorted(selected)) or "none"))
    return "; ".join(parts)


def _ensure_dir(path: Path) -> bool:
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
