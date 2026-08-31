from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from harness.config import HarnessConfig


@dataclass(frozen=True)
class PlatformConfig:
    project_root: Path
    data_dir: Path
    database_path: Path
    core_host: str
    core_port: int
    core_token: str
    core_url: str
    max_concurrent_jobs: int
    scheduler_poll_seconds: float

    codex_command: str
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str
    openai_organization: str | None
    openai_project: str | None
    openai_computer_tool: dict[str, Any]
    computer_use_enabled: bool
    computer_use_start_url: str
    computer_use_allowed_domains: tuple[str, ...]
    computer_use_headless: bool

    kaggle_command: str
    kaggle_api_token: str | None
    kaggle_username: str | None
    allow_kaggle_for_research: bool
    remote_workers: tuple[dict[str, Any], ...]
    paid_backends: tuple[str, ...]

    discord_bot_token: str | None
    discord_work_sessions_channel_id: str | None
    discord_approvals_channel_id: str | None
    discord_ops_channel_id: str | None
    discord_allowed_user_ids: tuple[str, ...]
    discord_event_poll_seconds: float

    @classmethod
    def from_env(
        cls,
        project_root: str | Path | None = None,
    ) -> "PlatformConfig":
        harness = HarnessConfig.from_env(project_root)
        root = harness.project_root
        data_dir = Path(os.getenv("RESEARCH_AGENT_DATA_DIR", str(root / "runtime"))).expanduser()
        if not data_dir.is_absolute():
            data_dir = (root / data_dir).resolve()
        else:
            data_dir = data_dir.resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        database_raw = os.getenv("RESEARCH_AGENT_DATABASE", "platform.sqlite3")
        database_path = Path(database_raw).expanduser()
        if not database_path.is_absolute():
            database_path = data_dir / database_path
        database_path = database_path.resolve()

        host = os.getenv("RESEARCH_AGENT_CORE_HOST", "0.0.0.0")
        port = _int("RESEARCH_AGENT_CORE_PORT", 8080, minimum=1, maximum=65535)
        public_host = os.getenv("RESEARCH_AGENT_CORE_PUBLIC_HOST", "127.0.0.1")
        core_url = os.getenv(
            "RESEARCH_AGENT_CORE_URL",
            f"http://{public_host}:{port}",
        ).rstrip("/")
        core_token = os.getenv("RESEARCH_AGENT_CORE_TOKEN", "").strip()

        computer_tool = _json_object("OPENAI_COMPUTER_TOOL_JSON")
        remote_workers = _json_list("REMOTE_COMPUTE_WORKERS_JSON")
        allowed_users = _csv("DISCORD_ALLOWED_USER_IDS")
        allowed_domains = _csv("COMPUTER_USE_ALLOWED_DOMAINS")

        return cls(
            project_root=root,
            data_dir=data_dir,
            database_path=database_path,
            core_host=host,
            core_port=port,
            core_token=core_token,
            core_url=core_url,
            max_concurrent_jobs=_int("MAX_CONCURRENT_COMPUTE_JOBS", 2, minimum=1, maximum=64),
            scheduler_poll_seconds=_float("COMPUTE_POLL_SECONDS", 15.0, minimum=0.2),
            codex_command=os.getenv("CODEX_COMMAND", "codex").strip() or "codex",
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_organization=os.getenv("OPENAI_ORGANIZATION") or None,
            openai_project=os.getenv("OPENAI_PROJECT") or None,
            openai_computer_tool=computer_tool,
            computer_use_enabled=_bool("COMPUTER_USE_ENABLED", False),
            computer_use_start_url=os.getenv("COMPUTER_USE_START_URL", "about:blank"),
            computer_use_allowed_domains=allowed_domains,
            computer_use_headless=_bool("COMPUTER_USE_HEADLESS", True),
            kaggle_command=os.getenv("KAGGLE_COMMAND", "kaggle").strip() or "kaggle",
            kaggle_api_token=os.getenv("KAGGLE_API_TOKEN") or None,
            kaggle_username=os.getenv("KAGGLE_USERNAME") or None,
            allow_kaggle_for_research=_bool("ALLOW_KAGGLE_FOR_RESEARCH", False),
            remote_workers=tuple(
                dict(item) for item in remote_workers if isinstance(item, Mapping)
            ),
            paid_backends=_csv("PAID_COMPUTE_BACKENDS") or ("gpu_vm",),
            discord_bot_token=os.getenv("DISCORD_BOT_TOKEN") or None,
            discord_work_sessions_channel_id=(
                os.getenv("DISCORD_WORK_SESSIONS_CHANNEL_ID") or None
            ),
            discord_approvals_channel_id=(
                os.getenv("DISCORD_APPROVALS_CHANNEL_ID") or None
            ),
            discord_ops_channel_id=os.getenv("DISCORD_OPS_CHANNEL_ID") or None,
            discord_allowed_user_ids=allowed_users,
            discord_event_poll_seconds=_float(
                "DISCORD_EVENT_POLL_SECONDS", 2.0, minimum=0.5
            ),
        )

    def validate_core(self) -> list[str]:
        errors: list[str] = []
        if not self.core_token:
            errors.append("RESEARCH_AGENT_CORE_TOKEN is required")
        if not self.openai_api_key and not self.codex_command:
            errors.append("Configure OPENAI_API_KEY or CODEX_COMMAND")
        return errors

    def validate_edge(self) -> list[str]:
        errors: list[str] = []
        if not self.discord_bot_token:
            errors.append("DISCORD_BOT_TOKEN is required")
        if not self.discord_work_sessions_channel_id:
            errors.append("DISCORD_WORK_SESSIONS_CHANNEL_ID is required")
        if not self.core_token:
            errors.append("RESEARCH_AGENT_CORE_TOKEN is required")
        if not self.discord_allowed_user_ids:
            errors.append("DISCORD_ALLOWED_USER_IDS must include at least one user")
        return errors


def _csv(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


def _float(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value)


def _json_object(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _json_list(name: str) -> list[Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return list(value) if isinstance(value, list) else []
