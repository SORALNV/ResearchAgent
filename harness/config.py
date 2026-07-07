from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class HarnessConfig:
    project_root: Path
    discord_bot_token: str | None = None
    discord_channel_id: str | None = None
    discord_important_channel_id: str | None = None
    discord_log_channel_id: str | None = None
    main_agent_command: str | None = None
    sub_agent_command: str | None = None
    review_agent_command: str | None = None
    fresh_agent_command: str | None = None
    claude_agent_command: str | None = None
    max_rounds: int = 3
    fresh_interval: int = 2
    convergence_patience: int = 2
    report_interval_seconds: int = 60
    max_turns_per_conversation: int = 4
    conversation_timeout_seconds: int = 60
    max_api_calls: int = 0
    max_total_tokens: int = 0
    max_agent_calls: int = 0
    max_command_seconds: int = 300
    paper_provider: str = "fake"
    research_archive_dir: Path | None = None

    @classmethod
    def from_env(
        cls,
        project_root: str | Path | None = None,
        research_archive_dir: str | Path | None = None,
    ) -> "HarnessConfig":
        root = Path(project_root or os.getenv("PROJECT_ROOT", ".")).expanduser().resolve()
        env_file = root / ".env"
        if env_file.exists():
            _load_dotenv(env_file)
        archive_raw = research_archive_dir or os.getenv("RESEARCH_ARCHIVE_DIR", "research_runs")
        archive_path = Path(archive_raw).expanduser()
        if not archive_path.is_absolute():
            archive_path = root / archive_path
        return cls(
            project_root=root,
            discord_bot_token=os.getenv("DISCORD_BOT_TOKEN") or None,
            discord_channel_id=os.getenv("DISCORD_CHANNEL_ID") or None,
            discord_important_channel_id=(
                os.getenv("DISCORD_IMPORTANT_CHANNEL_ID")
                or os.getenv("DISCORD_REPORT_CHANNEL_ID")
                or os.getenv("DISCORD_CHANNEL_ID")
                or None
            ),
            discord_log_channel_id=os.getenv("DISCORD_LOG_CHANNEL_ID") or None,
            main_agent_command=os.getenv("MAIN_AGENT_COMMAND") or None,
            sub_agent_command=os.getenv("SUB_AGENT_COMMAND") or None,
            review_agent_command=os.getenv("REVIEW_AGENT_COMMAND") or None,
            fresh_agent_command=os.getenv("FRESH_AGENT_COMMAND") or None,
            claude_agent_command=os.getenv("CLAUDE_AGENT_COMMAND") or None,
            max_rounds=_int_env("MAX_ROUNDS", 3),
            fresh_interval=_int_env("FRESH_INTERVAL", 2),
            convergence_patience=_int_env("CONVERGENCE_PATIENCE", 2),
            report_interval_seconds=_int_env("REPORT_INTERVAL_SECONDS", 60),
            max_turns_per_conversation=_int_env("MAX_TURNS_PER_CONVERSATION", 4),
            conversation_timeout_seconds=_int_env("CONVERSATION_TIMEOUT_SECONDS", 60),
            max_api_calls=_int_env("MAX_API_CALLS", 0),
            max_total_tokens=_int_env("MAX_TOTAL_TOKENS", 0),
            max_agent_calls=_int_env("MAX_AGENT_CALLS", 0),
            max_command_seconds=_int_env("MAX_COMMAND_SECONDS", 300),
            paper_provider=os.getenv("PAPER_PROVIDER", "fake"),
            research_archive_dir=archive_path,
        )

    @property
    def state_path(self) -> Path:
        return self.project_root / "state.json"

    @property
    def sessions_dir(self) -> Path:
        return self.research_archive_path

    def session_dir(self, session_id: str) -> Path:
        legacy = self.project_root / "sessions" / session_id
        if legacy.exists():
            return legacy
        for candidate in self.research_archive_path.glob(f"V*_{session_id}*"):
            if candidate.is_dir():
                return candidate
        return self.research_archive_path / session_id

    @property
    def research_archive_path(self) -> Path:
        return self.research_archive_dir or (self.project_root / "research_runs")

    @property
    def legacy_sessions_dir(self) -> Path:
        return self.project_root / "sessions"

    def allocate_research_dir(self, session_id: str, goal: str) -> tuple[str, Path]:
        self.research_archive_path.mkdir(parents=True, exist_ok=True)
        next_major = 1
        pattern = re.compile(r"^V(\d{3})\.0")
        for child in self.research_archive_path.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                next_major = max(next_major, int(match.group(1)) + 1)
        version_label = f"V{next_major:03d}.0"
        slug = _slugify(goal)
        folder = self.research_archive_path / f"{version_label}_{session_id}_{slug}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "artifacts").mkdir()
        return version_label, folder

    def session_state_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "state.json"

    def journal_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "journal.jsonl"

    def research_brief_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "research_brief.md"

    def papers_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "papers.jsonl"

    def research_ledger_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "research_ledger.jsonl"

    def artifacts_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "artifacts"

    def report_path(self, session_id: str) -> Path:
        return self.artifacts_dir(session_id) / "report.md"

    def run_summary_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "run_summary.md"

    @property
    def golden_questions_path(self) -> Path:
        return self.project_root / "eval" / "golden_questions.jsonl"


def _slugify(value: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    if not slug:
        slug = "research"
    return slug[:max_length].strip("-") or "research"


def _load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
