from pathlib import Path

from harness.config import HarnessConfig
from harness.process_manager import build_agent_environment
from harness.state import ResearchSession


def _session(tmp_path: Path) -> ResearchSession:
    session = ResearchSession.new("codex auth portability")
    research_dir = tmp_path / "run"
    research_dir.mkdir()
    session.research_dir = str(research_dir)
    return session


def test_isolated_home_preserves_only_the_mounted_codex_home(tmp_path):
    config = HarnessConfig(project_root=tmp_path, agent_home_mode="isolated")
    session = _session(tmp_path)
    codex_home = tmp_path / "persistent-codex"
    source = {
        "PATH": "/usr/bin",
        "HOME": "/host/home",
        "CODEX_HOME": str(codex_home),
        "DISCORD_BOT_TOKEN": "discord-secret",
        "OPENAI_API_KEY": "openai-secret",
    }

    environment = build_agent_environment(
        config,
        session,
        role="sub",
        stage="sub_execute",
        task_id="S1",
        working_dir=Path(session.research_dir),
        source=source,
    )

    assert environment["CODEX_HOME"] == str(codex_home)
    assert environment["HOME"] != source["HOME"]
    assert Path(environment["HOME"]).is_relative_to(Path(session.research_dir))
    assert "DISCORD_BOT_TOKEN" not in environment
    assert "OPENAI_API_KEY" not in environment


def test_discord_secret_is_denied_even_when_explicitly_allowlisted(tmp_path):
    config = HarnessConfig(
        project_root=tmp_path,
        agent_env_allowlist=("DISCORD_BOT_TOKEN", "OPENAI_API_KEY"),
    )
    session = _session(tmp_path)

    environment = build_agent_environment(
        config,
        session,
        role="sub",
        stage="sub_execute",
        task_id="S1",
        working_dir=Path(session.research_dir),
        source={
            "PATH": "/usr/bin",
            "DISCORD_BOT_TOKEN": "discord-secret",
            "OPENAI_API_KEY": "explicit-openai-secret",
        },
    )

    assert "DISCORD_BOT_TOKEN" not in environment
    assert environment["OPENAI_API_KEY"] == "explicit-openai-secret"
