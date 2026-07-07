from pathlib import Path

from harness.config import HarnessConfig


ROOT = Path(__file__).resolve().parents[1]


def test_required_docs_and_env_exist():
    assert (ROOT / "README.md").exists()
    assert (ROOT / ".env.example").exists()
    assert (ROOT / "main.py").exists()
    assert (ROOT / "harness").is_dir()


def test_env_example_contains_required_keys_and_no_obvious_secret_values():
    body = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in [
        "DISCORD_BOT_TOKEN",
        "DISCORD_CHANNEL_ID",
        "DISCORD_IMPORTANT_CHANNEL_ID",
        "DISCORD_LOG_CHANNEL_ID",
        "PROJECT_ROOT",
        "RESEARCH_ARCHIVE_DIR",
        "MAIN_AGENT_COMMAND",
        "SUB_AGENT_COMMAND",
        "REVIEW_AGENT_COMMAND",
        "FRESH_AGENT_COMMAND",
        "CLAUDE_AGENT_COMMAND",
        "PAPER_PROVIDER",
        "MAX_ROUNDS",
        "FRESH_INTERVAL",
        "CONVERGENCE_PATIENCE",
        "REPORT_INTERVAL_SECONDS",
        "MAX_TURNS_PER_CONVERSATION",
        "CONVERSATION_TIMEOUT_SECONDS",
        "MAX_API_CALLS",
        "MAX_TOTAL_TOKENS",
    ]:
        assert f"{key}=" in body
    assert "your-real-token" not in body
    assert "discord_token=" not in body.lower()


def test_readme_documents_setup_commands_and_flow():
    body = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in [
        "Setup",
        "Local Demo",
        "CLI Commands",
        "Discord Bot",
        "journal.jsonl",
        "research_brief.md",
        "approval gate",
        "/re new",
        "/re start",
        "/re stop",
        "/re search",
        "/re papers",
        "/re cost",
        "/re doctor",
        "/re runs",
        "papers.jsonl",
        "golden_questions.jsonl",
        "DISCORD_IMPORTANT_CHANNEL_ID",
        "DISCORD_LOG_CHANNEL_ID",
        "PLANNING",
        "RESEARCH",
    ]:
        assert required in body


def test_config_loads_dotenv_without_printing_values(tmp_path, monkeypatch):
    monkeypatch.delenv("SUB_AGENT_COMMAND", raising=False)
    monkeypatch.delenv("DISCORD_IMPORTANT_CHANNEL_ID", raising=False)
    (tmp_path / ".env").write_text(
        "SUB_AGENT_COMMAND=codex\nDISCORD_IMPORTANT_CHANNEL_ID=123\n",
        encoding="utf-8",
    )
    config = HarnessConfig.from_env(tmp_path)
    assert config.sub_agent_command == "codex"
    assert config.discord_important_channel_id == "123"
