from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.orchestrator import ResearchOrchestrator


def test_doctor_reports_core_checks(tmp_path):
    config = HarnessConfig(
        project_root=tmp_path,
        discord_important_channel_id="important",
        discord_log_channel_id="log",
        sub_agent_command="codex",
    )
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())
    result = orchestrator.doctor()
    assert "ResearchAgent doctor" in result
    assert "project_root" in result
    assert "research_archive_dir" in result
    assert "important_channel" in result
    assert "log_channel" in result
    assert "codex" in result


def test_runs_lists_versioned_research_folders(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "first run")
    discord.inject(orchestrator, "/re stop")
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "second run")

    runs = discord.inject(orchestrator, "/re runs")
    assert runs.ok
    assert "Research runs:" in runs.message
    assert "V001.0" in runs.message
    assert "V002.0" in runs.message
    assert "first run" in runs.message
    assert "second run" in runs.message
