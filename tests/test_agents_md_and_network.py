import json

from harness.agent_runner import SubAgentCommandRunner
from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.doctor import _gcloud_adc_check
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.state import ResearchSession


def _journal_events(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_write_agents_md_is_deterministic_and_includes_goal_criteria(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())
    session = ResearchSession.new("Gemini API experiment")
    session.goal_criteria.append("正答率がベースラインを5ポイント上回る")
    session.research_methods.append("Gemini API via ADCで比較実験")
    _, research_dir = config.allocate_research_dir(session.session_id, session.research_goal)
    session.research_dir = str(research_dir)
    orchestrator.store.save(session)

    path = orchestrator._write_agents_md(session)
    first_mtime = path.stat().st_mtime_ns
    first_events = _journal_events(config.journal_path(session.session_id))

    second = orchestrator._write_agents_md(session)
    second_events = _journal_events(config.journal_path(session.session_id))

    assert second == path
    assert path.exists()
    assert "正答率がベースラインを5ポイント上回る" in path.read_text(encoding="utf-8")
    assert path.stat().st_mtime_ns == first_mtime
    assert len(second_events) == len(first_events)
    assert sum(event.get("event_type") == "agents_md_updated" for event in second_events) == 1


def test_start_writes_agents_md(tmp_path):
    config = HarnessConfig(project_root=tmp_path, max_rounds=0)
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())
    session = ResearchSession.new("agent handoff goal")
    session.goal_criteria.append("AGENTS.mdが生成される")
    session.research_methods.append("MockAgentRunnerで開始確認")
    _, research_dir = config.allocate_research_dir(session.session_id, session.research_goal)
    session.research_dir = str(research_dir)
    orchestrator.store.save(session)

    message = orchestrator.start()
    updated = orchestrator.store.load()

    assert updated is not None
    assert "RESEARCH" in message or "完了" in message or "round" in message
    assert (config.session_dir(updated.session_id) / "AGENTS.md").exists()


def test_research_network_access_adds_codex_sandbox_network_config(tmp_path):
    session = ResearchSession.new("network goal")
    session.research_dir = str(tmp_path)

    enabled = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command="codex",
        research_network_access=True,
    )
    enabled_command = SubAgentCommandRunner(enabled)._build_command(session)

    assert "-c" in enabled_command
    assert "sandbox_workspace_write.network_access=true" in enabled_command
    assert enabled_command.index("-c") > enabled_command.index("workspace-write")

    disabled = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command="codex",
        research_network_access=False,
    )
    disabled_command = SubAgentCommandRunner(disabled)._build_command(session)

    assert "sandbox_workspace_write.network_access=true" not in disabled_command


def test_doctor_gcloud_adc_missing_command_is_ng_without_exception():
    check = _gcloud_adc_check(command_name="definitely-missing-gcloud-for-test")

    assert check.name == "gcloud_adc"
    assert check.ok is False
    assert "ADC未設定" in check.detail
