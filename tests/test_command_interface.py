from harness.commands import Command, CommandContext
from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter, bot_display_mode
from harness.modes import Mode
from harness.command_parser import parse_research_command
from harness.orchestrator import ResearchOrchestrator
from harness.planning_dialogue import PlanningDialogueRunner
from harness.state import ResearchSession


def test_single_command_entrypoint_tracks_context(tmp_path):
    orchestrator = ResearchOrchestrator(
        config=HarnessConfig(project_root=tmp_path),
        discord=FakeDiscordAdapter(),
    )
    result = orchestrator.handle(
        Command("goal", {"text": "Command DTOを検証する"}),
        CommandContext(actor="sora", source="test", correlation_id="c-1"),
    )
    assert result.ok
    assert result.mode == "PLANNING"
    assert result.data["context"]["actor"] == "sora"


def test_unknown_command_returns_failed_result(tmp_path):
    orchestrator = ResearchOrchestrator(
        config=HarnessConfig(project_root=tmp_path),
        discord=FakeDiscordAdapter(),
    )
    result = orchestrator.handle(Command("nope"))
    assert not result.ok
    assert "unknown command" in result.message


def test_phase_gate_commands_parse():
    assert parse_research_command("/re accept PG-1").name == "accept"
    revised = parse_research_command("/re revise PG-1 比較対象を変える")
    assert revised.name == "revise"
    assert revised.args["gate_id"] == "PG-1"
    assert revised.args["reason"] == "比較対象を変える"
    try:
        parse_research_command("/re discuss コスト重視で進めたい")
    except ValueError as exc:
        assert "unknown /re command" in str(exc)
    else:
        raise AssertionError("/re discuss should not be accepted")
    try:
        parse_research_command("/re scout")
    except ValueError as exc:
        assert "unknown /re command" in str(exc)
    else:
        raise AssertionError("/re scout should not be accepted")
    assert parse_research_command("/re new").name == "new_session"
    assert parse_research_command("/re plan").name == "enter_plan"
    try:
        parse_research_command("/re plan 小型LLMの研究支援")
    except ValueError as exc:
        assert "does not accept text" in str(exc)
    else:
        raise AssertionError("/re plan <text> should not be accepted")
    try:
        parse_research_command("/research start")
    except ValueError as exc:
        assert "expected /re" in str(exc)
    else:
        raise AssertionError("/research should not be accepted")


def test_planning_dialogue_extracts_llm_search_queries():
    output = """
SEARCH_NEEDED: yes
SEARCH_QUERY: research agent novelty detection literature review
SEARCH_QUERY: autonomous research agent paper search evaluation
REASON: 類似研究確認が必要
"""
    queries = PlanningDialogueRunner.extract_search_queries(output)
    assert queries == [
        "research agent novelty detection literature review",
        "autonomous research agent paper search evaluation",
    ]


def test_bot_display_mode_mapping():
    assert bot_display_mode(None) == "Neutral"
    session = ResearchSession.new("mode test")
    session.phase = "plan"
    assert bot_display_mode(session) == "plan"
    session.mode = Mode.RESEARCH
    assert bot_display_mode(session) == "researching"
    session.mode = Mode.APPROVAL_BLOCKED
    assert bot_display_mode(session) == "blocked"
    session.mode = Mode.DONE
    assert bot_display_mode(session) == "Neutral"
