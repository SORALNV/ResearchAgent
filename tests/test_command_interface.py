from harness.commands import Command, CommandContext
from harness.config import HarnessConfig, effective_model_name
from harness.discord_adapter import FakeDiscordAdapter, bot_display_mode
from harness.modes import Mode
from harness.command_parser import parse_research_command
from harness.orchestrator import ResearchOrchestrator
from harness.planning_dialogue import PlanningDialogueRunner, parse_plan_update
from harness.state import ResearchSession


class StaticPlanningDialogue:
    def propose_search_queries(self, session, user_text: str, purpose: str) -> list[str]:
        return []

    def respond(self, session, user_text: str, purpose: str, image_paths=None) -> str:
        return "短い壁打ち返答です。"


class StructuredPlanningDialogue:
    def propose_search_queries(self, session, user_text: str, purpose: str) -> list[str]:
        return []

    def respond(self, session, user_text: str, purpose: str, image_paths=None) -> str:
        return """ここは表示される壁打ち本文です。

PLAN_UPDATE:
GOAL_CRITERIA: 比較表と最小プロトタイプで有効性を確認できたら成功
METHOD: 既存研究との差分表を作る
OPEN_QUESTION: 評価対象データセットを何にするか
OPEN_QUESTION: 成果物をどこまで実装するか
RESOLVED_QUESTION: 成果物をどこまで実装するか
"""


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


def test_parse_plan_update_no_block():
    display, updates = parse_plan_update("普通の返答です。")
    assert display == "普通の返答です。"
    assert updates == {}


def test_parse_plan_update_extracts_multiple_lines():
    display, updates = parse_plan_update(
        """本文です。

PLAN_UPDATE:
GOAL_CRITERIA: レポートが完成している
GOAL_CRITERIA: 再現手順がある
METHOD: 論文比較
OPEN_QUESTION: 評価軸は何か
RESOLVED_QUESTION: 古い確認事項
"""
    )
    assert display == "本文です。"
    assert updates == {
        "goal_criteria": ["レポートが完成している", "再現手順がある"],
        "research_methods": ["論文比較"],
        "open_questions": ["評価軸は何か"],
        "resolved_questions": ["古い確認事項"],
    }


def test_parse_plan_update_ignores_body_before_block():
    display, updates = parse_plan_update(
        "本文中の GOAL_CRITERIA: これはただの会話\nPLAN_UPDATE:\nMETHOD: 実験ログを残す"
    )
    assert "GOAL_CRITERIA" in display
    assert updates == {"research_methods": ["実験ログを残す"]}


def test_planning_dialogue_codex_command_matches_current_cli(tmp_path):
    config = HarnessConfig(project_root=tmp_path, sub_agent_command="codex")
    runner = PlanningDialogueRunner(config)
    session = ResearchSession.new("command test")
    session.research_dir = str(tmp_path)
    command = runner._build_command("codex", session)
    assert "--ask-for-approval" not in command
    assert "-m" in command
    assert effective_model_name(config.plan_agent_model) in command
    assert "--sandbox" in command
    assert "--ephemeral" in command
    assert command[-1] == "-"


def test_planning_dialogue_adds_images_to_codex_command(tmp_path):
    config = HarnessConfig(project_root=tmp_path, sub_agent_command="codex")
    runner = PlanningDialogueRunner(config)
    session = ResearchSession.new("image command test")
    session.research_dir = str(tmp_path)
    command = runner._build_command("codex", session)
    output_path = runner._last_message_path(session)
    insert_at = command.index("-")
    command[insert_at:insert_at] = ["--output-last-message", str(output_path)]
    for image_path in [str(tmp_path / "sample.png")]:
        command[insert_at:insert_at] = ["--image", image_path]
        insert_at += 2
    assert "--image" in command
    assert str(tmp_path / "sample.png") in command


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


def test_fake_plan_message_is_not_duplicated(tmp_path):
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(
        config=HarnessConfig(project_root=tmp_path, paper_provider="fake"),
        discord=discord,
    )
    orchestrator.planning_dialogue = StaticPlanningDialogue()
    orchestrator.handle(Command("new_session"))
    orchestrator.handle(Command("enter_plan"))
    before = len(discord.important_messages)

    discord.inject_message(orchestrator, "テーマを詰めたいです")

    new_messages = discord.important_messages[before:]
    assert new_messages == ["短い壁打ち返答です。"]


def test_discuss_applies_plan_update_and_strips_block(tmp_path):
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(
        config=HarnessConfig(project_root=tmp_path, paper_provider="fake"),
        discord=discord,
    )
    orchestrator.planning_dialogue = StructuredPlanningDialogue()
    orchestrator.handle(Command("new_session"))
    orchestrator.handle(Command("enter_plan"))

    result = discord.inject_message(orchestrator, "ゴール条件と方法を決めたい")

    assert result.ok
    assert "PLAN_UPDATE" not in result.message
    assert "📌 プラン更新: ゴール条件+1, 方法+1, 確認事項+2, 確認事項-1" in result.message
    session = orchestrator.store.load()
    assert session.goal_criteria == ["比較表と最小プロトタイプで有効性を確認できたら成功"]
    assert session.research_methods == ["既存研究との差分表を作る"]
    assert session.open_questions == ["評価対象データセットを何にするか"]
    assert any("PLAN_UPDATE" not in item for item in session.accepted_ideas)
    journal = (
        orchestrator.config.journal_path(session.session_id)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert any("plan_structure_updated" in line for line in journal)
