import sys

from harness.agent_runner import MockAgentRunner, RoundOutput, SubAgentCommandRunner
from harness.agent_runner import parse_approval_required
from harness.approval import ApprovalGate, ProposedOperation
from harness.config import HarnessConfig
from harness.config import effective_model_name
from harness.discord_adapter import FakeDiscordAdapter
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.state import ResearchSession


def test_approval_policy_blocks_delete_but_not_long_running_notice():
    gate = ApprovalGate()
    assert gate.classify(
        ProposedOperation(operation="delete_folder: research_runs/V001.0_RS-x")
    ) == "approval_required"
    assert gate.classify(
        ProposedOperation(operation="long_running_command: pytest full suite")
    ) == "important_notice"
    assert gate.classify(
        ProposedOperation(operation="mass_file_generation: write many plots")
    ) == "important_notice"
    assert gate.classify(
        ProposedOperation(operation="sudo apt install graphviz")
    ) == "approval_required"


def test_sub_agent_approval_required_line_is_parsed():
    operation = parse_approval_required(
        "APPROVAL_REQUIRED: operation=sudo apt install graphviz; reason=依存が必要; impact=システム変更; dry_run_result=未実行"
    )
    assert operation is not None
    assert operation.operation == "sudo apt install graphviz"
    assert operation.reason == "依存が必要"


def test_sub_agent_command_runner_uses_generic_command_and_counts_call(tmp_path):
    session = ResearchSession.new("real sub test")
    session.research_dir = str(tmp_path)
    config = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command=f"{sys.executable} -c \"import sys; print('sub-ok'); print(len(sys.stdin.read()) > 0)\"",
        max_command_seconds=10,
    )
    output = SubAgentCommandRunner(config).run(session, round_number=1, task="test task")
    assert "sub-ok" in output
    assert "True" in output
    assert session.cost.agent_calls == 1


def test_real_sub_agent_round_does_not_create_dummy_danger_operation(tmp_path):
    session = ResearchSession.new("real research")
    session.research_dir = str(tmp_path)
    config = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command=f"{sys.executable} -c \"import sys; data=sys.stdin.read(); print('real-sub'); print('round1_notes.md' in data)\"",
        max_command_seconds=10,
        max_rounds=2,
    )
    output = MockAgentRunner(config).run_round(session)
    assert output.proposed_operation is None
    assert output.claude_consultation is None
    assert "delete_file: /tmp/research-harness-dangerous-demo.txt" not in output.sub_agent_output
    assert "Claude stub" not in output.main_agent_summary
    assert "real-sub" in output.sub_agent_output
    assert "True" in output.sub_agent_output
    assert output.next_action == "次ラウンドで続きを進める"


def test_build_round_task_has_three_stages(tmp_path):
    session = ResearchSession.new("staged research")
    real_runner = MockAgentRunner(HarnessConfig(project_root=tmp_path, sub_agent_command="codex"))
    first = real_runner._build_round_task(session, round_number=1, max_rounds=3)
    middle = real_runner._build_round_task(session, round_number=2, max_rounds=3)
    final = real_runner._build_round_task(session, round_number=3, max_rounds=3)
    assert "調査・実験準備" in first
    assert "artifacts/round1_notes.md" in first
    assert "前ラウンドの成果物と課題" in middle
    assert "artifacts/round2_notes.md" in middle
    assert "図・グラフを含む最終レポート" in final
    assert "artifacts/report.md" in final

    mock_runner = MockAgentRunner(HarnessConfig(project_root=tmp_path))
    mock = mock_runner._build_round_task(session, round_number=1, max_rounds=3)
    assert mock == f"Mock sub-agent: {mock_runner.prompts['sub']}"


def test_codex_command_is_built_safely(tmp_path):
    session = ResearchSession.new("codex command test")
    session.research_dir = str(tmp_path)
    config = HarnessConfig(project_root=tmp_path, sub_agent_command="codex")
    command = SubAgentCommandRunner(config)._build_command(session)
    assert command[:2] == ["codex", "exec"]
    assert "--cd" in command
    assert str(tmp_path) in command
    assert "--sandbox" in command
    assert "workspace-write" in command
    assert "--ask-for-approval" not in command
    assert "-m" in command
    assert effective_model_name(config.research_agent_model) in command
    assert "--ephemeral" in command


class NoticeRunner:
    def run_round(self, session):
        return RoundOutput(
            main_agent_summary="notice test",
            subtask="sub",
            sub_agent_output="ok",
            review_output="ok",
            claude_consultation=None,
            fresh_agent_output=None,
            conversation_sessions=[],
            proposed_operation=ProposedOperation(
                operation="long_running_command: extended benchmark",
                reason="ユーザー方針により重要チャンネル通知後に許可する。",
                impact="実行時間が長くなる可能性。",
                dry_run_result="通知のみ。",
            ),
            accepted_ideas=[],
            rejected_ideas=[],
            decision="allowed",
            confidence="mid",
            next_action="continue",
        )


def test_long_running_operation_reports_to_important_channel_without_blocking(tmp_path):
    config = HarnessConfig(project_root=tmp_path, max_rounds=1)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord, runner=NoticeRunner())
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "notice policy")
    discord.inject(orchestrator, "/re accept PG-1")
    discord.inject(orchestrator, "/re start")  # ゴール条件未確定のソフト警告
    result = discord.inject(orchestrator, "/re start")
    assert result.ok
    session = orchestrator.store.load()
    assert session.mode == Mode.DONE
    assert any("重要通知" in message for message in discord.important_messages)
    assert any("important_notice_sent" in message for message in discord.log_messages)


class SudoRequestRunner:
    def run_round(self, session):
        return RoundOutput(
            main_agent_summary="sudo request test",
            subtask="sub",
            sub_agent_output=(
                "APPROVAL_REQUIRED: operation=sudo apt install graphviz; "
                "reason=レポート図生成に必要; impact=システムパッケージ変更; dry_run_result=未実行"
            ),
            review_output="sudoは承認待ちにする。",
            claude_consultation=None,
            fresh_agent_output=None,
            conversation_sessions=[],
            proposed_operation=parse_approval_required(
                "APPROVAL_REQUIRED: operation=sudo apt install graphviz; "
                "reason=レポート図生成に必要; impact=システムパッケージ変更; dry_run_result=未実行"
            ),
            accepted_ideas=[],
            rejected_ideas=[],
            decision="approval required",
            confidence="high",
            next_action="承認待ち",
        )


def test_sudo_operation_can_be_approved_from_discord_command(tmp_path):
    config = HarnessConfig(project_root=tmp_path, max_rounds=2)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord, runner=SudoRequestRunner())
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "sudo policy")
    discord.inject(orchestrator, "/re accept PG-1")
    discord.inject(orchestrator, "/re start")  # ゴール条件未確定のソフト警告
    blocked = discord.inject(orchestrator, "/re start")
    assert blocked.ok
    session = orchestrator.store.load()
    assert session.mode == Mode.APPROVAL_BLOCKED
    assert session.approval_requests["AP-1"].operation == "sudo apt install graphviz"
    assert any("/re approve AP-1" in message for message in discord.important_messages)

    approved = discord.inject(orchestrator, "/re approve AP-1")
    assert approved.ok
    session = orchestrator.store.load()
    assert "AP-1" in session.approvals_received
