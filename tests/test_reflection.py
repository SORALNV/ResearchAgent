from harness.agent_runner import FINAL_REPORT_TASK, MockAgentRunner, RoundOutput, SubAgentCommandRunner
from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.report_review import ReviewResult, parse_review_output
from harness.round_director import (
    CriterionStatus,
    DirectorVerdict,
    FailureAnalysis,
    parse_director_output,
)
from harness.state import ResearchSession


def _config(tmp_path, *, max_rounds=3, stall_round_threshold=2):
    return HarnessConfig(
        project_root=tmp_path,
        max_rounds=max_rounds,
        stall_round_threshold=stall_round_threshold,
        sub_agent_command="codex",
    )


def _session(config, *, round_budget=3):
    session = ResearchSession.new("冷静な研究員を検証する")
    version, research_dir = config.allocate_research_dir(session.session_id, session.research_goal)
    session.version_label = version
    session.research_dir = str(research_dir)
    session.goal_criteria = ["C1を満たす", "C2を満たす"]
    session.research_methods = ["小さく実験する"]
    session.transition_to(Mode.RESEARCH)
    session.phase = "research"
    session.round_budget = round_budget
    return session


def _verdict(decision="CONTINUE", *, criteria=None, next_task="次を試す", analysis=None):
    return DirectorVerdict(
        decision=decision,
        criteria_status=criteria
        if criteria is not None
        else [CriterionStatus("C1", "partial", "まだ不足")],
        critique="批評",
        next_task=next_task,
        reason="理由",
        raw="raw",
        source="llm",
        failure_analysis=analysis,
    )


class SequenceDirector:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.stall_notes = []

    def assess(self, *args, **kwargs):
        self.stall_notes.append(kwargs.get("stall_note", ""))
        if len(self.verdicts) == 1:
            return self.verdicts[0]
        return self.verdicts.pop(0)


class RecordingRunner:
    def __init__(self, config):
        self.config = config
        self.builder = MockAgentRunner(config)
        self.subtasks = []

    def run_round(self, session):
        round_number = session.round_id + 1
        subtask = self.builder._build_round_task(
            session,
            round_number,
            session.round_budget or self.config.max_rounds,
        )
        self.subtasks.append(subtask)
        return RoundOutput(
            main_agent_summary=f"R{round_number}",
            subtask=subtask,
            sub_agent_output=f"output {round_number}",
            review_output="review",
            claude_consultation=None,
            fresh_agent_output=None,
            conversation_sessions=[],
            proposed_operation=None,
            accepted_ideas=[],
            rejected_ideas=[],
            decision="採用",
            confidence="high",
            next_action="next",
        )


class StubReviewer:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def review(self, *args, **kwargs):
        self.calls += 1
        if len(self.results) == 1:
            return self.results[0]
        return self.results.pop(0)


def _review(verdict, score=3, major=None):
    return ReviewResult(
        verdict=verdict,
        score=score,
        major_issues=major or [],
        minor_issues=[],
        raw="raw",
        source="llm",
    )


def test_parse_failure_analysis_variants():
    text = """CRITERIA_STATUS:
- C1: partial | 実行エラー
CRITIQUE: エラー原因が未分離
DECISION: CONTINUE
NEXT_TASK: 別ルート
REASON: 失敗
FAILURE_ANALYSIS:
- root_cause: 入力前提が誤っていた
- lesson: 症状と前提誤りを分ける
- alternative: 小さいデータで再現する
"""
    verdict = parse_director_output(text)

    assert verdict.failure_analysis == FailureAnalysis(
        root_cause="入力前提が誤っていた",
        lesson="症状と前提誤りを分ける",
        alternative="小さいデータで再現する",
    )
    assert parse_director_output("CRITERIA_STATUS:\n- C1: partial | x\nDECISION: CONTINUE\n").failure_analysis is None

    partial = parse_director_output(
        "CRITERIA_STATUS:\n- C1: partial | x\nDECISION: CONTINUE\n"
        "FAILURE_ANALYSIS:\n- lesson: まず環境をスモークテストする\n"
    )
    assert partial.failure_analysis.lesson == "まず環境をスモークテストする"
    assert partial.failure_analysis.root_cause == ""


def test_lessons_are_recorded_deduped_and_capped(tmp_path):
    config = _config(tmp_path)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=RecordingRunner(config))
    session = _session(config)

    orchestrator._apply_director_verdict(
        session,
        _verdict(analysis=FailureAnalysis("原因", "同じ教訓", "代替")),
        round_limit=3,
    )
    orchestrator._apply_director_verdict(
        session,
        _verdict(analysis=FailureAnalysis("原因2", "同じ教訓", "代替2")),
        round_limit=3,
    )
    for index in range(25):
        orchestrator._apply_director_verdict(
            session,
            _verdict(analysis=FailureAnalysis("原因", f"lesson-{index}", "代替")),
            round_limit=3,
        )

    assert session.lessons.count("同じ教訓") <= 1
    assert len(session.lessons) == 20
    assert session.lessons[0] == "lesson-5"
    events = orchestrator._journal(session).read_entries()
    assert any(event.get("event_type") == "failure_reflected" for event in events)


def test_sub_prompt_injects_lessons_and_empty_state(tmp_path):
    config = _config(tmp_path)
    session = _session(config)

    prompt = SubAgentCommandRunner(config)._build_prompt(session, 1, "task")
    assert "これまでの教訓（過去の失敗から。同じ轍を踏まない）:" in prompt
    assert "- なし" in prompt

    session.lessons = ["環境を先に確認する", "大きいデータに飛びつかない"]
    prompt = SubAgentCommandRunner(config)._build_prompt(session, 1, "task")
    assert "- 環境を先に確認する" in prompt
    assert "- 大きいデータに飛びつかない" in prompt


def test_stall_detection_adds_pivot_task_and_notification(tmp_path):
    config = _config(tmp_path, max_rounds=2, stall_round_threshold=2)
    discord = FakeDiscordAdapter()
    runner = RecordingRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=discord, runner=runner)
    orchestrator.round_director = SequenceDirector(
        [
            _verdict(criteria=[CriterionStatus("C1", "partial", "不足")]),
            _verdict(criteria=[CriterionStatus("C1", "partial", "まだ不足")]),
        ]
    )
    session = _session(config, round_budget=2)
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)
    saved = orchestrator.store.load()

    assert saved.stall_rounds == 2
    assert "【停滞検知】" in saved.next_round_task
    assert any("進捗停滞を検知" in message for message in discord.important_messages)
    events = orchestrator._journal(saved).read_entries()
    assert any(event.get("event_type") == "stall_detected" for event in events)


def test_stall_rounds_reset_when_progress_appears(tmp_path):
    config = _config(tmp_path, max_rounds=2, stall_round_threshold=2)
    runner = RecordingRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=runner)
    orchestrator.round_director = SequenceDirector(
        [
            _verdict(criteria=[CriterionStatus("C1", "partial", "不足")]),
            _verdict(criteria=[CriterionStatus("C1", "met", "EXP-001で確認")]),
        ]
    )
    session = _session(config, round_budget=2)
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)

    assert orchestrator.store.load().stall_rounds == 0


def test_report_review_revise_then_done_after_revision_limit(tmp_path):
    config = _config(tmp_path, max_rounds=3)
    discord = FakeDiscordAdapter()
    runner = RecordingRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=discord, runner=runner)
    orchestrator.round_director = SequenceDirector(
        [
            _verdict(
                "DONE",
                criteria=[
                    CriterionStatus("C1", "met", "EXP-001"),
                    CriterionStatus("C2", "met", "artifacts/report.md"),
                ],
            ),
            _verdict(
                "DONE",
                criteria=[
                    CriterionStatus("C1", "met", "EXP-001"),
                    CriterionStatus("C2", "met", "artifacts/report.md"),
                ],
            ),
        ]
    )
    orchestrator.report_reviewer = StubReviewer(
        [_review("REVISE", score=2, major=["C2の図表根拠が不足"])]
    )
    session = _session(config, round_budget=3)
    config.report_path(session.session_id).write_text("# Report\n", encoding="utf-8")
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)
    saved = orchestrator.store.load()

    assert saved.mode == Mode.DONE
    assert saved.report_review_rounds == 1
    assert "査読指摘を解消" not in saved.next_round_task
    assert orchestrator.report_reviewer.calls == 1
    assert any("査読で差し戻し" in message for message in discord.important_messages)


def test_report_review_accept_finishes_early_with_message(tmp_path):
    config = _config(tmp_path, max_rounds=3)
    discord = FakeDiscordAdapter()
    runner = RecordingRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=discord, runner=runner)
    orchestrator.round_director = SequenceDirector(
        [
            _verdict(
                "DONE",
                criteria=[
                    CriterionStatus("C1", "met", "EXP-001"),
                    CriterionStatus("C2", "met", "artifacts/report.md"),
                ],
            )
        ]
    )
    orchestrator.report_reviewer = StubReviewer([_review("ACCEPT", score=4)])
    session = _session(config, round_budget=3)
    config.report_path(session.session_id).write_text("# Report\n", encoding="utf-8")
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)

    assert orchestrator.store.load().mode == Mode.DONE
    assert any("査読: ACCEPT" in message for message in discord.important_messages)


def test_report_review_parser_variants():
    result = parse_review_output(
        """SCORE: 4
MAJOR_ISSUES:
- 根拠Aが弱い
MINOR_ISSUES:
- 表記ゆれ
VERDICT: REVISE
"""
    )
    assert result.verdict == "REVISE"
    assert result.score == 4
    assert result.major_issues == ["根拠Aが弱い"]
    assert result.minor_issues == ["表記ゆれ"]

    fallback = parse_review_output("SCORE: 3\nMAJOR_ISSUES:\n- x\n")
    assert fallback.verdict == "ACCEPT"
    assert fallback.source == "fallback"

    empty = parse_review_output(
        "SCORE: 5\nMAJOR_ISSUES:\n- なし\nMINOR_ISSUES:\n- なし\nVERDICT: ACCEPT\n"
    )
    assert empty.major_issues == []
    assert empty.minor_issues == []


def test_config_defaults_for_reflection_controls(tmp_path):
    config = HarnessConfig(project_root=tmp_path)

    assert config.stall_round_threshold == 2
    assert config.reviewer_model == ""
    assert config.max_report_revisions == 1
