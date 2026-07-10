import json
from pathlib import Path

from harness.agent_runner import FINAL_REPORT_TASK, MockAgentRunner, RoundOutput, SubAgentCommandRunner
from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.experiments import ExperimentEntry, ExperimentStore
from harness.ledger import ResearchLedger
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.report_review import ReviewResult
from harness.round_director import CriterionStatus, DirectorVerdict, ExperimentUpdate, parse_director_output
from harness.state import ResearchSession, utc_timestamp


def _config(tmp_path, *, max_rounds=3, sub_agent_command="codex"):
    return HarnessConfig(
        project_root=tmp_path,
        max_rounds=max_rounds,
        sub_agent_command=sub_agent_command,
    )


def _research_session(config, *, goal="動画検索を評価する", round_budget=3):
    session = ResearchSession.new(goal)
    version_label, research_dir = config.allocate_research_dir(session.session_id, goal)
    session.version_label = version_label
    session.research_dir = str(research_dir)
    session.goal_criteria = ["自然動画で比較できた", "最終レポートがある"]
    session.research_methods = ["小規模データで定量比較"]
    session.open_questions = ["動画本数が十分か"]
    session.transition_to(Mode.RESEARCH)
    session.phase = "research"
    session.round_budget = round_budget
    return session


def _verdict(decision, *, next_task="", critique="批評", criteria=None, reason="理由"):
    return DirectorVerdict(
        decision=decision,
        criteria_status=criteria
        if criteria is not None
        else [CriterionStatus("C1", "partial", "round1_notes.mdに途中結果")],
        critique=critique,
        next_task=next_task,
        reason=reason,
        raw="raw",
        source="llm",
    )


class SequenceDirector:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def assess(self, *args, **kwargs):
        if len(self.verdicts) == 1:
            return self.verdicts[0]
        return self.verdicts.pop(0)


class AcceptReviewer:
    def review(self, *args, **kwargs):
        return ReviewResult(
            verdict="ACCEPT",
            score=4,
            major_issues=[],
            minor_issues=[],
            raw="raw",
            source="llm",
        )


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


def test_parse_director_output_multiline_sections():
    text = """CRITERIA_STATUS:
- C1: met | artifacts/report.mdに成功率0.83を記載
- C2: partial | EXP-002はあるが自然動画は2本のみ
- C3: UNMET | 比較表が未作成
CRITIQUE: 自然動画の本数が不足している。
単一データセットへの一般化が強い。
DECISION: CONTINUE
NEXT_TASK: 自然動画5本で比較し、EXP-003に記録する。
結果をartifacts/round2_notes.mdへ表で残す。
REASON: C2/C3が未充足のため。
"""

    verdict = parse_director_output(text)

    assert verdict.source == "llm"
    assert verdict.decision == "CONTINUE"
    assert [item.status for item in verdict.criteria_status] == ["met", "partial", "unmet"]
    assert "一般化" in verdict.critique
    assert "自然動画5本" in verdict.next_task
    assert "未充足" in verdict.reason


def test_parse_director_output_invalid_decision_falls_back():
    verdict = parse_director_output("CRITERIA_STATUS:\n- C1: met | ok\nDECISION: MAYBE")

    assert verdict.source == "fallback"
    assert verdict.decision == "CONTINUE"
    assert "invalid or missing decision" in verdict.reason


def test_parse_director_output_experiment_updates():
    text = """CRITERIA_STATUS:
- C1: partial | EXP-001不足
CRITIQUE: 追加検証が必要
DECISION: CONTINUE
NEXT_TASK: 次を実行
REASON: まだ不足
EXPERIMENTS_UPDATE:
- add: natural-video-check | 自然動画で改善する | accuracy >= 0.8 | high
- retire: EXP-002 | 前提が崩れた
"""

    verdict = parse_director_output(text)

    assert len(verdict.experiment_updates) == 2
    assert verdict.experiment_updates[0].op == "add"
    assert verdict.experiment_updates[0].priority == "high"
    assert verdict.experiment_updates[1].op == "retire"
    assert verdict.experiment_updates[1].exp_id == "EXP-002"
    assert parse_director_output(
        "CRITERIA_STATUS:\n- C1: partial | x\nDECISION: CONTINUE\n"
    ).experiment_updates == []


def test_continue_next_task_is_used_by_next_round(tmp_path):
    config = _config(tmp_path, max_rounds=3)
    runner = RecordingRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=runner)
    orchestrator.round_director = SequenceDirector(
        [
            _verdict(
                "CONTINUE",
                next_task="自然動画5本で比較し、表にまとめる。",
                critique="前回は動画本数が不足。",
            ),
            _verdict("BLOCKED", reason="ここで停止"),
        ]
    )
    session = _research_session(config, round_budget=3)
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)

    assert len(runner.subtasks) >= 2
    assert "自然動画5本で比較" in runner.subtasks[1]
    assert "前回は動画本数が不足" in runner.subtasks[1]


def test_done_with_existing_report_finishes_early(tmp_path):
    config = _config(tmp_path, max_rounds=5)
    runner = RecordingRunner(config)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config, discord=discord, runner=runner)
    orchestrator.round_director = SequenceDirector(
        [
            _verdict(
                "DONE",
                criteria=[
                    CriterionStatus("C1", "met", "EXP-001で確認"),
                    CriterionStatus("C2", "met", "artifacts/report.md作成済み"),
                ],
            )
        ]
    )
    orchestrator.report_reviewer = AcceptReviewer()
    session = _research_session(config, round_budget=5)
    config.report_path(session.session_id).write_text("# Report\n", encoding="utf-8")
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)
    saved = orchestrator.store.load()

    assert saved.mode == Mode.DONE
    assert saved.completed_reason == "GOAL_CRITERIA_MET"
    assert saved.round_id == 1
    assert any("早期完了" in message for message in discord.important_messages)


def test_done_without_report_becomes_final_report_task(tmp_path):
    config = _config(tmp_path, max_rounds=1)
    runner = RecordingRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=runner)
    orchestrator.round_director = SequenceDirector(
        [
            _verdict(
                "DONE",
                criteria=[
                    CriterionStatus("C1", "met", "EXP-001で確認"),
                    CriterionStatus("C2", "met", "round1_notes.mdに実測値"),
                ],
            )
        ]
    )
    session = _research_session(config, round_budget=1)
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)
    saved = orchestrator.store.load()

    assert saved.next_round_task == FINAL_REPORT_TASK


def test_evidence_gate_downgrades_met_without_evidence(tmp_path):
    config = _config(tmp_path)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=RecordingRunner(config))
    session = _research_session(config, round_budget=3)
    session.round_id = 1

    orchestrator._apply_director_verdict(
        session,
        _verdict("CONTINUE", criteria=[CriterionStatus("C1", "met", "")]),
        round_limit=3,
    )

    assert session.last_criteria_status[0]["status"] == "partial"
    events = orchestrator._journal(session).read_entries()
    assert any(event.get("event_type") == "evidence_gate_downgraded" for event in events)


def test_done_gate_rejects_unmet_criteria(tmp_path):
    config = _config(tmp_path)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=RecordingRunner(config))
    session = _research_session(config, round_budget=3)
    session.round_id = 1

    orchestrator._apply_director_verdict(
        session,
        _verdict("DONE", criteria=[CriterionStatus("C1", "partial", "まだ不足")]),
        round_limit=3,
    )

    assert session.mode == Mode.RESEARCH
    assert session.next_round_task
    assert session.next_round_task != FINAL_REPORT_TASK
    assert "未達のゴール条件（C1）" in session.next_round_task
    events = orchestrator._journal(session).read_entries()
    assert any(event.get("event_type") == "done_gate_rejected" for event in events)


def test_director_backlog_updates_add_and_retire_planned_only(tmp_path):
    config = _config(tmp_path)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=RecordingRunner(config))
    session = _research_session(config, round_budget=3)
    store = ExperimentStore(config.experiments_path(session.session_id))
    now = utc_timestamp()
    store.append(
        ExperimentEntry(
            exp_id="EXP-001",
            name="planned",
            hypothesis="planned h",
            status="planned",
            round_id=0,
            metrics={},
            dir="",
            created_at=now,
            updated_at=now,
        )
    )
    store.append(
        ExperimentEntry(
            exp_id="EXP-002",
            name="done",
            hypothesis="done h",
            status="done",
            round_id=0,
            metrics={},
            dir="",
            created_at=now,
            updated_at=now,
        )
    )

    orchestrator._apply_director_verdict(
        session,
        DirectorVerdict(
            decision="CONTINUE",
            criteria_status=[CriterionStatus("C1", "partial", "x")],
            critique="",
            next_task="",
            reason="",
            raw="raw",
            source="llm",
            experiment_updates=[
                ExperimentUpdate(
                    op="add",
                    name="new",
                    hypothesis="new h",
                    success_metric="n >= 1",
                    priority="mid",
                ),
                ExperimentUpdate(op="retire", exp_id="EXP-001", reason="obsolete"),
                ExperimentUpdate(op="retire", exp_id="EXP-002", reason="done untouched"),
            ],
        ),
        round_limit=3,
    )

    entries = {entry.exp_id: entry for entry in store.read_all()}
    assert entries["EXP-001"].status == "retired"
    assert entries["EXP-002"].status == "done"
    assert entries["EXP-003"].status == "planned"
    assert entries["EXP-003"].success_metric == "n >= 1"


def test_max_rounds_report_mentions_unmet_criteria_and_restart(tmp_path):
    config = _config(tmp_path, max_rounds=1)
    runner = RecordingRunner(config)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config, discord=discord, runner=runner)
    orchestrator.round_director = SequenceDirector(
        [
            _verdict(
                "CONTINUE",
                next_task="追加実験をする",
                criteria=[CriterionStatus("C1", "partial", "EXP-001が不足")],
            )
        ]
    )
    session = _research_session(config, round_budget=1)
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)

    assert any("未達criteria" in message for message in discord.important_messages)
    assert any("/re start で継続できます" in message for message in discord.important_messages)


def test_blocked_verdict_pauses_session(tmp_path):
    config = _config(tmp_path)
    runner = RecordingRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=runner)
    orchestrator.round_director = SequenceDirector([_verdict("BLOCKED", reason="認証が必要")])
    session = _research_session(config, round_budget=3)
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)
    saved = orchestrator.store.load()

    assert saved.mode == Mode.PAUSED
    assert "認証が必要" in saved.next_action
    assert "/re resume" in saved.next_action


def test_penultimate_round_forces_final_report_task(tmp_path):
    config = _config(tmp_path)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=RecordingRunner(config))
    session = _research_session(config, round_budget=3)
    session.round_id = 2

    orchestrator._apply_director_verdict(
        session,
        _verdict("CONTINUE", next_task="追加実験をする"),
        round_limit=3,
    )

    assert session.next_round_task == FINAL_REPORT_TASK


def test_ledger_records_director_fields_and_reads_old_rows(tmp_path):
    config = _config(tmp_path)
    session = _research_session(config)
    output = RoundOutput(
        main_agent_summary="summary",
        subtask="task",
        sub_agent_output="result",
        review_output="review",
        claude_consultation=None,
        fresh_agent_output=None,
        conversation_sessions=[],
        proposed_operation=None,
        accepted_ideas=[],
        rejected_ideas=[],
        decision="decision",
        confidence="high",
        next_action="next",
    )
    path = config.research_ledger_path(session.session_id)
    ledger = ResearchLedger(path)

    ledger.append_round(
        session,
        output,
        verdict=_verdict("CONTINUE", critique="弱点あり"),
        evidence_ids=["EXP-001"],
    )
    data = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert data["director_decision"] == "CONTINUE"
    assert data["critique"] == "弱点あり"
    assert data["criteria_status"][0]["criterion"] == "C1"
    assert data["evidence_ids"] == ["EXP-001"]

    old_row = dict(data)
    old_row.pop("criteria_status")
    old_row.pop("critique")
    old_row.pop("director_decision")
    path.write_text(json.dumps(old_row, ensure_ascii=False) + "\n", encoding="utf-8")

    [entry] = ledger.read_entries()
    assert entry.criteria_status == []
    assert entry.critique == ""
    assert entry.director_decision == ""


def test_experiment_ids_are_added_to_ledger_evidence(tmp_path):
    config = _config(tmp_path, max_rounds=1, sub_agent_command=None)
    runner = RecordingRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=runner)
    orchestrator.round_director = None
    session = _research_session(config, round_budget=1)
    session.planning_scout = {"evidence_paper_ids": ["P-001"]}
    ExperimentStore(config.experiments_path(session.session_id)).append(
        ExperimentEntry(
            exp_id="EXP-001",
            name="video compare",
            hypothesis="natural videos improve evaluation",
            status="done",
            round_id=1,
            metrics={"n": 5},
            dir="experiments/EXP-001_video",
            created_at=utc_timestamp(),
            updated_at=utc_timestamp(),
        )
    )
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)

    rows = [
        json.loads(line)
        for line in config.research_ledger_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["evidence_ids"] == ["P-001", "EXP-001"]


def test_restart_after_done_clears_stale_final_report_task(tmp_path):
    config = _config(tmp_path, max_rounds=2)
    runner = RecordingRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=runner)
    orchestrator.round_director = SequenceDirector([_verdict("BLOCKED", reason="停止")])
    session = _research_session(config, round_budget=2)
    session.round_id = 2
    session.transition_to(Mode.DONE)
    session.completed_reason = "MAX_ROUNDS reached"
    session.next_round_task = FINAL_REPORT_TASK
    orchestrator.store.save(session)

    orchestrator.start()

    assert runner.subtasks
    assert runner.subtasks[0] != FINAL_REPORT_TASK


class ApprovalRunner(RecordingRunner):
    def run_round(self, session):
        from harness.approval import ProposedOperation

        output = super().run_round(session)
        output.proposed_operation = ProposedOperation(
            operation="delete_file: /tmp/x",
            reason="test",
            impact="test",
            dry_run_result="未実行",
        )
        return output


class ExplodingDirector:
    def assess(self, *args, **kwargs):
        raise AssertionError("director must not run while approval is pending")


def test_director_is_skipped_when_round_requires_approval(tmp_path):
    config = _config(tmp_path)
    runner = ApprovalRunner(config)
    orchestrator = ResearchOrchestrator(config, discord=FakeDiscordAdapter(), runner=runner)
    orchestrator.round_director = ExplodingDirector()
    session = _research_session(config, round_budget=3)
    orchestrator.store.save(session)

    orchestrator._run_until_blocked_or_done(session)
    saved = orchestrator.store.load()

    assert saved.mode == Mode.APPROVAL_BLOCKED


def test_artifact_listing_excludes_virtualenv_noise(tmp_path):
    config = _config(tmp_path)
    session = _research_session(config)
    artifacts = Path(session.research_dir) / "artifacts"
    (artifacts / "round1_notes.md").write_text("ok", encoding="utf-8")
    (artifacts / "venv_probe" / "lib").mkdir(parents=True)
    (artifacts / "venv_probe" / "lib" / "noise.py").write_text("x", encoding="utf-8")
    (artifacts / "__pycache__").mkdir()
    (artifacts / "__pycache__" / "mod.pyc").write_bytes(b"pyc")
    (artifacts / "keep.pyc").write_bytes(b"pyc")

    listing = SubAgentCommandRunner(config)._artifact_listing(session)

    assert "round1_notes.md" in listing
    assert "venv_probe" not in listing
    assert "__pycache__" not in listing
    assert "keep.pyc" not in listing
