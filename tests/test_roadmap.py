import json
from types import SimpleNamespace

from harness.agent_runner import SubAgentCommandRunner
from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.experiments import ExperimentEntry, ExperimentStore
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.roadmap import PlannedExperiment, parse_roadmap_output
from harness.state import ResearchSession, utc_timestamp


def _entry(exp_id, name, priority="", status="planned"):
    now = utc_timestamp()
    return ExperimentEntry(
        exp_id=exp_id,
        name=name,
        hypothesis=f"{name} hypothesis",
        status=status,
        round_id=0,
        metrics={},
        dir="",
        created_at=now,
        updated_at=now,
        success_metric=f"{name} metric",
        priority=priority,
    )


def _planned_session(config):
    session = ResearchSession.new("roadmap goal")
    _, research_dir = config.allocate_research_dir(session.session_id, session.research_goal)
    session.research_dir = str(research_dir)
    session.goal_criteria = ["C1を満たす"]
    session.research_methods = ["小さく検証する"]
    return session


def test_parse_roadmap_output_variants():
    text = """EXPERIMENT: smoke | env works | pass rate >= 1 | high
- EXPERIMENT: baseline | baseline works | accuracy >= 0.7 | mid
* experiment: ablation | ablation helps | delta >= 0.02 | low
EXPERIMENT: weird | invalid priority | n >= 3 | urgent
"""

    items = parse_roadmap_output(text)

    assert [item.name for item in items] == ["smoke", "baseline", "ablation", "weird"]
    assert [item.priority for item in items] == ["high", "mid", "low", "mid"]
    assert parse_roadmap_output("no roadmap here") == []


def test_start_generates_initial_roadmap_and_skips_non_initial_resume(tmp_path):
    config = HarnessConfig(project_root=tmp_path, max_rounds=0, sub_agent_command="codex")
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())
    session = _planned_session(config)
    orchestrator.store.save(session)

    class StubPlanner:
        def __init__(self):
            self.calls = 0

        def generate(self, session, plan_text):
            self.calls += 1
            assert "Research Plan" in plan_text
            return [
                PlannedExperiment("smoke", "env works", "pass count >= 1", "high"),
                PlannedExperiment("goal-check", "goal can be checked", "score >= 0.8", "mid"),
            ]

    planner = StubPlanner()
    orchestrator.roadmap_planner = planner

    orchestrator.start()

    entries = ExperimentStore(config.experiments_path(session.session_id)).read_all()
    assert planner.calls == 1
    assert [entry.status for entry in entries] == ["planned", "planned"]
    assert entries[0].success_metric == "pass count >= 1"
    assert "## Experiment Roadmap" in config.plan_path(session.session_id).read_text(encoding="utf-8")

    session = orchestrator.store.load()
    assert session is not None
    session.round_id = 1
    session.mode = Mode.DONE
    orchestrator.store.save(session)

    class ExplodingPlanner:
        def generate(self, *args, **kwargs):
            raise AssertionError("planner must not run on non-initial resume")

    orchestrator.roadmap_planner = ExplodingPlanner()
    orchestrator.start()


def test_planned_backlog_priority_order(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.jsonl")
    store.append(_entry("EXP-003", "low", "low"))
    store.append(_entry("EXP-002", "high", "high"))
    store.append(_entry("EXP-001", "mid", "mid"))
    store.append(_entry("EXP-004", "done", "high", status="done"))
    store.append(_entry("EXP-005", "none", ""))

    assert [entry.exp_id for entry in store.planned_backlog()] == [
        "EXP-002",
        "EXP-001",
        "EXP-003",
        "EXP-005",
    ]


def test_sub_experiment_summary_prioritizes_backlog(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    session = _planned_session(config)
    store = ExperimentStore(config.experiments_path(session.session_id))
    store.append(_entry("EXP-001", "done", status="done"))
    store.append(_entry("EXP-002", "planned-high", priority="high"))

    summary = SubAgentCommandRunner(config)._experiment_summary(session)

    assert summary.index("実験バックログ") < summary.index("実行済み/進行中")
    assert "EXP-002 [high] planned-high" in summary
    assert "EXP-001 [done] done" in summary


def test_sub_agent_retries_hard_failure_then_success(monkeypatch, tmp_path):
    config = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command="fake-agent",
        max_round_retries=1,
    )
    session = _planned_session(config)
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs["input"])
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("harness.agent_runner.subprocess.run", fake_run)

    output = SubAgentCommandRunner(config).run(session, round_number=1, task="task")

    assert "(リトライ1回目で成功)" in output
    assert "ok" in output
    assert "前回試行の失敗" in calls[1]
    assert session.cost.agent_calls == 2


def test_sub_agent_retries_all_failures(monkeypatch, tmp_path):
    config = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command="fake-agent",
        max_round_retries=1,
    )
    session = _planned_session(config)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="bad", stderr="boom")

    monkeypatch.setattr("harness.agent_runner.subprocess.run", fake_run)

    output = SubAgentCommandRunner(config).run(session, round_number=1, task="task")

    assert "Real sub-agent failed: returncode=1" in output
    assert session.cost.agent_calls == 2


def test_old_experiment_rows_read_with_new_defaults(tmp_path):
    path = tmp_path / "experiments.jsonl"
    old = {
        "exp_id": "EXP-001",
        "name": "old",
        "hypothesis": "old hypothesis",
        "status": "planned",
        "round_id": 0,
        "metrics": {},
        "dir": "",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    path.write_text(json.dumps(old, ensure_ascii=False) + "\n", encoding="utf-8")

    [entry] = ExperimentStore(path).read_all()

    assert entry.success_metric == ""
    assert entry.priority == ""
