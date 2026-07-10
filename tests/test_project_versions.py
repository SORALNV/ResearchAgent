import json

import pytest

from harness.command_parser import parse_research_command
from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.experiments import ExperimentEntry, ExperimentStore
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.project_catalog import ProjectCatalog
from harness.reporting import render_report, render_run_summary
from harness.state import ResearchSession, utc_timestamp


def _session(config, goal="goal", project="project"):
    session = ResearchSession.new(goal, project_name=project)
    version, research_dir = config.allocate_research_dir(session.session_id, goal)
    session.version_label = version
    session.research_dir = str(research_dir)
    return session


def _exp(exp_id, status="planned", round_id=2, metrics=None):
    now = utc_timestamp()
    return ExperimentEntry(
        exp_id=exp_id,
        name=f"{exp_id} name",
        hypothesis=f"{exp_id} hypothesis",
        status=status,
        round_id=round_id,
        metrics=metrics or {},
        dir="",
        created_at=now,
        updated_at=now,
        success_metric="metric",
        priority="high",
    )


def test_allocate_child_research_dir_increments_minor_and_creates_subdirs(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    parent = tmp_path / "research_runs" / "V009.0_RS-parent_slug"
    parent.mkdir(parents=True)

    version, folder = config.allocate_child_research_dir("V009.0", "RS-child1", "slug")
    assert version == "V009.1"
    assert folder.name == "V009.1_RS-child1_slug"
    assert (folder / "artifacts").is_dir()
    assert (folder / "experiments").is_dir()
    assert (folder / "paper" / "figures").is_dir()

    version, folder = config.allocate_child_research_dir("V009.1", "RS-child2", "slug")
    assert version == "V009.2"
    assert folder.name == "V009.2_RS-child2_slug"


def test_allocate_research_dir_counts_existing_minor_versions_for_next_major(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    (tmp_path / "research_runs" / "V001.1_RS-child_slug").mkdir(parents=True)

    version, folder = config.allocate_research_dir("RS-new", "new goal")

    assert version == "V002.0"
    assert folder.name.startswith("V002.0_RS-new_")


def test_fork_inherits_parent_files_and_planned_backlog(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())
    parent = _session(config, goal="parent goal", project="project alpha")
    parent.mode = Mode.DONE
    parent.goal_criteria = ["criterion 1"]
    parent.research_methods = ["method 1"]
    parent.constraints = ["constraint 1"]
    parent.open_questions = ["question 1"]
    parent.lessons = ["lesson 1"]
    parent.completed_reason = "MAX_ROUNDS reached"
    parent.last_criteria_status = [
        {"criterion": "criterion 1", "status": "met", "evidence": "EXP-001"}
    ]
    orchestrator.store.save(parent)
    parent_dir = config.session_dir(parent.session_id)
    (parent_dir / "papers.jsonl").write_text('{"paper_id":"P-001"}\n', encoding="utf-8")
    parent_exps = ExperimentStore(parent_dir / "experiments.jsonl")
    parent_exps.append(_exp("EXP-001", status="planned", round_id=3))
    parent_exps.append(_exp("EXP-002", status="done", round_id=2, metrics={"score": 1}))

    message = orchestrator.fork_session(parent.version_label)
    child = orchestrator.store.load()

    assert child is not None
    assert child.parent_version == parent.version_label
    assert child.parent_session_id == parent.session_id
    assert child.mode == Mode.PLANNING
    assert child.phase == "plan"
    assert child.goal_criteria == parent.goal_criteria
    assert child.research_methods == parent.research_methods
    assert child.constraints == parent.constraints
    assert child.open_questions == parent.open_questions
    assert child.lessons == parent.lessons
    assert "継承セッションを作成しました" in message
    child_dir = config.session_dir(child.session_id)
    assert (child_dir / "papers.jsonl").read_text(encoding="utf-8") == '{"paper_id":"P-001"}\n'
    child_exps = ExperimentStore(child_dir / "experiments.jsonl").read_all()
    assert [(entry.exp_id, entry.status, entry.round_id) for entry in child_exps] == [
        ("EXP-001", "planned", 0)
    ]
    plan = (child_dir / "plan.md").read_text(encoding="utf-8")
    assert f"## 継承元（{parent.version_label}）" in plan
    assert "MAX_ROUNDS reached" in plan
    assert "EXP-002 [done]" in plan
    assert "lesson 1" in plan


def test_fork_without_argument_uses_latest_done_and_reports_when_missing(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())
    first = _session(config, goal="first", project="p1")
    first.mode = Mode.DONE
    orchestrator.store.save(first)
    second = _session(config, goal="second", project="p2")
    second.mode = Mode.DONE
    orchestrator.store.save(second)

    orchestrator.fork_session()
    child = orchestrator.store.load()
    assert child.parent_version == second.version_label

    empty = ResearchOrchestrator(
        config=HarnessConfig(project_root=tmp_path / "empty"),
        discord=FakeDiscordAdapter(),
    )
    assert "継承元がありません" in empty.fork_session()


def test_fork_argument_resolution_major_latest_lowercase_and_missing(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())
    parent = _session(config, goal="root", project="p")
    parent.mode = Mode.DONE
    orchestrator.store.save(parent)
    orchestrator.fork_session(parent.version_label)
    child1 = orchestrator.store.load()
    child1.mode = Mode.DONE
    orchestrator.store.save(child1)

    orchestrator.fork_session(parent.version_label.split(".")[0])
    child2 = orchestrator.store.load()
    assert child2.parent_version == child1.version_label

    orchestrator.fork_session(parent.version_label.lower())
    child3 = orchestrator.store.load()
    assert child3.parent_version == parent.version_label
    assert "V999 が見つかりません" in orchestrator.fork_session("V999")


def test_start_skips_roadmap_generation_when_planned_backlog_exists(tmp_path):
    config = HarnessConfig(project_root=tmp_path, sub_agent_command="codex", max_rounds=0)
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())
    session = _session(config, goal="roadmap", project="p")
    session.goal_criteria = ["criterion"]
    orchestrator.store.save(session)
    store = ExperimentStore(config.experiments_path(session.session_id))
    store.append(_exp("EXP-001", status="planned", round_id=0))

    class ExplodingPlanner:
        def generate(self, *args, **kwargs):
            raise AssertionError("planner must be skipped")

    orchestrator.roadmap_planner = ExplodingPlanner()
    orchestrator.start()

    entries = ExperimentStore(config.experiments_path(session.session_id)).read_all()
    assert [entry.exp_id for entry in entries] == ["EXP-001"]
    assert "experiment_roadmap_skipped" in config.journal_path(session.session_id).read_text(
        encoding="utf-8"
    )


def test_project_catalog_scan_render_and_write_index(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    session = _session(config, goal="catalog goal", project="catalog project")
    session.mode = Mode.DONE
    session.last_criteria_status = [
        {"criterion": "c1", "status": "met", "evidence": "e"},
        {"criterion": "c2", "status": "unmet", "evidence": ""},
    ]
    session.completed_reason = "done reason"
    ResearchOrchestrator(config=config, discord=FakeDiscordAdapter()).store.save(session)
    child_version, child_dir = config.allocate_child_research_dir(
        session.version_label,
        "RS-child",
        "catalog-goal",
    )
    child = ResearchSession.new("child goal", project_name="catalog project")
    child.session_id = "RS-child"
    child.version_label = child_version
    child.research_dir = str(child_dir)
    child.parent_version = session.version_label
    child_dir.joinpath("state.json").write_text(
        json.dumps(child.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    broken = tmp_path / "research_runs" / "V999.0_RS-broken_bad"
    broken.mkdir(parents=True)
    (broken / "state.json").write_text("{bad", encoding="utf-8")

    catalog = ProjectCatalog(config.research_archive_path)
    infos = catalog.scan()
    assert [info.version_label for info in infos] == [session.version_label, child_version, "V999.0"]
    assert infos[0].met == 1
    assert infos[-1].mode == "broken"
    index = catalog.render_index(infos)
    assert "## V001 catalog project" in index
    assert f"  - {child_version}" in index
    assert f"← {session.version_label}" in index
    assert "met=1/2 done reason" in index
    written = catalog.write_index().read_text(encoding="utf-8")
    assert "## V001 catalog project" in written
    assert f"← {session.version_label}" in written


def test_runs_groups_versions_with_criteria_reason_and_parent(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())
    parent = _session(config, goal="runs goal", project="runs project")
    parent.mode = Mode.DONE
    parent.completed_reason = "MAX_ROUNDS reached"
    parent.last_criteria_status = [{"criterion": "c", "status": "met", "evidence": "e"}]
    orchestrator.store.save(parent)
    orchestrator.fork_session(parent.version_label)

    message = orchestrator.runs()

    assert "[V001] runs project" in message
    assert "met=1/1 MAX_ROUNDS reached" in message
    assert f"← {parent.version_label}" in message


def test_render_report_and_run_summary_include_modern_sections():
    session = ResearchSession.new("report goal", project_name="report project")
    session.version_label = "V001.1"
    session.parent_version = "V001.0"
    session.round_id = 2
    session.completed_reason = "done"
    session.goal_criteria = ["c1"]
    session.last_criteria_status = [{"criterion": "c1", "status": "met", "evidence": "EXP-001"}]
    session.lessons = ["lesson"]
    session.last_report_review = {
        "verdict": "ACCEPT",
        "score": 5,
        "major_issues": [],
    }
    experiments = [_exp("EXP-001", status="done", metrics={"score": 1}), _exp("EXP-002")]
    report = render_report(session, [], [], "mock", experiments=experiments)
    summary = render_run_summary(session, [], [], experiments=experiments)

    assert "## Version" in report
    assert "- Parent: V001.0" in report
    assert "## Goal Criteria" in report
    assert "- c1: met | EXP-001" in report
    assert "## Lessons Learned" in report
    assert "## Report Review" in report
    assert "EXP-001 [done]" in report
    assert "- Criteria Met: 1/1" in summary
    assert "- Completed Reason: done" in summary
    assert "- Review: ACCEPT score=5" in summary
    assert "- Lessons: 1" in summary
    assert "- Experiments: done 1 / planned 1" in summary


def test_re_fork_parser_variants():
    empty = parse_research_command("/re fork")
    assert empty.name == "fork"
    assert empty.args == {"version": ""}
    versioned = parse_research_command("/re fork V009.0")
    assert versioned.name == "fork"
    assert versioned.args == {"version": "V009.0"}
    with pytest.raises(ValueError, match="at most one version"):
        parse_research_command("/re fork V009 extra")
