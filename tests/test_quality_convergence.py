import json
from pathlib import Path
from types import SimpleNamespace

from harness.config import HarnessConfig
from harness.convergence import ConvergenceTracker
from harness.discord_adapter import FakeDiscordAdapter
from harness.eval import run_golden_eval
from harness.hardened_orchestrator import HardenedResearchOrchestrator
from harness.papers import PaperStore, make_paper
from harness.sandbox import sandbox_capability
from harness.state import ResearchSession


def make_session(tmp_path: Path) -> ResearchSession:
    session = ResearchSession.new("quality and observability")
    session.research_dir = str(tmp_path / "run")
    Path(session.research_dir).mkdir()
    return session


def test_evaluation_scores_generated_answer_not_only_paper_metadata(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "question": "How should reliability improve?",
                "expected_keywords": ["restartability", "journal"],
                "required_source_types": ["arxiv"],
                "must_not_claim": ["perfectly autonomous"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = PaperStore(tmp_path / "papers.jsonl")
    store.upsert_many(
        [
            make_paper(
                title="Reliable Research Agents",
                authors=["A. Author"],
                year=2026,
                venue="arXiv",
                url="https://example.test/paper",
                doi="10.0000/reliable",
                arxiv_id=None,
                abstract="restartability and journal design",
                source="arxiv",
                relevance_score=0.9,
                confidence="high",
            )
        ]
    )
    journal = [
        {
            "conversation_sessions": [
                {
                    "stage": "review",
                    "returncode": 0,
                    "skipped": False,
                    "timed_out": False,
                    "cancelled": False,
                    "output": json.dumps(
                        {
                            "verdict": "accept",
                            "summary": "checked",
                            "revisions": [],
                            "confidence": "high",
                        }
                    ),
                }
            ]
        }
    ]

    result = run_golden_eval(
        golden,
        store,
        report_text="restartability is supported by an append-only journal [P-001]",
        journal_entries=journal,
        research_dir=tmp_path,
    )

    assert result.expected_keyword_hits == 2
    assert result.must_not_claim_hits == 0
    assert result.citation_ready is True
    assert result.review_successes == 1
    assert result.overall_score > 80
    assert "answer_keyword_coverage=2/2" in result.summary


def test_evaluation_detects_invalid_citation_and_forbidden_claim(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "question": "Safety",
                "expected_keywords": ["approval"],
                "required_source_types": [],
                "must_not_claim": ["perfectly autonomous"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = PaperStore(tmp_path / "papers.jsonl")

    result = run_golden_eval(
        golden,
        store,
        report_text="perfectly autonomous approval [P-999]",
    )

    assert result.must_not_claim_hits == 1
    assert result.invalid_citations == ("P-999",)
    assert result.citation_ready is False


def test_convergence_tracker_completes_and_detects_stagnation(tmp_path):
    config = HarnessConfig(
        project_root=tmp_path,
        convergence_patience=1,
        convergence_no_evidence_patience=3,
        convergence_min_progress=0.1,
    )
    session = make_session(tmp_path)
    tracker = ConvergenceTracker(session, config)

    low_progress = SimpleNamespace(
        decision="continue",
        next_action="same",
        accepted_ideas=[],
        new_evidence_ids=[],
        unresolved_blockers=[],
        round_status="continue",
        progress_score=0.0,
        confidence="high",
        round_number=1,
    )
    decision = tracker.evaluate(low_progress)
    assert decision.needs_human_review is True

    completed_session = ResearchSession.new("complete")
    completed_session.research_dir = str(tmp_path / "complete")
    Path(completed_session.research_dir).mkdir()
    complete_tracker = ConvergenceTracker(completed_session, config)
    completed = SimpleNamespace(
        decision="done",
        next_action="report",
        accepted_ideas=["verified"],
        new_evidence_ids=["P-001"],
        unresolved_blockers=[],
        round_status="completed",
        progress_score=1.0,
        confidence="high",
        round_number=1,
    )
    complete_decision = complete_tracker.evaluate(completed)
    assert complete_decision.should_complete is True


class StatusRunner:
    def runtime_snapshot(self, session):
        return {
            "current_stage": "review_1",
            "checkpoint_status": "running",
            "active_agents": 2,
            "completed_subtasks": 1,
            "failed_subtasks": 0,
            "total_subtasks": 3,
            "last_error": None,
        }


def test_status_exposes_runtime_observability_without_mutation(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    orchestrator = HardenedResearchOrchestrator(
        config=config,
        discord=FakeDiscordAdapter(),
        runner=StatusRunner(),
    )
    session = make_session(tmp_path)
    orchestrator.store.save(session)

    before = config.state_path.read_text(encoding="utf-8")
    status = orchestrator.status()
    after = config.state_path.read_text(encoding="utf-8")

    assert "実行ステージ: review_1" in status
    assert "実行中Agent: 2" in status
    assert "completed=1" in status
    assert before == after


def test_secure_from_env_defaults_deny_unsandboxed_generic(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_SANDBOX_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_ALLOW_UNSANDBOXED_GENERIC", raising=False)
    config = HarnessConfig.from_env(tmp_path)

    assert config.agent_sandbox_backend == "auto"
    assert config.agent_allow_unsandboxed_generic is False
    ok, detail = sandbox_capability(config)
    assert isinstance(ok, bool)
    assert detail
