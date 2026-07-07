import json

from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.paper_search import ArxivPaperSearchProvider


def make_orchestrator(tmp_path, **overrides):
    config = HarnessConfig(project_root=tmp_path, **overrides)
    discord = FakeDiscordAdapter()
    return ResearchOrchestrator(config=config, discord=discord), discord, config


def test_fake_search_writes_papers_dedupes_and_lists_with_citations(tmp_path):
    orchestrator, discord, config = make_orchestrator(tmp_path)
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "literature test")

    first = discord.inject(orchestrator, "/re search harness approval citation")
    assert first.ok
    assert "P-001" in first.message
    assert "[P-001]" in first.message
    session = orchestrator.store.load()
    papers_path = config.papers_path(session.session_id)
    assert papers_path.exists()
    papers = [json.loads(line) for line in papers_path.read_text(encoding="utf-8").splitlines()]
    assert len(papers) == 2
    assert all(paper["summary"] for paper in papers)

    second = discord.inject(orchestrator, "/re search harness approval citation")
    assert second.ok
    papers_after = [json.loads(line) for line in papers_path.read_text(encoding="utf-8").splitlines()]
    assert len(papers_after) == 2

    listing = discord.inject(orchestrator, "/re papers")
    assert "P-001" in listing.message
    detail = discord.inject(orchestrator, "/re paper P-001")
    assert "summary:" in detail.message

    entries = [
        json.loads(line)
        for line in config.journal_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["event_type"] == "literature_search_completed" for entry in entries)
    assert any(entry["event_type"] == "paper_summaries_created" for entry in entries)


def test_cost_limit_blocks_after_search(tmp_path):
    orchestrator, discord, _ = make_orchestrator(tmp_path, max_api_calls=1)
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "cost limit")
    result = discord.inject(orchestrator, "/re search harness")
    assert result.ok
    assert "コスト上限に到達" in result.message
    session = orchestrator.store.load()
    assert session.mode == Mode.APPROVAL_BLOCKED
    assert session.cost.api_calls == 1
    assert any("コスト上限" in message for message in discord.messages)


def test_cost_and_eval_commands(tmp_path):
    orchestrator, discord, _ = make_orchestrator(tmp_path)
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "eval test")
    discord.inject(orchestrator, "/re search citation")
    cost = discord.inject(orchestrator, "/re cost")
    assert "api_calls: 1" in cost.message
    evaluation = discord.inject(orchestrator, "/re eval")
    assert "questions=" in evaluation.message
    assert "citation_ready=True" in evaluation.message


def test_restore_session_from_journal_when_state_file_missing(tmp_path):
    orchestrator, discord, config = make_orchestrator(tmp_path)
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "restore test")
    discord.inject(orchestrator, "/re search citation")
    session = orchestrator.store.load()
    config.session_state_path(session.session_id).unlink()

    restored = orchestrator.store.load()
    assert restored is not None
    assert restored.session_id == session.session_id
    assert restored.cost.api_calls == 1


def test_arxiv_provider_parses_atom_xml():
    xml_body = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <title>Example Arxiv Paper</title>
        <summary>This is an abstract about agent evaluation.</summary>
        <published>2024-01-01T00:00:00Z</published>
        <author><name>A. Author</name></author>
        <arxiv:doi>10.1234/example</arxiv:doi>
      </entry>
    </feed>
    """
    papers = ArxivPaperSearchProvider()._parse(xml_body, max_results=5)
    assert len(papers) == 1
    assert papers[0].title == "Example Arxiv Paper"
    assert papers[0].year == 2024
    assert papers[0].arxiv_id == "2401.00001v1"
    assert papers[0].doi == "10.1234/example"
