from __future__ import annotations

import re
from dataclasses import asdict

from harness.ledger import LedgerEntry
from harness.papers import Paper
from harness.state import ResearchSession, utc_timestamp


def render_report(
    session: ResearchSession,
    papers: list[Paper],
    ledger_entries: list[LedgerEntry],
    model_or_command: str,
) -> str:
    paper_lines = [
        f"- [{paper.paper_id}] {paper.title} ({paper.year or '未確認'}, {paper.source})"
        for paper in papers
    ] or ["- 未確認: 出典候補なし"]
    ledger_lines = [
        f"- {entry.node_id}: {entry.hypothesis} -> {entry.next_action}"
        for entry in ledger_entries
    ] or ["- 未実行"]
    scout = session.planning_scout or {}
    unverified = _unverified_claims(session, papers)
    provenance = {
        "generated_at": utc_timestamp(),
        "generator_role": "deterministic_report_writer",
        "model_or_command": model_or_command or "mock",
        "paper_provider": _paper_provider_summary(papers),
        "fake_or_real_sources": _source_mix(papers),
        "human_approvals": session.approvals_received,
        "major_directives": session.redirects,
    }
    return "\n".join(
        [
            "# Research Report",
            "",
            "## Summary",
            f"- Goal: {session.research_goal}",
            f"- Mode: {session.mode.value}",
            f"- Rounds: {session.round_id}",
            f"- Novelty Status: {scout.get('novelty_status') or '未評価'}",
            "",
            "## Sources",
            *paper_lines,
            "",
            "## Known Facts",
            *_known_facts(papers),
            "",
            "## Hypotheses",
            *_hypotheses(session),
            "",
            "## Unverified Claims",
            *unverified,
            "",
            "## Experiments or Analysis",
            *ledger_lines,
            "",
            "## Risks",
            *_risks(session),
            "",
            "## Next Steps",
            f"- {session.next_action or '未設定'}",
            "",
            "## AI Provenance",
            *[f"- {key}: {value}" for key, value in provenance.items()],
        ]
    )


def render_run_summary(session: ResearchSession, papers: list[Paper], ledger_entries: list[LedgerEntry]) -> str:
    paper_ids = ", ".join(paper.paper_id for paper in papers[:5]) or "なし"
    latest = ledger_entries[-1].next_action if ledger_entries else session.next_action or "未設定"
    return "\n".join(
        [
            "# Run Summary",
            "",
            f"- Session: {session.session_id}",
            f"- Version: {session.version_label or '未設定'}",
            f"- Goal: {session.research_goal}",
            f"- Rounds: {session.round_id}",
            f"- Papers: {paper_ids}",
            f"- Novelty Status: {session.planning_scout.get('novelty_status') or '未評価'}",
            f"- Next: {latest}",
        ]
    )


def review_report(report: str, papers: list[Paper], session: ResearchSession) -> dict[str, object]:
    valid_ids = {paper.paper_id for paper in papers}
    cited_ids = set(re.findall(r"\[(P-\d{3})\]", report))
    invalid = sorted(cited_ids - valid_ids)
    warnings: list[str] = []
    if session.planning_scout.get("blocking"):
        warnings.append("blocking novelty gateのまま終了しています。")
    if not papers:
        warnings.append("出典候補がありません。Known Factsは未確認扱いです。")
    if invalid:
        warnings.append(f"存在しないpaper_idを参照しています: {', '.join(invalid)}")
    if "未確認" not in report and session.planning_scout.get("novelty_status") != "supported":
        warnings.append("未確認事項の明示が不足している可能性があります。")
    return {
        "warning_count": len(warnings),
        "warnings": warnings,
        "valid_paper_ids": sorted(valid_ids),
        "cited_paper_ids": sorted(cited_ids),
    }


def _known_facts(papers: list[Paper]) -> list[str]:
    if not papers:
        return ["- 未確認: 出典候補がないため、既知事実は確定しない。"]
    return [f"- {paper.summary} [{paper.paper_id}]" for paper in papers[:5]]


def _hypotheses(session: ResearchSession) -> list[str]:
    scout = session.planning_scout or {}
    values = scout.get("differentiation_hypotheses") or scout.get("suggested_directions") or []
    return [f"- {item}" for item in values] or ["- 未確認"]


def _unverified_claims(session: ResearchSession, papers: list[Paper]) -> list[str]:
    claims = []
    if session.planning_scout.get("novelty_status") != "supported":
        claims.append("- 新規性は未確認または人間判断待ち。")
    if not papers:
        claims.append("- 文献根拠がないため、外部研究との比較は未確認。")
    return claims or ["- なし"]


def _risks(session: ResearchSession) -> list[str]:
    risks = session.planning_scout.get("risks") or []
    return [f"- {risk}" for risk in risks] or ["- 未確認"]


def _paper_provider_summary(papers: list[Paper]) -> str:
    sources = sorted({paper.source for paper in papers})
    return ", ".join(sources) or "none"


def _source_mix(papers: list[Paper]) -> dict[str, int]:
    fake = sum(1 for paper in papers if paper.source == "fake")
    return {"fake": fake, "real": len(papers) - fake}
