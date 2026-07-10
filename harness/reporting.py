from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from harness.ledger import LedgerEntry
from harness.papers import Paper
from harness.state import ResearchSession, utc_timestamp


def render_report(
    session: ResearchSession,
    papers: list[Paper],
    ledger_entries: list[LedgerEntry],
    model_or_command: str,
    experiments: list | None = None,
) -> str:
    paper_lines = [
        f"- [{paper.paper_id}] {paper.title} ({paper.year or '未確認'}, {paper.source})"
        for paper in papers
    ] or ["- 未確認: 出典候補なし"]
    ledger_lines = [
        f"- {entry.node_id}: {entry.hypothesis} -> {entry.next_action}"
        for entry in ledger_entries
    ] or ["- 未実行"]
    experiment_lines = _experiment_lines(experiments) if experiments else ledger_lines
    scout = session.planning_scout or {}
    unverified = _unverified_claims(session, papers)
    met, total = _criteria_counts(session)
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
            f"- Criteria Met: {met}/{total}",
            f"- Novelty Status: {scout.get('novelty_status') or '未評価'}",
            "",
            "## Version",
            f"- Project: {session.project_name or '未設定'}",
            f"- Version: {session.version_label or '未設定'}",
            f"- Parent: {session.parent_version or 'なし'}",
            f"- Session: {session.session_id}",
            "",
            "## Goal Criteria",
            *_criteria_lines(session),
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
            "## Experiments",
            *experiment_lines,
            "",
            "## Risks",
            *_risks(session),
            "",
            "## Next Steps",
            f"- {session.next_action or '未設定'}",
            "",
            "## Lessons Learned",
            *([f"- {lesson}" for lesson in session.lessons] or ["- なし"]),
            "",
            "## Report Review",
            *_review_lines(session),
            "",
            "## AI Provenance",
            *[f"- {key}: {value}" for key, value in provenance.items()],
        ]
    )


def render_run_summary(
    session: ResearchSession,
    papers: list[Paper],
    ledger_entries: list[LedgerEntry],
    experiments: list | None = None,
) -> str:
    paper_ids = ", ".join(paper.paper_id for paper in papers[:5]) or "なし"
    latest = ledger_entries[-1].next_action if ledger_entries else session.next_action or "未設定"
    met, total = _criteria_counts(session)
    review = session.last_report_review or {}
    review_text = (
        f"{review.get('verdict', '未実施')} score={review.get('score', 0)}"
        if review
        else "未実施"
    )
    experiments = experiments or []
    done_count = sum(1 for entry in experiments if getattr(entry, "status", "") == "done")
    planned_count = sum(1 for entry in experiments if getattr(entry, "status", "") == "planned")
    return "\n".join(
        [
            "# Run Summary",
            "",
            f"- Session: {session.session_id}",
            f"- Version: {session.version_label or '未設定'}",
            f"- Project: {session.project_name or '未設定'}",
            f"- Parent: {session.parent_version or 'なし'}",
            f"- Goal: {session.research_goal}",
            f"- Rounds: {session.round_id}",
            f"- Criteria Met: {met}/{total}",
            f"- Completed Reason: {session.completed_reason or 'なし'}",
            f"- Review: {review_text}",
            f"- Lessons: {len(session.lessons)}",
            f"- Experiments: done {done_count} / planned {planned_count}",
            f"- Papers: {paper_ids}",
            f"- Novelty Status: {session.planning_scout.get('novelty_status') or '未評価'}",
            f"- Next: {latest}",
        ]
    )


def _criteria_counts(session: ResearchSession) -> tuple[int, int]:
    statuses = session.last_criteria_status or []
    return (
        sum(1 for item in statuses if str(item.get("status", "")).lower() == "met"),
        len(statuses),
    )


def _criteria_lines(session: ResearchSession) -> list[str]:
    if not session.last_criteria_status:
        return ["- 未評価"]
    return [
        f"- {item.get('criterion', '?')}: {item.get('status', 'unknown')} | {item.get('evidence', '')}"
        for item in session.last_criteria_status
    ]


def _experiment_lines(experiments: list | None) -> list[str]:
    if not experiments:
        return ["- 未実行"]
    return [
        "- {exp_id} [{status}] {name} | metrics={metrics}".format(
            exp_id=getattr(entry, "exp_id", ""),
            status=getattr(entry, "status", ""),
            name=getattr(entry, "name", ""),
            metrics=getattr(entry, "metrics", {}),
        )
        for entry in experiments
    ]


def _review_lines(session: ResearchSession) -> list[str]:
    review = session.last_report_review or {}
    if not review:
        return ["- 未実施"]
    lines = [
        f"- Verdict: {review.get('verdict', '未実施')}",
        f"- Score: {review.get('score', 0)}",
    ]
    issues = review.get("major_issues") or []
    if issues:
        lines.append("- Major Issues: " + "; ".join(str(item) for item in issues))
    else:
        lines.append("- Major Issues: なし")
    return lines


def review_report(
    report: str,
    papers: list[Paper],
    session: ResearchSession,
    research_dir: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, object]:
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
    missing_images = _missing_image_links(report, research_dir=research_dir, report_path=report_path)
    for image_path in missing_images:
        warnings.append(f"存在しない画像を参照しています: {image_path}")
    return {
        "warning_count": len(warnings),
        "warnings": warnings,
        "valid_paper_ids": sorted(valid_ids),
        "cited_paper_ids": sorted(cited_ids),
    }


def _missing_image_links(
    report: str,
    research_dir: str | Path | None = None,
    report_path: str | Path | None = None,
) -> list[str]:
    if report_path is not None:
        base_dir = Path(report_path).parent
    elif research_dir is not None:
        candidate = Path(research_dir)
        base_dir = candidate.parent if candidate.suffix == ".md" else candidate / "artifacts"
    else:
        return []
    missing: list[str] = []
    for match in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", report):
        raw_path = match.group(1).strip().strip("<>")
        if not raw_path:
            continue
        lowered = raw_path.lower()
        if (
            lowered.startswith("http://")
            or lowered.startswith("https://")
            or lowered.startswith("data:")
            or lowered.startswith("mailto:")
            or lowered.startswith("#")
        ):
            continue
        image_path = Path(raw_path)
        target = image_path if image_path.is_absolute() else base_dir / image_path
        if not target.exists():
            missing.append(raw_path)
    return missing


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
