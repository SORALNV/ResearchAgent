from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass

from harness.config import HarnessConfig, effective_model_name
from harness.state import ResearchSession


@dataclass
class ReviewResult:
    verdict: str
    score: int
    major_issues: list[str]
    minor_issues: list[str]
    raw: str
    source: str


class ReportReviewer:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def review(
        self,
        session: ResearchSession,
        report_text: str | None = None,
        experiment_summary: str = "",
    ) -> ReviewResult:
        if report_text is None:
            report_path = self.config.report_path(session.session_id)
            report_text = (
                report_path.read_text(encoding="utf-8") if report_path.exists() else "レポート未作成"
            )
        command = self._build_command(session)
        prompt = self._build_prompt(session, report_text, experiment_summary)
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                timeout=self.config.max_command_seconds,
                capture_output=True,
                text=True,
                cwd=session.research_dir or str(self.config.project_root),
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return _fallback(str(exc))
        raw = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        combined = "\n".join(part for part in [raw, stderr] if part)
        if completed.returncode != 0:
            return _fallback(combined)
        result = parse_review_output(raw or combined)
        if result.source == "llm":
            session.cost.api_calls += 1
        return result

    def _build_command(self, session: ResearchSession) -> list[str]:
        model = effective_model_name(
            self.config.reviewer_model or self.config.director_model or self.config.plan_agent_model
        )
        return [
            self._codex_executable(),
            "exec",
            "-m",
            model,
            "--cd",
            session.research_dir or str(self.config.project_root),
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--color",
            "never",
            "-",
        ]

    def _codex_executable(self) -> str:
        command_text = (
            self.config.main_agent_command
            or self.config.claude_agent_command
            or self.config.sub_agent_command
            or "codex"
        )
        parts = shlex.split(command_text)
        return parts[0] if parts else "codex"

    def _build_prompt(
        self,
        session: ResearchSession,
        report_text: str,
        experiment_summary: str,
    ) -> str:
        criteria = "\n".join(
            f"C{index}: {item}" for index, item in enumerate(session.goal_criteria, start=1)
        ) or "未確定"
        constraints = "\n".join(f"- {item}" for item in session.constraints) or "- なし"
        return f"""あなたは研究レポートの査読者。完了前の最終ゲートとして、証拠に基づいて冷静に判定する。

研究ゴール:
{session.research_goal}

ゴール条件:
{criteria}

制約:
{constraints}

実験台帳サマリ:
{experiment_summary or "なし"}

artifacts/report.md（先頭12000字）:
{(report_text or "レポート未作成")[:12000]}

査読観点:
- 各ゴール条件がレポート内の証拠（実測値・図・出典）で裏付けられているか
- 数値・図表の整合が取れているか
- 限界と未確認事項が正直に記載されているか
- 再現に必要な情報（手順・パラメータ）があるか

出力フォーマットを厳守:
SCORE: <1-5の整数>
MAJOR_ISSUES:（完了を妨げる重大指摘。無ければ「- なし」）
- <指摘>
MINOR_ISSUES:（軽微な指摘。無ければ「- なし」）
- <指摘>
VERDICT: ACCEPT|REVISE
"""


def parse_review_output(text: str) -> ReviewResult:
    verdict = _parse_verdict(text)
    if verdict not in {"ACCEPT", "REVISE"}:
        return _fallback(text)
    return ReviewResult(
        verdict=verdict,
        score=_parse_score(text),
        major_issues=_parse_issue_list(text, "MAJOR_ISSUES"),
        minor_issues=_parse_issue_list(text, "MINOR_ISSUES"),
        raw=text,
        source="llm",
    )


def _parse_verdict(text: str) -> str:
    match = re.search(r"(?im)^VERDICT\s*[:：]\s*(ACCEPT|REVISE)\b", text)
    return match.group(1).upper() if match else ""


def _parse_score(text: str) -> int:
    match = re.search(r"(?im)^SCORE\s*[:：]\s*([1-5])\b", text)
    return int(match.group(1)) if match else 0


def _parse_issue_list(text: str, section: str) -> list[str]:
    issues: list[str] = []
    capture = False
    for line in text.splitlines():
        stripped = line.strip()
        heading = _heading_name(stripped)
        if heading == section:
            capture = True
            after = re.split(r"[:：]", line, maxsplit=1)
            if len(after) == 2 and after[1].strip() and not after[1].strip().startswith("（"):
                item = after[1].strip().lstrip("-*").strip()
                if item not in {"なし", "無し", "none", "None", "NONE"}:
                    issues.append(item)
            continue
        if capture and heading:
            break
        if not capture:
            continue
        match = re.match(r"^[-*]\s*(.+)$", stripped)
        if match:
            item = match.group(1).strip()
            if item not in {"なし", "無し", "none", "None", "NONE"}:
                issues.append(item)
    return issues


def _heading_name(line: str) -> str:
    match = re.match(r"^([A-Z_]+)\s*[:：]", line.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _fallback(raw: str) -> ReviewResult:
    return ReviewResult(
        verdict="ACCEPT",
        score=0,
        major_issues=[],
        minor_issues=[],
        raw=raw,
        source="fallback",
    )
