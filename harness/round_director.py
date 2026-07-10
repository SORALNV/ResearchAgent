from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass

from harness.config import HarnessConfig, effective_model_name
from harness.state import ResearchSession


VALID_DECISIONS = {"CONTINUE", "FINAL_REPORT", "DONE", "BLOCKED"}
VALID_STATUSES = {"met", "partial", "unmet"}


@dataclass
class CriterionStatus:
    criterion: str
    status: str
    evidence: str


@dataclass
class DirectorVerdict:
    decision: str
    criteria_status: list[CriterionStatus]
    critique: str
    next_task: str
    reason: str
    raw: str
    source: str


class RoundDirector:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def assess(
        self,
        session: ResearchSession,
        round_number: int,
        round_limit: int,
        subtask: str,
        sub_output: str,
        experiment_summary: str,
    ) -> DirectorVerdict:
        command = self._build_command(session)
        prompt = self._build_prompt(
            session=session,
            round_number=round_number,
            round_limit=round_limit,
            subtask=subtask,
            sub_output=sub_output,
            experiment_summary=experiment_summary,
        )
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
        except subprocess.TimeoutExpired as exc:
            raw = _tail(f"{exc.stdout or ''}\n{exc.stderr or ''}\ntimeout")
            return _fallback(f"director failed: timeout after {self.config.max_command_seconds}s", raw)
        except OSError as exc:
            return _fallback(f"director failed: {exc}", "")

        raw = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        combined = "\n".join(part for part in [raw, stderr] if part)
        if completed.returncode != 0:
            return _fallback(f"director failed: returncode={completed.returncode}", _tail(combined))
        session.cost.api_calls += 1
        return parse_director_output(raw or combined)

    def _build_command(self, session: ResearchSession) -> list[str]:
        model = effective_model_name(self.config.director_model or self.config.plan_agent_model)
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
        round_number: int,
        round_limit: int,
        subtask: str,
        sub_output: str,
        experiment_summary: str,
    ) -> str:
        criteria = _numbered_items(session.goal_criteria, empty="未確定")
        methods = _bullet_items(session.research_methods, empty="未確定")
        questions = _bullet_items(session.open_questions, empty="なし")
        remaining = max(round_limit - round_number, 0)
        return f"""あなたは研究ディレクター。subエージェントのラウンド結果を検証し、次の一手を決める。

必要なら研究フォルダ内の artifacts/ や experiments/、notes.md、experiments.jsonl などのファイルを読んで確認してよい。

研究ゴール:
{session.research_goal}

ゴール条件:
{criteria}

研究方法:
{methods}

未解決の確認事項:
{questions}

今回のラウンド:
R{round_number} / round_limit={round_limit}
残りラウンド数: {remaining}

今回subに与えたタスク:
{subtask}

subの出力（先頭8000字）:
{sub_output[:8000]}

実験台帳サマリ:
{experiment_summary}

判断規範:
- 証拠（実測値・ファイル・出典）に基づいて criteria を判定する。subの自己申告を鵜呑みにしない
- 未検証の主張・矛盾・過度な一般化を critique で指摘する
- next_task は「subが1ラウンドで実行可能な具体的作業」1〜3文。前ラウンドの失敗や未確認事項の解消を優先する
- 全criteriaがmetで最終レポート（artifacts/report.md）が最新なら DONE
- 残りラウンドが少なく統合レポートが未執筆なら FINAL_REPORT
- 環境・認証・データ入手など、ユーザーの介入なしに進めない場合のみ BLOCKED

出力フォーマットを厳守:
CRITERIA_STATUS:
- C1: met|partial|unmet | <根拠一行>
- C2: met|partial|unmet | <根拠一行>
CRITIQUE: <2〜4行。改行可>
DECISION: CONTINUE|FINAL_REPORT|DONE|BLOCKED
NEXT_TASK: <1〜3文。DONE/BLOCKEDなら「なし」>
REASON: <1〜2行>
"""


def parse_director_output(text: str) -> DirectorVerdict:
    criteria = _parse_criteria(text)
    decision = _parse_decision(text)
    if decision not in VALID_DECISIONS:
        return _fallback("director failed: invalid or missing decision", _tail(text))
    return DirectorVerdict(
        decision=decision,
        criteria_status=criteria,
        critique=_section_text(text, "CRITIQUE"),
        next_task=_normalize_none(_section_text(text, "NEXT_TASK")),
        reason=_section_text(text, "REASON"),
        raw=text,
        source="llm",
    )


def _parse_criteria(text: str) -> list[CriterionStatus]:
    statuses: list[CriterionStatus] = []
    capture = False
    for line in text.splitlines():
        stripped = line.strip()
        if _heading_name(stripped) == "CRITERIA_STATUS":
            capture = True
            continue
        if capture and _heading_name(stripped):
            break
        if not capture:
            continue
        match = re.match(
            r"^[-*]\s*(C\d+)\s*[:：]\s*([A-Za-z]+)\s*(?:\|\s*)?(.*)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        status = match.group(2).lower()
        if status not in VALID_STATUSES:
            continue
        statuses.append(
            CriterionStatus(
                criterion=match.group(1).upper(),
                status=status,
                evidence=match.group(3).strip(),
            )
        )
    return statuses


def _parse_decision(text: str) -> str:
    match = re.search(r"(?im)^DECISION\s*[:：]\s*([A-Z_]+)\b", text)
    if not match:
        return ""
    return match.group(1).upper()


def _section_text(text: str, section: str) -> str:
    lines = text.splitlines()
    capturing = False
    captured: list[str] = []
    for line in lines:
        stripped = line.strip()
        heading = _heading_name(stripped)
        if heading == section:
            capturing = True
            after = re.split(r"[:：]", line, maxsplit=1)
            if len(after) == 2 and after[1].strip():
                captured.append(after[1].strip())
            continue
        if capturing and heading:
            break
        if capturing:
            captured.append(line.rstrip())
    return "\n".join(captured).strip()


def _heading_name(line: str) -> str:
    match = re.match(r"^([A-Z_]+)\s*[:：]", line.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _normalize_none(text: str) -> str:
    return "" if text.strip() in {"なし", "無し", "none", "None", "NONE"} else text


def _fallback(reason: str, raw: str) -> DirectorVerdict:
    return DirectorVerdict(
        decision="CONTINUE",
        criteria_status=[],
        critique="",
        next_task="",
        reason=reason,
        raw=raw,
        source="fallback",
    )


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:]


def _numbered_items(items: list[str], empty: str) -> str:
    if not items:
        return empty
    return "\n".join(f"C{index}: {item}" for index, item in enumerate(items, start=1))


def _bullet_items(items: list[str], empty: str) -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)
