from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass

from harness.config import HarnessConfig, effective_model_name
from harness.state import ResearchSession


VALID_PRIORITIES = {"high", "mid", "low"}


@dataclass
class PlannedExperiment:
    name: str
    hypothesis: str
    success_metric: str
    priority: str


class ExperimentRoadmapPlanner:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def generate(self, session: ResearchSession, plan_text: str) -> list[PlannedExperiment]:
        command = self._build_command(session)
        prompt = self._build_prompt(session, plan_text)
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
        except (subprocess.TimeoutExpired, OSError):
            return []
        if completed.returncode != 0:
            return []
        session.cost.api_calls += 1
        return parse_roadmap_output(completed.stdout or "")

    def _build_command(self, session: ResearchSession) -> list[str]:
        model = effective_model_name(self.config.roadmap_model or self.config.research_agent_model)
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
        command_text = self.config.sub_agent_command or self.config.main_agent_command or "codex"
        parts = shlex.split(command_text)
        return parts[0] if parts else "codex"

    def _build_prompt(self, session: ResearchSession, plan_text: str) -> str:
        criteria = "\n".join(
            f"C{index}: {item}" for index, item in enumerate(session.goal_criteria, start=1)
        ) or "未確定"
        methods = "\n".join(f"- {item}" for item in session.research_methods) or "- 未確定"
        constraints = "\n".join(f"- {item}" for item in session.constraints) or "- なし"
        questions = "\n".join(f"- {item}" for item in session.open_questions) or "- なし"
        lessons = "\n".join(f"- {item}" for item in session.lessons[-5:]) or "- なし"
        return f"""あなたは研究の実験計画者。ゴール条件を検証可能にする最小の実験系列を設計する。

研究ゴール:
{session.research_goal}

ゴール条件:
{criteria}

研究方法:
{methods}

制約:
{constraints}

未解決の確認事項:
{questions}

これまでの教訓（過去の失敗から。同じ轍を踏まない）:
{lessons}

plan.md（先頭6000字）:
{plan_text[:6000]}

判断規範:
- 各実験は1ラウンドで実行可能な粒度
- 各実験に反証可能な仮説と、数値で判定できる成功指標を付ける
- 環境検証（スモークテスト）を最初に置く
- ゴール条件を直接検証する実験を必ず含める
- 2〜6件

出力フォーマットを厳守:
EXPERIMENT: <name(英語短slug可)> | <仮説1文> | <成功指標1文> | <high|mid|low>
"""


def parse_roadmap_output(text: str) -> list[PlannedExperiment]:
    items: list[PlannedExperiment] = []
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^(?:[-*]\s*)?EXPERIMENT\s*[:：]\s*(.+)$", stripped, flags=re.IGNORECASE)
        if not match:
            continue
        parts = [part.strip() for part in match.group(1).split("|", 3)]
        if len(parts) != 4:
            continue
        priority = parts[3].lower()
        if priority not in VALID_PRIORITIES:
            priority = "mid"
        items.append(
            PlannedExperiment(
                name=parts[0],
                hypothesis=parts[1],
                success_metric=parts[2],
                priority=priority,
            )
        )
    return items
