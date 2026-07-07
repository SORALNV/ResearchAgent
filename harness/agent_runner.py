from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.approval import ProposedOperation
from harness.config import HarnessConfig
from harness.conversation import ConversationSession
from harness.state import ResearchSession


DEFAULT_PROMPTS = {
    "main": "研究状態を整理し、次のタスクを決める。",
    "sub": "タスクを実行した体で構造化結果を返す。",
    "review": "subの結果に対して懸念、反証、追加確認を返す。",
    "fresh": "既出案と重複しない新規アイデアを返す。",
}


@dataclass
class RoundOutput:
    main_agent_summary: str
    subtask: str
    sub_agent_output: str
    review_output: str
    claude_consultation: str | None
    fresh_agent_output: str | None
    conversation_sessions: list[dict[str, object]]
    proposed_operation: ProposedOperation | None
    accepted_ideas: list[str]
    rejected_ideas: list[str]
    decision: str
    confidence: str
    next_action: str


class MockAgentRunner:
    """Deterministic runner that proves harness flow before real agents exist."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.prompts = self._load_prompts(config.project_root / "prompts")

    def run_round(self, session: ResearchSession) -> RoundOutput:
        round_number = session.round_id + 1
        current_question = f"R{round_number}: {session.research_goal}"
        conversation = ConversationSession(
            topic=f"R{round_number}の実験設計確認",
            participants=["main", "sub"],
            max_turns=self.config.max_turns_per_conversation,
            timeout_seconds=self.config.conversation_timeout_seconds,
        ).run_scripted(
            [
                "MVPでは状態遷移と記録の再現性を優先する。",
                "MockAgentRunnerで安全にE2Eを通す。",
                "MockAgentRunnerで安全にE2Eを通す。",
                "MockAgentRunnerで安全にE2Eを通す。",
            ]
        )
        fresh = None
        if round_number % self.config.fresh_interval == 0:
            fresh = f"Fresh stub: {self.prompts['fresh']} 実エージェント接続前に失敗時再開シナリオを試す。"
        claude = None
        if round_number == 1:
            claude = "Claude stub: 重要判断ではPLANNING承認と承認ゲートを分離する。"
        proposed_operation = None
        if round_number == 1:
            proposed_operation = ProposedOperation(
                operation="delete_file: /tmp/research-harness-dangerous-demo.txt",
                reason="承認ゲートが危険操作を止めることをMVPで検証するため。",
                impact="MVPでは実ファイル削除は行わない。承認状態だけを検証する。",
                dry_run_result="危険操作として検出。@Sora の /re approve が来るまで停止する。",
            )
        subtask = f"Mock sub-agent: {self.prompts['sub']}"
        sub_agent_output = "Mock sub-agent output: 変更diffなし。E2E観点は満たせる見込み。"
        if self.config.sub_agent_command:
            sub_agent_output = SubAgentCommandRunner(self.config).run(
                session=session,
                round_number=round_number,
                task=subtask,
            )
            proposed_operation = proposed_operation or parse_approval_required(sub_agent_output)
        return RoundOutput(
            main_agent_summary=f"{self.prompts['main']} {current_question} を分解し、MVP配線の進捗を統合した。",
            subtask=subtask,
            sub_agent_output=sub_agent_output,
            review_output=f"Review stub: {self.prompts['review']} /re start前にRESEARCHへ進まないこと、承認なし危険操作を止めることを確認。",
            claude_consultation=claude,
            fresh_agent_output=fresh,
            conversation_sessions=[conversation.to_journal_dict()],
            proposed_operation=proposed_operation,
            accepted_ideas=[f"R{round_number}: MockAgentRunnerで検証可能な最小ループを維持"],
            rejected_ideas=[],
            decision="採用: MVP範囲内で次のラウンドへ進める",
            confidence="mid" if round_number == 1 else "high",
            next_action="承認待ちを解消する" if proposed_operation else "次の研究ラウンドへ進む",
        )

    def _load_prompts(self, prompt_dir: Path) -> dict[str, str]:
        prompts = dict(DEFAULT_PROMPTS)
        for role in prompts:
            path = prompt_dir / f"{role}.md"
            if path.exists():
                prompts[role] = path.read_text(encoding="utf-8").strip() or prompts[role]
        return prompts


class SubAgentCommandRunner:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def run(self, session: ResearchSession, round_number: int, task: str) -> str:
        if self.config.max_agent_calls > 0 and session.cost.agent_calls >= self.config.max_agent_calls:
            return (
                "Real sub-agent skipped: MAX_AGENT_CALLS reached. "
                "続行するには設定変更または承認が必要。"
            )
        session.cost.agent_calls += 1
        command = self._build_command(session)
        prompt = self._build_prompt(session, round_number, task)
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.config.max_command_seconds,
                cwd=session.research_dir or str(self.config.project_root),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return (
                "Real sub-agent timeout: "
                f"{self.config.sub_agent_command} exceeded {self.config.max_command_seconds}s. "
                "結果は未確認。"
            )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            return (
                f"Real sub-agent failed: returncode={completed.returncode}. "
                f"stdout={stdout[-1200:] or 'なし'} stderr={stderr[-1200:] or 'なし'}"
            )
        return stdout or "Real sub-agent completed without output. 結果は未確認。"

    def _build_command(self, session: ResearchSession) -> list[str]:
        assert self.config.sub_agent_command is not None
        parts = shlex.split(self.config.sub_agent_command)
        executable = Path(parts[0]).name if parts else ""
        if executable == "codex" and len(parts) == 1:
            return [
                parts[0],
                "exec",
                "--cd",
                session.research_dir or str(self.config.project_root),
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--ask-for-approval",
                "never",
                "-",
            ]
        return parts

    def _build_prompt(self, session: ResearchSession, round_number: int, task: str) -> str:
        return f"""あなたは研究ハーネスのsubエージェントです。
単一タスクだけを実行し、結果を構造化して返してください。

制約:
- 作業場所は研究フォルダ内に限定する: {session.research_dir}
- ファイル削除、外部投稿、git push、秘密情報送信は禁止
- sudo/chmod/chown、ファイル削除、外部投稿、git push、秘密情報送信、課金API呼び出しは禁止
- 危険操作が必要なら実行せず、次の形式で1行だけ報告する:
  APPROVAL_REQUIRED: operation=<操作>; reason=<理由>; impact=<影響>; dry_run_result=<実行していない確認結果>
- 大量ファイル生成や長時間コマンドが必要なら、その必要性を報告だけして実行しない
- 返答には、結果、使ったコマンド、変更ファイル、失敗、未確認事項、次の提案を含める

session_id: {session.session_id}
round: {round_number}
goal: {session.research_goal}
task: {task}
"""


def parse_approval_required(text: str) -> ProposedOperation | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("APPROVAL_REQUIRED:"):
            continue
        payload = stripped.split(":", 1)[1].strip()
        parts: dict[str, str] = {}
        for chunk in payload.split(";"):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            parts[key.strip()] = value.strip()
        operation = parts.get("operation")
        if not operation:
            return ProposedOperation(
                operation="unstructured_approval_required",
                reason=payload or "SubAgentが承認必要操作を報告した。",
                impact="詳細未確認。MVPでは実行しない。",
                dry_run_result="未実行。Discord承認待ちに変換した。",
            )
        return ProposedOperation(
            operation=operation,
            reason=parts.get("reason", "SubAgentが承認必要操作を報告した。"),
            impact=parts.get("impact", "影響は未確認。MVPでは実行しない。"),
            dry_run_result=parts.get("dry_run_result", "未実行。Discord承認待ちに変換した。"),
        )
    return None
