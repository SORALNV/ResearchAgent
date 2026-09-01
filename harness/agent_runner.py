from __future__ import annotations

import os
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path

from harness.approval import ProposedOperation
from harness.config import HarnessConfig
from harness.conversation import ConversationSession
from harness.process_manager import ProcessCancellationController
from harness.provider_runtime import ProviderAwareAgentCommandExecutor
from harness.state import ResearchSession


DEFAULT_PROMPTS = {
    "main": "研究状態を整理し、次のタスクを決める。",
    "sub": "タスクを実行した体で構造化結果を返す。",
    "review": "subの結果に対して懸念、反証、追加確認を返す。",
    "fresh": "既出案と重複しない新規アイデアを返す。",
}

_RUNTIME_ORDER_ENV_NAMES = (
    "AGENT_RUNTIME_ORDER",
    "MAIN_AGENT_RUNTIME_ORDER",
    "SUB_AGENT_RUNTIME_ORDER",
    "REVIEW_AGENT_RUNTIME_ORDER",
    "FRESH_AGENT_RUNTIME_ORDER",
    "PLANNING_AGENT_RUNTIME_ORDER",
    "CLAUDE_AGENT_RUNTIME_ORDER",
)


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
    """Compatibility facade: deterministic mock or portable real runner."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.prompts = self._load_prompts(config.project_root / "prompts")
        self._real_runner = None
        if _runtime_requested(config):
            from harness.portable_multi_agent_runner import MultiAgentRunner

            self._real_runner = MultiAgentRunner(config)

    def run_round(self, session: ResearchSession):
        if self._real_runner is not None:
            return self._real_runner.run_round(session)

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
        if self.config.fresh_interval > 0 and round_number % self.config.fresh_interval == 0:
            fresh = (
                f"Fresh stub: {self.prompts['fresh']} "
                "実エージェント接続前に失敗時再開シナリオを試す。"
            )
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
        return RoundOutput(
            main_agent_summary=(
                f"{self.prompts['main']} {current_question} を分解し、"
                "MVP配線の進捗を統合した。"
            ),
            subtask=subtask,
            sub_agent_output=sub_agent_output,
            review_output=(
                f"Review stub: {self.prompts['review']} "
                "/re start前にRESEARCHへ進まないこと、承認なし危険操作を止めることを確認。"
            ),
            claude_consultation=claude,
            fresh_agent_output=fresh,
            conversation_sessions=[conversation.to_journal_dict()],
            proposed_operation=proposed_operation,
            accepted_ideas=[
                f"R{round_number}: MockAgentRunnerで検証可能な最小ループを維持"
            ],
            rejected_ideas=[],
            decision="採用: MVP範囲内で次のラウンドへ進める",
            confidence="mid" if round_number == 1 else "high",
            next_action=(
                "承認待ちを解消する"
                if proposed_operation
                else "次の研究ラウンドへ進む"
            ),
        )

    def cancel_active(self, reason: str = "cancel requested") -> int:
        if self._real_runner is None:
            return 0
        return self._real_runner.cancel_active(reason)

    def reset_cancellation(self) -> None:
        if self._real_runner is not None:
            self._real_runner.reset_cancellation()

    def runtime_snapshot(self, session: ResearchSession) -> dict[str, object]:
        if self._real_runner is None:
            return {
                "active_agents": 0,
                "checkpoint_status": "mock",
                "current_stage": "mock",
                "completed_subtasks": 0,
                "failed_subtasks": 0,
                "total_subtasks": 0,
            }
        return self._real_runner.runtime_snapshot(session)

    def _load_prompts(self, prompt_dir: Path) -> dict[str, str]:
        prompts = dict(DEFAULT_PROMPTS)
        for role in prompts:
            path = prompt_dir / f"{role}.md"
            if path.exists():
                prompts[role] = path.read_text(encoding="utf-8").strip() or prompts[role]
        return prompts


class SubAgentCommandRunner:
    """Backward-compatible single-agent entry point using the same provider router."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.cancellation = ProcessCancellationController(
            config.agent_cancel_grace_seconds
        )
        self.executor = ProviderAwareAgentCommandExecutor(
            config,
            threading.RLock(),
            self.cancellation,
        )

    def run(self, session: ResearchSession, round_number: int, task: str) -> str:
        invocation = self.executor.run(
            session=session,
            role="sub",
            stage="legacy_sub_execute",
            prompt=self._build_prompt(session, round_number, task),
            command_text=self.config.sub_agent_command,
            sandbox="workspace-write",
            working_dir=Path(session.research_dir or self.config.project_root),
        )
        return invocation.output

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
                "-c",
                'approval_policy="never"',
                "-",
            ]
        return parts

    def _build_prompt(
        self,
        session: ResearchSession,
        round_number: int,
        task: str,
    ) -> str:
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
            dry_run_result=parts.get(
                "dry_run_result", "未実行。Discord承認待ちに変換した。"
            ),
        )
    return None


def _runtime_requested(config: HarnessConfig) -> bool:
    commands = (
        config.main_agent_command,
        config.sub_agent_command,
        config.review_agent_command,
        config.fresh_agent_command,
        config.claude_agent_command,
    )
    return any(commands) or any(
        os.getenv(name, "").strip() for name in _RUNTIME_ORDER_ENV_NAMES
    )
