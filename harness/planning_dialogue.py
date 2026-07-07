from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from harness.config import HarnessConfig
from harness.planning import render_planning_scout
from harness.state import ResearchSession


class PlanningDialogueRunner:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def respond(self, session: ResearchSession, user_text: str, purpose: str) -> str:
        prompt = self._build_prompt(session, user_text, purpose)
        output = self._run_llm(session, prompt)
        return output or self._fallback_response(session, user_text)

    def propose_search_queries(self, session: ResearchSession, user_text: str, purpose: str) -> list[str]:
        prompt = self._build_search_prompt(session, user_text, purpose)
        output = self._run_llm(session, prompt)
        if output:
            queries = self.extract_search_queries(output)
            if queries:
                return queries[:2]
            if self._declines_search(output):
                return []
        return self._fallback_search_queries(session, user_text)

    @staticmethod
    def extract_search_queries(text: str) -> list[str]:
        queries: list[str] = []
        for line in text.splitlines():
            stripped = line.strip().lstrip("-*").strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if not (
                lower.startswith("search_query")
                or lower.startswith("search query")
                or stripped.startswith("検索クエリ")
                or stripped.startswith("検索語")
            ):
                continue
            if ":" in stripped:
                value = stripped.split(":", 1)[1]
            elif "：" in stripped:
                value = stripped.split("：", 1)[1]
            else:
                continue
            for part in value.replace("；", ";").split(";"):
                query = part.strip().strip('"').strip("'").strip()
                if query and query not in queries:
                    queries.append(query)
        return queries

    def _run_llm(self, session: ResearchSession, prompt: str) -> str:
        command_text = (
            self.config.main_agent_command
            or self.config.claude_agent_command
            or self.config.sub_agent_command
        )
        if not command_text:
            return ""
        command = self._build_command(command_text, session)
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
            return ""
        stdout = completed.stdout.strip()
        if completed.returncode != 0:
            return ""
        return stdout

    def _build_command(self, command_text: str, session: ResearchSession) -> list[str]:
        parts = shlex.split(command_text)
        executable = Path(parts[0]).name if parts else ""
        if executable == "codex" and len(parts) == 1:
            return [
                parts[0],
                "exec",
                "--cd",
                session.research_dir or str(self.config.project_root),
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--ask-for-approval",
                "never",
                "-",
            ]
        return parts

    def _build_prompt(self, session: ResearchSession, user_text: str, purpose: str) -> str:
        scout = render_planning_scout(session.planning_scout) if session.planning_scout else "未実行"
        history = self._dialogue_history(session)
        return f"""あなたはResearchAgentのPLANNING壁打ち担当です。
目的: {purpose}

設計思想:
- ResearchAgentハーネスは研究問題を単独で解く主体ではなく、大きめのAGENTS.mdのように方針・制約・証跡・役割分担を保持する制御文脈である
- 単体のLLM回答で結論を出さない
- 研究はmain/sub/review/freshなどの複数役割、論文検索ツール、承認ゲートを組み合わせたオーケストレーションで進める
- 壁打ち担当の役割は、Soraの意図を明確にし、必要なツールやエージェント呼び出しを提案し、未確認点を分離すること

制約:
- 日本語で返す
- ChatGPTと普通に話しているような自然な壁打ちにする
- レポート形式、長い見出し、網羅的な箇条書きを避ける
- 返答は原則400〜700字以内。長くても900字を超えない
- 最初に結論を短く返し、その後に理由を1〜2点だけ添える
- 質問は1〜2個まで
- 研究テーマを勝手に確定しない
- 類似研究がある場合は、paper_idを使って比較対象を示す
- 新規性は断定せず、未確認点を分ける
- 1つのLLMとして最終判断や実行結果をでっち上げない
- 必要なら「次にmain/sub/review/freshのどの役割を使うべきか」を示す
- コマンド案は必要な時だけ最後に1つだけ示す
- ファイル編集やコマンド実行はしない

研究ゴール:
{session.research_goal}

Soraの入力:
{user_text}

直近の対話履歴:
{history}

現在の類似研究スカウト:
{scout}
"""

    def _build_search_prompt(self, session: ResearchSession, user_text: str, purpose: str) -> str:
        scout = render_planning_scout(session.planning_scout) if session.planning_scout else "未実行"
        history = self._dialogue_history(session)
        return f"""あなたはResearchAgentの類似研究検索クエリ設計担当です。
目的: {purpose}

設計思想:
- 論文検索は壁打ちLLMが必要に応じて使うツールであり、ハーネスが常に自動実行するものではない
- 単体LLMで研究判断を完結させず、検索結果は後続のmain/sub/review/freshのオーケストレーション材料として使う

制約:
- 日本語で考えてよいが、検索クエリはarXiv等で通りやすい英語中心にする
- 類似研究調査が不要なら SEARCH_NEEDED: no とだけ返す
- 必要なら最大2件だけ SEARCH_QUERY: <query> を返す
- ファイル編集やコマンド実行はしない
- 出力形式を守る

出力形式:
SEARCH_NEEDED: yes または no
SEARCH_QUERY: <query 1>
SEARCH_QUERY: <query 2>
REASON: <短い理由>

研究ゴール:
{session.research_goal}

Soraの入力:
{user_text}

直近の対話履歴:
{history}

現在の類似研究スカウト:
{scout}
"""

    def _dialogue_history(self, session: ResearchSession) -> str:
        items = [
            item
            for item in session.accepted_ideas[-12:]
            if item.startswith("Sora discuss:") or item.startswith("PLANNING dialogue:")
        ]
        if not items:
            return "なし"
        return "\n".join(f"- {item[:700]}" for item in items[-8:])

    def _declines_search(self, text: str) -> bool:
        compact = text.lower().replace(" ", "")
        return "search_needed:no" in compact or "検索不要" in text

    def _fallback_search_queries(self, session: ResearchSession, user_text: str) -> list[str]:
        return []

    def _fallback_response(self, session: ResearchSession, user_text: str) -> str:
        scout = session.planning_scout or {}
        primary = scout.get("primary_comparison") or {}
        questions = scout.get("required_decisions") or [
            "主要比較対象をどれにするか。",
            "差分をどこに置くか。",
        ]
        comparison = (
            f"{primary.get('title')} [{primary.get('paper_id')}]"
            if primary.get("paper_id") and primary.get("title")
            else "まだ主要比較対象は未確認"
        )
        return (
            f"いいと思います。今の方向だと、まずは「何を既存研究と違うと言うのか」を絞るのが先です。\n\n"
            f"比較対象は {comparison} です。ここを基準に、性能で勝つ話にするのか、運用コストや再現性の条件を変える話にするのかを決めたいです。新規性はまだ断定せず、仮説扱いで進めるのが安全です。\n\n"
            f"いま決めるなら、{questions[0]} もう一つ、最初の成果物は比較表・小さな実装・調査メモのどれに寄せますか？"
        )
