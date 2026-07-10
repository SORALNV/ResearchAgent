from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from harness.config import HarnessConfig, effective_model_name
from harness.memory import AgentMemory
from harness.state import ResearchSession


PLAN_UPDATE_KEYS = {
    "GOAL_CRITERIA": "goal_criteria",
    "METHOD": "research_methods",
    "OPEN_QUESTION": "open_questions",
    "RESOLVED_QUESTION": "resolved_questions",
    "REMEMBER": "remembers",
}


def parse_plan_update(text: str) -> tuple[str, dict[str, list[str]]]:
    lines = text.splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == "PLAN_UPDATE:":
            start_index = index
            break
    if start_index is None:
        return text, {}

    updates: dict[str, list[str]] = {
        "goal_criteria": [],
        "research_methods": [],
        "open_questions": [],
        "resolved_questions": [],
        "remembers": [],
    }
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        update_key = PLAN_UPDATE_KEYS.get(key.strip())
        if not update_key:
            continue
        item = value.strip()
        if item:
            updates[update_key].append(item)

    return "\n".join(lines[:start_index]).rstrip(), {
        key: values for key, values in updates.items() if values
    }


def _render_items(items: list[str], empty: str) -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


@dataclass
class DialogueTurnResult:
    text: str
    thread_id: str
    searches: list[str]
    error: str


@dataclass
class CodexRunResult:
    returncode: int
    stdout: str
    stderr: str


class PlanningDialogueRunner:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def converse(
        self,
        session: ResearchSession,
        user_text: str,
        search_callback: Callable[[str], str],
        image_paths: list[str] | None = None,
    ) -> DialogueTurnResult:
        if not (
            self.config.main_agent_command
            or self.config.claude_agent_command
            or self.config.sub_agent_command
        ):
            return DialogueTurnResult(
                text="",
                thread_id="",
                searches=[],
                error="PLAN LLMコマンド未設定（SUB_AGENT_COMMAND等）",
            )
        self._append_dialogue_log(session, "sora", user_text)
        output_path = self._last_message_path(session)
        thread_id = session.plan_thread_id
        searches: list[str] = []

        if thread_id:
            command = self._build_resume_command(thread_id, session, output_path, image_paths=image_paths)
            run = self._run_codex(command, user_text, session, output_path)
            if run.returncode != 0:
                summary = self._recent_dialogue_summary(session)
                prompt = self._build_initial_prompt(session, user_text, dialogue_summary=summary)
                command = self._build_start_command(session, output_path, image_paths=image_paths)
                run = self._run_codex(command, prompt, session, output_path)
                thread_id, first_text = self._extract_run_payload(run, output_path)
            else:
                parsed_thread_id, first_text = self._extract_run_payload(run, output_path)
                thread_id = parsed_thread_id or thread_id
        else:
            prompt = self._build_initial_prompt(session, user_text)
            command = self._build_start_command(session, output_path, image_paths=image_paths)
            run = self._run_codex(command, prompt, session, output_path)
            thread_id, first_text = self._extract_run_payload(run, output_path)

        if run.returncode != 0:
            return DialogueTurnResult(
                text="",
                thread_id=thread_id,
                searches=searches,
                error=self._short_error(run),
            )
        if not thread_id:
            return DialogueTurnResult(
                text="",
                thread_id="",
                searches=searches,
                error="thread_idを取得できませんでした",
            )

        queries = self.extract_search_queries(first_text)[:2]
        final_text = first_text
        if queries:
            result_blocks: list[str] = []
            for query in queries:
                searches.append(query)
                try:
                    search_text = search_callback(query)
                except Exception as exc:
                    search_text = f"検索に失敗しました: {exc}"
                self._append_dialogue_log(session, "search", f"SEARCH_QUERY: {query}\n{search_text}")
                result_blocks.append(f"SEARCH_QUERY: {query}\n{search_text}")
            followup = (
                "検索結果:\n"
                + "\n\n".join(result_blocks)
                + "\n\nこの結果を踏まえて、Soraへの返答を完成させてください。再検索は不要です。"
                + "論文の内容説明は英語のまま貼らず、日本語に翻訳して紹介してください。"
            )
            command = self._build_resume_command(thread_id, session, output_path)
            run = self._run_codex(command, followup, session, output_path)
            parsed_thread_id, followup_text = self._extract_run_payload(run, output_path)
            if run.returncode != 0:
                return DialogueTurnResult(
                    text="",
                    thread_id=parsed_thread_id or thread_id,
                    searches=searches,
                    error=self._short_error(run),
                )
            thread_id = parsed_thread_id or thread_id
            final_text = self._strip_search_queries(followup_text)

        self._append_dialogue_log(session, "assistant", final_text)
        return DialogueTurnResult(text=final_text, thread_id=thread_id, searches=searches, error="")

    def _build_start_command(
        self,
        session: ResearchSession,
        output_path: Path | None,
        image_paths: list[str] | None = None,
    ) -> list[str]:
        command = [
            self._codex_executable(),
            "exec",
            "-m",
            effective_model_name(self.config.plan_agent_model),
            "--cd",
            session.research_dir or str(self.config.project_root),
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--json",
        ]
        if output_path is not None:
            command.extend(["--output-last-message", str(output_path)])
        for image_path in image_paths or []:
            command.extend(["--image", image_path])
        command.append("-")
        return command

    def _build_resume_command(
        self,
        thread_id: str,
        session: ResearchSession,
        output_path: Path | None,
        image_paths: list[str] | None = None,
    ) -> list[str]:
        command = [
            self._codex_executable(),
            "exec",
            "resume",
            thread_id,
            "-m",
            effective_model_name(self.config.plan_agent_model),
            "--skip-git-repo-check",
            "--json",
        ]
        if output_path is not None:
            command.extend(["--output-last-message", str(output_path)])
        for image_path in image_paths or []:
            command.extend(["-i", image_path])
        command.append("-")
        return command

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

    def _run_codex(
        self,
        command: list[str],
        prompt: str,
        session: ResearchSession,
        output_path: Path | None,
    ) -> CodexRunResult:
        if output_path is not None and output_path.exists():
            output_path.unlink()
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
        except subprocess.TimeoutExpired as exc:
            return CodexRunResult(124, exc.stdout or "", "timeout")
        return CodexRunResult(completed.returncode, completed.stdout, completed.stderr)

    def _extract_run_payload(self, run: CodexRunResult, output_path: Path | None) -> tuple[str, str]:
        thread_id = ""
        last_agent_text = ""
        for line in run.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") == "thread.started":
                thread_id = str(item.get("thread_id") or "")
            if item.get("type") != "item.completed":
                continue
            payload = item.get("item") or item
            text = self._extract_agent_text(payload)
            if text:
                last_agent_text = text
        if output_path is not None and output_path.exists():
            final = output_path.read_text(encoding="utf-8").strip()
            if final:
                return thread_id, final
        return thread_id, last_agent_text or run.stdout.strip()

    def _extract_agent_text(self, payload: object) -> str:
        if not isinstance(payload, dict):
            return ""
        if payload.get("type") == "agent_message" and isinstance(payload.get("text"), str):
            return payload["text"]
        if isinstance(payload.get("agent_message"), dict):
            message = payload["agent_message"]
            if isinstance(message.get("text"), str):
                return message["text"]
        content = payload.get("content")
        if isinstance(content, list):
            parts = [
                str(part.get("text"))
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            return "\n".join(parts).strip()
        return ""

    def _codex_executable(self) -> str:
        command_text = (
            self.config.main_agent_command
            or self.config.claude_agent_command
            or self.config.sub_agent_command
            or "codex"
        )
        return command_text.split()[0]

    def _short_error(self, run: CodexRunResult) -> str:
        detail = (run.stderr or run.stdout or "").strip()
        if not detail:
            detail = f"returncode={run.returncode}"
        return detail[-200:]

    def _last_message_path(self, session: ResearchSession) -> Path | None:
        if not session.research_dir:
            return None
        path = Path(session.research_dir) / "artifacts" / "planning_llm_last_message.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _build_initial_prompt(self, session: ResearchSession, user_text: str, dialogue_summary: str = "") -> str:
        goal_criteria = _render_items(session.goal_criteria, "未確定")
        research_methods = _render_items(session.research_methods, "未確定")
        open_questions = _render_items(session.open_questions, "なし")
        summary_block = f"\nこれまでの会話の要約:\n{dialogue_summary}\n" if dialogue_summary else ""
        memory_digest = AgentMemory(self.config.agent_memory_path).digest()
        return f"""あなたはResearchAgentのPLANNING壁打ち担当です。

設計思想:
- ResearchAgentハーネスは研究問題を単独で解く主体ではなく、大きめのAGENTS.mdのように方針・制約・証跡・役割分担を保持する制御文脈である
- 単体のLLM回答で結論を出さない
- 研究はmain/sub/review/freshなどの複数役割、論文検索ツール、承認ゲートを組み合わせたオーケストレーションで進める
- 壁打ち担当の役割は、Soraの意図を明確にし、必要なツールやエージェント呼び出しを提案し、未確認点を分離すること
- これは継続的な会話であり、あなたは過去のやり取りを覚えている。同じ説明を繰り返さず、会話を前に進める

壁打ちの目的:
- 研究テーマ・主要比較対象との差分・ゴール条件（何をもって完了/成功とみなすか）・研究方法・成果物を確定させること
- 未確定項目を自然な会話の中で1つずつ詰める

制約:
- 日本語で返す
- 論文・類似研究の紹介も必ず日本語で行う。タイトルは原文のままでよいが、内容・手法・結果の説明は日本語に翻訳して伝える。英語のabstractをそのまま貼らない
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
- ファイル編集やコマンド実行はしない。テキスト返答のみ
- SEARCH_QUERYは学術論文や類似研究を探す時だけ使う。製品仕様、公式ドキュメント、API仕様、料金、最新モデル仕様の確認が必要な場合はSEARCH_QUERYを出さず、「mainで公式一次資料を確認する」と返す
- arXiv検索に向かない固有製品名や公式仕様確認を、無理に論文検索へ流さない

構造化記録:
- 類似研究・文献を調べたい場合は、返答の最後に SEARCH_QUERY: <英語クエリ> 行を最大2つ付ける
- ハーネスが検索を実行し、結果を次のメッセージで渡す。必要なときだけ使う
- ユーザーとの合意が取れた項目、またはユーザーが明示指定した方法・条件があれば、返答の最後にPLAN_UPDATEブロックで記録する。合意していないものは書かない
- PLAN_UPDATEブロックは字数制限の外とする
- あなた側で研究を進める上で明確にしておきたい点があればOPEN_QUESTIONとして記録し、会話の中でユーザーに聞く。解決したらRESOLVED_QUESTIONで消す
- 会話の中でユーザーの恒久的な好み・環境制約・重要な決定を知ったら、PLAN_UPDATEブロックに REMEMBER: <1行> を含めてよい。本当に今後も使う事実だけを書く

PLAN_UPDATE形式:
PLAN_UPDATE:
GOAL_CRITERIA: <ゴール条件1行>
METHOD: <研究方法1行>
OPEN_QUESTION: <確認したい点1行>
RESOLVED_QUESTION: <解決した確認事項の本文>
REMEMBER: <今後も使う恒久的な記憶1行>

研究ゴール:
{session.research_goal}

現在の確定状況:
確定済みゴール条件:
{goal_criteria}
確定済み研究方法:
{research_methods}
未解決の確認事項:
{open_questions}

長期記憶（過去の研究・ユーザーの好み）:
{memory_digest}

Soraの入力:
{user_text}{summary_block}
"""

    def _dialogue_log_path(self, session: ResearchSession) -> Path | None:
        if not session.research_dir:
            return None
        return Path(session.research_dir) / "dialogue.jsonl"

    def _append_dialogue_log(self, session: ResearchSession, role: str, text: str) -> None:
        path = self._dialogue_log_path(session)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "role": role,
            "text": text,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _recent_dialogue_summary(self, session: ResearchSession) -> str:
        path = self._dialogue_log_path(session)
        if path is None or not path.exists():
            return "なし"
        entries: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines()[-10:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = item.get("role") or "unknown"
            text = str(item.get("text") or "").replace("\n", " ")[:400]
            entries.append(f"- {role}: {text}")
        return "\n".join(entries) if entries else "なし"

    def _strip_search_queries(self, text: str) -> str:
        lines = [
            line
            for line in text.splitlines()
            if not line.strip().lower().startswith("search_query")
        ]
        return "\n".join(lines).rstrip()
