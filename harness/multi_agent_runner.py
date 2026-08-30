from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from harness.approval import ProposedOperation
from harness.config import HarnessConfig
from harness.state import ResearchSession


@dataclass(frozen=True)
class AgentInvocation:
    role: str
    stage: str
    task_id: str | None
    command: tuple[str, ...]
    output: str
    stderr: str
    returncode: int
    duration_seconds: float
    skipped: bool = False
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return not self.skipped and not self.timed_out and self.returncode == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "stage": self.stage,
            "task_id": self.task_id,
            "command": list(self.command),
            "returncode": self.returncode,
            "duration_seconds": round(self.duration_seconds, 3),
            "skipped": self.skipped,
            "timed_out": self.timed_out,
            "output": self.output,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class SubTask:
    task_id: str
    task: str
    deliverable: str


@dataclass
class SubTaskRun:
    task: SubTask
    attempts: list[AgentInvocation] = field(default_factory=list)

    @property
    def latest(self) -> AgentInvocation:
        return self.attempts[-1]


@dataclass
class RealRoundOutput:
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


class AgentCommandExecutor:
    def __init__(self, config: HarnessConfig, lock: threading.Lock) -> None:
        self.config = config
        self.lock = lock
        self.output_limit = _env_int("AGENT_OUTPUT_CHAR_LIMIT", 12000)

    def run(
        self,
        *,
        session: ResearchSession,
        role: str,
        stage: str,
        prompt: str,
        command_text: str | None,
        sandbox: str,
        task_id: str | None = None,
        working_dir: Path | None = None,
    ) -> AgentInvocation:
        started = time.monotonic()
        if not command_text:
            return AgentInvocation(role, stage, task_id, (), f"Real {role} agent skipped: command not configured.", "", 127, 0, True)
        with self.lock:
            if self.config.max_agent_calls > 0 and session.cost.agent_calls >= self.config.max_agent_calls:
                return AgentInvocation(role, stage, task_id, (), "Real agent skipped: MAX_AGENT_CALLS reached.", "", 125, 0, True)
            session.cost.agent_calls += 1
        cwd = working_dir or Path(session.research_dir or self.config.project_root)
        cwd.mkdir(parents=True, exist_ok=True)
        command = self._build_command(command_text, session, sandbox, cwd)
        redacted = tuple(_redact(command))
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.config.max_command_seconds,
                cwd=cwd,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = _text(exc.stdout) or f"Real {role} agent timeout after {self.config.max_command_seconds}s."
            return AgentInvocation(role, stage, task_id, redacted, _clip(output, self.output_limit), _clip(_text(exc.stderr), self.output_limit), 124, time.monotonic() - started, timed_out=True)
        except OSError as exc:
            return AgentInvocation(role, stage, task_id, redacted, f"Real {role} agent failed to start: {exc}", str(exc), 127, time.monotonic() - started)
        output = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode and not output:
            output = f"Real {role} agent failed: returncode={completed.returncode}; stderr={stderr[-2000:] or 'なし'}"
        if not completed.returncode and not output:
            output = f"Real {role} agent completed without output. 結果は未確認。"
        return AgentInvocation(role, stage, task_id, redacted, _clip(output, self.output_limit), _clip(stderr, self.output_limit), completed.returncode, time.monotonic() - started)

    def _build_command(self, command_text: str, session: ResearchSession, sandbox: str, cwd: Path) -> list[str]:
        parts = shlex.split(command_text)
        if not parts:
            raise ValueError("agent command is empty")
        if Path(parts[0]).name == "codex" and len(parts) == 1:
            return [parts[0], "exec", "--cd", str(cwd), "--skip-git-repo-check", "--sandbox", sandbox, "--ask-for-approval", "never", "-"]
        return parts


class MultiAgentRunner:
    """main plan -> parallel sub -> review/selective retry -> fresh -> main integration."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.sub_count = max(1, _env_int("SUB_AGENT_COUNT", 3))
        self.parallelism = max(1, _env_int("AGENT_PARALLELISM", self.sub_count))
        self.max_review_retries = max(0, _env_int("MAX_REVIEW_RETRIES", 1))
        self.executor = AgentCommandExecutor(config, threading.Lock())

    def run_round(self, session: ResearchSession) -> RealRoundOutput:
        round_number = session.round_id + 1
        trace: list[AgentInvocation] = []

        plan_call = self.executor.run(
            session=session,
            role="main",
            stage="main_plan",
            prompt=self._plan_prompt(session, round_number),
            command_text=self._command_for("main"),
            sandbox="read-only",
        )
        trace.append(plan_call)
        plan = _json(plan_call.output) or {"summary": plan_call.output, "subtasks": []}
        tasks = self._normalize_tasks(plan.get("subtasks"), session, round_number)
        runs = self._run_subs(session, round_number, tasks, "sub_execute")
        trace.extend(run.latest for run in runs.values())

        reviews: list[dict[str, object]] = []
        for attempt in range(self.max_review_retries + 1):
            review_call = self.executor.run(
                session=session,
                role="review",
                stage="review",
                prompt=self._review_prompt(session, round_number, plan, runs, attempt),
                command_text=self._command_for("review"),
                sandbox="read-only",
            )
            trace.append(review_call)
            review = self._parse_review(review_call.output, runs)
            reviews.append({"attempt": attempt + 1, **review})
            if review["verdict"] != "revise" or attempt >= self.max_review_retries:
                break
            retry_tasks = self._revision_tasks(review["revisions"], runs)
            instructions = {
                str(item.get("task_id")): str(item.get("instructions") or review["summary"])
                for item in review["revisions"]
                if isinstance(item, dict)
            }
            runs = self._run_subs(session, round_number, retry_tasks, "sub_retry", runs, instructions)
            trace.extend(runs[task.task_id].latest for task in retry_tasks)

        fresh_call = None
        if self.config.fresh_interval > 0 and round_number % self.config.fresh_interval == 0:
            fresh_call = self.executor.run(
                session=session,
                role="fresh",
                stage="fresh",
                prompt=self._fresh_prompt(session, round_number, runs, reviews),
                command_text=self._command_for("fresh"),
                sandbox="read-only",
            )
            trace.append(fresh_call)

        claude_call = None
        if self.config.claude_agent_command:
            claude_call = self.executor.run(
                session=session,
                role="claude",
                stage="claude_consultation",
                prompt=self._claude_prompt(session, round_number, runs, reviews),
                command_text=self.config.claude_agent_command,
                sandbox="read-only",
            )
            trace.append(claude_call)

        integration_call = self.executor.run(
            session=session,
            role="main",
            stage="main_integrate",
            prompt=self._integration_prompt(session, round_number, plan, runs, reviews, fresh_call, claude_call),
            command_text=self._command_for("main"),
            sandbox="read-only",
        )
        trace.append(integration_call)
        integration = _json(integration_call.output) or {
            "summary": integration_call.output,
            "decision": "追加検証",
            "confidence": "low",
            "next_action": "未確認事項を次ラウンドで検証する",
            "accepted_ideas": [],
            "rejected_ideas": [],
        }
        proposed = _first_operation(item.output for item in trace)
        return RealRoundOutput(
            main_agent_summary=f"Initial plan: {plan.get('summary', '未確認')}\nFinal integration: {integration.get('summary', '未確認')}",
            subtask="\n".join(f"{task.task_id}: {task.task}" for task in tasks),
            sub_agent_output=_render_runs(runs),
            review_output=_render_reviews(reviews),
            claude_consultation=claude_call.output if claude_call else None,
            fresh_agent_output=fresh_call.output if fresh_call else None,
            conversation_sessions=[item.to_dict() for item in trace],
            proposed_operation=proposed,
            accepted_ideas=_strings(integration.get("accepted_ideas")),
            rejected_ideas=_strings(integration.get("rejected_ideas")),
            decision=str(integration.get("decision") or "追加検証"),
            confidence=_confidence(integration.get("confidence")),
            next_action=str(integration.get("next_action") or "未確認事項を次ラウンドで検証する"),
        )

    def _command_for(self, role: str) -> str | None:
        choices = {
            "main": (self.config.main_agent_command, self.config.claude_agent_command, self.config.sub_agent_command, self.config.review_agent_command, self.config.fresh_agent_command),
            "sub": (self.config.sub_agent_command, self.config.main_agent_command, self.config.claude_agent_command, self.config.review_agent_command, self.config.fresh_agent_command),
            "review": (self.config.review_agent_command, self.config.claude_agent_command, self.config.main_agent_command, self.config.sub_agent_command, self.config.fresh_agent_command),
            "fresh": (self.config.fresh_agent_command, self.config.claude_agent_command, self.config.main_agent_command, self.config.sub_agent_command, self.config.review_agent_command),
        }
        return next((item for item in choices[role] if item), None)

    def _normalize_tasks(self, raw: object, session: ResearchSession, round_number: int) -> list[SubTask]:
        tasks: list[SubTask] = []
        if isinstance(raw, list):
            for index, item in enumerate(raw, 1):
                if isinstance(item, str):
                    text, deliverable, task_id = item.strip(), "検証可能な結果", f"S{index}"
                elif isinstance(item, dict):
                    text = str(item.get("task") or item.get("description") or "").strip()
                    deliverable = str(item.get("deliverable") or "検証可能な結果")
                    task_id = str(item.get("id") or item.get("task_id") or f"S{index}")
                else:
                    continue
                if text:
                    tasks.append(SubTask(task_id, text, deliverable))
                if len(tasks) >= self.sub_count:
                    break
        defaults = [
            "根拠・既存成果・反証候補を整理する",
            "実装・実験・再現手順を検証する",
            "失敗条件・安全性・コスト・未確認事項を監査する",
            "別仮説・比較軸・より単純な方法を検討する",
        ]
        while len(tasks) < self.sub_count:
            index = len(tasks)
            description = defaults[index % len(defaults)]
            tasks.append(SubTask(f"S{index + 1}", f"R{round_number}: {session.research_goal}について{description}", "根拠、実行内容、失敗、未確認事項、次の提案"))
        return tasks

    def _run_subs(
        self,
        session: ResearchSession,
        round_number: int,
        tasks: list[SubTask],
        stage: str,
        runs: dict[str, SubTaskRun] | None = None,
        instructions: dict[str, str] | None = None,
    ) -> dict[str, SubTaskRun]:
        result = dict(runs or {})
        futures = {}
        with ThreadPoolExecutor(max_workers=min(self.parallelism, len(tasks)), thread_name_prefix="research-sub") as pool:
            for task in tasks:
                previous = result.get(task.task_id)
                attempt = len(previous.attempts) + 1 if previous else 1
                workspace = self._workspace(session, round_number, task.task_id, attempt)
                future = pool.submit(
                    self.executor.run,
                    session=session,
                    role="sub",
                    stage=stage,
                    task_id=task.task_id,
                    prompt=self._sub_prompt(session, round_number, task, workspace, previous.latest.output if previous else None, (instructions or {}).get(task.task_id)),
                    command_text=self._command_for("sub"),
                    sandbox="workspace-write",
                    working_dir=workspace,
                )
                futures[future] = task
            completed = {futures[future].task_id: future.result() for future in as_completed(futures)}
        for task in tasks:
            run = result.get(task.task_id) or SubTaskRun(task)
            run.attempts.append(completed[task.task_id])
            result[task.task_id] = run
        return result

    def _workspace(self, session: ResearchSession, round_number: int, task_id: str, attempt: int) -> Path:
        root = Path(session.research_dir or self.config.project_root)
        path = root / "artifacts" / "agent_workspaces" / f"R{round_number:03d}" / _safe(task_id) / f"attempt-{attempt:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _parse_review(self, output: str, runs: dict[str, SubTaskRun]) -> dict[str, object]:
        parsed = _json(output)
        if parsed:
            verdict = str(parsed.get("verdict") or "accept").lower()
            verdict = "revise" if verdict not in {"accept", "revise"} and ("revis" in verdict or "修正" in verdict) else verdict
            if verdict not in {"accept", "revise"}:
                verdict = "accept"
            revisions = parsed.get("revisions") if isinstance(parsed.get("revisions"), list) else []
            return {"verdict": verdict, "summary": str(parsed.get("summary") or output), "revisions": revisions}
        revise = "revise_required" in output.lower() or "verdict: revise" in output.lower() or "修正必要" in output or "再実行" in output
        return {"verdict": "revise" if revise else "accept", "summary": output, "revisions": [{"task_id": key, "instructions": "review指摘を踏まえて再検証"} for key in runs] if revise else []}

    def _revision_tasks(self, revisions: object, runs: dict[str, SubTaskRun]) -> list[SubTask]:
        selected: list[SubTask] = []
        if isinstance(revisions, list):
            for item in revisions:
                if isinstance(item, dict):
                    task_id = str(item.get("task_id") or item.get("id") or "")
                    if task_id in runs:
                        selected.append(runs[task_id].task)
        return selected or [run.task for run in runs.values()]

    def _plan_prompt(self, session: ResearchSession, round_number: int) -> str:
        return f'''STAGE: main_plan\nROLE: main\nROUND: {round_number}\nGOAL: {session.research_goal}\nCURRENT_QUESTION: {session.current_question or "未設定"}\n\n研究ゴールを最大{self.sub_count}個の相互依存を最小化したsubタスクへ分解してください。証拠収集、実装・実験、反証・リスク監査など異なる観点にしてください。\nJSONのみ: {{"summary":"...","subtasks":[{{"id":"S1","task":"...","deliverable":"..."}}],"confidence":"low|mid|high"}}'''

    def _sub_prompt(self, session: ResearchSession, round_number: int, task: SubTask, workspace: Path, previous: str | None, instruction: str | None) -> str:
        return f'''STAGE: sub_retry\nROLE: sub\nROUND: {round_number}\nTASK_ID: {task.task_id}\nGOAL: {session.research_goal}\nTASK: {task.task}\nDELIVERABLE: {task.deliverable}\nREVISION_INSTRUCTION: {instruction or "なし"}\nPRIOR_OUTPUT: {previous or "なし"}\n\nこのタスクだけを実行してください。このsub専用workspaceにのみ書き込む: {workspace}\n共有研究フォルダは参照用: {session.research_dir}\n他subのworkspaceへ書き込まない。ファイル削除、外部投稿、git push、秘密情報送信、課金API、sudo/chmod/chownは禁止。危険操作が必要なら実行せず1行で: APPROVAL_REQUIRED: operation=<操作>; reason=<理由>; impact=<影響>; dry_run_result=<未実行結果>\n長時間・大量生成が必要なら: IMPORTANT_NOTICE: operation=long_running_command:<操作>; reason=<理由>; impact=<影響>; dry_run_result=<未実行結果>\n結果、根拠、コマンド、変更ファイル、失敗、未確認事項、次の提案を返してください。'''

    def _review_prompt(self, session: ResearchSession, round_number: int, plan: dict[str, object], runs: dict[str, SubTaskRun], attempt: int) -> str:
        return f'''STAGE: review\nROLE: review\nROUND: {round_number}\nREVIEW_ATTEMPT: {attempt}\nGOAL: {session.research_goal}\nPLAN: {json.dumps(plan, ensure_ascii=False)}\nSUB_OUTPUTS:\n{_render_runs(runs)}\n\n根拠、再現性、相互矛盾、未確認事項、安全性を批判的に確認してください。修正が必要なら対象TASK_IDと具体的再実行指示を返してください。\nJSONのみ: {{"verdict":"accept|revise","summary":"...","revisions":[{{"task_id":"S1","instructions":"..."}}],"confidence":"low|mid|high"}}'''

    def _fresh_prompt(self, session: ResearchSession, round_number: int, runs: dict[str, SubTaskRun], reviews: list[dict[str, object]]) -> str:
        return f'''STAGE: fresh\nROLE: fresh\nROUND: {round_number}\nGOAL: {session.research_goal}\nSUB_OUTPUTS:\n{_render_runs(runs)}\nREVIEWS:\n{_render_reviews(reviews)}\n\n既出案の言い換えではなく、別仮説、反証例、見落とした比較軸、より単純な方法を根拠と検証方法つきで返してください。'''

    def _claude_prompt(self, session: ResearchSession, round_number: int, runs: dict[str, SubTaskRun], reviews: list[dict[str, object]]) -> str:
        return f'''STAGE: claude_consultation\nROLE: claude\nROUND: {round_number}\nGOAL: {session.research_goal}\nSUB_OUTPUTS:\n{_render_runs(runs)}\nREVIEWS:\n{_render_reviews(reviews)}\n\n重要判断だけ独立監査し、事実・推論・未確認事項を分離してください。'''

    def _integration_prompt(self, session: ResearchSession, round_number: int, plan: dict[str, object], runs: dict[str, SubTaskRun], reviews: list[dict[str, object]], fresh: AgentInvocation | None, claude: AgentInvocation | None) -> str:
        return f'''STAGE: main_integrate\nROLE: main\nROUND: {round_number}\nGOAL: {session.research_goal}\nPLAN: {json.dumps(plan, ensure_ascii=False)}\nSUB_OUTPUTS:\n{_render_runs(runs)}\nREVIEWS:\n{_render_reviews(reviews)}\nFRESH:\n{fresh.output if fresh else "未実行"}\nCLAUDE:\n{claude.output if claude else "未実行"}\n\n根拠とreviewを優先して統合し、失敗・矛盾・未確認事項を隠さないでください。\nJSONのみ: {{"summary":"...","decision":"...","confidence":"low|mid|high","next_action":"...","accepted_ideas":["..."],"rejected_ideas":["..."]}}'''


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _json(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                candidates.append(block)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _render_runs(runs: dict[str, SubTaskRun]) -> str:
    parts = []
    for task_id, run in runs.items():
        parts.append(f"## {task_id}: {run.task.task}\nDeliverable: {run.task.deliverable}")
        for index, attempt in enumerate(run.attempts, 1):
            parts.append(f"### attempt {index} stage={attempt.stage} ok={attempt.ok}\n{attempt.output}")
    return "\n\n".join(parts) or "sub outputなし"


def _render_reviews(reviews: list[dict[str, object]]) -> str:
    return "\n\n".join(f"review {item.get('attempt')} verdict={item.get('verdict')}\n{item.get('summary')}" for item in reviews) or "review outputなし"


def _first_operation(outputs: Iterable[str]) -> ProposedOperation | None:
    items = list(outputs)
    for prefix in ("APPROVAL_REQUIRED:", "IMPORTANT_NOTICE:"):
        for output in items:
            for line in output.splitlines():
                if not line.strip().startswith(prefix):
                    continue
                payload = line.split(":", 1)[1].strip()
                fields = {}
                for chunk in payload.split(";"):
                    if "=" in chunk:
                        key, value = chunk.split("=", 1)
                        fields[key.strip()] = value.strip()
                operation = fields.get("operation") or ("unstructured_approval_required" if prefix.startswith("APPROVAL") else "long_running_command:unstructured_notice")
                return ProposedOperation(operation, fields.get("reason", "Agentが操作候補を報告した。"), fields.get("impact", "影響未確認。"), fields.get("dry_run_result", "未実行。"))
    return None


def _strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _confidence(value: object) -> str:
    value = str(value or "mid").lower()
    return value if value in {"low", "mid", "high"} else "mid"


def _clip(text: str, limit: int) -> str:
    return text if limit <= 0 or len(text) <= limit else text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _safe(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
    return cleaned.strip("-")[:64] or "task"


def _redact(command: list[str]) -> list[str]:
    result, secret_next = [], False
    for part in command:
        lower = part.lower()
        if secret_next:
            result.append("***")
            secret_next = False
        elif any(key in lower for key in ("token=", "api_key=", "apikey=", "password=")):
            result.append(part.split("=", 1)[0] + "=***")
        else:
            result.append(part)
            secret_next = lower in {"--token", "--api-key", "--password"}
    return result
