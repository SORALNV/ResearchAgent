from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.approval import ProposedOperation
from harness.artifacts import ArtifactRecord, build_artifact_manifest
from harness.config import HarnessConfig
from harness.cost import estimate_tokens
from harness.process_manager import ProcessCancellationController, build_agent_environment
from harness.sandbox import SandboxUnavailableError, build_agent_command
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
    cancelled: bool = False
    workspace: str | None = None
    artifacts: tuple[ArtifactRecord, ...] = ()
    artifact_warnings: tuple[str, ...] = ()
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0

    @property
    def ok(self) -> bool:
        return not (self.skipped or self.timed_out or self.cancelled) and self.returncode == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "stage": self.stage,
            "task_id": self.task_id,
            "command": list(self.command),
            "output": self.output,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "duration_seconds": round(self.duration_seconds, 3),
            "skipped": self.skipped,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "workspace": self.workspace,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "artifact_warnings": list(self.artifact_warnings),
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentInvocation":
        return cls(
            role=str(data.get("role") or "unknown"),
            stage=str(data.get("stage") or "unknown"),
            task_id=str(data["task_id"]) if data.get("task_id") is not None else None,
            command=tuple(str(item) for item in data.get("command", [])),
            output=str(data.get("output") or ""),
            stderr=str(data.get("stderr") or ""),
            returncode=int(data.get("returncode") or 0),
            duration_seconds=float(data.get("duration_seconds") or 0.0),
            skipped=bool(data.get("skipped", False)),
            timed_out=bool(data.get("timed_out", False)),
            cancelled=bool(data.get("cancelled", False)),
            workspace=str(data["workspace"]) if data.get("workspace") else None,
            artifacts=tuple(
                ArtifactRecord.from_dict(item)
                for item in data.get("artifacts", [])
                if isinstance(item, dict)
            ),
            artifact_warnings=tuple(str(item) for item in data.get("artifact_warnings", [])),
            estimated_input_tokens=int(data.get("estimated_input_tokens") or 0),
            estimated_output_tokens=int(data.get("estimated_output_tokens") or 0),
        )


@dataclass(frozen=True)
class SubTask:
    task_id: str
    task: str
    deliverable: str

    def to_dict(self) -> dict[str, str]:
        return {"task_id": self.task_id, "task": self.task, "deliverable": self.deliverable}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubTask":
        return cls(
            task_id=str(data["task_id"]),
            task=str(data["task"]),
            deliverable=str(data.get("deliverable") or "検証可能な結果"),
        )


@dataclass
class SubTaskRun:
    task: SubTask
    attempts: list[AgentInvocation] = field(default_factory=list)

    @property
    def latest(self) -> AgentInvocation:
        return self.attempts[-1]

    def to_dict(self) -> dict[str, object]:
        return {"task": self.task.to_dict(), "attempts": [item.to_dict() for item in self.attempts]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubTaskRun":
        return cls(
            task=SubTask.from_dict(dict(data["task"])),
            attempts=[
                AgentInvocation.from_dict(item)
                for item in data.get("attempts", [])
                if isinstance(item, dict)
            ],
        )


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
    proposed_operations: list[ProposedOperation] = field(default_factory=list)
    promoted_artifacts: list[dict[str, object]] = field(default_factory=list)
    protocol_errors: list[str] = field(default_factory=list)
    round_status: str = "continue"
    progress_score: float = 0.5
    new_evidence_ids: list[str] = field(default_factory=list)
    unresolved_blockers: list[str] = field(default_factory=list)
    round_number: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "main_agent_summary": self.main_agent_summary,
            "subtask": self.subtask,
            "sub_agent_output": self.sub_agent_output,
            "review_output": self.review_output,
            "claude_consultation": self.claude_consultation,
            "fresh_agent_output": self.fresh_agent_output,
            "conversation_sessions": self.conversation_sessions,
            "proposed_operation": _operation_dict(self.proposed_operation),
            "proposed_operations": [_operation_dict(item) for item in self.proposed_operations],
            "accepted_ideas": self.accepted_ideas,
            "rejected_ideas": self.rejected_ideas,
            "decision": self.decision,
            "confidence": self.confidence,
            "next_action": self.next_action,
            "promoted_artifacts": self.promoted_artifacts,
            "protocol_errors": self.protocol_errors,
            "round_status": self.round_status,
            "progress_score": self.progress_score,
            "new_evidence_ids": self.new_evidence_ids,
            "unresolved_blockers": self.unresolved_blockers,
            "round_number": self.round_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RealRoundOutput":
        operations = [
            _operation_from_dict(item)
            for item in data.get("proposed_operations", [])
            if isinstance(item, dict)
        ]
        first = _operation_from_dict(data.get("proposed_operation"))
        operations = [item for item in operations if item is not None]
        if first and not operations:
            operations = [first]
        return cls(
            main_agent_summary=str(data.get("main_agent_summary") or ""),
            subtask=str(data.get("subtask") or ""),
            sub_agent_output=str(data.get("sub_agent_output") or ""),
            review_output=str(data.get("review_output") or ""),
            claude_consultation=(
                str(data["claude_consultation"])
                if data.get("claude_consultation") is not None
                else None
            ),
            fresh_agent_output=(
                str(data["fresh_agent_output"])
                if data.get("fresh_agent_output") is not None
                else None
            ),
            conversation_sessions=[
                dict(item)
                for item in data.get("conversation_sessions", [])
                if isinstance(item, dict)
            ],
            proposed_operation=first or (operations[0] if operations else None),
            proposed_operations=operations,
            accepted_ideas=_string_list(data.get("accepted_ideas")),
            rejected_ideas=_string_list(data.get("rejected_ideas")),
            decision=str(data.get("decision") or "blocked"),
            confidence=_confidence(data.get("confidence")),
            next_action=str(data.get("next_action") or "人間確認"),
            promoted_artifacts=[
                dict(item)
                for item in data.get("promoted_artifacts", [])
                if isinstance(item, dict)
            ],
            protocol_errors=_string_list(data.get("protocol_errors")),
            round_status=_round_status(data.get("round_status")),
            progress_score=_progress_score(data.get("progress_score")),
            new_evidence_ids=_string_list(data.get("new_evidence_ids")),
            unresolved_blockers=_string_list(data.get("unresolved_blockers")),
            round_number=int(data.get("round_number") or 0),
        )


class AgentCommandExecutor:
    def __init__(
        self,
        config: HarnessConfig,
        lock: threading.Lock,
        cancellation: ProcessCancellationController,
    ) -> None:
        self.config = config
        self.lock = lock
        self.cancellation = cancellation

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
        input_tokens = estimate_tokens(prompt)
        if self.cancellation.cancelled:
            return AgentInvocation(
                role=role,
                stage=stage,
                task_id=task_id,
                command=(),
                output=f"Agent cancelled before launch: {self.cancellation.reason}",
                stderr="",
                returncode=130,
                duration_seconds=0.0,
                cancelled=True,
                estimated_input_tokens=input_tokens,
            )
        if not command_text:
            return AgentInvocation(
                role=role,
                stage=stage,
                task_id=task_id,
                command=(),
                output=f"Real {role} agent skipped: command not configured.",
                stderr="",
                returncode=127,
                duration_seconds=0.0,
                skipped=True,
                estimated_input_tokens=input_tokens,
            )

        cwd = working_dir or Path(session.research_dir or self.config.project_root)
        cwd.mkdir(parents=True, exist_ok=True)
        environment = build_agent_environment(
            self.config,
            session,
            role=role,
            stage=stage,
            task_id=task_id,
            working_dir=cwd,
        )
        try:
            command = build_agent_command(
                self.config,
                command_text=command_text,
                sandbox_mode=sandbox,
                working_dir=cwd,
                research_root=Path(session.research_dir or self.config.project_root),
                environment=environment,
            )
        except (SandboxUnavailableError, ValueError, OSError) as exc:
            return AgentInvocation(
                role=role,
                stage=stage,
                task_id=task_id,
                command=(),
                output=f"Agent sandbox setup failed: {exc}",
                stderr=str(exc),
                returncode=126,
                duration_seconds=time.monotonic() - started,
                skipped=True,
                workspace=str(cwd),
                estimated_input_tokens=input_tokens,
            )

        with self.lock:
            if self.config.max_agent_calls > 0 and session.cost.agent_calls >= self.config.max_agent_calls:
                return AgentInvocation(
                    role=role,
                    stage=stage,
                    task_id=task_id,
                    command=tuple(_redact(command)),
                    output="Real agent skipped: MAX_AGENT_CALLS reached.",
                    stderr="",
                    returncode=125,
                    duration_seconds=time.monotonic() - started,
                    skipped=True,
                    workspace=str(cwd),
                    estimated_input_tokens=input_tokens,
                )
            session.cost.agent_calls += 1
            session.cost.estimated_tokens += input_tokens

        redacted = tuple(_redact(command))
        process: subprocess.Popen[str] | None = None
        output = ""
        stderr = ""
        returncode = 127
        timed_out = False
        cancelled = False
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=environment,
                start_new_session=os.name != "nt",
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            if not self.cancellation.register(process):
                self.cancellation.terminate_one(process)
                cancelled = True
                output = f"Agent cancelled before execution: {self.cancellation.reason}"
                returncode = 130
            else:
                try:
                    output, stderr = process.communicate(
                        input=prompt,
                        timeout=self.config.max_command_seconds,
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self.cancellation.terminate_one(process)
                    output, stderr = process.communicate()
                returncode = process.returncode if process.returncode is not None else 127
                cancelled = self.cancellation.cancelled
        except OSError as exc:
            output = f"Real {role} agent failed to start: {exc}"
            stderr = str(exc)
        finally:
            if process is not None:
                self.cancellation.unregister(process)

        output = (output or "").strip()
        stderr = (stderr or "").strip()
        if cancelled:
            output = output or f"Agent cancelled: {self.cancellation.reason}"
            returncode = returncode or 130
        elif timed_out:
            output = output or f"Real {role} agent timeout after {self.config.max_command_seconds}s."
            returncode = 124
        elif returncode and not output:
            output = (
                f"Real {role} agent failed: returncode={returncode}; "
                f"stderr={stderr[-2000:] or 'なし'}"
            )
        elif not returncode and not output:
            output = "Real agent completed without output. 結果は未確認。"

        output_tokens = estimate_tokens(output) + estimate_tokens(stderr)
        artifacts: list[ArtifactRecord] = []
        artifact_warnings: list[str] = []
        if sandbox == "workspace-write":
            artifacts, artifact_warnings = build_artifact_manifest(
                cwd,
                max_files=self.config.artifact_max_files,
                max_bytes=self.config.artifact_max_bytes,
            )
        with self.lock:
            session.cost.estimated_tokens += output_tokens

        return AgentInvocation(
            role=role,
            stage=stage,
            task_id=task_id,
            command=redacted,
            output=_clip(output, self.config.agent_output_char_limit),
            stderr=_clip(stderr, self.config.agent_output_char_limit),
            returncode=returncode,
            duration_seconds=time.monotonic() - started,
            timed_out=timed_out,
            cancelled=cancelled,
            workspace=str(cwd),
            artifacts=tuple(artifacts),
            artifact_warnings=tuple(artifact_warnings),
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
        )


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _redact(command: list[str]) -> list[str]:
    result: list[str] = []
    secret_next = False
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


def _operation_dict(operation: ProposedOperation | None) -> dict[str, str] | None:
    if operation is None:
        return None
    return {
        "operation": operation.operation,
        "reason": operation.reason,
        "impact": operation.impact,
        "dry_run_result": operation.dry_run_result,
    }


def _operation_from_dict(data: object) -> ProposedOperation | None:
    if not isinstance(data, dict):
        return None
    return ProposedOperation(
        operation=str(data.get("operation") or ""),
        reason=str(data.get("reason") or ""),
        impact=str(data.get("impact") or ""),
        dry_run_result=str(data.get("dry_run_result") or ""),
    )


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _confidence(value: object) -> str:
    normalized = str(value or "mid").lower()
    return normalized if normalized in {"low", "mid", "high"} else "mid"


def _round_status(value: object) -> str:
    normalized = str(value or "continue").lower()
    return normalized if normalized in {"continue", "completed", "blocked", "failed"} else "continue"


def _progress_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, score))
