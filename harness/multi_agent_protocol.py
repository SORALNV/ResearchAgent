from __future__ import annotations

import json
from typing import Any, Iterable

from harness.approval import ProposedOperation
from harness.multi_agent_types import AgentInvocation, SubTaskRun
from harness.process_manager import AgentCancelledError
from harness.state import ResearchSession


def _validate_plan(
    invocation: AgentInvocation,
    max_tasks: int = 100,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = _invocation_errors(invocation)
    parsed = _json_object(invocation.output)
    if parsed is None:
        return None, errors + ["main_plan output is not a JSON object"]

    summary = parsed.get("summary")
    subtasks = parsed.get("subtasks")
    confidence = str(parsed.get("confidence") or "").lower()
    if not isinstance(summary, str) or not summary.strip():
        errors.append("main_plan.summary must be a non-empty string")
    if confidence not in {"low", "mid", "high"}:
        errors.append("main_plan.confidence must be low|mid|high")
    if not isinstance(subtasks, list) or not (1 <= len(subtasks) <= max_tasks):
        errors.append(f"main_plan.subtasks must contain 1..{max_tasks} items")

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    if isinstance(subtasks, list):
        for index, item in enumerate(subtasks, 1):
            if not isinstance(item, dict):
                errors.append(f"subtask {index} must be an object")
                continue
            task_id = str(item.get("id") or item.get("task_id") or "").strip()
            task = str(item.get("task") or "").strip()
            deliverable = str(item.get("deliverable") or "").strip()
            if not task_id or task_id in seen:
                errors.append(f"subtask {index} has missing or duplicate id")
            if not task:
                errors.append(f"subtask {index}.task must be non-empty")
            if not deliverable:
                errors.append(f"subtask {index}.deliverable must be non-empty")
            if task_id and task and deliverable and task_id not in seen:
                seen.add(task_id)
                normalized.append(
                    {"id": task_id, "task": task, "deliverable": deliverable}
                )
    if errors:
        return None, errors
    return {
        "summary": summary.strip(),
        "subtasks": normalized,
        "confidence": confidence,
    }, []


def _validate_review(
    invocation: AgentInvocation,
    task_ids: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = _invocation_errors(invocation)
    parsed = _json_object(invocation.output)
    if parsed is None:
        return None, errors + ["review output is not a JSON object"]

    verdict = str(parsed.get("verdict") or "").lower()
    summary = parsed.get("summary")
    confidence = str(parsed.get("confidence") or "").lower()
    revisions = parsed.get("revisions")
    if verdict not in {"accept", "revise"}:
        errors.append("review.verdict must be accept|revise")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("review.summary must be a non-empty string")
    if confidence not in {"low", "mid", "high"}:
        errors.append("review.confidence must be low|mid|high")
    if not isinstance(revisions, list):
        errors.append("review.revisions must be a list")
        revisions = []

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(revisions):
        if not isinstance(item, dict):
            errors.append(f"review.revisions[{index}] must be an object")
            continue
        task_id = str(item.get("task_id") or item.get("id") or "").strip()
        instructions = str(item.get("instructions") or "").strip()
        if task_id not in task_ids:
            errors.append(f"review revision references unknown task: {task_id}")
        if not instructions:
            errors.append(f"review revision instructions missing for {task_id}")
        if task_id in task_ids and instructions:
            normalized.append({"task_id": task_id, "instructions": instructions})

    if verdict == "revise" and not normalized:
        errors.append("review verdict=revise requires at least one valid revision")
    if verdict == "accept" and normalized:
        errors.append("review verdict=accept must not include revisions")
    if errors:
        return None, errors
    return {
        "verdict": verdict,
        "summary": summary.strip(),
        "revisions": normalized,
        "confidence": confidence,
    }, []


def _validate_integration(
    invocation: AgentInvocation,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = _invocation_errors(invocation)
    parsed = _json_object(invocation.output)
    if parsed is None:
        return None, errors + ["main_integrate output is not a JSON object"]

    for key in ("summary", "decision", "next_action"):
        if not isinstance(parsed.get(key), str) or not str(parsed[key]).strip():
            errors.append(f"main_integrate.{key} must be a non-empty string")

    confidence = str(parsed.get("confidence") or "").lower()
    if confidence not in {"low", "mid", "high"}:
        errors.append("main_integrate.confidence must be low|mid|high")

    accepted = parsed.get("accepted_ideas")
    rejected = parsed.get("rejected_ideas")
    promotions = parsed.get("promote_artifacts", [])
    if not isinstance(accepted, list) or not all(isinstance(item, str) for item in accepted):
        errors.append("main_integrate.accepted_ideas must be a string list")
    if not isinstance(rejected, list) or not all(isinstance(item, str) for item in rejected):
        errors.append("main_integrate.rejected_ideas must be a string list")
    if not isinstance(promotions, list):
        errors.append("main_integrate.promote_artifacts must be a list")

    round_status = str(
        parsed.get("round_status")
        or _infer_round_status(str(parsed.get("decision") or ""))
    ).lower()
    if round_status not in {"continue", "completed", "blocked", "failed"}:
        errors.append(
            "main_integrate.round_status must be continue|completed|blocked|failed"
        )

    try:
        progress_score = float(parsed.get("progress_score", 0.5))
    except (TypeError, ValueError):
        errors.append("main_integrate.progress_score must be numeric")
        progress_score = 0.0
    if not 0.0 <= progress_score <= 1.0:
        errors.append("main_integrate.progress_score must be between 0 and 1")

    evidence = parsed.get("new_evidence_ids", [])
    blockers = parsed.get("unresolved_blockers", [])
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        errors.append("main_integrate.new_evidence_ids must be a string list")
        evidence = []
    if not isinstance(blockers, list) or not all(isinstance(item, str) for item in blockers):
        errors.append("main_integrate.unresolved_blockers must be a string list")
        blockers = []

    if errors:
        return None, errors
    return {
        "summary": str(parsed["summary"]).strip(),
        "decision": str(parsed["decision"]).strip(),
        "confidence": confidence,
        "next_action": str(parsed["next_action"]).strip(),
        "accepted_ideas": [str(item) for item in accepted],
        "rejected_ideas": [str(item) for item in rejected],
        "promote_artifacts": promotions,
        "round_status": round_status,
        "progress_score": progress_score,
        "new_evidence_ids": [str(item) for item in evidence],
        "unresolved_blockers": [str(item) for item in blockers],
    }, []


def _invocation_errors(invocation: AgentInvocation) -> list[str]:
    errors: list[str] = []
    if invocation.cancelled:
        errors.append(f"{invocation.stage} was cancelled")
    if invocation.timed_out:
        errors.append(f"{invocation.stage} timed out")
    if invocation.skipped:
        errors.append(f"{invocation.stage} was skipped")
    if invocation.returncode != 0:
        errors.append(
            f"{invocation.stage} returned {invocation.returncode}: "
            f"{invocation.stderr[-500:] or invocation.output[-500:]}"
        )
    return errors


def _json_object(text: str) -> dict[str, Any] | None:
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
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _runs_from_state(state: dict[str, Any]) -> dict[str, SubTaskRun]:
    return {
        str(task_id): SubTaskRun.from_dict(item)
        for task_id, item in dict(state.get("runs") or {}).items()
        if isinstance(item, dict)
    }


def _attempt_invocations(value: object) -> list[AgentInvocation]:
    result: list[AgentInvocation] = []
    if not isinstance(value, list):
        return result
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("invocation"), dict):
            result.append(AgentInvocation.from_dict(item["invocation"]))
    return result


def _raise_cancelled_runs(
    runs: Iterable[SubTaskRun],
    checkpoint,
    state: dict[str, Any],
    reason: str,
) -> None:
    for run in runs:
        if run.attempts and run.latest.cancelled:
            checkpoint.mark_status(
                state,
                "interrupted",
                stage=run.latest.stage,
                error=reason,
            )
            raise AgentCancelledError(reason)


def _raise_cancelled_invocation(
    invocation: AgentInvocation,
    checkpoint,
    state: dict[str, Any],
    reason: str,
) -> None:
    if invocation.cancelled:
        checkpoint.mark_status(
            state,
            "interrupted",
            stage=invocation.stage,
            error=reason,
        )
        raise AgentCancelledError(reason)


def _render_runs(runs: dict[str, SubTaskRun]) -> str:
    parts: list[str] = []
    for task_id, run in runs.items():
        parts.append(f"## {task_id}: {run.task.task}\nDeliverable: {run.task.deliverable}")
        for index, attempt in enumerate(run.attempts, 1):
            artifacts = "\n".join(
                f"- {item.path} sha256={item.sha256} size={item.size_bytes}"
                for item in attempt.artifacts
            ) or "- なし"
            parts.append(
                f"### attempt {index} stage={attempt.stage} ok={attempt.ok}\n"
                f"workspace: {attempt.workspace or 'なし'}\n"
                f"artifacts:\n{artifacts}\noutput:\n{attempt.output}"
            )
    return "\n\n".join(parts) or "sub outputなし"


def _render_reviews(reviews: list[dict[str, object]]) -> str:
    return "\n\n".join(
        f"review {item.get('attempt')} verdict={item.get('verdict')}\n"
        f"{item.get('summary')}"
        for item in reviews
    ) or "review outputなし"


def _collect_operations(outputs: Iterable[str]) -> list[ProposedOperation]:
    operations: list[ProposedOperation] = []
    for output in outputs:
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("APPROVAL_REQUIRED:"):
                operations.append(_parse_operation_line(stripped, approval=True))
            elif stripped.startswith("IMPORTANT_NOTICE:"):
                operations.append(_parse_operation_line(stripped, approval=False))
    return _dedupe_operations(operations)


def _parse_operation_line(line: str, *, approval: bool) -> ProposedOperation:
    payload = line.split(":", 1)[1].strip()
    fields: dict[str, str] = {}
    for chunk in payload.split(";"):
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            fields[key.strip()] = value.strip()
    default = (
        "unstructured_approval_required"
        if approval
        else "long_running_command:unstructured_notice"
    )
    return ProposedOperation(
        operation=fields.get("operation") or default,
        reason=fields.get("reason", "Agentが操作候補を報告した。"),
        impact=fields.get("impact", "影響未確認。"),
        dry_run_result=fields.get("dry_run_result", "未実行。"),
    )


def _dedupe_operations(
    operations: Iterable[ProposedOperation],
) -> list[ProposedOperation]:
    result: list[ProposedOperation] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in operations:
        key = (item.operation, item.reason, item.impact, item.dry_run_result)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _dedupe_trace(trace: Iterable[AgentInvocation]) -> list[AgentInvocation]:
    result: list[AgentInvocation] = []
    seen: set[tuple[str, str, str | None, str | None, str]] = set()
    for item in trace:
        key = (item.role, item.stage, item.task_id, item.workspace, item.output)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _restore_cost_from_checkpoint(
    session: ResearchSession,
    state: dict[str, Any],
) -> None:
    cost = state.get("cost")
    if not isinstance(cost, dict):
        return
    for key in ("api_calls", "estimated_tokens", "literature_searches", "agent_calls"):
        try:
            value = int(cost.get(key) or 0)
        except (TypeError, ValueError):
            continue
        setattr(session.cost, key, max(int(getattr(session.cost, key, 0)), value))


def _infer_round_status(decision: str) -> str:
    normalized = decision.strip().lower()
    if any(token in normalized for token in ("complete", "done", "完了", "終了")):
        return "completed"
    if any(token in normalized for token in ("block", "保留", "承認待ち")):
        return "blocked"
    if any(token in normalized for token in ("fail", "失敗")):
        return "failed"
    return "continue"


# Compatibility helpers imported by MultiAgentRunnerSupport.
def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _safe(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )
    return cleaned.strip("-")[:64] or "task"
