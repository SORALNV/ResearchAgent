from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from harness.platform.models import (
    Domain,
    JobSpec,
    ResourceRequest,
    SteeringEvent,
    SteeringKind,
)
from harness.platform.registry import PlatformRegistry


@dataclass(frozen=True)
class ToolExecutionContext:
    registry: PlatformRegistry
    project_id: str | None = None
    work_session_id: str | None = None
    job_id: str | None = None
    actor: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolExecutionContext], Mapping[str, Any]]
    mutating: bool = False
    approval_required: bool = False

    def openai_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


class HarnessToolRegistry:
    """Allowlisted tool surface exposed to an internal model.

    Tool execution never gives a model direct access to secrets. Mutating tools
    operate through the durable PlatformRegistry and compute scheduler. Tools
    marked approval_required return a proposal instead of performing the action.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RuntimeTool] = {}

    def register(self, tool: RuntimeTool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"duplicate or empty tool name: {tool.name}")
        self._tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.openai_definition() for tool in self._tools.values()]

    def execute(
        self,
        name: str,
        arguments: str | Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            parsed = (
                json.loads(arguments)
                if isinstance(arguments, str)
                else dict(arguments)
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return {"ok": False, "error": f"invalid tool arguments: {exc}"}
        if not isinstance(parsed, dict):
            return {"ok": False, "error": "tool arguments must be an object"}
        if tool.approval_required:
            return {
                "ok": False,
                "requires_approval": True,
                "proposal": {
                    "tool": tool.name,
                    "arguments": parsed,
                    "actor": context.actor,
                    "work_session_id": context.work_session_id,
                },
            }
        try:
            result = dict(tool.handler(parsed, context))
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"ok": True, "result": result}

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def build_default_harness_tools() -> HarnessToolRegistry:
    registry = HarnessToolRegistry()
    registry.register(
        RuntimeTool(
            name="get_work_session",
            description="Read the current work session and its project.",
            parameters={
                "type": "object",
                "properties": {
                    "work_session_id": {"type": "string"},
                },
                "required": ["work_session_id"],
                "additionalProperties": False,
            },
            handler=_get_work_session,
        )
    )
    registry.register(
        RuntimeTool(
            name="list_jobs",
            description="List durable jobs for a work session.",
            parameters={
                "type": "object",
                "properties": {
                    "work_session_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["work_session_id", "limit"],
                "additionalProperties": False,
            },
            handler=_list_jobs,
        )
    )
    registry.register(
        RuntimeTool(
            name="get_job_status",
            description="Read one job, its result, and recent progress events.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "event_limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["job_id", "event_limit"],
                "additionalProperties": False,
            },
            handler=_get_job_status,
        )
    )
    registry.register(
        RuntimeTool(
            name="add_steering",
            description=(
                "Record a question, constraint, redirect, or hypothesis for the "
                "next safe job checkpoint."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "work_session_id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            SteeringKind.QUESTION.value,
                            SteeringKind.CONSTRAINT.value,
                            SteeringKind.REDIRECT.value,
                            SteeringKind.NEW_HYPOTHESIS.value,
                        ],
                    },
                    "instruction": {"type": "string", "minLength": 1},
                    "apply_after": {"type": "string"},
                    "job_id": {"type": ["string", "null"]},
                },
                "required": [
                    "work_session_id",
                    "kind",
                    "instruction",
                    "apply_after",
                    "job_id",
                ],
                "additionalProperties": False,
            },
            handler=_add_steering,
            mutating=True,
        )
    )
    registry.register(
        RuntimeTool(
            name="propose_job",
            description=(
                "Create a durable compute job proposal. The scheduler selects "
                "Kaggle, a remote GPU, a GPU VM, or local CPU later."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "work_session_id": {"type": "string"},
                    "domain": {
                        "type": "string",
                        "enum": [Domain.RESEARCH.value, Domain.KAGGLE.value],
                    },
                    "task_type": {"type": "string", "minLength": 1},
                    "entrypoint": {"type": "string"},
                    "backend_preferences": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "resources": {
                        "type": "object",
                        "properties": {
                            "accelerator": {"type": "string"},
                            "min_vram_gb": {"type": "number", "minimum": 0},
                            "preferred_gpu_count": {"type": "integer", "minimum": 0},
                            "cpu_cores": {"type": "integer", "minimum": 1},
                            "ram_gb": {"type": "number", "minimum": 0.1},
                            "max_runtime_minutes": {"type": "integer", "minimum": 1},
                            "network_required": {"type": "boolean"},
                            "capabilities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "accelerator",
                            "min_vram_gb",
                            "preferred_gpu_count",
                            "cpu_cores",
                            "ram_gb",
                            "max_runtime_minutes",
                            "network_required",
                            "capabilities",
                        ],
                        "additionalProperties": False,
                    },
                    "inputs": {"type": "object"},
                    "outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "metadata": {"type": "object"},
                },
                "required": [
                    "work_session_id",
                    "domain",
                    "task_type",
                    "entrypoint",
                    "backend_preferences",
                    "resources",
                    "inputs",
                    "outputs",
                    "metadata",
                ],
                "additionalProperties": False,
            },
            handler=_propose_job,
            mutating=True,
        )
    )
    registry.register(
        RuntimeTool(
            name="request_computer_use",
            description=(
                "Request an interactive browser/computer session only when no "
                "API or deterministic tool can complete the task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "purpose": {"type": "string", "minLength": 1},
                    "target": {"type": "string", "minLength": 1},
                    "reason_api_is_insufficient": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": [
                    "purpose",
                    "target",
                    "reason_api_is_insufficient",
                ],
                "additionalProperties": False,
            },
            handler=lambda arguments, context: arguments,
            approval_required=True,
        )
    )
    return registry


def _get_work_session(
    arguments: dict[str, Any], context: ToolExecutionContext
) -> Mapping[str, Any]:
    session = context.registry.get_work_session(str(arguments["work_session_id"]))
    if session is None:
        raise KeyError(f"unknown work session: {arguments['work_session_id']}")
    project = context.registry.get_project(session.project_id)
    return {
        "work_session": session.to_dict(),
        "project": project.to_dict() if project else None,
        "pending_steering": [
            item.to_dict()
            for item in context.registry.list_pending_steering(session.session_id)
        ],
    }


def _list_jobs(
    arguments: dict[str, Any], context: ToolExecutionContext
) -> Mapping[str, Any]:
    jobs = context.registry.list_jobs(
        work_session_id=str(arguments["work_session_id"]),
        limit=int(arguments["limit"]),
    )
    return {"jobs": [job.to_dict() for job in jobs]}


def _get_job_status(
    arguments: dict[str, Any], context: ToolExecutionContext
) -> Mapping[str, Any]:
    job = context.registry.get_job(str(arguments["job_id"]))
    if job is None:
        raise KeyError(f"unknown job: {arguments['job_id']}")
    events = context.registry.list_events(
        job.spec.work_session_id,
        limit=int(arguments["event_limit"]),
    )
    return {
        "job": job.to_dict(),
        "events": [
            event.to_dict() for event in events if event.job_id == job.spec.job_id
        ][-int(arguments["event_limit"]):],
    }


def _add_steering(
    arguments: dict[str, Any], context: ToolExecutionContext
) -> Mapping[str, Any]:
    event = SteeringEvent.new(
        work_session_id=str(arguments["work_session_id"]),
        kind=SteeringKind(str(arguments["kind"])),
        instruction=str(arguments["instruction"]),
        apply_after=str(arguments.get("apply_after") or "next_checkpoint"),
        job_id=(str(arguments["job_id"]) if arguments.get("job_id") else None),
        metadata={"actor": context.actor},
    )
    return context.registry.add_steering(event).to_dict()


def _propose_job(
    arguments: dict[str, Any], context: ToolExecutionContext
) -> Mapping[str, Any]:
    spec = JobSpec.new(
        work_session_id=str(arguments["work_session_id"]),
        domain=Domain(str(arguments["domain"])),
        task_type=str(arguments["task_type"]),
        entrypoint=str(arguments.get("entrypoint") or ""),
        backend_preferences=[
            str(item) for item in arguments.get("backend_preferences", [])
        ],
        resources=ResourceRequest.from_dict(
            arguments.get("resources") if isinstance(arguments.get("resources"), dict) else None
        ),
        inputs=(
            dict(arguments.get("inputs") or {})
            if isinstance(arguments.get("inputs"), dict)
            else {}
        ),
        outputs=[str(item) for item in arguments.get("outputs", [])],
        metadata={
            **(
                dict(arguments.get("metadata") or {})
                if isinstance(arguments.get("metadata"), dict)
                else {}
            ),
            "proposed_by": context.actor,
        },
    )
    return context.registry.create_job(spec).to_dict()
