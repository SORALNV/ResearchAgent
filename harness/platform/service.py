from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from harness.compute.base import BackendCapabilities, ComputeBroker
from harness.compute.kaggle import KaggleNotebookBackend
from harness.compute.local import LocalProcessBackend
from harness.compute.remote import RemoteWorkerBackend, RemoteWorkerDescriptor
from harness.compute.scheduler import JobScheduler
from harness.config import HarnessConfig
from harness.platform.config import PlatformConfig
from harness.platform.models import (
    Domain,
    EventKind,
    JobEvent,
    JobStatus,
    Project,
    SteeringEvent,
    SteeringKind,
    WorkSession,
    WorkSessionStatus,
)
from harness.platform.registry import PlatformRegistry
from harness.runtime.base import (
    RuntimeCapability,
    RuntimeContext,
    RuntimeRequest,
    RuntimeResult,
)
from harness.runtime.codex_cli import CodexCliRuntime
from harness.runtime.computer import ComputerUsePolicy, PlaywrightComputerDriver
from harness.runtime.openai_responses import (
    OpenAIResponsesRuntime,
    UrllibResponsesTransport,
)
from harness.runtime.router import AgentRuntimeRouter
from harness.runtime.tools import ToolExecutionContext, build_default_harness_tools


@dataclass
class CoreComponents:
    config: PlatformConfig
    harness_config: HarnessConfig
    registry: PlatformRegistry
    runtime_router: AgentRuntimeRouter
    compute_broker: ComputeBroker
    scheduler: JobScheduler


class PlatformService:
    """Domain-neutral Core used by the Windows host and Jetson-compatible Edge.

    Discord, HTTP, and future native clients all use this service. The service
    records conversation and steering as durable events, lets internal models use
    typed harness tools, and delegates long-running work to ComputeBackend.
    """

    def __init__(self, components: CoreComponents) -> None:
        self.components = components
        self.config = components.config
        self.registry = components.registry
        self.runtime_router = components.runtime_router
        self.scheduler = components.scheduler

    @classmethod
    def from_env(
        cls,
        project_root: str | Path | None = None,
    ) -> "PlatformService":
        platform_config = PlatformConfig.from_env(project_root)
        harness_config = HarnessConfig.from_env(project_root)
        registry = PlatformRegistry(platform_config.database_path)
        tools = build_default_harness_tools()

        runtimes = []
        if platform_config.codex_command:
            runtimes.append(
                CodexCliRuntime(
                    harness_config,
                    command=platform_config.codex_command,
                )
            )
        if platform_config.openai_api_key and platform_config.openai_model:
            computer_factory = None
            if platform_config.computer_use_enabled:
                computer_factory = lambda request: PlaywrightComputerDriver(
                    start_url=str(
                        request.metadata.get("computer_start_url")
                        or platform_config.computer_use_start_url
                    ),
                    headless=platform_config.computer_use_headless,
                    policy=ComputerUsePolicy(
                        allowed_domains=platform_config.computer_use_allowed_domains,
                        max_actions=int(request.metadata.get("computer_max_actions") or 80),
                    ),
                )
            runtimes.append(
                OpenAIResponsesRuntime(
                    model=platform_config.openai_model,
                    transport=UrllibResponsesTransport(
                        api_key=platform_config.openai_api_key,
                        base_url=platform_config.openai_base_url,
                        organization=platform_config.openai_organization,
                        project=platform_config.openai_project,
                    ),
                    tools=tools,
                    tool_context_factory=lambda request: ToolExecutionContext(
                        registry=registry,
                        project_id=request.context.project_id,
                        work_session_id=request.context.work_session_id,
                        job_id=request.context.job_id,
                        actor="openai_runtime",
                        metadata=request.context.metadata,
                    ),
                    computer_tool=platform_config.openai_computer_tool,
                    computer_driver_factory=computer_factory,
                )
            )

        runtime_router = AgentRuntimeRouter(
            runtimes,
            role_preferences={
                "conversation": ("openai_responses", "codex_cli"),
                "planner": ("openai_responses", "codex_cli"),
                "coder": ("codex_cli", "openai_responses"),
                "reviewer": ("openai_responses", "codex_cli"),
            },
        )

        backends = [
            LocalProcessBackend(
                max_cpu_cores=max(1, os.cpu_count() or 1),
            ),
            KaggleNotebookBackend(
                command=platform_config.kaggle_command,
                api_token=platform_config.kaggle_api_token,
                username=platform_config.kaggle_username,
            ),
        ]
        paid_names = set(platform_config.paid_backends)
        for raw in platform_config.remote_workers:
            name = str(raw.get("name") or "remote_gpu").strip()
            capabilities = _remote_capabilities(raw)
            backends.append(
                RemoteWorkerBackend(
                    RemoteWorkerDescriptor(
                        name=name,
                        base_url=str(raw.get("base_url") or ""),
                        token=str(raw.get("token") or ""),
                        paid=bool(raw.get("paid", False)),
                        capabilities=capabilities,
                    )
                )
            )
            if bool(raw.get("paid", False)):
                paid_names.add(name)

        broker = ComputeBroker(
            backends,
            paid_backends=tuple(sorted(paid_names)),
            allow_kaggle_for_research=platform_config.allow_kaggle_for_research,
        )
        scheduler = JobScheduler(
            registry=registry,
            broker=broker,
            root_dir=platform_config.data_dir,
            max_concurrent_jobs=platform_config.max_concurrent_jobs,
            poll_interval_seconds=platform_config.scheduler_poll_seconds,
        )
        return cls(
            CoreComponents(
                config=platform_config,
                harness_config=harness_config,
                registry=registry,
                runtime_router=runtime_router,
                compute_broker=broker,
                scheduler=scheduler,
            )
        )

    def start(self) -> None:
        self.scheduler.start(recover=True)

    def stop(self) -> None:
        self.runtime_router.cancel_all("Core shutdown")
        self.scheduler.stop(wait=False, cancel_active=False)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "database": str(self.config.database_path),
            "runtimes": self.runtime_router.available_runtimes(),
            "compute_backends": {
                name: {
                    "available": backend.available()[0],
                    "detail": backend.available()[1],
                    "capabilities": backend.capabilities.to_dict(),
                }
                for name, backend in self.components.compute_broker.backends.items()
            },
            "scheduler": self.scheduler.snapshot(),
        }

    def create_project(
        self,
        *,
        domain: Domain | str,
        title: str,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Project:
        project = Project.new(
            domain=domain,
            title=title,
            description=description,
            metadata=metadata,
        )
        stored = self.registry.create_project(project)
        self._project_workspace(stored).mkdir(parents=True, exist_ok=True)
        return stored

    def create_work_session(
        self,
        *,
        project_id: str,
        title: str,
        objective: str,
        parent_session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkSession:
        project = self._require_project(project_id)
        session = WorkSession.new(
            project_id=project_id,
            title=title,
            objective=objective,
            parent_session_id=parent_session_id,
            metadata=metadata,
        )
        stored = self.registry.create_work_session(session)
        workspace = self._session_workspace(stored)
        workspace.mkdir(parents=True, exist_ok=True)
        self._write_session_context(project, stored, workspace)
        self._append_event(
            JobEvent.new(
                work_session_id=stored.session_id,
                kind=EventKind.STATUS,
                message=f"Work session created: {stored.title}",
                payload={
                    "project_id": project_id,
                    "domain": project.domain.value,
                    "status": stored.status.value,
                    "workspace": str(workspace),
                },
            )
        )
        return stored

    def attach_discord_route(
        self,
        session_id: str,
        *,
        guild_id: str | int,
        parent_channel_id: str | int,
        thread_id: str | int,
        live_message_id: str | int | None = None,
    ) -> WorkSession:
        return self.registry.update_work_session(
            session_id,
            discord_guild_id=guild_id,
            discord_parent_channel_id=parent_channel_id,
            discord_thread_id=thread_id,
            discord_live_message_id=live_message_id,
        )

    def handle_message(
        self,
        *,
        session_id: str,
        text: str,
        actor: str,
        correlation_id: str,
        mode: str = "auto",
        steering_kind: SteeringKind | str | None = None,
        computer_use_allowed: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        project = self._require_project(session.project_id)
        if not self.registry.claim_interaction(
            correlation_id,
            work_session_id=session_id,
        ):
            previous = self.registry.get_interaction_result(correlation_id)
            return previous or {
                "ok": True,
                "duplicate": True,
                "work_session_id": session_id,
            }

        self._append_event(
            JobEvent.new(
                work_session_id=session_id,
                kind=EventKind.USER_MESSAGE,
                message=text,
                payload={"actor": actor, "mode": mode},
            )
        )

        inferred = steering_kind or _infer_steering(text)
        if inferred is not None and SteeringKind(str(inferred)) != SteeringKind.QUESTION:
            event = self.registry.add_steering(
                SteeringEvent.new(
                    work_session_id=session_id,
                    kind=SteeringKind(str(inferred)),
                    instruction=text,
                    apply_after=str((metadata or {}).get("apply_after") or "next_checkpoint"),
                    job_id=(str((metadata or {}).get("job_id")) if (metadata or {}).get("job_id") else None),
                    metadata={"actor": actor, **dict(metadata or {})},
                )
            )
            acknowledgement = (
                f"補足を記録しました。{event.apply_after} で適用します。"
                if event.kind in {SteeringKind.CONSTRAINT, SteeringKind.REDIRECT}
                else "新しい仮説を記録しました。現在のJobは壊さずchild Job候補として扱います。"
            )
            self._append_event(
                JobEvent.new(
                    work_session_id=session_id,
                    kind=EventKind.ASSISTANT_MESSAGE,
                    message=acknowledgement,
                    payload={"steering": event.to_dict()},
                )
            )
            result = {
                "ok": True,
                "work_session_id": session_id,
                "message": acknowledgement,
                "steering": event.to_dict(),
            }
            self.registry.store_interaction_result(correlation_id, result)
            return result

        runtime_request = self._runtime_request(
            project=project,
            session=session,
            text=text,
            mode=mode,
            computer_use_allowed=computer_use_allowed,
            metadata=metadata,
        )
        runtime_result = self.runtime_router.run(runtime_request)
        enqueued = self._enqueue_tool_jobs(runtime_result)
        message = runtime_result.output_text or (
            "内部Runtimeが承認待ちです。" if runtime_result.requires_approval else runtime_result.error or "応答を生成できませんでした。"
        )
        event_kind = EventKind.APPROVAL if runtime_result.requires_approval else (
            EventKind.ERROR if runtime_result.error else EventKind.ASSISTANT_MESSAGE
        )
        self._append_event(
            JobEvent.new(
                work_session_id=session_id,
                kind=event_kind,
                message=message,
                payload={
                    "runtime": runtime_result.runtime,
                    "model": runtime_result.model,
                    "tool_calls": list(runtime_result.tool_calls),
                    "tool_results": list(runtime_result.tool_results),
                    "pending_actions": list(runtime_result.pending_actions),
                    "requires_approval": runtime_result.requires_approval,
                    "enqueued_jobs": enqueued,
                    "duration_seconds": runtime_result.duration_seconds,
                    "error": runtime_result.error,
                },
            )
        )
        updated_status = (
            WorkSessionStatus.WAITING_APPROVAL
            if runtime_result.requires_approval
            else WorkSessionStatus.RUNNING
            if enqueued
            else WorkSessionStatus.PLANNING
        )
        self.registry.update_work_session(
            session_id,
            status=updated_status,
            current_stage=("waiting_approval" if runtime_result.requires_approval else "jobs_queued" if enqueued else "conversation"),
        )
        result = {
            "ok": runtime_result.ok or runtime_result.requires_approval,
            "work_session_id": session_id,
            "message": message,
            "runtime": runtime_result.to_dict(),
            "enqueued_jobs": enqueued,
        }
        self.registry.store_interaction_result(correlation_id, result)
        return result

    def session_status(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        project = self._require_project(session.project_id)
        jobs = self.registry.list_jobs(work_session_id=session_id, limit=200)
        events = self.registry.list_events(session_id, after_sequence=0, limit=2000)
        active = [
            job
            for job in jobs
            if job.status
            not in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        ]
        latest_event = events[-1].to_dict() if events else None
        return {
            "project": project.to_dict(),
            "work_session": session.to_dict(),
            "jobs": [job.to_dict() for job in jobs],
            "active_job_count": len(active),
            "latest_event": latest_event,
            "pending_steering": [
                item.to_dict()
                for item in self.registry.list_pending_steering(session_id)
            ],
            "scheduler": self.scheduler.snapshot(),
        }

    def cancel_session(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        jobs = self.registry.list_jobs(work_session_id=session_id, limit=500)
        cancelled: list[str] = []
        for job in jobs:
            if job.status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                continue
            self.scheduler.cancel(job.spec.job_id)
            cancelled.append(job.spec.job_id)
        runtime_count = self.runtime_router.cancel_all(
            f"work session {session_id} cancelled"
        )
        updated = self.registry.update_work_session(
            session_id,
            status=WorkSessionStatus.PAUSED,
            current_stage="cancelled",
        )
        self._append_event(
            JobEvent.new(
                work_session_id=session_id,
                kind=EventKind.MILESTONE,
                message="Work session paused and active work cancelled",
                payload={
                    "jobs": cancelled,
                    "runtime_processes": runtime_count,
                },
            )
        )
        return {
            "work_session": updated.to_dict(),
            "cancelled_jobs": cancelled,
            "cancelled_runtime_processes": runtime_count,
        }

    def approve_job(self, job_id: str) -> dict[str, Any]:
        return self.scheduler.approve_job(job_id).to_dict()

    def _runtime_request(
        self,
        *,
        project: Project,
        session: WorkSession,
        text: str,
        mode: str,
        computer_use_allowed: bool,
        metadata: Mapping[str, Any] | None,
    ) -> RuntimeRequest:
        normalized_mode = mode.strip().lower()
        capabilities: list[RuntimeCapability] = [
            RuntimeCapability.CHAT,
            RuntimeCapability.REASONING,
        ]
        preferred_runtime = None
        role = "conversation"
        if normalized_mode in {"code", "coding", "implement", "execute_code"}:
            capabilities.extend(
                [
                    RuntimeCapability.CODING,
                    RuntimeCapability.FILE_EDIT,
                    RuntimeCapability.SHELL,
                ]
            )
            preferred_runtime = "codex_cli"
            role = "coder"
        elif normalized_mode in {"computer", "browser", "computer_use"}:
            capabilities.append(RuntimeCapability.COMPUTER_USE)
            preferred_runtime = "openai_responses"
        elif "openai_responses" in self.runtime_router.available_runtimes() and self.runtime_router.available_runtimes()["openai_responses"]["available"]:
            capabilities.append(RuntimeCapability.FUNCTION_TOOLS)
            preferred_runtime = "openai_responses"
            role = "planner"

        workspace = self._session_workspace(session)
        context = self._context_text(project, session)
        system_prompt = (
            "あなたはResearchAgent Core内部の会話・計画担当です。ResearchとKaggleの両方を扱います。\n"
            "相談には直接答え、実行が必要ならtyped harness toolでJobを提案してください。\n"
            "長時間処理を会話Runtime内で実行せずComputeBackendへ委譲してください。\n"
            "Kaggle提出、課金GPU、computer-use、外部公開は勝手に実行せず承認を要求してください。\n"
            "既存Jobを上書きせず、別仮説はparent/child関係を持つ新Jobとして扱ってください。\n"
            "事実、推論、未確認事項を分離してください。\n\n"
            "<UNTRUSTED_SESSION_CONTEXT>\n"
            + context
            + "\n</UNTRUSTED_SESSION_CONTEXT>"
        )
        return RuntimeRequest(
            prompt=text,
            system_prompt=system_prompt,
            capabilities=tuple(dict.fromkeys(capabilities)),
            preferred_runtime=preferred_runtime,
            context=RuntimeContext(
                project_id=project.project_id,
                work_session_id=session.session_id,
                role=role,
                stage=normalized_mode or "auto",
                working_dir=str(workspace),
                metadata={
                    "goal": session.objective,
                    "domain": project.domain.value,
                },
            ),
            tools_enabled=True,
            computer_use_allowed=computer_use_allowed,
            metadata=dict(metadata or {}),
        )

    def _context_text(self, project: Project, session: WorkSession) -> str:
        jobs = self.registry.list_jobs(work_session_id=session.session_id, limit=20)
        pending = self.registry.list_pending_steering(session.session_id)
        return json.dumps(
            {
                "project": project.to_dict(),
                "work_session": session.to_dict(),
                "recent_jobs": [job.to_dict() for job in jobs[:20]],
                "pending_steering": [item.to_dict() for item in pending],
            },
            ensure_ascii=False,
            indent=2,
        )

    def _enqueue_tool_jobs(self, result: RuntimeResult) -> list[str]:
        job_ids: list[str] = []
        for item in result.tool_results:
            if item.get("name") != "propose_job":
                continue
            wrapped = item.get("result")
            if not isinstance(wrapped, Mapping) or not wrapped.get("ok"):
                continue
            record = wrapped.get("result")
            if not isinstance(record, Mapping):
                continue
            spec = record.get("spec")
            if not isinstance(spec, Mapping) or not spec.get("job_id"):
                continue
            job_id = str(spec["job_id"])
            self.scheduler.enqueue(job_id)
            job_ids.append(job_id)
        return job_ids

    def _append_event(self, event: JobEvent) -> JobEvent:
        return self.registry.append_event(event)

    def _project_workspace(self, project: Project) -> Path:
        return self.config.data_dir / "projects" / _safe(project.project_id)

    def _session_workspace(self, session: WorkSession) -> Path:
        return (
            self.config.data_dir
            / "projects"
            / _safe(session.project_id)
            / "sessions"
            / _safe(session.session_id)
        )

    @staticmethod
    def _write_session_context(
        project: Project,
        session: WorkSession,
        workspace: Path,
    ) -> None:
        (workspace / "SESSION_CONTEXT.md").write_text(
            "\n".join(
                [
                    f"# {session.title}",
                    "",
                    f"- Project: {project.project_id}",
                    f"- Domain: {project.domain.value}",
                    f"- Work session: {session.session_id}",
                    "",
                    "## Objective",
                    session.objective,
                    "",
                    "## Operating policy",
                    "- Do not bypass approval gates.",
                    "- Long-running compute must be represented as a durable Job.",
                    "- Do not expose Discord, Kaggle, OpenAI, or worker credentials.",
                    "- Preserve failed attempts and create child Jobs for new hypotheses.",
                    "- Store structured result.json and progress.json when executing code.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _require_project(self, project_id: str) -> Project:
        project = self.registry.get_project(project_id)
        if project is None:
            raise KeyError(f"unknown project: {project_id}")
        return project

    def _require_session(self, session_id: str) -> WorkSession:
        session = self.registry.get_work_session(session_id)
        if session is None:
            raise KeyError(f"unknown work session: {session_id}")
        return session


def _infer_steering(text: str) -> SteeringKind | None:
    stripped = text.strip()
    if stripped.startswith("?") or re.match(r"^(今|現在).*(状況|進捗|どこ)", stripped):
        return SteeringKind.QUESTION
    if re.match(r"^(補足|制約|条件)[:：]", stripped):
        return SteeringKind.CONSTRAINT
    if re.match(r"^(方針変更|変更|リダイレクト)[:：]", stripped):
        return SteeringKind.REDIRECT
    if re.match(r"^(仮説|別案|追加実験)[:：]", stripped):
        return SteeringKind.NEW_HYPOTHESIS
    return None


def _safe(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:100] or "item"


def _remote_capabilities(raw: Mapping[str, Any]) -> BackendCapabilities:
    capabilities = raw.get("capabilities")
    value = dict(capabilities) if isinstance(capabilities, Mapping) else {}
    domains: list[Domain] = []
    for item in value.get("domains", ["research", "kaggle"]):
        try:
            domains.append(Domain(str(item)))
        except ValueError:
            continue
    return BackendCapabilities(
        accelerators=tuple(str(item) for item in value.get("accelerators", ["cpu", "gpu"])),
        max_vram_gb=(float(value["max_vram_gb"]) if value.get("max_vram_gb") is not None else None),
        max_gpu_count=(int(value["max_gpu_count"]) if value.get("max_gpu_count") is not None else None),
        max_cpu_cores=(int(value["max_cpu_cores"]) if value.get("max_cpu_cores") is not None else None),
        max_ram_gb=(float(value["max_ram_gb"]) if value.get("max_ram_gb") is not None else None),
        max_runtime_minutes=(int(value["max_runtime_minutes"]) if value.get("max_runtime_minutes") is not None else None),
        network_available=bool(value.get("network_available", True)),
        detailed_progress=bool(value.get("detailed_progress", True)),
        supports_cancel=bool(value.get("supports_cancel", True)),
        supports_kaggle_data=bool(value.get("supports_kaggle_data", False)),
        domains=tuple(domains) or (Domain.RESEARCH, Domain.KAGGLE),
        tags=tuple(str(item) for item in value.get("tags", ["training", "inference", "remote_worker"])),
    )
