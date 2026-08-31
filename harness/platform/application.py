from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from harness.compute.base import BackendCapabilities, ComputeBroker
from harness.compute.kaggle import KaggleNotebookBackend
from harness.compute.portable_local import PortableLocalProcessBackend
from harness.compute.remote import RemoteWorkerBackend, RemoteWorkerDescriptor
from harness.compute.scheduler import JobScheduler
from harness.config import HarnessConfig
from harness.domains.kaggle.gateway import KaggleGateway, create_submission_candidate
from harness.domains.kaggle.models import (
    CVSpec,
    KaggleCompetitionState,
    SubmissionCandidate,
)
from harness.domains.kaggle.registry import KaggleRegistry
from harness.domains.kaggle.tools import register_kaggle_tools
from harness.domains.kaggle.validation import validate_submission, write_validation_report
from harness.domains.kaggle.workspace import KaggleWorkspace
from harness.platform.config import PlatformConfig
from harness.platform.models import Domain, EventKind, JobEvent, Project
from harness.platform.registry import PlatformRegistry
from harness.platform.service import CoreComponents, PlatformService
from harness.runtime.computer import ComputerUsePolicy, PlaywrightComputerDriver
from harness.runtime.openai_responses import OpenAIResponsesRuntime, UrllibResponsesTransport
from harness.runtime.router import AgentRuntimeRouter
from harness.runtime.tools import ToolExecutionContext, build_default_harness_tools
from harness.runtime.verified_codex import VerifiedCodexCliRuntime


class PortablePlatformService(PlatformService):
    """Integrated ResearchAgent and KaggleAgent application."""

    def __init__(
        self,
        components: CoreComponents,
        *,
        kaggle_registry: KaggleRegistry,
        kaggle_gateway: KaggleGateway,
    ) -> None:
        super().__init__(components)
        self.kaggle_registry = kaggle_registry
        self.kaggle_gateway = kaggle_gateway

    def create_project(
        self,
        *,
        domain: Domain | str,
        title: str,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Project:
        normalized = Domain(str(domain))
        project = super().create_project(
            domain=normalized,
            title=title,
            description=description,
            metadata=metadata,
        )
        if normalized != Domain.KAGGLE:
            return project
        raw_metadata = dict(metadata or {})
        competition_url = str(raw_metadata.get("competition_url") or "").strip()
        slug = str(raw_metadata.get("competition_slug") or "").strip()
        if not slug:
            slug = _competition_slug(competition_url) or _slug(title)
        competition = KaggleCompetitionState.new(
            project_id=project.project_id,
            slug=slug,
            url=competition_url,
            title=title,
            metadata={
                "source": "ResearchAgent",
                **raw_metadata,
            },
        )
        self.kaggle_registry.create_competition(competition)
        workspace = self.kaggle_workspace(project.project_id)
        files = KaggleWorkspace(workspace).initialize(competition)
        updated = self.registry.update_project(
            project.project_id,
            metadata={
                **project.metadata,
                "competition_id": competition.competition_id,
                "competition_slug": competition.slug,
                "competition_url": competition.url,
                "kaggle_workspace": str(workspace),
                "workspace_files": files,
            },
        )
        return updated

    def kaggle_workspace(self, project_id: str) -> Path:
        return self.config.data_dir / "projects" / _safe(project_id) / "kaggle"

    def kaggle_status(self, project_id: str) -> dict[str, Any]:
        competition = self.kaggle_registry.get_competition(project_id=project_id)
        if competition is None:
            raise KeyError(f"Kaggle competition not found for project {project_id}")
        cv = (
            self.kaggle_registry.get_cv_spec(competition.active_cv_spec_id)
            if competition.active_cv_spec_id
            else None
        )
        experiments = self.kaggle_registry.list_experiments(
            competition.competition_id,
            limit=500,
        )
        return {
            "competition": competition.to_dict(),
            "active_cv": cv.to_dict() if cv else None,
            "experiments": [item.to_dict() for item in experiments],
            "workspace": str(self.kaggle_workspace(project_id)),
            "gateway": {
                "available": self.kaggle_gateway.available()[0],
                "detail": self.kaggle_gateway.available()[1],
            },
        }

    def acknowledge_kaggle_rules(
        self,
        project_id: str,
        *,
        rules_text: str,
        actor: str,
    ) -> dict[str, Any]:
        competition = self._competition(project_id)
        digest = hashlib.sha256(rules_text.encode("utf-8")).hexdigest()
        updated = self.kaggle_registry.update_competition(
            competition.competition_id,
            rules_acknowledged=True,
            rules_hash=digest,
            metadata={
                **competition.metadata,
                "rules_acknowledged_by": actor,
            },
        )
        rules_path = self.kaggle_workspace(project_id) / "docs" / "rules.md"
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        rules_path.write_text(rules_text.rstrip() + "\n", encoding="utf-8")
        return updated.to_dict()

    def create_kaggle_cv_spec(
        self,
        project_id: str,
        *,
        strategy: str,
        n_splits: int,
        metric: str,
        seed: int = 42,
        shuffle: bool = True,
        group_column: str | None = None,
        time_column: str | None = None,
        stratify_column: str | None = None,
    ) -> dict[str, Any]:
        competition = self._competition(project_id)
        spec = CVSpec.new(
            competition_id=competition.competition_id,
            strategy=strategy,
            n_splits=n_splits,
            metric=metric,
            seed=seed,
            shuffle=shuffle,
            group_column=group_column,
            time_column=time_column,
            stratify_column=stratify_column,
        )
        self.kaggle_registry.create_cv_spec(spec)
        return spec.to_dict()

    def lock_kaggle_cv_spec(
        self,
        project_id: str,
        cv_spec_id: str,
        *,
        actor: str,
    ) -> dict[str, Any]:
        competition = self._competition(project_id)
        spec = self.kaggle_registry.get_cv_spec(cv_spec_id)
        if spec is None or spec.competition_id != competition.competition_id:
            raise KeyError(cv_spec_id)
        locked = self.kaggle_registry.lock_cv_spec(cv_spec_id)
        self.kaggle_registry.update_competition(
            competition.competition_id,
            active_cv_spec_id=cv_spec_id,
            metadata={
                **competition.metadata,
                "cv_locked_by": actor,
            },
        )
        KaggleWorkspace(self.kaggle_workspace(project_id)).initialize(
            competition,
            cv_spec=locked,
            overwrite_generated=True,
        )
        return locked.to_dict()

    def prepare_submission_candidate(
        self,
        project_id: str,
        *,
        experiment_id: str,
        file_path: str,
        sample_submission_path: str,
        message: str,
        id_columns: list[str] | tuple[str, ...] = (),
        probability_columns: list[str] | tuple[str, ...] = (),
        probability_groups: list[list[str]] | tuple[tuple[str, ...], ...] = (),
        cv_score: float | None = None,
        previous_best_cv: float | None = None,
        risks: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        competition = self._competition(project_id)
        result = validate_submission(
            file_path,
            sample_submission_path,
            id_columns=id_columns,
            probability_columns=probability_columns,
            probability_groups=probability_groups,
        )
        report = (
            self.kaggle_workspace(project_id)
            / "submissions"
            / "candidates"
            / f"validation-{result.submission_sha256[:16]}.json"
        )
        write_validation_report(result, report)
        candidate = create_submission_candidate(
            self.kaggle_registry,
            competition_id=competition.competition_id,
            experiment_id=experiment_id,
            file_path=file_path,
            message=message,
            validation={**result.to_dict(), "report_path": str(report)},
            cv_score=cv_score,
            previous_best_cv=previous_best_cv,
            risks=risks,
        )
        return candidate.to_dict()

    def approve_submission_candidate(
        self,
        candidate_id: str,
        *,
        approval_id: str,
    ) -> dict[str, Any]:
        return self.kaggle_gateway.approve_candidate(
            candidate_id,
            approval_id=approval_id,
        ).to_dict()

    def submit_candidate(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.kaggle_registry.get_submission(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        competition = self.kaggle_registry.get_competition(
            candidate.competition_id
        )
        if competition is None:
            raise KeyError(candidate.competition_id)
        return self.kaggle_gateway.submit_candidate(
            candidate_id,
            cwd=self.kaggle_workspace(competition.project_id),
        ).to_dict()

    def _competition(self, project_id: str) -> KaggleCompetitionState:
        competition = self.kaggle_registry.get_competition(project_id=project_id)
        if competition is None:
            raise KeyError(f"Kaggle competition not found for project {project_id}")
        return competition


def build_application(
    project_root: str | Path | None = None,
) -> PortablePlatformService:
    platform_config = PlatformConfig.from_env(project_root)
    harness_config = HarnessConfig.from_env(project_root)
    registry = PlatformRegistry(platform_config.database_path)
    kaggle_registry = KaggleRegistry(platform_config.database_path)
    tools = build_default_harness_tools()
    register_kaggle_tools(
        tools,
        kaggle_registry,
        data_dir=platform_config.data_dir,
    )

    runtimes = []
    if platform_config.codex_command:
        runtimes.append(
            VerifiedCodexCliRuntime(
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
        PortableLocalProcessBackend(max_cpu_cores=max(1, os.cpu_count() or 1)),
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
        paid = bool(raw.get("paid", False))
        backends.append(
            RemoteWorkerBackend(
                RemoteWorkerDescriptor(
                    name=name,
                    base_url=str(raw.get("base_url") or ""),
                    token=str(raw.get("token") or ""),
                    paid=paid,
                    capabilities=capabilities,
                )
            )
        )
        if paid:
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
    gateway = KaggleGateway(
        registry=kaggle_registry,
        command=platform_config.kaggle_command,
        api_token=platform_config.kaggle_api_token,
        username=platform_config.kaggle_username,
    )
    return PortablePlatformService(
        CoreComponents(
            config=platform_config,
            harness_config=harness_config,
            registry=registry,
            runtime_router=runtime_router,
            compute_broker=broker,
            scheduler=scheduler,
        ),
        kaggle_registry=kaggle_registry,
        kaggle_gateway=gateway,
    )


def _competition_slug(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if "competitions" in parts:
        index = parts.index("competitions")
        if index + 1 < len(parts):
            return parts[index + 1]
    return parts[-1] if parts else ""


def _slug(value: str) -> str:
    cleaned = "-".join(value.lower().split())
    cleaned = "".join(character if character.isalnum() or character == "-" else "-" for character in cleaned)
    return cleaned.strip("-")[:80] or "competition"


def _safe(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:100] or "project"


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
