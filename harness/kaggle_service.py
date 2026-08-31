from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from harness.control_plane import ControlPlaneRegistry, JobEvent, JobSpec
from harness.job_scheduler import JobScheduler
from harness.kaggle_domain import (
    CVSpec,
    ExperimentRecord,
    KaggleCompetition,
    KaggleStore,
    SubmissionCandidate,
)
from harness.kaggle_gateway import KaggleSubmissionGateway
from harness.kaggle_validation import (
    CVValidationReport,
    SubmissionValidationReport,
    fingerprint_paths,
    validate_cv_spec,
)
from harness.work_sessions import WorkSessionService


class KaggleApplicationService:
    """High-level Kaggle workflow above generic WorkSession and ComputeBackend layers."""

    def __init__(
        self,
        *,
        registry: ControlPlaneRegistry,
        work_sessions: WorkSessionService,
        scheduler: JobScheduler,
        kaggle_store: KaggleStore,
        submission_gateway: KaggleSubmissionGateway | None = None,
    ) -> None:
        self.registry = registry
        self.work_sessions = work_sessions
        self.scheduler = scheduler
        self.store = kaggle_store
        self.submission_gateway = submission_gateway
        self.scheduler.subscribe(self.on_job_event)

    def create_competition(
        self,
        competition_url_or_slug: str,
        *,
        title: str | None = None,
        project_root: str | Path,
        metric: str | None = None,
        target_column: str | None = None,
        id_column: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[KaggleCompetition, Any]:
        slug = parse_competition_slug(competition_url_or_slug)
        url = (
            competition_url_or_slug.strip()
            if "://" in competition_url_or_slug
            else f"https://www.kaggle.com/competitions/{slug}"
        )
        display_title = title or slug.replace("-", " ").title()
        project, session = self.work_sessions.create_session(
            domain="kaggle",
            title=f"[KG] {display_title}",
            project_root=project_root,
            project_metadata={
                "competition_slug": slug,
                "competition_url": url,
                **dict(metadata or {}),
            },
            session_metadata={"purpose": "competition_setup"},
        )
        competition = self.store.create_competition(
            project_id=project.project_id,
            competition_slug=slug,
            competition_url=url,
            title=display_title,
            metric=metric,
            target_column=target_column,
            id_column=id_column,
            metadata=metadata,
        )
        self.work_sessions.record_assistant_message(
            session.work_session_id,
            (
                f"Kaggleコンペ `{slug}` を登録しました。\n"
                "現在はRules Gateです。ルール・評価指標・外部データ条件を確認するまで、"
                "学習や提出は開始しません。"
            ),
            metadata={"phase": competition.phase},
        )
        return competition, session

    def confirm_rules(
        self,
        project_id: str,
        *,
        confirmed_by: str,
        notes: str = "",
    ) -> KaggleCompetition:
        competition = self._competition(project_id)
        metadata = {
            **competition.metadata,
            "rules_confirmed_by": confirmed_by,
            "rules_confirmation_notes": notes,
        }
        return self.store.update_competition(
            project_id,
            phase="data_preparation",
            rules_confirmed=True,
            metadata=metadata,
        )

    def register_dataset(
        self,
        project_id: str,
        *,
        paths: list[str | Path],
        sample_submission_path: str | Path | None = None,
    ) -> KaggleCompetition:
        competition = self._competition(project_id)
        if not competition.rules_confirmed:
            raise PermissionError("rules must be confirmed before dataset registration")
        fingerprint = fingerprint_paths(paths)
        return self.store.update_competition(
            project_id,
            phase="baseline_building",
            sample_submission_path=(
                str(sample_submission_path) if sample_submission_path else None
            ),
            dataset_fingerprint=fingerprint,
        )

    def propose_cv(
        self,
        project_id: str,
        *,
        strategy: str,
        n_splits: int,
        metric: str | None = None,
        shuffle: bool = True,
        seed: int | None = 42,
        group_column: str | None = None,
        time_column: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[CVSpec, CVValidationReport]:
        competition = self._competition(project_id)
        selected_metric = metric or competition.metric
        if not selected_metric:
            raise ValueError("competition metric must be known before CV design")
        spec = self.store.create_cv_spec(
            project_id=project_id,
            strategy=strategy,
            n_splits=n_splits,
            metric=selected_metric,
            shuffle=shuffle,
            seed=seed,
            group_column=group_column,
            time_column=time_column,
            parameters=parameters,
            split_hash=(
                f"{competition.dataset_fingerprint[:12]}:pending"
                if competition.dataset_fingerprint
                else None
            ),
        )
        report = validate_cv_spec(
            spec,
            expected_metric=competition.metric,
            dataset_fingerprint=competition.dataset_fingerprint,
        )
        return spec, report

    def approve_cv(
        self,
        cv_spec_id: str,
        *,
        split_hash: str,
    ) -> CVSpec:
        spec = self.store.get_cv_spec(cv_spec_id)
        if spec is None:
            raise KeyError(cv_spec_id)
        report = validate_cv_spec(spec)
        if not report.valid:
            raise ValueError("CV specification is invalid")
        approved = self.store.approve_cv_spec(cv_spec_id, split_hash=split_hash)
        self.store.update_competition(
            approved.project_id,
            phase="experimenting",
        )
        return approved

    def create_experiment(
        self,
        *,
        project_id: str,
        hypothesis: str,
        config: dict[str, Any] | None = None,
        parent_experiment_id: str | None = None,
        cv_spec_id: str | None = None,
        work_session_id: str | None = None,
        code_snapshot: str | None = None,
    ) -> tuple[ExperimentRecord, Any]:
        competition = self._competition(project_id)
        if not competition.rules_confirmed:
            raise PermissionError("rules are not confirmed")
        if cv_spec_id:
            cv = self.store.get_cv_spec(cv_spec_id)
            if cv is None or not cv.approved:
                raise PermissionError("CVSpec must be approved before experiment")
        if work_session_id is None:
            parent_sessions = [
                session
                for session in self._project_sessions(project_id)
                if session.parent_work_session_id is None
            ]
            if not parent_sessions:
                raise ValueError("project has no root WorkSession")
            parent = parent_sessions[0]
            session = self.work_sessions.create_child_session(
                parent.work_session_id,
                title=f"[KG] {hypothesis}"[:72],
                kind="experiment",
                metadata={
                    "parent_experiment_id": parent_experiment_id,
                    "competition_slug": competition.competition_slug,
                },
            )
        else:
            session = self.registry.get_work_session(work_session_id)
            if session is None or session.project_id != project_id:
                raise ValueError("work_session_id does not belong to project")
        experiment = self.store.create_experiment(
            project_id=project_id,
            work_session_id=session.work_session_id,
            parent_experiment_id=parent_experiment_id,
            hypothesis=hypothesis,
            config=config,
            code_snapshot=code_snapshot,
            dataset_fingerprint=competition.dataset_fingerprint,
            cv_spec_id=cv_spec_id,
        )
        self.work_sessions.record_assistant_message(
            session.work_session_id,
            (
                f"実験 `{experiment.experiment_id}` を作成しました。\n"
                f"仮説: {experiment.hypothesis}\n"
                "既存実験を変更せず、独立したchild experimentとして記録します。"
            ),
            metadata={"experiment_id": experiment.experiment_id},
        )
        return experiment, session

    def queue_experiment(
        self,
        experiment_id: str,
        *,
        source_dir: str | Path,
        entrypoint: str = "run.py",
        backend_preferences: tuple[str, ...] = (
            "kaggle_notebook",
            "remote_gpu",
            "gpu_vm",
            "local_cpu",
        ),
        resources: dict[str, Any] | None = None,
        outputs: tuple[str, ...] = (
            "result.json",
            "metrics.json",
        ),
        max_runtime_seconds: int = 0,
        kernel_ref: str | None = None,
        kernel_title: str | None = None,
        competition_sources: tuple[str, ...] = (),
        enable_internet: bool = False,
        smoke_test: bool = False,
    ) -> Any:
        experiment = self._experiment(experiment_id)
        competition = self._competition(experiment.project_id)
        cv = self.store.get_cv_spec(experiment.cv_spec_id) if experiment.cv_spec_id else None
        payload = {
            "experiment_id": experiment.experiment_id,
            "hypothesis": experiment.hypothesis,
            "experiment_config": experiment.config,
            "source_dir": str(Path(source_dir).expanduser().resolve()),
            "entrypoint": entrypoint,
            "kernel_ref": kernel_ref,
            "kernel_title": kernel_title or f"{competition.competition_slug}-{experiment.experiment_id}",
            "competition_sources": list(
                competition_sources or (competition.competition_slug,)
            ),
            "enable_internet": bool(enable_internet),
            "dataset_fingerprint": experiment.dataset_fingerprint,
            "cv_spec": (cv and as_cv_dict(cv)) or None,
            "smoke_test": bool(smoke_test),
        }
        spec = JobSpec.new(
            project_id=experiment.project_id,
            work_session_id=experiment.work_session_id,
            domain="kaggle",
            task_type="smoke_test" if smoke_test else "train_cv",
            payload=payload,
            resources=resources or {"accelerator": "gpu"},
            backend_preferences=backend_preferences,
            outputs=outputs,
            max_runtime_seconds=max_runtime_seconds,
        )
        self.store.update_experiment(
            experiment_id,
            status="queued",
            job_id=spec.job_id,
        )
        return self.work_sessions.queue_job(spec)

    def prepare_submission(
        self,
        experiment_id: str,
        *,
        submission_path: str | Path,
        message: str,
        prediction_ranges: Mapping[
            str,
            tuple[float | None, float | None],
        ]
        | None = None,
        cv_score: float | None = None,
    ) -> tuple[SubmissionCandidate, SubmissionValidationReport]:
        if self.submission_gateway is None:
            raise RuntimeError("Kaggle submission gateway is not configured")
        experiment = self._experiment(experiment_id)
        competition = self._competition(experiment.project_id)
        if not competition.sample_submission_path:
            raise ValueError("sample submission path is not registered")
        candidate, report = self.submission_gateway.prepare(
            project_id=experiment.project_id,
            experiment_id=experiment_id,
            submission_path=submission_path,
            sample_submission_path=competition.sample_submission_path,
            message=message,
            id_column=competition.id_column,
            prediction_ranges=prediction_ranges,
            cv_score=cv_score,
        )
        self.store.update_competition(
            experiment.project_id,
            phase="submission_review",
        )
        return candidate, report

    def approve_submission(
        self,
        candidate_id: str,
        approval_id: str,
    ) -> SubmissionCandidate:
        if self.submission_gateway is None:
            raise RuntimeError("Kaggle submission gateway is not configured")
        return self.submission_gateway.approve(candidate_id, approval_id)

    def submit_candidate(self, candidate_id: str) -> SubmissionCandidate:
        if self.submission_gateway is None:
            raise RuntimeError("Kaggle submission gateway is not configured")
        candidate = self.submission_gateway.submit(candidate_id)
        self.store.update_competition(candidate.project_id, phase="lb_analysis")
        return candidate

    def status(self, project_id: str) -> dict[str, Any]:
        competition = self._competition(project_id)
        experiments = self.store.list_experiments(project_id)
        completed = [item for item in experiments if item.status == "completed"]
        failed = [item for item in experiments if item.status == "failed"]
        best = _best_experiment(completed)
        return {
            "project_id": project_id,
            "competition_slug": competition.competition_slug,
            "phase": competition.phase,
            "metric": competition.metric,
            "rules_confirmed": competition.rules_confirmed,
            "dataset_fingerprint": competition.dataset_fingerprint,
            "experiments_total": len(experiments),
            "experiments_completed": len(completed),
            "experiments_failed": len(failed),
            "best_experiment": best.experiment_id if best else None,
            "best_metrics": best.metrics if best else {},
        }

    def on_job_event(self, event: JobEvent) -> None:
        job = self.registry.get_job(event.job_id)
        if job is None:
            return
        experiment_id = str(job.spec.payload.get("experiment_id") or "")
        if not experiment_id:
            return
        experiment = self.store.get_experiment(experiment_id)
        if experiment is None:
            return
        if event.event_type in {"backend_started", "job_running"}:
            self.store.update_experiment(
                experiment_id,
                status=(
                    "smoke_running"
                    if bool(job.spec.payload.get("smoke_test"))
                    else "training"
                ),
                job_id=job.spec.job_id,
            )
            return
        if event.event_type not in {
            "job_completed",
            "job_failed",
            "job_cancelled",
            "job_summary",
        }:
            return
        refreshed = self.registry.get_job(event.job_id)
        if refreshed is None:
            return
        if refreshed.status == "completed":
            metrics = _extract_metrics(refreshed.result)
            artifacts = _extract_artifacts(refreshed.result)
            self.store.update_experiment(
                experiment_id,
                status="review",
                metrics=metrics,
                artifacts=artifacts,
            )
        elif refreshed.status == "failed":
            self.store.update_experiment(
                experiment_id,
                status="failed",
                failure_reason=refreshed.error or "job failed",
            )
        elif refreshed.status == "cancelled":
            self.store.update_experiment(
                experiment_id,
                status="cancelled",
                failure_reason=refreshed.error or "job cancelled",
            )

    def accept_experiment(self, experiment_id: str) -> ExperimentRecord:
        experiment = self._experiment(experiment_id)
        if experiment.status != "review":
            raise ValueError(f"experiment is not in review: {experiment.status}")
        return self.store.update_experiment(experiment_id, status="completed")

    def reject_experiment(
        self,
        experiment_id: str,
        reason: str,
    ) -> ExperimentRecord:
        experiment = self._experiment(experiment_id)
        if experiment.status not in {"review", "completed"}:
            raise ValueError(f"experiment cannot be rejected from {experiment.status}")
        return self.store.update_experiment(
            experiment_id,
            status="rejected",
            failure_reason=reason,
        )

    def _competition(self, project_id: str) -> KaggleCompetition:
        competition = self.store.get_competition(project_id)
        if competition is None:
            raise KeyError(project_id)
        return competition

    def _experiment(self, experiment_id: str) -> ExperimentRecord:
        experiment = self.store.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        return experiment

    def _project_sessions(self, project_id: str) -> list[Any]:
        # ControlPlaneRegistry intentionally keeps a compact API; use known jobs
        # and root session metadata to find a project session without exposing SQL.
        jobs = self.registry.list_jobs(limit=1000)
        session_ids = {
            job.spec.work_session_id
            for job in jobs
            if job.spec.project_id == project_id
        }
        candidates = [
            self.registry.get_work_session(session_id)
            for session_id in session_ids
        ]
        result = [item for item in candidates if item is not None]
        if result:
            return result
        # The root session may not have a job yet. Its ID is stored in the
        # WorkSession table; callers should normally pass work_session_id.
        return []


def parse_competition_slug(value: str) -> str:
    stripped = value.strip().rstrip("/")
    match = re.search(r"kaggle\.com/competitions/([^/?#]+)", stripped, re.I)
    slug = match.group(1) if match else stripped
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", slug):
        raise ValueError("invalid Kaggle competition URL or slug")
    return slug


def as_cv_dict(spec: CVSpec) -> dict[str, Any]:
    return {
        "cv_spec_id": spec.cv_spec_id,
        "strategy": spec.strategy,
        "n_splits": spec.n_splits,
        "shuffle": spec.shuffle,
        "seed": spec.seed,
        "metric": spec.metric,
        "group_column": spec.group_column,
        "time_column": spec.time_column,
        "parameters": spec.parameters,
        "split_hash": spec.split_hash,
        "approved": spec.approved,
    }


def _extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
    for key in ("metrics", "result"):
        value = result.get(key)
        if isinstance(value, dict):
            nested = value.get("metrics") if key == "result" else None
            return dict(nested) if isinstance(nested, dict) else dict(value)
    return {}


def _extract_artifacts(result: dict[str, Any]) -> list[dict[str, Any]]:
    files = result.get("files")
    if not isinstance(files, list):
        return []
    root = result.get("output_dir") or result.get("workspace")
    return [
        {
            "path": str(item),
            "root": str(root) if root else None,
        }
        for item in files
    ]


def _best_experiment(experiments: list[ExperimentRecord]) -> ExperimentRecord | None:
    def score(item: ExperimentRecord) -> float:
        for key in ("cv_mean", "score", "metric", "value"):
            value = item.metrics.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float("-inf")

    return max(experiments, key=score, default=None)
