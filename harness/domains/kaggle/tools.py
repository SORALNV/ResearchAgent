from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from harness.domains.kaggle.gateway import create_submission_candidate
from harness.domains.kaggle.models import CVSpec, ExperimentRecord
from harness.domains.kaggle.registry import KaggleRegistry
from harness.domains.kaggle.validation import validate_submission, write_validation_report
from harness.domains.kaggle.workspace import KaggleWorkspace
from harness.runtime.tools import HarnessToolRegistry, RuntimeTool, ToolExecutionContext


class KaggleToolContext(ToolExecutionContext):
    kaggle_registry: KaggleRegistry


def register_kaggle_tools(
    tools: HarnessToolRegistry,
    kaggle_registry: KaggleRegistry,
    *,
    data_dir: Path,
) -> None:
    tools.register(
        RuntimeTool(
            name="get_kaggle_competition",
            description="Read Kaggle competition, locked CV, experiments, and submissions.",
            parameters={
                "type": "object",
                "properties": {"project_id": {"type": "string"}},
                "required": ["project_id"],
                "additionalProperties": False,
            },
            handler=lambda args, context: _get_competition(
                args, context, kaggle_registry
            ),
        )
    )
    tools.register(
        RuntimeTool(
            name="initialize_kaggle_workspace",
            description=(
                "Create the standard safe Kaggle directory, policy, docs, source "
                "templates, and competition metadata."
            ),
            parameters={
                "type": "object",
                "properties": {"project_id": {"type": "string"}},
                "required": ["project_id"],
                "additionalProperties": False,
            },
            handler=lambda args, context: _initialize_workspace(
                args, context, kaggle_registry, data_dir
            ),
            mutating=True,
        )
    )
    tools.register(
        RuntimeTool(
            name="create_kaggle_cv_spec",
            description=(
                "Create an unlocked CV proposal. A human must lock it before "
                "scores are treated as comparable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "strategy": {"type": "string", "minLength": 1},
                    "n_splits": {"type": "integer", "minimum": 2},
                    "metric": {"type": "string", "minLength": 1},
                    "seed": {"type": "integer"},
                    "shuffle": {"type": "boolean"},
                    "group_column": {"type": ["string", "null"]},
                    "time_column": {"type": ["string", "null"]},
                    "stratify_column": {"type": ["string", "null"]},
                },
                "required": [
                    "project_id",
                    "strategy",
                    "n_splits",
                    "metric",
                    "seed",
                    "shuffle",
                    "group_column",
                    "time_column",
                    "stratify_column",
                ],
                "additionalProperties": False,
            },
            handler=lambda args, context: _create_cv(
                args, context, kaggle_registry
            ),
            mutating=True,
        )
    )
    tools.register(
        RuntimeTool(
            name="create_kaggle_experiment",
            description=(
                "Create a child experiment without changing its parent. The "
                "experiment is only a proposal until a durable Job is linked."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "hypothesis": {"type": "string", "minLength": 1},
                    "parent_experiment_id": {"type": ["string", "null"]},
                    "config": {"type": "object"},
                    "config_diff": {"type": "object"},
                },
                "required": [
                    "project_id",
                    "hypothesis",
                    "parent_experiment_id",
                    "config",
                    "config_diff",
                ],
                "additionalProperties": False,
            },
            handler=lambda args, context: _create_experiment(
                args, context, kaggle_registry, data_dir
            ),
            mutating=True,
        )
    )
    tools.register(
        RuntimeTool(
            name="validate_kaggle_submission",
            description=(
                "Validate a candidate CSV against sample_submission. This never "
                "submits the file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "submission_path": {"type": "string"},
                    "sample_submission_path": {"type": "string"},
                    "id_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "probability_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "probability_groups": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "required": [
                    "submission_path",
                    "sample_submission_path",
                    "id_columns",
                    "probability_columns",
                    "probability_groups",
                ],
                "additionalProperties": False,
            },
            handler=lambda args, context: _validate_candidate(
                args, context, data_dir
            ),
        )
    )
    tools.register(
        RuntimeTool(
            name="create_kaggle_submission_candidate",
            description=(
                "Record a hash-bound submission candidate after validation. "
                "This does not approve or submit it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "file_path": {"type": "string"},
                    "message": {"type": "string"},
                    "validation": {"type": "object"},
                    "cv_score": {"type": ["number", "null"]},
                    "previous_best_cv": {"type": ["number", "null"]},
                    "risks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "project_id",
                    "experiment_id",
                    "file_path",
                    "message",
                    "validation",
                    "cv_score",
                    "previous_best_cv",
                    "risks",
                ],
                "additionalProperties": False,
            },
            handler=lambda args, context: _create_candidate(
                args, context, kaggle_registry
            ),
            mutating=True,
        )
    )
    tools.register(
        RuntimeTool(
            name="request_kaggle_submission",
            description=(
                "Request human approval for one immutable submission candidate. "
                "The model cannot submit directly."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["candidate_id", "reason"],
                "additionalProperties": False,
            },
            handler=lambda args, context: args,
            approval_required=True,
        )
    )


def _get_competition(
    args: dict[str, Any],
    context: ToolExecutionContext,
    registry: KaggleRegistry,
) -> Mapping[str, Any]:
    competition = registry.get_competition(project_id=str(args["project_id"]))
    if competition is None:
        raise KeyError(f"Kaggle competition is not initialized: {args['project_id']}")
    cv = (
        registry.get_cv_spec(competition.active_cv_spec_id)
        if competition.active_cv_spec_id
        else None
    )
    experiments = registry.list_experiments(competition.competition_id, limit=100)
    return {
        "competition": competition.to_dict(),
        "active_cv": cv.to_dict() if cv else None,
        "experiments": [item.to_dict() for item in experiments],
    }


def _initialize_workspace(
    args: dict[str, Any],
    context: ToolExecutionContext,
    registry: KaggleRegistry,
    data_dir: Path,
) -> Mapping[str, Any]:
    competition = registry.get_competition(project_id=str(args["project_id"]))
    if competition is None:
        raise KeyError(args["project_id"])
    cv = (
        registry.get_cv_spec(competition.active_cv_spec_id)
        if competition.active_cv_spec_id
        else None
    )
    workspace = _workspace(data_dir, competition.project_id)
    files = KaggleWorkspace(workspace).initialize(competition, cv_spec=cv)
    return {"workspace": str(workspace), "files": files}


def _create_cv(
    args: dict[str, Any],
    context: ToolExecutionContext,
    registry: KaggleRegistry,
) -> Mapping[str, Any]:
    competition = registry.get_competition(project_id=str(args["project_id"]))
    if competition is None:
        raise KeyError(args["project_id"])
    spec = CVSpec.new(
        competition_id=competition.competition_id,
        strategy=str(args["strategy"]),
        n_splits=int(args["n_splits"]),
        metric=str(args["metric"]),
        seed=int(args["seed"]),
        shuffle=bool(args["shuffle"]),
        group_column=(str(args["group_column"]) if args.get("group_column") else None),
        time_column=(str(args["time_column"]) if args.get("time_column") else None),
        stratify_column=(
            str(args["stratify_column"]) if args.get("stratify_column") else None
        ),
        metadata={"proposed_by": context.actor},
    )
    registry.create_cv_spec(spec)
    return spec.to_dict()


def _create_experiment(
    args: dict[str, Any],
    context: ToolExecutionContext,
    registry: KaggleRegistry,
    data_dir: Path,
) -> Mapping[str, Any]:
    competition = registry.get_competition(project_id=str(args["project_id"]))
    if competition is None:
        raise KeyError(args["project_id"])
    experiment = ExperimentRecord.new(
        competition_id=competition.competition_id,
        hypothesis=str(args["hypothesis"]),
        parent_experiment_id=(
            str(args["parent_experiment_id"])
            if args.get("parent_experiment_id")
            else None
        ),
        work_session_id=context.work_session_id,
        cv_spec_id=competition.active_cv_spec_id,
        dataset_fingerprint=competition.dataset_fingerprint,
        config=(dict(args["config"]) if isinstance(args.get("config"), dict) else {}),
        config_diff=(
            dict(args["config_diff"])
            if isinstance(args.get("config_diff"), dict)
            else {}
        ),
        metadata={"proposed_by": context.actor},
    )
    registry.create_experiment(experiment)
    directory = KaggleWorkspace(_workspace(data_dir, competition.project_id)).create_experiment_directory(
        experiment.experiment_id,
        parent_experiment_id=experiment.parent_experiment_id,
        hypothesis=experiment.hypothesis,
        config=experiment.config,
        config_diff=experiment.config_diff,
    )
    return {"experiment": experiment.to_dict(), "directory": str(directory)}


def _validate_candidate(
    args: dict[str, Any],
    context: ToolExecutionContext,
    data_dir: Path,
) -> Mapping[str, Any]:
    result = validate_submission(
        str(args["submission_path"]),
        str(args["sample_submission_path"]),
        id_columns=[str(item) for item in args.get("id_columns", [])],
        probability_columns=[
            str(item) for item in args.get("probability_columns", [])
        ],
        probability_groups=[
            [str(column) for column in group]
            for group in args.get("probability_groups", [])
            if isinstance(group, list)
        ],
    )
    report_dir = data_dir / "validation"
    report = report_dir / f"submission-{result.submission_sha256[:16]}.json"
    write_validation_report(result, report)
    return {**result.to_dict(), "report_path": str(report)}


def _create_candidate(
    args: dict[str, Any],
    context: ToolExecutionContext,
    registry: KaggleRegistry,
) -> Mapping[str, Any]:
    competition = registry.get_competition(project_id=str(args["project_id"]))
    if competition is None:
        raise KeyError(args["project_id"])
    candidate = create_submission_candidate(
        registry,
        competition_id=competition.competition_id,
        experiment_id=str(args["experiment_id"]),
        file_path=str(args["file_path"]),
        message=str(args.get("message") or ""),
        validation=(
            dict(args["validation"])
            if isinstance(args.get("validation"), dict)
            else {}
        ),
        cv_score=(float(args["cv_score"]) if args.get("cv_score") is not None else None),
        previous_best_cv=(
            float(args["previous_best_cv"])
            if args.get("previous_best_cv") is not None
            else None
        ),
        risks=[str(item) for item in args.get("risks", [])],
        metadata={"created_by": context.actor},
    )
    return candidate.to_dict()


def _workspace(data_dir: Path, project_id: str) -> Path:
    return data_dir / "projects" / _safe(project_id) / "kaggle"


def _safe(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:100] or "project"
