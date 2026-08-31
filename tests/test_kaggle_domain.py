from __future__ import annotations

from pathlib import Path

import pytest

from harness.domains.kaggle.gateway import KaggleGateway, create_submission_candidate
from harness.domains.kaggle.models import (
    CVSpec,
    ExperimentRecord,
    KaggleCompetitionState,
    SubmissionStatus,
)
from harness.domains.kaggle.registry import KaggleRegistry
from harness.domains.kaggle.validation import dataset_fingerprint, validate_submission
from harness.domains.kaggle.workspace import KaggleWorkspace


def test_kaggle_registry_workspace_cv_and_child_experiments(tmp_path):
    registry = KaggleRegistry(tmp_path / "platform.sqlite3")
    competition = registry.create_competition(
        KaggleCompetitionState.new(
            project_id="KG-1",
            slug="example-competition",
            url="https://www.kaggle.com/competitions/example-competition",
            title="Example",
        )
    )
    cv = registry.create_cv_spec(
        CVSpec.new(
            competition_id=competition.competition_id,
            strategy="StratifiedKFold",
            n_splits=5,
            metric="roc_auc",
            stratify_column="target",
        )
    )
    locked = registry.lock_cv_spec(cv.cv_spec_id)
    assert locked.locked
    competition = registry.update_competition(
        competition.competition_id,
        active_cv_spec_id=cv.cv_spec_id,
    )

    parent = registry.create_experiment(
        ExperimentRecord.new(
            competition_id=competition.competition_id,
            hypothesis="minimal baseline",
            cv_spec_id=cv.cv_spec_id,
        )
    )
    child = registry.create_experiment(
        ExperimentRecord.new(
            competition_id=competition.competition_id,
            hypothesis="add one leakage-safe feature",
            parent_experiment_id=parent.experiment_id,
            cv_spec_id=cv.cv_spec_id,
            config_diff={"feature": "safe_feature"},
        )
    )
    assert registry.get_experiment(child.experiment_id).parent_experiment_id == parent.experiment_id

    workspace = KaggleWorkspace(tmp_path / "competition")
    files = workspace.initialize(competition, cv_spec=locked)
    assert Path(files["policy"]).is_file()
    assert (tmp_path / "competition" / "data" / "raw" / "README.md").is_file()
    exp_dir = workspace.create_experiment_directory(
        child.experiment_id,
        parent_experiment_id=parent.experiment_id,
        hypothesis=child.hypothesis,
        config=child.config,
        config_diff=child.config_diff,
    )
    assert (exp_dir / "experiment.json").is_file()


def test_submission_validation_and_hash_bound_gateway(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,pred\n1,0.5\n2,0.5\n", encoding="utf-8")
    submission.write_text("id,pred\n1,0.2\n2,0.8\n", encoding="utf-8")
    result = validate_submission(
        submission,
        sample,
        id_columns=("id",),
        probability_columns=("pred",),
    )
    assert result.valid
    assert result.checks["columns"] == "pass"
    assert result.checks["id_alignment"] == "pass"
    assert result.submission_sha256

    registry = KaggleRegistry(tmp_path / "platform.sqlite3")
    competition = registry.create_competition(
        KaggleCompetitionState.new(
            project_id="KG-2",
            slug="hash-test",
            url="https://www.kaggle.com/competitions/hash-test",
            title="Hash test",
        )
    )
    competition = registry.update_competition(
        competition.competition_id,
        rules_acknowledged=True,
        rules_hash="rules-hash",
    )
    experiment = registry.create_experiment(
        ExperimentRecord.new(
            competition_id=competition.competition_id,
            hypothesis="baseline",
        )
    )
    candidate = create_submission_candidate(
        registry,
        competition_id=competition.competition_id,
        experiment_id=experiment.experiment_id,
        file_path=submission,
        message="baseline",
        validation=result.to_dict(),
    )
    assert candidate.status == SubmissionStatus.WAITING_APPROVAL

    commands = []

    def runner(command, cwd, environment):
        commands.append(list(command))
        assert "DISCORD_BOT_TOKEN" not in environment
        return 0, "Successfully submitted https://www.kaggle.com/test", ""

    gateway = KaggleGateway(
        registry=registry,
        command_runner=runner,
        api_token="test-kaggle-token",
    )
    approved = gateway.approve_candidate(
        candidate.candidate_id,
        approval_id="AP-DISCORD-1",
    )
    assert approved.status == SubmissionStatus.APPROVED

    submission.write_text("id,pred\n1,0.3\n2,0.7\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        gateway.submit_candidate(candidate.candidate_id)
    assert commands == []

    validation2 = validate_submission(
        submission,
        sample,
        id_columns=("id",),
        probability_columns=("pred",),
    )
    candidate2 = create_submission_candidate(
        registry,
        competition_id=competition.competition_id,
        experiment_id=experiment.experiment_id,
        file_path=submission,
        message="baseline corrected",
        validation=validation2.to_dict(),
    )
    gateway.approve_candidate(candidate2.candidate_id, approval_id="AP-DISCORD-2")
    submitted = gateway.submit_candidate(candidate2.candidate_id)
    assert submitted.status == SubmissionStatus.SUBMITTED
    assert commands[0][1:3] == ["competitions", "submit"]


def test_dataset_fingerprint_detects_content_change(tmp_path):
    first = tmp_path / "train.csv"
    second = tmp_path / "test.csv"
    first.write_text("x,y\n1,2\n", encoding="utf-8")
    second.write_text("x\n3\n", encoding="utf-8")
    before = dataset_fingerprint([first, second], root=tmp_path)
    first.write_text("x,y\n1,9\n", encoding="utf-8")
    after = dataset_fingerprint([first, second], root=tmp_path)
    assert before != after
