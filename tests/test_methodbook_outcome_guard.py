from pathlib import Path

from harness.control_plane import (
    ControlPlaneStore,
    Domain,
    JobSpec,
    ResourceRequirements,
)
from harness.kaggle_methodbook import MethodCardStore
from harness.methodbook_planner import MethodBookAwareMemoPlanner


class WrongOutcomePlanner:
    def summarize(self, *, job, result, result_ref, proposals, backend):
        return {
            "outcome": "regressed",
            "lesson_summary": "model-supplied classification must not be trusted",
            "method_candidates": [],
        }


def test_model_cannot_override_metric_derived_iteration_outcome(tmp_path: Path) -> None:
    store = ControlPlaneStore(tmp_path / "control-plane")
    project = store.create_project("guard", Domain.KAGGLE)
    session = store.create_work_session(project.project_id, "guard")
    job = store.create_job(
        JobSpec(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain=Domain.KAGGLE,
            kind="experiment",
            payload={"hypothesis": "test", "competition_slug": "guard-comp"},
            resources=ResourceRequirements(accelerator="cpu"),
            experiment_id="guard-exp",
        )
    )
    planner = MethodBookAwareMemoPlanner(
        WrongOutcomePlanner(),
        MethodCardStore(tmp_path / "knowledge"),
    )
    output = planner.summarize(
        job=job,
        result={
            "status": "succeeded",
            "primary_metric": {
                "name": "cv_score",
                "value": 0.84,
                "baseline": 0.82,
                "direction": "maximize",
            },
        },
        result_ref="result:guard",
        proposals=(),
        backend="fake",
    )
    assert output["outcome"] == "improved"
    assert output["observed_outcome"] == "improved"
