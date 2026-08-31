from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from harness.domains.kaggle.models import CVSpec, KaggleCompetitionState


class KaggleWorkspace:
    """Create the durable directory contract used by Kaggle jobs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def initialize(
        self,
        competition: KaggleCompetitionState,
        *,
        cv_spec: CVSpec | None = None,
        overwrite_generated: bool = False,
    ) -> dict[str, str]:
        directories = [
            self.root / "docs",
            self.root / "data" / "raw",
            self.root / "data" / "fingerprints",
            self.root / "data" / "processed",
            self.root / "data" / "sample_submission",
            self.root / "cv" / "splits",
            self.root / "src",
            self.root / "experiments",
            self.root / "submissions" / "candidates",
            self.root / "submissions" / "submitted",
            self.root / "logs",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        files = {
            "competition": self.root / "competition.json",
            "policy": self.root / "COMPETITION_POLICY.md",
            "agents": self.root / "AGENTS.md",
            "summary": self.root / "EXP_SUMMARY.md",
            "overview": self.root / "docs" / "overview.md",
            "evaluation": self.root / "docs" / "evaluation.md",
            "ideas": self.root / "docs" / "ideas.md",
            "cv": self.root / "cv" / "active_cv.json",
            "train": self.root / "src" / "train.py",
            "infer": self.root / "src" / "infer.py",
            "submission_history": self.root / "submissions" / "submission_history.jsonl",
        }
        self._write_json(
            files["competition"],
            competition.to_dict(),
            overwrite=overwrite_generated,
        )
        self._write_text(
            files["policy"],
            _policy(competition),
            overwrite=overwrite_generated,
        )
        self._write_text(
            files["agents"],
            _agents(competition),
            overwrite=overwrite_generated,
        )
        self._write_text(
            files["summary"],
            "# Experiment Summary\n\nNo experiments have completed.\n",
            overwrite=False,
        )
        self._write_text(
            files["overview"],
            f"# {competition.title}\n\n- URL: {competition.url}\n- Slug: `{competition.slug}`\n- Rules: not yet acknowledged\n",
            overwrite=False,
        )
        self._write_text(
            files["evaluation"],
            f"# Evaluation\n\nMetric: {competition.evaluation_metric or 'unconfirmed'}\n",
            overwrite=False,
        )
        self._write_text(
            files["ideas"],
            "# Ideas\n\nRecord human hypotheses and rejected directions here.\n",
            overwrite=False,
        )
        if cv_spec is not None:
            self._write_json(files["cv"], cv_spec.to_dict(), overwrite=True)
        self._write_text(files["train"], _train_template(), overwrite=False)
        self._write_text(files["infer"], _infer_template(), overwrite=False)
        files["submission_history"].touch(exist_ok=True)
        self._make_raw_readme()
        return {key: str(path) for key, path in files.items()}

    def create_experiment_directory(
        self,
        experiment_id: str,
        *,
        parent_experiment_id: str | None,
        hypothesis: str,
        config: Mapping[str, Any] | None = None,
        config_diff: Mapping[str, Any] | None = None,
    ) -> Path:
        directory = self.root / "experiments" / _safe(experiment_id)
        directory.mkdir(parents=True, exist_ok=False)
        for child in ("source_snapshot", "logs", "model", "outputs"):
            (directory / child).mkdir()
        self._write_json(
            directory / "experiment.json",
            {
                "experiment_id": experiment_id,
                "parent_experiment_id": parent_experiment_id,
                "hypothesis": hypothesis,
                "config": dict(config or {}),
                "config_diff": dict(config_diff or {}),
                "status": "proposed",
            },
            overwrite=True,
        )
        return directory

    def notebook_package(
        self,
        experiment_dir: str | Path,
        *,
        owner: str,
        kernel_slug: str,
        title: str,
        competition_slug: str,
        enable_gpu: bool,
        enable_internet: bool,
    ) -> Path:
        source = Path(experiment_dir).expanduser().resolve()
        try:
            source.relative_to((self.root / "experiments").resolve())
        except ValueError as exc:
            raise PermissionError("experiment directory is outside workspace") from exc
        package = source / "kaggle_kernel"
        package.mkdir(parents=True, exist_ok=True)
        run_path = package / "run.py"
        if not run_path.exists():
            run_path.write_text(_kernel_runner(), encoding="utf-8")
        metadata = {
            "id": f"{owner}/{kernel_slug}",
            "title": title,
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": enable_gpu,
            "enable_internet": enable_internet,
            "dataset_sources": [],
            "competition_sources": [competition_slug],
            "kernel_sources": [],
        }
        self._write_json(
            package / "kernel-metadata.json",
            metadata,
            overwrite=True,
        )
        return package

    def _make_raw_readme(self) -> None:
        path = self.root / "data" / "raw" / "README.md"
        self._write_text(
            path,
            "# Raw data\n\nTreat this directory as read-only. Do not overwrite competition data.\n",
            overwrite=False,
        )

    @staticmethod
    def _write_text(path: Path, content: str, *, overwrite: bool) -> None:
        if path.exists() and not overwrite:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
        KaggleWorkspace._write_text(
            path,
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            overwrite=overwrite,
        )


def _policy(competition: KaggleCompetitionState) -> str:
    return f"""# Competition Policy

- Competition: {competition.title}
- Slug: `{competition.slug}`
- URL: {competition.url}

## Non-negotiable rules

- Do not submit without a human approval bound to the exact SHA-256 file hash.
- Do not upload or publish datasets/notebooks without a separate approval.
- Do not overwrite `data/raw/`.
- Confirm the competition rules, external-data policy, team/private-sharing policy, and metric before training.
- Build a valid minimal submission before complex modeling.
- Lock a CV specification before comparing experiments.
- Fit preprocessing only inside each training fold.
- Preserve OOF predictions, failed experiments, configuration diffs, and provenance.
- Public leaderboard score alone is not sufficient evidence.
- A new hypothesis creates a child experiment; do not mutate the current best experiment.
"""


def _agents(competition: KaggleCompetitionState) -> str:
    return f"""# Agent Instructions

This workspace belongs to Kaggle competition `{competition.slug}`.

Roles:
- Lead: convert human hypotheses into bounded experiments.
- Researcher: inspect public rules, discussions, writeups, and notebooks with citations.
- Builder: implement train/infer/config without changing the locked CV contract.
- Analyst: compare OOF/error/runtime and CV/LB correlation.
- Reviewer: detect leakage, format errors, non-reproducibility, and unsafe operations.
- Submission operator: submit only an approved immutable file hash.

Every executable job must write `progress.json` and terminal `result.json`.
Do not expose Kaggle, Discord, OpenAI, or worker credentials to model-generated code.
"""


def _train_template() -> str:
    return '''"""Competition training entrypoint.

Write fold-safe training code here. The finished program must produce:
- progress.json during execution
- result.json with metric/fold_scores/runtime/errors
- optional oof.parquet/model artifacts
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    Path("progress.json").write_text(
        json.dumps({"stage": "smoke_test", "progress": 0.05}) + "\\n",
        encoding="utf-8",
    )
    raise NotImplementedError("Implement a fold-safe baseline before running")


if __name__ == "__main__":
    main()
'''


def _infer_template() -> str:
    return '''"""Competition inference entrypoint.

Generate a candidate only. Submission is performed by the gated Kaggle Gateway.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Implement inference after the baseline is reviewed")


if __name__ == "__main__":
    main()
'''


def _kernel_runner() -> str:
    return '''"""Generated Kaggle kernel runner.

Copy or import the reviewed experiment code into this package before push.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    Path("progress.json").write_text(
        json.dumps({"stage": "kernel_started", "progress": 0.1}) + "\\n",
        encoding="utf-8",
    )
    raise NotImplementedError("Package reviewed experiment code before submission")


if __name__ == "__main__":
    main()
'''


def _safe(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:100] or "experiment"
