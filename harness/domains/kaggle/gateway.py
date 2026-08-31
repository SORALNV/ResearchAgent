from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from harness.domains.kaggle.models import (
    KaggleCompetitionState,
    SubmissionCandidate,
    SubmissionStatus,
)
from harness.domains.kaggle.registry import KaggleRegistry
from harness.state import utc_timestamp


class KaggleGatewayError(RuntimeError):
    pass


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], tuple[int, str, str]]


class KaggleGateway:
    """Credential-holding Kaggle operations isolated from model runtimes."""

    def __init__(
        self,
        *,
        registry: KaggleRegistry,
        command: str = "kaggle",
        api_token: str | None = None,
        username: str | None = None,
        command_runner: CommandRunner | None = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.registry = registry
        self.command = command
        self.api_token = api_token
        self.username = username
        self.command_runner = command_runner or self._default_runner
        self.timeout_seconds = max(10, timeout_seconds)

    def available(self) -> tuple[bool, str]:
        if not self.command.strip():
            return False, "Kaggle command is empty"
        if not self.api_token and not _has_kaggle_config():
            return False, "Kaggle credential is not configured in Core"
        return True, self.command

    def competition_files(
        self,
        competition: KaggleCompetitionState,
        *,
        cwd: str | Path,
    ) -> list[dict[str, Any]]:
        code, stdout, stderr = self._run(
            [self.command, "competitions", "files", "-c", competition.slug, "-v"],
            Path(cwd),
        )
        if code != 0:
            raise KaggleGatewayError((stderr or stdout)[-4000:])
        return _parse_csv_or_lines(stdout)

    def download_competition_data(
        self,
        competition: KaggleCompetitionState,
        *,
        destination: str | Path,
        force: bool = False,
    ) -> dict[str, Any]:
        if not competition.rules_acknowledged:
            raise PermissionError("competition rules must be acknowledged before download")
        target = Path(destination).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        command = [
            self.command,
            "competitions",
            "download",
            "-c",
            competition.slug,
            "-p",
            str(target),
        ]
        if force:
            command.append("--force")
        code, stdout, stderr = self._run(command, target)
        if code != 0:
            raise KaggleGatewayError((stderr or stdout)[-4000:])
        return {
            "competition": competition.slug,
            "destination": str(target),
            "stdout": stdout[-4000:],
        }

    def approve_candidate(
        self,
        candidate_id: str,
        *,
        approval_id: str,
    ) -> SubmissionCandidate:
        candidate = self._require_candidate(candidate_id)
        path = Path(candidate.file_path).expanduser().resolve()
        self._assert_hash(candidate, path)
        if not bool(candidate.validation.get("valid")):
            raise ValueError("submission candidate validation has not passed")
        return self.registry.update_submission(
            candidate_id,
            status=SubmissionStatus.APPROVED,
            approval_id=approval_id,
            metadata={
                **candidate.metadata,
                "approved_sha256": candidate.file_sha256,
                "approved_at": utc_timestamp(),
            },
        )

    def submit_candidate(
        self,
        candidate_id: str,
        *,
        cwd: str | Path | None = None,
    ) -> SubmissionCandidate:
        candidate = self._require_candidate(candidate_id)
        if candidate.status != SubmissionStatus.APPROVED:
            raise PermissionError(
                f"candidate is not approved: {candidate.status.value}"
            )
        approved_hash = str(candidate.metadata.get("approved_sha256") or "")
        if not approved_hash or approved_hash != candidate.file_sha256:
            raise PermissionError("submission approval is not bound to this hash")
        path = Path(candidate.file_path).expanduser().resolve()
        self._assert_hash(candidate, path)
        competition = self.registry.get_competition(candidate.competition_id)
        if competition is None:
            raise KeyError(candidate.competition_id)
        if not competition.rules_acknowledged:
            raise PermissionError("competition rules are not acknowledged")
        workdir = Path(cwd).expanduser().resolve() if cwd else path.parent
        code, stdout, stderr = self._run(
            [
                self.command,
                "competitions",
                "submit",
                "-c",
                competition.slug,
                "-f",
                str(path),
                "-m",
                candidate.message or candidate.experiment_id,
            ],
            workdir,
        )
        if code != 0:
            return self.registry.update_submission(
                candidate_id,
                status=SubmissionStatus.FAILED,
                metadata={
                    **candidate.metadata,
                    "submit_stdout": stdout[-4000:],
                    "submit_stderr": stderr[-4000:],
                },
            )
        kaggle_ref = _extract_reference(stdout)
        return self.registry.update_submission(
            candidate_id,
            status=SubmissionStatus.SUBMITTED,
            submitted_at=utc_timestamp(),
            kaggle_ref=kaggle_ref,
            metadata={
                **candidate.metadata,
                "submit_stdout": stdout[-4000:],
            },
        )

    def submission_history(
        self,
        competition: KaggleCompetitionState,
        *,
        cwd: str | Path,
    ) -> list[dict[str, Any]]:
        code, stdout, stderr = self._run(
            [
                self.command,
                "competitions",
                "submissions",
                "-c",
                competition.slug,
                "-v",
            ],
            Path(cwd),
        )
        if code != 0:
            raise KaggleGatewayError((stderr or stdout)[-4000:])
        return _parse_csv_or_lines(stdout)

    def _require_candidate(self, candidate_id: str) -> SubmissionCandidate:
        candidate = self.registry.get_submission(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        return candidate

    @staticmethod
    def _assert_hash(candidate: SubmissionCandidate, path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != candidate.file_sha256:
            raise PermissionError(
                "submission file changed after candidate creation; approval is invalid"
            )

    def _run(self, command: Sequence[str], cwd: Path) -> tuple[int, str, str]:
        cwd.mkdir(parents=True, exist_ok=True)
        return self.command_runner(command, cwd, self._environment())

    def _environment(self) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "HOME",
                "USERPROFILE",
                "LANG",
                "LC_ALL",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "KAGGLE_CONFIG_DIR",
                "KAGGLE_USERNAME",
                "KAGGLE_KEY",
                "KAGGLE_API_TOKEN",
            }
        }
        if self.api_token:
            environment["KAGGLE_API_TOKEN"] = self.api_token
        if self.username:
            environment["KAGGLE_USERNAME"] = self.username
        return environment

    def _default_runner(
        self,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> tuple[int, str, str]:
        try:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(environment),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 127, "", f"{type(exc).__name__}: {exc}"
        return completed.returncode, completed.stdout, completed.stderr


def create_submission_candidate(
    registry: KaggleRegistry,
    *,
    competition_id: str,
    experiment_id: str,
    file_path: str | Path,
    message: str,
    validation: Mapping[str, Any],
    cv_score: float | None = None,
    previous_best_cv: float | None = None,
    risks: tuple[str, ...] | list[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> SubmissionCandidate:
    path = Path(file_path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    candidate = SubmissionCandidate.new(
        competition_id=competition_id,
        experiment_id=experiment_id,
        file_path=str(path),
        file_sha256=_sha256(path),
        message=message,
        cv_score=cv_score,
        previous_best_cv=previous_best_cv,
        validation=validation,
        risks=risks,
        metadata=metadata,
    )
    if bool(validation.get("valid")):
        candidate = replace(
            candidate,
            status=SubmissionStatus.WAITING_APPROVAL,
        )
    else:
        candidate = replace(candidate, status=SubmissionStatus.INVALID)
    return registry.create_submission(candidate)


def _has_kaggle_config() -> bool:
    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    return bool(
        os.getenv("KAGGLE_API_TOKEN")
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        or (config_dir / "kaggle.json").is_file()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv_or_lines(text: str) -> list[dict[str, Any]]:
    import csv
    import io

    stripped = text.strip()
    if not stripped:
        return []
    try:
        reader = csv.DictReader(io.StringIO(stripped))
        rows = [dict(row) for row in reader]
        if reader.fieldnames and rows:
            return rows
    except csv.Error:
        pass
    return [{"line": line} for line in stripped.splitlines() if line.strip()]


def _extract_reference(text: str) -> str | None:
    for token in text.replace("\n", " ").split():
        if token.startswith("http://") or token.startswith("https://"):
            return token.strip(".,)")
    return None
