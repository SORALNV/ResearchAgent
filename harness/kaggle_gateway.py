from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from harness.kaggle_domain import KaggleStore, SubmissionCandidate
from harness.kaggle_validation import (
    SubmissionValidationReport,
    validate_submission,
    verify_submission_hash,
)


@dataclass(frozen=True)
class KaggleSubmissionResult:
    ok: bool
    reference: str | None
    stdout: str
    stderr: str


class KaggleSubmissionTransport(Protocol):
    def submit(
        self,
        competition_slug: str,
        file_path: Path,
        message: str,
    ) -> KaggleSubmissionResult: ...


class KaggleCliSubmissionTransport:
    """Submission-only gateway. Credentials never enter Agent subprocesses."""

    SAFE_ENV = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "KAGGLE_CONFIG_DIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SYSTEMROOT",
        "WINDIR",
        "TMP",
        "TEMP",
        "TMPDIR",
    }

    def __init__(
        self,
        *,
        executable: str = "kaggle",
        timeout_seconds: int = 120,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.executable = executable
        self.timeout_seconds = max(1, int(timeout_seconds))
        source = dict(os.environ if environment is None else environment)
        self.environment = {
            key: value
            for key, value in source.items()
            if key in self.SAFE_ENV and value
        }

    def submit(
        self,
        competition_slug: str,
        file_path: Path,
        message: str,
    ) -> KaggleSubmissionResult:
        command = [
            self.executable,
            "competitions",
            "submit",
            "-c",
            competition_slug,
            "-f",
            str(file_path),
            "-m",
            message[:100],
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                env=self.environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return KaggleSubmissionResult(
                ok=False,
                reference=None,
                stdout=_text(exc.stdout),
                stderr=_text(exc.stderr) or "submission command timed out",
            )
        except OSError as exc:
            return KaggleSubmissionResult(
                ok=False,
                reference=None,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
            )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        return KaggleSubmissionResult(
            ok=completed.returncode == 0,
            reference=_extract_submission_reference(stdout) if completed.returncode == 0 else None,
            stdout=stdout,
            stderr=stderr,
        )


class KaggleSubmissionGateway:
    """Hash-bound, human-approved Kaggle submission workflow."""

    def __init__(
        self,
        store: KaggleStore,
        transport: KaggleSubmissionTransport,
    ) -> None:
        self.store = store
        self.transport = transport

    def prepare(
        self,
        *,
        project_id: str,
        experiment_id: str,
        submission_path: str | Path,
        sample_submission_path: str | Path,
        message: str,
        id_column: str | None = None,
        prediction_ranges: Mapping[str, tuple[float | None, float | None]] | None = None,
        cv_score: float | None = None,
    ) -> tuple[SubmissionCandidate, SubmissionValidationReport]:
        report = validate_submission(
            submission_path,
            sample_submission_path,
            id_column=id_column,
            prediction_ranges=prediction_ranges,
        )
        candidate = self.store.create_submission_candidate(
            project_id=project_id,
            experiment_id=experiment_id,
            file_path=submission_path,
            sha256=report.sha256,
            validation=report.to_dict(),
            message=message,
            cv_score=cv_score,
        )
        return candidate, report

    def attach_approval(
        self,
        candidate_id: str,
        approval_id: str,
    ) -> SubmissionCandidate:
        candidate = self._require(candidate_id)
        if candidate.status != "waiting_approval":
            raise ValueError(f"candidate is not waiting for approval: {candidate.status}")
        return self.store.update_submission_candidate(
            candidate_id,
            status="waiting_approval",
            approval_id=approval_id,
        )

    def approve(
        self,
        candidate_id: str,
        approval_id: str,
    ) -> SubmissionCandidate:
        candidate = self._require(candidate_id)
        if candidate.status not in {"waiting_approval", "prepared"}:
            raise ValueError(f"candidate cannot be approved from {candidate.status}")
        if candidate.approval_id and candidate.approval_id != approval_id:
            raise ValueError("approval id does not match candidate")
        hash_check = verify_submission_hash(candidate.file_path, candidate.sha256)
        if not hash_check.ok:
            self.store.update_submission_candidate(candidate_id, status="invalid")
            raise ValueError("submission file changed after validation")
        return self.store.update_submission_candidate(
            candidate_id,
            status="approved",
            approval_id=approval_id,
        )

    def submit(self, candidate_id: str) -> SubmissionCandidate:
        candidate = self._require(candidate_id)
        if candidate.status == "submitted":
            return candidate
        if candidate.status != "approved" or not candidate.approval_id:
            raise PermissionError("candidate requires explicit approval before submission")
        hash_check = verify_submission_hash(candidate.file_path, candidate.sha256)
        if not hash_check.ok:
            self.store.update_submission_candidate(candidate_id, status="invalid")
            raise ValueError("approved submission hash no longer matches")
        competition = self.store.get_competition(candidate.project_id)
        if competition is None:
            raise KeyError(candidate.project_id)
        result = self.transport.submit(
            competition.competition_slug,
            Path(candidate.file_path),
            candidate.message,
        )
        if not result.ok:
            self.store.update_submission_candidate(candidate_id, status="failed")
            raise RuntimeError(
                "Kaggle submission failed: " + (result.stderr or result.stdout)
            )
        reference = result.reference or _fallback_reference(candidate)
        return self.store.update_submission_candidate(
            candidate_id,
            status="submitted",
            kaggle_submission_ref=reference,
        )

    def _require(self, candidate_id: str) -> SubmissionCandidate:
        candidate = self.store.get_submission_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        return candidate


@dataclass
class FakeKaggleSubmissionTransport:
    fail: bool = False

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def submit(
        self,
        competition_slug: str,
        file_path: Path,
        message: str,
    ) -> KaggleSubmissionResult:
        self.calls.append(
            {
                "competition_slug": competition_slug,
                "file_path": str(file_path),
                "message": message,
            }
        )
        if self.fail:
            return KaggleSubmissionResult(False, None, "", "fake submit failure")
        return KaggleSubmissionResult(
            True,
            f"fake:{competition_slug}:{file_path.name}",
            json.dumps({"submitted": True}),
            "",
        )


def _extract_submission_reference(stdout: str) -> str | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:500]
    return None


def _fallback_reference(candidate: SubmissionCandidate) -> str:
    return f"submitted:{candidate.candidate_id}:{candidate.sha256[:12]}"


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
