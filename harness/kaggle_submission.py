from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from harness.compute_feedback import ResultFeedbackEngine
from harness.control_plane import ControlPlaneStore, Domain, Event, EventLane, Job
from harness.discord_thread_router import DiscordThreadRoute, DiscordThreadRouter
from harness.human_decision_policy import ControlledAction
from harness.state import utc_timestamp


class SubmissionState(str, Enum):
    READY = "ready"
    INVALID = "invalid"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    SUBMITTED_UNCONFIRMED = "submitted_unconfirmed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class KaggleSubmissionError(RuntimeError):
    pass


class SubmissionBlockedError(KaggleSubmissionError):
    """The action is not currently executable but may become executable later."""


class SubmissionUncertainError(KaggleSubmissionError):
    """A prior submit may have reached Kaggle; blind re-submission is forbidden."""


@dataclass(frozen=True)
class SubmissionCandidate:
    candidate_id: str
    project_id: str
    work_session_id: str
    job_id: str | None
    result_ref: str
    source_event_id: str
    competition_slug: str
    file_path: str
    relative_path: str
    file_sha256: str
    file_size: int
    message: str
    marker: str
    validation: dict[str, Any] = field(default_factory=dict)
    risks: tuple[str, ...] = ()
    state: SubmissionState = SubmissionState.READY
    submitted_at: str | None = None
    confirmed_at: str | None = None
    public_score: float | None = None
    private_score: float | None = None
    kaggle_status: str | None = None
    kaggle_ref: str | None = None
    history_row: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    @property
    def subject_ref(self) -> str:
        return f"sha256:{self.file_sha256}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["risks"] = list(self.risks)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubmissionCandidate":
        return cls(
            candidate_id=str(data["candidate_id"]),
            project_id=str(data["project_id"]),
            work_session_id=str(data["work_session_id"]),
            job_id=(str(data["job_id"]) if data.get("job_id") else None),
            result_ref=str(data.get("result_ref") or ""),
            source_event_id=str(data.get("source_event_id") or ""),
            competition_slug=str(data.get("competition_slug") or ""),
            file_path=str(data.get("file_path") or ""),
            relative_path=str(data.get("relative_path") or ""),
            file_sha256=str(data.get("file_sha256") or ""),
            file_size=int(data.get("file_size") or 0),
            message=str(data.get("message") or ""),
            marker=str(data.get("marker") or ""),
            validation=_json_dict(data.get("validation")),
            risks=tuple(str(item) for item in data.get("risks", [])),
            state=SubmissionState(
                str(data.get("state") or SubmissionState.READY.value)
            ),
            submitted_at=_optional_str(data.get("submitted_at")),
            confirmed_at=_optional_str(data.get("confirmed_at")),
            public_score=_optional_float(data.get("public_score")),
            private_score=_optional_float(data.get("private_score")),
            kaggle_status=_optional_str(data.get("kaggle_status")),
            kaggle_ref=_optional_str(data.get("kaggle_ref")),
            history_row=_json_dict(data.get("history_row")),
            last_error=_optional_str(data.get("last_error")),
            created_at=str(data.get("created_at") or utc_timestamp()),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


class SubmissionCandidateStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.candidates_dir = self.root / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def save(self, candidate: SubmissionCandidate) -> SubmissionCandidate:
        updated = replace(candidate, updated_at=utc_timestamp())
        with self._lock:
            _atomic_json(self._path(updated.candidate_id), updated.to_dict())
        return updated

    def get(self, candidate_id: str) -> SubmissionCandidate | None:
        path = self._path(candidate_id)
        if not path.is_file():
            return None
        with self._lock:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return (
            SubmissionCandidate.from_dict(value)
            if isinstance(value, Mapping)
            else None
        )

    def list(self, *, work_session_id: str | None = None) -> list[SubmissionCandidate]:
        result: list[SubmissionCandidate] = []
        for path in sorted(self.candidates_dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, Mapping):
                    continue
                candidate = SubmissionCandidate.from_dict(value)
            except Exception:
                continue
            if work_session_id and candidate.work_session_id != work_session_id:
                continue
            result.append(candidate)
        return sorted(result, key=lambda item: (item.created_at, item.candidate_id))

    def find_by_hash(
        self,
        *,
        work_session_id: str,
        file_sha256: str,
    ) -> SubmissionCandidate | None:
        digest = _normalize_digest(file_sha256)
        matches = [
            item
            for item in self.list(work_session_id=work_session_id)
            if item.file_sha256 == digest
        ]
        if not matches:
            return None
        scopes = {(item.competition_slug, item.file_path) for item in matches}
        if len(scopes) > 1:
            raise KaggleSubmissionError(
                "the approved SHA-256 resolves to multiple Kaggle candidates in "
                "this WorkSession; refusing an ambiguous submission"
            )
        return matches[-1]

    def _path(self, candidate_id: str) -> Path:
        safe = _safe_component(candidate_id)
        return self.candidates_dir / f"{safe}.json"


@dataclass(frozen=True)
class KaggleCommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[
    [Sequence[str], Path, Mapping[str, str]], KaggleCommandResult
]


class KaggleCliTransport:
    """Credential-holding Kaggle CLI boundary used only by the Core process."""

    def __init__(
        self,
        *,
        command: str = "kaggle",
        api_token: str | None = None,
        username: str | None = None,
        key: str | None = None,
        timeout_seconds: int = 180,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.command = command.strip()
        self.api_token = api_token
        self.username = username
        self.key = key
        self.timeout_seconds = max(10, int(timeout_seconds))
        self.command_runner = command_runner

    def available(self) -> tuple[bool, str]:
        command = self._base_command()
        if not command:
            return False, "KAGGLE_COMMAND is empty"
        if self.command_runner is None and shutil.which(command[0]) is None:
            return False, f"Kaggle command not found: {command[0]}"
        if self.command_runner is None and not self._credentials_available():
            return False, "Kaggle credentials are not configured in Core"
        return True, command[0]

    def submission_history(
        self,
        competition_slug: str,
        *,
        cwd: str | Path,
    ) -> list[dict[str, str]]:
        command = [
            *self._base_command(),
            "competitions",
            "submissions",
            "-c",
            competition_slug,
            "-v",
        ]
        result = self._run(command, Path(cwd))
        if result.returncode != 0:
            raise KaggleSubmissionError(
                "Kaggle submission-history query failed: "
                + (result.stderr or result.stdout)[-4000:]
            )
        return _parse_history(result.stdout)

    def submit(
        self,
        *,
        competition_slug: str,
        file_path: str | Path,
        message: str,
        cwd: str | Path,
    ) -> KaggleCommandResult:
        command = [
            *self._base_command(),
            "competitions",
            "submit",
            "-c",
            competition_slug,
            "-f",
            str(Path(file_path).expanduser().resolve()),
            "-m",
            message,
        ]
        return self._run(command, Path(cwd))

    def _run(self, command: Sequence[str], cwd: Path) -> KaggleCommandResult:
        target = cwd.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        environment = self._environment()
        if self.command_runner is not None:
            return self.command_runner(tuple(command), target, environment)
        try:
            completed = subprocess.run(
                list(command),
                cwd=target,
                env=environment,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return KaggleCommandResult(127, "", f"{type(exc).__name__}: {exc}")
        return KaggleCommandResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )

    def _base_command(self) -> tuple[str, ...]:
        if not self.command:
            return ()
        try:
            return tuple(shlex.split(self.command))
        except ValueError:
            return ()

    def _credentials_available(self) -> bool:
        config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
        return bool(
            self.api_token
            or (self.username and self.key)
            or os.getenv("KAGGLE_API_TOKEN")
            or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
            or (config_dir / "kaggle.json").is_file()
        )

    def _environment(self) -> dict[str, str]:
        allowed_names = {
            "PATH",
            "HOME",
            "USERPROFILE",
            "LANG",
            "LC_ALL",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "KAGGLE_CONFIG_DIR",
            "KAGGLE_USERNAME",
            "KAGGLE_KEY",
            "KAGGLE_API_TOKEN",
        }
        environment = {
            key: value for key, value in os.environ.items() if key in allowed_names
        }
        if self.api_token:
            environment["KAGGLE_API_TOKEN"] = self.api_token
        if self.username:
            environment["KAGGLE_USERNAME"] = self.username
        if self.key:
            environment["KAGGLE_KEY"] = self.key
        return environment


class KaggleSubmissionPipeline:
    CANDIDATE_EVENT = "kaggle.submission.candidate"
    STARTED_EVENT = "kaggle.submission.started"
    COMPLETED_EVENT = "kaggle.submission.completed"
    HISTORY_EVENT = "kaggle.submission.history_updated"
    FAILED_EVENT = "kaggle.submission.failed"

    def __init__(
        self,
        *,
        router: DiscordThreadRouter,
        root_dir: str | Path,
        transport: KaggleCliTransport,
        rules_acknowledged: Sequence[str] = (),
        default_competition: str = "",
        max_file_bytes: int = 512 * 1024 * 1024,
        history_poll_seconds: float = 5.0,
        history_timeout_seconds: float = 90.0,
    ) -> None:
        self.router = router
        self.store: ControlPlaneStore = router.store
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.candidates = SubmissionCandidateStore(self.root_dir)
        self.transport = transport
        self.rules_acknowledged = {
            str(item).strip().lower()
            for item in rules_acknowledged
            if str(item).strip()
        }
        self.default_competition = default_competition.strip()
        self.max_file_bytes = max(1, int(max_file_bytes))
        self.history_poll_seconds = max(0.0, float(history_poll_seconds))
        self.history_timeout_seconds = max(0.0, float(history_timeout_seconds))

    def discover_work_session(self, work_session_id: str) -> tuple[SubmissionCandidate, ...]:
        session = self.store.get_work_session(work_session_id)
        project = self.store.get_project(session.project_id)
        if project.domain != Domain.KAGGLE:
            return ()
        discovered: list[SubmissionCandidate] = []
        for event in self.store.latest_events(
            work_session_id=work_session_id,
            lanes=[EventLane.DATA],
            limit=5000,
        ):
            if event.event_type != ResultFeedbackEngine.RESULT_EVENT:
                continue
            for candidate in self._discover_from_result_event(event):
                discovered.append(candidate)
        return tuple(discovered)

    def execute(
        self,
        route: DiscordThreadRoute,
        *,
        subject_ref: str,
    ) -> SubmissionCandidate:
        if route.domain != Domain.KAGGLE:
            raise ValueError("Kaggle submission is only valid in the kaggle domain")
        digest = _normalize_digest(subject_ref)
        self.discover_work_session(route.work_session.work_session_id)
        candidate = self.candidates.find_by_hash(
            work_session_id=route.work_session.work_session_id,
            file_sha256=digest,
        )
        if candidate is None:
            raise SubmissionBlockedError(
                "no validated submission candidate matches the approved SHA-256"
            )
        if candidate.state == SubmissionState.INVALID or not bool(
            candidate.validation.get("valid")
        ):
            raise SubmissionBlockedError(
                "the approved file is not a valid submission candidate"
            )

        gate = self.router.check_human_gate(
            route,
            action=ControlledAction.SUBMIT_KAGGLE,
            subject_ref=candidate.subject_ref,
        )
        if not gate.allowed:
            raise PermissionError(gate.reason)
        self._assert_rules_acknowledged(candidate.competition_slug)
        available, detail = self.transport.available()
        if not available:
            raise SubmissionBlockedError(detail)
        self._assert_file(candidate)

        history = self.transport.submission_history(
            candidate.competition_slug,
            cwd=Path(candidate.file_path).parent,
        )
        matched = _find_marker(history, candidate.marker)
        if matched is not None:
            return self._mark_submitted(
                candidate,
                matched,
                reconciled=True,
                confirmed=True,
            )
        if candidate.state in {
            SubmissionState.SUBMITTING,
            SubmissionState.UNCERTAIN,
            SubmissionState.SUBMITTED_UNCONFIRMED,
        }:
            uncertain = self.candidates.save(
                replace(
                    candidate,
                    state=SubmissionState.UNCERTAIN,
                    last_error=(
                        "a prior submit may have reached Kaggle, but the marker is not "
                        "visible in submission history; refusing a blind duplicate"
                    ),
                )
            )
            raise SubmissionUncertainError(uncertain.last_error or "submission uncertain")

        candidate = self.candidates.save(
            replace(
                candidate,
                state=SubmissionState.SUBMITTING,
                last_error=None,
            )
        )
        self._append_event(
            candidate,
            event_type=self.STARTED_EVENT,
            lane=EventLane.CONTROL,
            payload={
                "candidate_id": candidate.candidate_id,
                "subject_ref": candidate.subject_ref,
                "competition_slug": candidate.competition_slug,
                "marker": candidate.marker,
                "decision_event_id": gate.event_id,
            },
            idempotency_key=f"kaggle-submit:{candidate.candidate_id}:started",
        )

        result = self.transport.submit(
            competition_slug=candidate.competition_slug,
            file_path=candidate.file_path,
            message=_submission_message(candidate),
            cwd=Path(candidate.file_path).parent,
        )
        if result.returncode != 0:
            failed = self.candidates.save(
                replace(
                    candidate,
                    state=SubmissionState.FAILED,
                    last_error=(result.stderr or result.stdout)[-4000:],
                )
            )
            self._append_event(
                failed,
                event_type=self.FAILED_EVENT,
                lane=EventLane.STATUS,
                payload={
                    "candidate_id": failed.candidate_id,
                    "subject_ref": failed.subject_ref,
                    "competition_slug": failed.competition_slug,
                    "error": failed.last_error,
                    "returncode": result.returncode,
                },
                idempotency_key=(
                    f"kaggle-submit:{failed.candidate_id}:failed:"
                    f"{hashlib.sha256((failed.last_error or '').encode()).hexdigest()[:16]}"
                ),
            )
            raise KaggleSubmissionError(
                "Kaggle submission failed: " + (failed.last_error or "unknown error")
            )

        submitted_at = utc_timestamp()
        candidate = self.candidates.save(
            replace(
                candidate,
                state=SubmissionState.SUBMITTED_UNCONFIRMED,
                submitted_at=submitted_at,
                last_error=None,
                kaggle_ref=_extract_url(result.stdout),
            )
        )
        matched = self._poll_for_marker(candidate)
        if matched is not None:
            return self._mark_submitted(
                candidate,
                matched,
                reconciled=False,
                confirmed=True,
            )
        completed = self.candidates.save(candidate)
        self._append_event(
            completed,
            event_type=self.COMPLETED_EVENT,
            lane=EventLane.STATUS,
            payload={
                "candidate_id": completed.candidate_id,
                "subject_ref": completed.subject_ref,
                "competition_slug": completed.competition_slug,
                "marker": completed.marker,
                "confirmed": False,
                "submitted_at": completed.submitted_at,
                "kaggle_ref": completed.kaggle_ref,
                "note": (
                    "Kaggle CLI returned success, but the submission was not yet visible "
                    "in history. The marker will be reconciled without re-submitting."
                ),
            },
            idempotency_key=f"kaggle-submit:{completed.candidate_id}:submitted",
        )
        return completed

    def refresh_history(
        self,
        *,
        work_session_id: str | None = None,
    ) -> tuple[SubmissionCandidate, ...]:
        candidates = [
            item
            for item in self.candidates.list(work_session_id=work_session_id)
            if item.state
            in {
                SubmissionState.SUBMITTED,
                SubmissionState.SUBMITTED_UNCONFIRMED,
                SubmissionState.SUBMITTING,
                SubmissionState.UNCERTAIN,
            }
        ]
        by_competition: dict[str, list[dict[str, str]]] = {}
        updated: list[SubmissionCandidate] = []
        for candidate in candidates:
            if candidate.competition_slug not in by_competition:
                try:
                    by_competition[candidate.competition_slug] = (
                        self.transport.submission_history(
                            candidate.competition_slug,
                            cwd=Path(candidate.file_path).parent,
                        )
                    )
                except Exception:
                    continue
            matched = _find_marker(
                by_competition[candidate.competition_slug],
                candidate.marker,
            )
            if matched is None:
                continue
            updated.append(
                self._mark_submitted(
                    candidate,
                    matched,
                    reconciled=True,
                    confirmed=True,
                )
            )
        return tuple(updated)

    def status_lines(self, work_session_id: str) -> tuple[str, ...]:
        self.discover_work_session(work_session_id)
        items = self.candidates.list(work_session_id=work_session_id)
        if not items:
            return ("- なし",)
        return tuple(
            (
                f"- {item.subject_ref}: {item.state.value}; "
                f"competition={item.competition_slug or '-'}; "
                f"file={item.relative_path}; "
                f"public={item.public_score if item.public_score is not None else '-'}; "
                f"private={item.private_score if item.private_score is not None else '-'}"
            )
            for item in items[-20:]
        )

    def _discover_from_result_event(self, event: Event) -> list[SubmissionCandidate]:
        raw_result = event.payload.get("result")
        result = dict(raw_result) if isinstance(raw_result, Mapping) else {}
        artifact_root_text = str(event.payload.get("artifacts_dir") or "").strip()
        if not artifact_root_text:
            return []
        artifact_root = Path(artifact_root_text).expanduser().resolve()
        if not artifact_root.is_dir():
            return []
        job = self.store.get_job(event.job_id) if event.job_id else None
        raw_candidates = _explicit_candidates(result)
        if not raw_candidates:
            raw_candidates = [
                {"path": path.relative_to(artifact_root).as_posix()}
                for path in sorted(artifact_root.rglob("*.csv"))
                if _looks_like_submission(path)
            ][:20]
        discovered: list[SubmissionCandidate] = []
        for raw in raw_candidates[:20]:
            try:
                candidate = self._build_candidate(
                    event=event,
                    job=job,
                    result=result,
                    raw=raw,
                    artifact_root=artifact_root,
                )
            except (OSError, ValueError, PermissionError):
                continue
            existing = self.candidates.get(candidate.candidate_id)
            if existing is not None and existing.state in {
                SubmissionState.SUBMITTING,
                SubmissionState.SUBMITTED,
                SubmissionState.SUBMITTED_UNCONFIRMED,
                SubmissionState.UNCERTAIN,
            }:
                candidate = replace(
                    candidate,
                    state=existing.state,
                    submitted_at=existing.submitted_at,
                    confirmed_at=existing.confirmed_at,
                    public_score=existing.public_score,
                    private_score=existing.private_score,
                    kaggle_status=existing.kaggle_status,
                    kaggle_ref=existing.kaggle_ref,
                    history_row=existing.history_row,
                    last_error=existing.last_error,
                    created_at=existing.created_at,
                )
            candidate = self.candidates.save(candidate)
            self._append_event(
                candidate,
                event_type=self.CANDIDATE_EVENT,
                lane=EventLane.DATA,
                payload={
                    "candidate": candidate.to_dict(),
                    "subject_ref": candidate.subject_ref,
                    "requires_human_submission_decision": True,
                },
                idempotency_key=f"kaggle-candidate:{candidate.candidate_id}",
            )
            discovered.append(candidate)
        return discovered

    def _build_candidate(
        self,
        *,
        event: Event,
        job: Job | None,
        result: Mapping[str, Any],
        raw: Mapping[str, Any],
        artifact_root: Path,
    ) -> SubmissionCandidate:
        raw_path = str(
            raw.get("path")
            or raw.get("file")
            or raw.get("file_path")
            or raw.get("submission_path")
            or ""
        ).strip()
        if not raw_path:
            raise ValueError("submission candidate path is missing")
        candidate_path = Path(raw_path).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = artifact_root / candidate_path
        candidate_path = candidate_path.resolve()
        try:
            relative = candidate_path.relative_to(artifact_root).as_posix()
        except ValueError as exc:
            raise PermissionError("submission candidate escapes the artifact root") from exc
        if candidate_path.is_symlink() or not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        size = candidate_path.stat().st_size
        if size <= 0 or size > self.max_file_bytes:
            raise ValueError("submission candidate size is outside the configured limit")
        digest = _sha256(candidate_path)
        competition = _first_text(
            raw.get("competition_slug"),
            raw.get("competition"),
            result.get("competition_slug"),
            result.get("competition"),
            _job_value(job, "competition_slug"),
            _job_value(job, "competition"),
            _job_value(job, "kaggle_competition"),
            self.default_competition,
        )
        message = _first_text(
            raw.get("message"),
            result.get("submission_message"),
            _job_value(job, "submission_message"),
            _job_value(job, "title"),
            _job_value(job, "hypothesis"),
            event.job_id or "ResearchAgent submission",
        )
        metadata_validation = (
            dict(raw.get("validation"))
            if isinstance(raw.get("validation"), Mapping)
            else dict(result.get("submission_validation"))
            if isinstance(result.get("submission_validation"), Mapping)
            else {}
        )
        expected_columns = raw.get("expected_columns") or result.get(
            "submission_columns"
        )
        sample_path = raw.get("sample_submission_path") or result.get(
            "sample_submission_path"
        )
        resolved_sample: Path | None = None
        if sample_path:
            candidate_sample = Path(str(sample_path)).expanduser()
            if not candidate_sample.is_absolute():
                candidate_sample = artifact_root / candidate_sample
            candidate_sample = candidate_sample.resolve()
            try:
                candidate_sample.relative_to(artifact_root)
            except ValueError:
                candidate_sample = Path()
            if candidate_sample and candidate_sample.is_file() and not candidate_sample.is_symlink():
                resolved_sample = candidate_sample
        validation = _validate_csv(
            candidate_path,
            max_bytes=self.max_file_bytes,
            expected_columns=expected_columns,
            sample_submission_path=resolved_sample,
        )
        if sample_path and resolved_sample is None:
            validation.setdefault("warnings", []).append(
                "sample_submission_path was outside the artifact root or unavailable"
            )
        declared_valid = metadata_validation.get("valid")
        if declared_valid is False:
            validation["valid"] = False
            validation.setdefault("errors", []).append(
                "experiment result explicitly marked the candidate invalid"
            )
        validation["declared_validation"] = metadata_validation
        if not competition:
            validation["valid"] = False
            validation.setdefault("errors", []).append(
                "competition_slug is missing"
            )
        raw_risks = raw.get("risks") or result.get("submission_risks") or []
        if isinstance(raw_risks, str):
            raw_risks = [raw_risks]
        risks = tuple(str(item) for item in raw_risks if str(item).strip())
        candidate_id = "SUB-" + hashlib.sha256(
            (
                event.work_session_id
                + "\0"
                + competition.lower()
                + "\0"
                + digest
            ).encode("utf-8")
        ).hexdigest()[:24]
        marker = f"[ra:{digest[:12]}]"
        return SubmissionCandidate(
            candidate_id=candidate_id,
            project_id=event.project_id,
            work_session_id=event.work_session_id,
            job_id=event.job_id,
            result_ref=str(event.payload.get("result_ref") or ""),
            source_event_id=event.event_id,
            competition_slug=competition,
            file_path=str(candidate_path),
            relative_path=relative,
            file_sha256=digest,
            file_size=size,
            message=message,
            marker=marker,
            validation=validation,
            risks=risks,
            state=(
                SubmissionState.READY
                if bool(validation.get("valid"))
                else SubmissionState.INVALID
            ),
        )

    def _poll_for_marker(
        self,
        candidate: SubmissionCandidate,
    ) -> dict[str, str] | None:
        deadline = time.monotonic() + self.history_timeout_seconds
        while True:
            try:
                history = self.transport.submission_history(
                    candidate.competition_slug,
                    cwd=Path(candidate.file_path).parent,
                )
            except Exception:
                history = []
            matched = _find_marker(history, candidate.marker)
            if matched is not None:
                return matched
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.history_poll_seconds)

    def _mark_submitted(
        self,
        candidate: SubmissionCandidate,
        history_row: Mapping[str, Any],
        *,
        reconciled: bool,
        confirmed: bool,
    ) -> SubmissionCandidate:
        row = {str(key): value for key, value in history_row.items()}
        updated = self.candidates.save(
            replace(
                candidate,
                state=SubmissionState.SUBMITTED,
                submitted_at=(candidate.submitted_at or utc_timestamp()),
                confirmed_at=utc_timestamp() if confirmed else candidate.confirmed_at,
                public_score=_history_score(row, "public"),
                private_score=_history_score(row, "private"),
                kaggle_status=_history_text(row, "status"),
                kaggle_ref=(
                    _history_text(row, "url", "ref", "id")
                    or candidate.kaggle_ref
                ),
                history_row=_json_dict(row),
                last_error=None,
            )
        )
        digest = hashlib.sha256(
            json.dumps(row, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:20]
        self._append_event(
            updated,
            event_type=self.COMPLETED_EVENT,
            lane=EventLane.STATUS,
            payload={
                "candidate_id": updated.candidate_id,
                "subject_ref": updated.subject_ref,
                "competition_slug": updated.competition_slug,
                "marker": updated.marker,
                "confirmed": confirmed,
                "reconciled": reconciled,
                "submitted_at": updated.submitted_at,
                "public_score": updated.public_score,
                "private_score": updated.private_score,
                "kaggle_status": updated.kaggle_status,
                "kaggle_ref": updated.kaggle_ref,
                "history_row": updated.history_row,
            },
            idempotency_key=(
                f"kaggle-submit:{updated.candidate_id}:completed:{digest}"
            ),
        )
        self._append_event(
            updated,
            event_type=self.HISTORY_EVENT,
            lane=EventLane.DATA,
            payload={
                "candidate_id": updated.candidate_id,
                "subject_ref": updated.subject_ref,
                "competition_slug": updated.competition_slug,
                "public_score": updated.public_score,
                "private_score": updated.private_score,
                "kaggle_status": updated.kaggle_status,
                "history_row": updated.history_row,
            },
            idempotency_key=(
                f"kaggle-submit:{updated.candidate_id}:history:{digest}"
            ),
        )
        return updated

    def _assert_rules_acknowledged(self, competition_slug: str) -> None:
        normalized = competition_slug.strip().lower()
        if normalized not in self.rules_acknowledged and "*" not in self.rules_acknowledged:
            raise SubmissionBlockedError(
                "Kaggle competition rules are not marked as acknowledged in Core "
                f"configuration for {competition_slug!r}"
            )

    @staticmethod
    def _assert_file(candidate: SubmissionCandidate) -> None:
        path = Path(candidate.file_path).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != candidate.file_sha256:
            raise PermissionError(
                "submission file changed after candidate creation; the SHA-256 "
                "approval does not apply"
            )

    def _append_event(
        self,
        candidate: SubmissionCandidate,
        *,
        event_type: str,
        lane: EventLane,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> Event:
        return self.store.append_event(
            event_type=event_type,
            lane=lane,
            project_id=candidate.project_id,
            work_session_id=candidate.work_session_id,
            job_id=candidate.job_id,
            actor="core:kaggle-submission",
            payload=payload,
            idempotency_key=idempotency_key,
        )


def build_kaggle_submission_pipeline(
    *,
    router: DiscordThreadRouter,
    project_root: str | Path,
    transport: KaggleCliTransport | None = None,
) -> KaggleSubmissionPipeline:
    root = Path(
        os.getenv("FINAL_ACTION_RUNTIME_DIR", "final_actions")
    ).expanduser()
    if not root.is_absolute():
        root = Path(project_root).expanduser().resolve() / root
    configured_rules = tuple(
        item.strip()
        for item in os.getenv("KAGGLE_RULES_ACKNOWLEDGED", "").split(",")
        if item.strip()
    )
    return KaggleSubmissionPipeline(
        router=router,
        root_dir=root / "kaggle",
        transport=transport
        or KaggleCliTransport(
            command=os.getenv("KAGGLE_COMMAND", "kaggle"),
            api_token=os.getenv("KAGGLE_API_TOKEN") or None,
            username=os.getenv("KAGGLE_USERNAME") or None,
            key=os.getenv("KAGGLE_KEY") or None,
            timeout_seconds=_int_env("KAGGLE_COMMAND_TIMEOUT_SECONDS", 180),
        ),
        rules_acknowledged=configured_rules,
        default_competition=os.getenv("KAGGLE_DEFAULT_COMPETITION", ""),
        max_file_bytes=_int_env(
            "KAGGLE_SUBMISSION_MAX_BYTES", 512 * 1024 * 1024
        ),
        history_poll_seconds=_float_env(
            "KAGGLE_SUBMISSION_HISTORY_POLL_SECONDS", 5.0
        ),
        history_timeout_seconds=_float_env(
            "KAGGLE_SUBMISSION_HISTORY_TIMEOUT_SECONDS", 90.0
        ),
    )


def _explicit_candidates(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = result.get("submission_candidates")
    if isinstance(raw, list):
        candidates = [dict(item) for item in raw if isinstance(item, Mapping)]
        if candidates:
            return candidates
    single = result.get("submission_candidate")
    if isinstance(single, Mapping):
        return [dict(single)]
    path = result.get("submission_path") or result.get("submission_file")
    if path:
        return [
            {
                "path": str(path),
                "competition_slug": result.get("competition_slug")
                or result.get("competition"),
                "message": result.get("submission_message"),
                "validation": result.get("submission_validation"),
                "risks": result.get("submission_risks"),
            }
        ]
    return []


def _looks_like_submission(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.is_file()
        and not path.is_symlink()
        and "submission" in name
        and not name.startswith("sample_submission")
    )


def _validate_csv(
    path: Path,
    *,
    max_bytes: int,
    expected_columns: Any = None,
    sample_submission_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    columns: list[str] = []
    row_count = 0
    size = path.stat().st_size
    if size <= 0:
        errors.append("CSV is empty")
    if size > max_bytes:
        errors.append(f"CSV exceeds max bytes: {max_bytes}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                columns = [str(item).strip() for item in next(reader)]
            except StopIteration:
                columns = []
            for row in reader:
                row_count += 1
                if len(row) != len(columns):
                    errors.append(
                        f"row {row_count + 1} has {len(row)} columns; expected {len(columns)}"
                    )
                    if len(errors) >= 20:
                        break
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"CSV parse failed: {type(exc).__name__}: {exc}")
    if not columns or any(not column for column in columns):
        errors.append("CSV header is missing or contains an empty column")
    if len(set(columns)) != len(columns):
        errors.append("CSV header contains duplicate columns")
    if row_count <= 0:
        errors.append("CSV contains no prediction rows")

    normalized_expected: list[str] = []
    if isinstance(expected_columns, str):
        normalized_expected = [
            item.strip() for item in expected_columns.split(",") if item.strip()
        ]
    elif isinstance(expected_columns, (list, tuple)):
        normalized_expected = [str(item).strip() for item in expected_columns]
    if normalized_expected and columns != normalized_expected:
        errors.append(
            "CSV columns do not match expected_columns: "
            + json.dumps(normalized_expected, ensure_ascii=False)
        )

    if sample_submission_path is not None:
        sample = sample_submission_path.expanduser().resolve()
        if sample.is_file() and not sample.is_symlink():
            try:
                with sample.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    sample_columns = [str(item).strip() for item in next(reader)]
                    sample_rows = sum(1 for _ in reader)
                if columns != sample_columns:
                    errors.append("CSV columns differ from sample_submission")
                if row_count != sample_rows:
                    errors.append(
                        f"CSV row count {row_count} differs from sample {sample_rows}"
                    )
            except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
                warnings.append(
                    f"sample_submission comparison failed: {type(exc).__name__}: {exc}"
                )
        else:
            warnings.append("sample_submission_path was provided but is unavailable")

    return {
        "valid": not errors,
        "file_size": size,
        "columns": columns,
        "row_count": row_count,
        "errors": errors,
        "warnings": warnings,
    }


def _submission_message(candidate: SubmissionCandidate) -> str:
    marker = candidate.marker
    base = " ".join(candidate.message.split()).strip()
    max_base = max(0, 100 - len(marker) - 1)
    base = base[:max_base].rstrip()
    return f"{base} {marker}".strip()


def _parse_history(text: str) -> list[dict[str, str]]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        reader = csv.DictReader(io.StringIO(stripped))
        rows = [
            {str(key): str(value or "") for key, value in row.items() if key is not None}
            for row in reader
        ]
        if reader.fieldnames and rows:
            return rows
    except csv.Error:
        pass
    return [{"raw": line} for line in stripped.splitlines() if line.strip()]


def _find_marker(
    rows: Sequence[Mapping[str, Any]],
    marker: str,
) -> dict[str, str] | None:
    for row in rows:
        normalized = {str(key): str(value or "") for key, value in row.items()}
        if marker in " ".join(normalized.values()):
            return normalized
    return None


def _history_text(row: Mapping[str, Any], *needles: str) -> str | None:
    lowered = {str(key).lower(): str(value or "").strip() for key, value in row.items()}
    for needle in needles:
        for key, value in lowered.items():
            if needle in key and value:
                return value
    return None


def _history_score(row: Mapping[str, Any], kind: str) -> float | None:
    value = _history_text(row, f"{kind}score", f"{kind}_score", kind)
    return _optional_float(value)


def _extract_url(text: str) -> str | None:
    for token in text.replace("\n", " ").split():
        if token.startswith("https://") or token.startswith("http://"):
            return token.strip(".,;()[]<>")
    return None


def _normalize_digest(value: str) -> str:
    digest = str(value).strip().lower().removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("submission subject_ref must be a SHA-256 digest")
    return digest


def _job_value(job: Job | None, key: str) -> Any:
    return job.spec.payload.get(key) if job is not None else None


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value)
    )
    return cleaned.strip("-")[:120] or "candidate"


def _json_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    encoded = json.dumps(dict(value), ensure_ascii=False, allow_nan=False, default=str)
    decoded = json.loads(encoded)
    return dict(decoded) if isinstance(decoded, dict) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default
