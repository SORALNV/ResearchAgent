from __future__ import annotations

import hmac
from typing import Any, Mapping

from harness.platform.application import PortablePlatformService


def register_portable_routes(app, service: PortablePlatformService) -> None:
    try:
        from fastapi import Header, HTTPException
    except ImportError as exc:
        raise RuntimeError("Install platform API dependencies") from exc

    def authorize(authorization: str | None = Header(default=None)) -> None:
        supplied = _bearer(authorization)
        expected = service.config.core_token
        if not expected or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="invalid Core token")

    @app.get("/v1/kaggle/projects/{project_id}")
    def kaggle_status(
        project_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return service.kaggle_status(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/kaggle/projects/{project_id}/rules/acknowledge")
    def acknowledge_rules(
        project_id: str,
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return service.acknowledge_kaggle_rules(
                project_id,
                rules_text=_required(body, "rules_text"),
                actor=str(body.get("actor") or "api"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/kaggle/projects/{project_id}/cv")
    def create_cv(
        project_id: str,
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return service.create_kaggle_cv_spec(
                project_id,
                strategy=_required(body, "strategy"),
                n_splits=int(body.get("n_splits") or 0),
                metric=_required(body, "metric"),
                seed=int(body.get("seed") or 42),
                shuffle=bool(body.get("shuffle", True)),
                group_column=_optional(body.get("group_column")),
                time_column=_optional(body.get("time_column")),
                stratify_column=_optional(body.get("stratify_column")),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/kaggle/projects/{project_id}/cv/{cv_spec_id}/lock")
    def lock_cv(
        project_id: str,
        cv_spec_id: str,
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return service.lock_kaggle_cv_spec(
                project_id,
                cv_spec_id,
                actor=str(body.get("actor") or "api"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/kaggle/projects/{project_id}/submission-candidates")
    def prepare_candidate(
        project_id: str,
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return service.prepare_submission_candidate(
                project_id,
                experiment_id=_required(body, "experiment_id"),
                file_path=_required(body, "file_path"),
                sample_submission_path=_required(body, "sample_submission_path"),
                message=str(body.get("message") or ""),
                id_columns=[str(item) for item in body.get("id_columns", [])],
                probability_columns=[
                    str(item) for item in body.get("probability_columns", [])
                ],
                probability_groups=[
                    [str(column) for column in group]
                    for group in body.get("probability_groups", [])
                    if isinstance(group, list)
                ],
                cv_score=_optional_float(body.get("cv_score")),
                previous_best_cv=_optional_float(body.get("previous_best_cv")),
                risks=[str(item) for item in body.get("risks", [])],
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/kaggle/submission-candidates/{candidate_id}/approve")
    def approve_candidate(
        candidate_id: str,
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return service.approve_submission_candidate(
                candidate_id,
                approval_id=_required(body, "approval_id"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/kaggle/submission-candidates/{candidate_id}/submit")
    def submit_candidate(
        candidate_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return service.submit_candidate(candidate_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def _bearer(value: str | None) -> str:
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _required(body: Mapping[str, Any], key: str) -> str:
    value = str(body.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
