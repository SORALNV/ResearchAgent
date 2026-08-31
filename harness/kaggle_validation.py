from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from harness.kaggle_domain import CVSpec


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class SubmissionValidationReport:
    valid: bool
    sha256: str
    size_bytes: int
    row_count: int
    columns: tuple[str, ...]
    checks: tuple[ValidationCheck, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "checks": [asdict(check) for check in self.checks],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CVValidationReport:
    valid: bool
    checks: tuple[ValidationCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": [asdict(check) for check in self.checks],
        }


def validate_cv_spec(
    spec: CVSpec,
    *,
    expected_metric: str | None = None,
    dataset_fingerprint: str | None = None,
) -> CVValidationReport:
    strategy = spec.strategy.strip().lower().replace("_", "")
    group_strategy = "group" in strategy
    time_strategy = "time" in strategy
    checks = [
        ValidationCheck("n_splits", spec.n_splits >= 2, str(spec.n_splits)),
        ValidationCheck("metric", bool(spec.metric.strip()), spec.metric or "missing"),
        ValidationCheck(
            "metric_matches_competition",
            not expected_metric
            or spec.metric.strip().lower() == expected_metric.strip().lower(),
            f"cv={spec.metric}, competition={expected_metric or 'unspecified'}",
        ),
        ValidationCheck(
            "group_column",
            not group_strategy or bool(spec.group_column),
            spec.group_column or "not configured",
        ),
        ValidationCheck(
            "time_column",
            not time_strategy or bool(spec.time_column),
            spec.time_column or "not configured",
        ),
        ValidationCheck(
            "time_shuffle_disabled",
            not time_strategy or not spec.shuffle,
            f"shuffle={spec.shuffle}",
        ),
        ValidationCheck(
            "shuffle_seed",
            not spec.shuffle or spec.seed is not None,
            f"shuffle={spec.shuffle}, seed={spec.seed}",
        ),
        ValidationCheck(
            "dataset_fingerprint",
            not dataset_fingerprint
            or not spec.split_hash
            or spec.split_hash.startswith(dataset_fingerprint[:12]),
            (
                f"split_hash={spec.split_hash or 'missing'}, "
                f"dataset={dataset_fingerprint or 'unspecified'}"
            ),
        ),
    ]
    return CVValidationReport(
        valid=all(check.ok for check in checks),
        checks=tuple(checks),
    )


def validate_submission(
    candidate_path: str | Path,
    sample_submission_path: str | Path,
    *,
    id_column: str | None = None,
    prediction_ranges: Mapping[str, tuple[float | None, float | None]] | None = None,
    max_bytes: int = 2 * 1024 * 1024 * 1024,
) -> SubmissionValidationReport:
    candidate = Path(candidate_path)
    sample = Path(sample_submission_path)
    checks: list[ValidationCheck] = []
    warnings: list[str] = []

    candidate_exists = candidate.is_file() and not candidate.is_symlink()
    sample_exists = sample.is_file() and not sample.is_symlink()
    checks.append(ValidationCheck("candidate_exists", candidate_exists, str(candidate)))
    checks.append(ValidationCheck("sample_exists", sample_exists, str(sample)))
    if not candidate_exists or not sample_exists:
        return SubmissionValidationReport(
            valid=False,
            sha256=_sha256(candidate) if candidate_exists else "",
            size_bytes=candidate.stat().st_size if candidate_exists else 0,
            row_count=0,
            columns=(),
            checks=tuple(checks),
        )

    size = candidate.stat().st_size
    checks.append(
        ValidationCheck(
            "file_size",
            0 < size <= max(1, int(max_bytes)),
            f"{size} bytes",
        )
    )

    try:
        sample_header, sample_rows = _read_csv(sample)
        candidate_header, candidate_rows = _read_csv(candidate)
    except (OSError, UnicodeError, csv.Error) as exc:
        checks.append(
            ValidationCheck("csv_parse", False, f"{type(exc).__name__}: {exc}")
        )
        return SubmissionValidationReport(
            valid=False,
            sha256=_sha256(candidate),
            size_bytes=size,
            row_count=0,
            columns=(),
            checks=tuple(checks),
        )

    checks.append(ValidationCheck("csv_parse", True, "UTF-8 CSV parsed"))
    checks.append(
        ValidationCheck(
            "columns_exact",
            candidate_header == sample_header,
            f"candidate={candidate_header}, sample={sample_header}",
        )
    )
    checks.append(
        ValidationCheck(
            "row_count",
            len(candidate_rows) == len(sample_rows),
            f"candidate={len(candidate_rows)}, sample={len(sample_rows)}",
        )
    )

    duplicate_rows = len({tuple(row) for row in candidate_rows}) != len(candidate_rows)
    if duplicate_rows:
        warnings.append("candidate contains duplicate complete rows")

    missing_count = 0
    nonfinite_count = 0
    range_errors: list[str] = []
    ranges = dict(prediction_ranges or {})
    for row_index, row in enumerate(candidate_rows, 2):
        for column_index, value in enumerate(row):
            column = (
                candidate_header[column_index]
                if column_index < len(candidate_header)
                else f"column_{column_index}"
            )
            normalized = value.strip()
            if normalized == "":
                missing_count += 1
            if normalized.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                nonfinite_count += 1
            if column in ranges and normalized:
                minimum, maximum = ranges[column]
                try:
                    numeric = float(normalized)
                except ValueError:
                    range_errors.append(f"row {row_index} {column}: non-numeric {normalized!r}")
                    continue
                if not math.isfinite(numeric):
                    range_errors.append(f"row {row_index} {column}: non-finite")
                elif minimum is not None and numeric < minimum:
                    range_errors.append(f"row {row_index} {column}: {numeric} < {minimum}")
                elif maximum is not None and numeric > maximum:
                    range_errors.append(f"row {row_index} {column}: {numeric} > {maximum}")

    checks.append(
        ValidationCheck("missing_values", missing_count == 0, f"count={missing_count}")
    )
    checks.append(
        ValidationCheck("nonfinite_values", nonfinite_count == 0, f"count={nonfinite_count}")
    )
    checks.append(
        ValidationCheck(
            "prediction_ranges",
            not range_errors,
            "; ".join(range_errors[:20]) or "within configured ranges",
        )
    )

    if id_column:
        if id_column not in sample_header or id_column not in candidate_header:
            checks.append(
                ValidationCheck("id_column", False, f"missing id column {id_column}")
            )
        else:
            sample_index = sample_header.index(id_column)
            candidate_index = candidate_header.index(id_column)
            sample_ids = [row[sample_index] for row in sample_rows]
            candidate_ids = [row[candidate_index] for row in candidate_rows]
            checks.extend(
                [
                    ValidationCheck(
                        "id_alignment",
                        candidate_ids == sample_ids,
                        "candidate IDs match sample order"
                        if candidate_ids == sample_ids
                        else "candidate IDs differ from sample",
                    ),
                    ValidationCheck(
                        "id_uniqueness",
                        len(set(candidate_ids)) == len(candidate_ids),
                        f"unique={len(set(candidate_ids))}, rows={len(candidate_ids)}",
                    ),
                ]
            )

    return SubmissionValidationReport(
        valid=all(check.ok for check in checks),
        sha256=_sha256(candidate),
        size_bytes=size,
        row_count=len(candidate_rows),
        columns=tuple(candidate_header),
        checks=tuple(checks),
        warnings=tuple(warnings),
    )


def verify_submission_hash(path: str | Path, expected_sha256: str) -> ValidationCheck:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        return ValidationCheck("approved_hash", False, "file missing or unsafe")
    actual = _sha256(candidate)
    return ValidationCheck(
        "approved_hash",
        actual == expected_sha256,
        f"expected={expected_sha256}, actual={actual}",
    )


def fingerprint_paths(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    normalized = sorted(Path(path).expanduser().resolve() for path in paths)
    for root in normalized:
        if root.is_symlink():
            raise ValueError(f"symlink cannot be fingerprinted: {root}")
        if root.is_file():
            _update_file_digest(digest, root.name, root)
            continue
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlink cannot be fingerprinted: {path}")
            if path.is_file():
                _update_file_digest(digest, path.relative_to(root).as_posix(), path)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        rows = [list(row) for row in reader]
    return list(header), rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_file_digest(digest, relative: str, path: Path) -> None:
    encoded = relative.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(path.stat().st_size.to_bytes(8, "big"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
