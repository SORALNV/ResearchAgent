from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.state import ResearchSession, utc_timestamp


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str
    size_bytes: int
    media_type: str
    executable: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data["size_bytes"]),
            media_type=str(data.get("media_type") or "application/octet-stream"),
            executable=bool(data.get("executable", False)),
        )


def build_artifact_manifest(
    workspace: Path,
    *,
    max_files: int,
    max_bytes: int,
) -> tuple[list[ArtifactRecord], list[str]]:
    """Hash regular files in a task workspace. Symlinks and excessive output are rejected."""

    records: list[ArtifactRecord] = []
    warnings: list[str] = []
    total_bytes = 0
    workspace = workspace.resolve()
    manifest_path = workspace / "artifact_manifest.json"

    for path in sorted(workspace.rglob("*")):
        if path == manifest_path:
            continue
        if path.is_symlink():
            warnings.append(f"symlink skipped: {path.relative_to(workspace)}")
            continue
        if not path.is_file():
            continue
        if len(records) >= max_files:
            warnings.append(f"artifact file limit reached: {max_files}")
            break
        size = path.stat().st_size
        if total_bytes + size > max_bytes:
            warnings.append(f"artifact byte limit reached: {max_bytes}")
            break
        relative = path.relative_to(workspace).as_posix()
        records.append(
            ArtifactRecord(
                path=relative,
                sha256=_sha256(path),
                size_bytes=size,
                media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                executable=bool(path.stat().st_mode & 0o111),
            )
        )
        total_bytes += size

    payload = {
        "generated_at": utc_timestamp(),
        "workspace": str(workspace),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "warnings": warnings,
        "artifacts": [record.to_dict() for record in records],
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return records, warnings


def promote_selected_artifacts(
    session: ResearchSession,
    round_number: int,
    selections: object,
    sources: dict[str, tuple[Path, list[ArtifactRecord]]],
) -> tuple[list[dict[str, object]], list[str]]:
    """Promote only explicitly selected, hash-verified files into artifacts/final."""

    requested = _normalize_selections(selections)
    if not requested:
        return [], []

    final_root = Path(session.research_dir) / "artifacts" / "final" / f"R{round_number:03d}"
    promoted: list[dict[str, object]] = []
    errors: list[str] = []

    for task_id, relative in requested:
        if task_id not in sources:
            errors.append(f"unknown task_id for promotion: {task_id}")
            continue
        workspace, manifest = sources[task_id]
        workspace = workspace.resolve()
        by_path = {record.path: record for record in manifest}
        record = by_path.get(relative)
        if record is None:
            errors.append(f"artifact not found in manifest: {task_id}:{relative}")
            continue
        source = (workspace / relative).resolve()
        try:
            source.relative_to(workspace)
        except ValueError:
            errors.append(f"artifact escapes workspace: {task_id}:{relative}")
            continue
        if source.is_symlink() or not source.is_file():
            errors.append(f"artifact is not a regular file: {task_id}:{relative}")
            continue
        actual_hash = _sha256(source)
        if actual_hash != record.sha256:
            errors.append(f"artifact hash changed after review: {task_id}:{relative}")
            continue

        destination = final_root / _safe_component(task_id) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                errors.append(f"promotion destination is unsafe: {destination}")
                continue
            if _sha256(destination) != actual_hash:
                errors.append(f"promotion would overwrite different content: {destination}")
                continue
        else:
            shutil.copy2(source, destination)

        promoted.append(
            {
                "task_id": task_id,
                "source": str(source),
                "destination": str(destination),
                "path": relative,
                "sha256": actual_hash,
                "size_bytes": record.size_bytes,
                "media_type": record.media_type,
            }
        )

    manifest_path = final_root / "promotion_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": utc_timestamp(),
                "session_id": session.session_id,
                "round_number": round_number,
                "promoted": promoted,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return promoted, errors


def _normalize_selections(value: object) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if not isinstance(value, list):
        return result
    for item in value:
        task_id = ""
        path = ""
        if isinstance(item, str) and ":" in item:
            task_id, path = item.split(":", 1)
        elif isinstance(item, dict):
            task_id = str(item.get("task_id") or item.get("id") or "")
            path = str(item.get("path") or item.get("artifact") or "")
        task_id = task_id.strip()
        path = _safe_relative_path(path)
        if task_id and path:
            pair = (task_id, path)
            if pair not in result:
                result.append(pair)
    return result


def _safe_relative_path(value: str) -> str:
    path = Path(value.strip())
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        return ""
    return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:64] or "task"
