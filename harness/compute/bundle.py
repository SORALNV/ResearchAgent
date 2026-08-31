from __future__ import annotations

import base64
import hashlib
import io
import os
import stat
import zipfile
from pathlib import Path
from typing import Iterable


_DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "data",
    "models",
    "checkpoints",
}


def build_source_bundle(
    root: str | Path,
    *,
    max_files: int = 5000,
    max_bytes: int = 64 * 1024 * 1024,
    exclude_names: Iterable[str] = (),
) -> dict[str, object]:
    source = Path(root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    excludes = _DEFAULT_EXCLUDES | {str(item) for item in exclude_names}
    memory = io.BytesIO()
    file_count = 0
    source_bytes = 0
    digest = hashlib.sha256()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(source)
            if any(part in excludes for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"source bundle contains symlink: {relative}")
            if not path.is_file():
                continue
            file_count += 1
            if file_count > max_files:
                raise ValueError(f"source bundle file limit exceeded: {max_files}")
            size = path.stat().st_size
            source_bytes += size
            if source_bytes > max_bytes:
                raise ValueError(f"source bundle byte limit exceeded: {max_bytes}")
            data = path.read_bytes()
            name = relative.as_posix()
            archive.writestr(name, data)
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(data).digest())
            digest.update(b"\0")
    compressed = memory.getvalue()
    if len(compressed) > max_bytes:
        raise ValueError(f"compressed source bundle exceeds limit: {max_bytes}")
    return {
        "encoding": "base64+zip",
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "content_sha256": digest.hexdigest(),
        "file_count": file_count,
        "source_bytes": source_bytes,
        "compressed_bytes": len(compressed),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def extract_source_bundle(
    bundle: dict[str, object],
    destination: str | Path,
    *,
    max_files: int = 5000,
    max_bytes: int = 64 * 1024 * 1024,
) -> dict[str, object]:
    if bundle.get("encoding") != "base64+zip":
        raise ValueError("unsupported source bundle encoding")
    raw = base64.b64decode(str(bundle.get("data") or ""), validate=True)
    expected = str(bundle.get("sha256") or "")
    actual = hashlib.sha256(raw).hexdigest()
    if not expected or not _constant_equal(actual, expected):
        raise ValueError("source bundle hash mismatch")
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    extracted = 0
    total = 0
    content_digest = hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            extracted += 1
            if extracted > max_files:
                raise ValueError(f"source bundle file limit exceeded: {max_files}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"source bundle symlink rejected: {info.filename}")
            relative = _safe_relative(info.filename)
            if relative is None:
                raise ValueError(f"unsafe source bundle path: {info.filename}")
            total += int(info.file_size)
            if total > max_bytes:
                raise ValueError(f"source bundle byte limit exceeded: {max_bytes}")
            output = (target / relative).resolve()
            try:
                output.relative_to(target)
            except ValueError as exc:
                raise ValueError(f"source bundle path escapes target: {relative}") from exc
            output.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(info)
            if len(data) != info.file_size:
                raise ValueError(f"source bundle size mismatch: {relative}")
            output.write_bytes(data)
            content_digest.update(relative.as_posix().encode("utf-8"))
            content_digest.update(b"\0")
            content_digest.update(hashlib.sha256(data).digest())
            content_digest.update(b"\0")
    expected_content = str(bundle.get("content_sha256") or "")
    actual_content = content_digest.hexdigest()
    if expected_content and not _constant_equal(actual_content, expected_content):
        raise ValueError("source bundle content hash mismatch")
    return {
        "sha256": actual,
        "content_sha256": actual_content,
        "file_count": extracted,
        "bytes": total,
        "destination": str(target),
    }


def _safe_relative(value: str) -> Path | None:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        return None
    return path


def _constant_equal(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left, right)
