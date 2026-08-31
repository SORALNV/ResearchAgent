from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from harness.control_plane_json import json_dict

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
EXTERNAL_IDENTITY_KEYS = (
    "thread_id",
    "forum_post_id",
    "conversation_id",
    "session_key",
)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def validate_id(value: str) -> str:
    value = str(value)
    if not ID_PATTERN.fullmatch(value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


def idempotency_path(directory: Path, key: str) -> Path:
    if not key:
        raise ValueError("idempotency key must be non-empty")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return directory / f"{digest}.json"


def normalize_external_ref(value: Mapping[str, Any] | None) -> dict[str, str]:
    return {
        str(key): str(item)
        for key, item in json_dict(value).items()
        if item is not None and str(item)
    }


def external_identity(value: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value[key]
        for key in EXTERNAL_IDENTITY_KEYS
        if value.get(key)
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        private_permissions(tmp)
        os.replace(tmp, path)
        private_permissions(path)
        _fsync_directory(path.parent)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def private_permissions(path: Path, *, directory: bool = False) -> None:
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    """Persist a successful rename on filesystems that support directory fsync."""
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def cross_process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
