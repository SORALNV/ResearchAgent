from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from harness.state import utc_timestamp


_CHILD: subprocess.Popen[bytes] | None = None


def run_spec(spec_path: str | Path) -> int:
    """Execute one argv command and atomically persist its exit status.

    The wrapper itself is started in a dedicated process group by the backend.
    Its experiment child is started in another process group so timeout and
    signal handling can terminate the complete child tree before the wrapper
    writes the durable exit marker.
    """

    global _CHILD
    path = Path(spec_path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("process spec must be an object")
    command = raw.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise ValueError("process spec command must be a non-empty argv list")
    cwd = Path(str(raw.get("cwd") or path.parent)).expanduser().resolve()
    exit_path = Path(
        str(raw.get("exit_path") or (cwd / ".compute_exit.json"))
    ).expanduser().resolve()
    timeout_seconds = raw.get("timeout_seconds")
    timeout = int(timeout_seconds) if timeout_seconds is not None else None
    started_at = utc_timestamp()
    returncode = 127
    error: str | None = None
    timed_out = False

    try:
        _CHILD = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
        try:
            returncode = _CHILD.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_child()
            returncode = 124
            error = f"process exceeded timeout_seconds={timeout}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        returncode = 127
    finally:
        _CHILD = None
        payload: dict[str, Any] = {
            "returncode": int(returncode),
            "command": command,
            "cwd": str(cwd),
            "started_at": started_at,
            "finished_at": utc_timestamp(),
            "timed_out": timed_out,
            "error": error,
        }
        exit_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = exit_path.with_suffix(exit_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(exit_path)
    return int(returncode)


def _terminate_child() -> None:
    child = _CHILD
    if child is None or child.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                child.kill()
            except OSError:
                pass
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except OSError:
        try:
            child.terminate()
        except OSError:
            return
    try:
        child.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except OSError:
        try:
            child.kill()
        except OSError:
            pass


def _handle_signal(signum: int, _frame: Any) -> None:
    _terminate_child()
    raise SystemExit(128 + signum)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent compute process wrapper")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, _handle_signal)
        except (ValueError, OSError):
            pass
    raise SystemExit(run_spec(args.spec))


if __name__ == "__main__":
    main()
