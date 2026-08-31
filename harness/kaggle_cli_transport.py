from __future__ import annotations

import os
from pathlib import Path

from harness.kaggle_submission import (
    KaggleCliTransport,
    KaggleCommandResult,
    KaggleSubmissionError,
    _parse_history,
)


class CurrentKaggleCliTransport(KaggleCliTransport):
    """Kaggle CLI 2.x transport using the documented positional competition ref."""

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
            competition_slug,
            "-v",
            "-q",
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
            competition_slug,
            "-f",
            str(Path(file_path).expanduser().resolve()),
            "-m",
            message,
        ]
        return self._run(command, Path(cwd))


def current_kaggle_transport_from_env() -> CurrentKaggleCliTransport:
    return CurrentKaggleCliTransport(
        command=os.getenv("KAGGLE_COMMAND", "kaggle"),
        api_token=os.getenv("KAGGLE_API_TOKEN") or None,
        username=os.getenv("KAGGLE_USERNAME") or None,
        key=os.getenv("KAGGLE_KEY") or None,
        timeout_seconds=_int_env("KAGGLE_COMMAND_TIMEOUT_SECONDS", 180),
    )


def _int_env(name: str, default: int) -> int:
    try:
        return max(10, int(os.getenv(name, str(default))))
    except ValueError:
        return default
