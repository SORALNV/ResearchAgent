from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from harness.runtime.codex_cli import CodexCliRuntime


class VerifiedCodexCliRuntime(CodexCliRuntime):
    """Codex runtime that reports missing executables before routing work."""

    def available(self) -> tuple[bool, str]:
        parts = shlex.split(self.command)
        if not parts:
            return False, "Codex command is empty"
        executable = parts[0]
        if Path(executable).is_absolute():
            exists = Path(executable).is_file()
        else:
            exists = shutil.which(executable) is not None
        if not exists:
            return False, f"Codex executable not found: {executable}"
        return True, self.command
