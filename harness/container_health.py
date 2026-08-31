from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    checks: dict[str, object] = {}
    ok = True

    for name, default in (
        ("PROJECT_ROOT", "/data/runtime"),
        ("RESEARCH_ARCHIVE_DIR", "/data/research_runs"),
        ("CODEX_HOME", "/data/codex"),
    ):
        path = Path(os.getenv(name, default))
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".health-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            checks[name] = {"path": str(path), "writable": True}
        except OSError as exc:
            checks[name] = {
                "path": str(path),
                "writable": False,
                "error": type(exc).__name__,
            }
            ok = False

    codex = shutil.which("codex")
    checks["codex"] = {"available": bool(codex), "path": codex}
    if _bool_env("CONTAINER_HEALTH_REQUIRE_CODEX", False) and not codex:
        ok = False

    openai_selected = "openai" in os.getenv("AGENT_RUNTIME_ORDER", "")
    checks["openai"] = {
        "selected": openai_selected,
        "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "model_present": bool(os.getenv("OPENAI_MODEL")),
    }
    if openai_selected and (
        not checks["openai"]["api_key_present"]
        or not checks["openai"]["model_present"]
    ):
        # A missing OpenAI credential does not make the container unhealthy when
        # Codex is available as the primary provider. It is reported only.
        checks["openai"]["warning"] = "OpenAI fallback is not fully configured"

    print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
