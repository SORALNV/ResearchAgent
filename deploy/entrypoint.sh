#!/bin/sh
set -eu

mkdir -p \
  "${PROJECT_ROOT:-/data/runtime}" \
  "${RESEARCH_ARCHIVE_DIR:-/data/research_runs}" \
  "${CODEX_HOME:-/data/codex}"

if [ "${CONTAINER_PRINT_PLATFORM:-false}" = "true" ]; then
  python - <<'PY'
import json
import platform
print(json.dumps({
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python": platform.python_version(),
}, ensure_ascii=False))
PY
fi

exec "$@"
