#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git -C "$REPO_ROOT" config core.hooksPath "$REPO_ROOT/.githooks"
chmod +x "$REPO_ROOT/.githooks/pre-push"
chmod +x "$REPO_ROOT/scripts/pre-push-secret-scan.sh"
echo "Configured git hooks path: $REPO_ROOT/.githooks"
echo "Run: git config --get core.hooksPath"

