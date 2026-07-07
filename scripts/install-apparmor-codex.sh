#!/usr/bin/env bash
# Ubuntu 24.04系の kernel.apparmor_restrict_unprivileged_userns=1 環境で
# codexのbwrapサンドボックスにuser namespace作成を許可する。
# 実行にはsudoが必要: sudo ./scripts/install-apparmor-codex.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILE_SRC="$SCRIPT_DIR/apparmor-codex-bwrap.profile"
PROFILE_DST="/etc/apparmor.d/codex-bwrap"

if [ "$(id -u)" -ne 0 ]; then
  echo "sudoで実行してください: sudo $0" >&2
  exit 1
fi

cp "$PROFILE_SRC" "$PROFILE_DST"
apparmor_parser -r "$PROFILE_DST"
echo "installed: $PROFILE_DST"
echo "動作確認: codex exec --sandbox workspace-write で書き込みできれば成功"
