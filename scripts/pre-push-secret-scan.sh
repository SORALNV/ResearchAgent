#!/usr/bin/env bash
set -euo pipefail

ZERO_SHA="0000000000000000000000000000000000000000"
MODEL="${PRE_PUSH_SCAN_MODEL:-5.4mini}"
MAX_COMMITS_NEW_BRANCH="${PRE_PUSH_SCAN_NEW_BRANCH_COMMITS:-100}"
SCAN_PERSONAL="${PRE_PUSH_SCAN_PERSONAL:-0}"
AI_SCAN="${PRE_PUSH_AI_SCAN:-0}"
AI_SCAN_COMMAND="${PRE_PUSH_AI_SCAN_COMMAND:-}"

declare -a SECRET_PATTERNS=(
  "AKIA[0-9A-Z]{16}"
  "(?i)aws_secret_access_key\\s*[=:]\\s*['\\\"][A-Za-z0-9/+=]{40,45}['\\\"]"
  "(?i)aws_access_key_id\\s*[=:]\\s*['\\\"]?(?:A3T|AKIA)[0-9A-Z]{16}['\\\"]?"
  "(?i)(github_pat|ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{35,}"
  "xox[baprs]-[A-Za-z0-9-]{10,}"
  "sk-[A-Za-z0-9]{20,}"
  "AIza[0-9A-Za-z_-]{35}"
  "eyJ[a-zA-Z0-9_-]{10,}\\.[a-zA-Z0-9._-]{10,}\\.[a-zA-Z0-9._-]{10,}"
  "-----BEGIN (RSA|EC|DSA|OPENSSH|PRIVATE) PRIVATE KEY-----"
  "ssh-rsa AAAA[0-9A-Za-z+/=]+"
)

declare -a PERSONAL_PATTERNS=(
  "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"
  "\\+?[0-9][0-9\\-() ]{8,}[0-9]"
)

SCAN_HITS=0
TMP_DIR="$(mktemp -d)"
DIFF_TMP="$TMP_DIR/full_diff.patch"
trap 'rm -rf "$TMP_DIR"' EXIT

scan_text_block() {
  local source="$1"
  local diff="$2"

  local current_file=""
  while IFS= read -r line || [ -n "${line:-}" ]; do
    if [[ "$line" == diff\ --git* ]]; then
      current_file="${line#diff --git }"
      continue
    fi
    if [[ "$line" == +++* || "$line" == ---* ]]; then
      continue
    fi
    if [[ "$line" == +* ]]; then
      local added_line="${line:1}"
      for pattern in "${SECRET_PATTERNS[@]}"; do
        if printf '%s\n' "$added_line" | grep -Pq "$pattern"; then
          SCAN_HITS=$((SCAN_HITS + 1))
          echo "[pre-push] secret-like pattern detected (${pattern})"
          echo "  source: $source"
          echo "  file: $current_file"
          echo "  line: +$added_line"
          return
        fi
      done
      if [ "$SCAN_PERSONAL" = "1" ]; then
        for pattern in "${PERSONAL_PATTERNS[@]}"; do
          if printf '%s\n' "$added_line" | grep -Pq "$pattern"; then
            SCAN_HITS=$((SCAN_HITS + 1))
            echo "[pre-push] personal-data pattern detected (${pattern})"
            echo "  source: $source"
            echo "  file: $current_file"
            echo "  line: +$added_line"
            return
          fi
        done
      fi
    fi
  done <<< "$diff"
}

scan_commits() {
  local local_ref="$1"
  local remote_sha="$2"
  local local_sha="$3"

  if [ "$local_sha" = "$ZERO_SHA" ]; then
    return
  fi

  if [ "$remote_sha" = "$ZERO_SHA" ]; then
    while IFS= read -r commit; do
      local diff
      diff="$(git show --unified=0 --no-color --no-ext-diff --format= "$commit" --)"
      printf '%s\n' "$diff" >> "$DIFF_TMP"
      scan_text_block "$local_ref:$commit" "$diff"
    done < <(git rev-list --max-count="$MAX_COMMITS_NEW_BRANCH" "$local_sha")
    return
  fi

  local diff_range="$remote_sha..$local_sha"
  local diff
  diff="$(git diff --unified=0 "$diff_range" --)"
  if [ -n "$diff" ]; then
    printf '%s\n' "$diff" >> "$DIFF_TMP"
    scan_text_block "$local_ref:$diff_range" "$diff"
  fi
}

while IFS=' ' read -r local_ref local_sha remote_ref remote_sha; do
  if [ "$local_sha" = "$ZERO_SHA" ]; then
    continue
  fi
  scan_commits "$local_ref" "$remote_sha" "$local_sha"
done

if [ "$AI_SCAN" = "1" ]; then
  if [ -z "$AI_SCAN_COMMAND" ]; then
    echo "[pre-push] PRE_PUSH_AI_SCAN=1 but PRE_PUSH_AI_SCAN_COMMAND is empty."
    echo "[pre-push] Configure an AI command and pass it in." 
    exit 1
  fi
  export PRE_PUSH_AI_MODEL="$MODEL"
  if ! /bin/sh -lc "$AI_SCAN_COMMAND" < "$DIFF_TMP"; then
    SCAN_HITS=$((SCAN_HITS + 1))
    echo "[pre-push] AI scan reported potential leak."
    echo "  model: $MODEL"
    echo "  command: $AI_SCAN_COMMAND"
  fi
fi

if [ "$SCAN_HITS" -gt 0 ]; then
  echo "[pre-push] scan blocked. Please remove or redact secret-like values."
  exit 1
fi

exit 0
