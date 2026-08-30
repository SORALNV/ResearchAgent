from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_VERSION = "2026-03-10"
RULESET_PATH = Path(__file__).resolve().parents[1] / ".github" / "rulesets" / "main.json"


def main() -> int:
    token = os.getenv("GITHUB_ADMIN_TOKEN") or os.getenv("GH_TOKEN")
    repository = os.getenv("GITHUB_REPOSITORY", "SORALNV/ResearchAgent")
    if not token:
        print("GITHUB_ADMIN_TOKEN or GH_TOKEN is required", file=sys.stderr)
        return 2
    if "/" not in repository:
        print("GITHUB_REPOSITORY must be owner/repo", file=sys.stderr)
        return 2

    payload = json.loads(RULESET_PATH.read_text(encoding="utf-8"))
    rulesets = request_json(
        f"https://api.github.com/repos/{repository}/rulesets",
        token=token,
        method="GET",
    )
    existing = next(
        (
            item
            for item in rulesets
            if isinstance(item, dict) and item.get("name") == payload["name"]
        ),
        None,
    )
    if existing:
        ruleset_id = int(existing["id"])
        result = request_json(
            f"https://api.github.com/repos/{repository}/rulesets/{ruleset_id}",
            token=token,
            method="PUT",
            payload=payload,
        )
        action = "updated"
    else:
        result = request_json(
            f"https://api.github.com/repos/{repository}/rulesets",
            token=token,
            method="POST",
            payload=payload,
        )
        action = "created"

    print(
        json.dumps(
            {
                "action": action,
                "id": result.get("id"),
                "name": result.get("name"),
                "enforcement": result.get("enforcement"),
                "repository": repository,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def request_json(
    url: str,
    *,
    token: str,
    method: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "ResearchAgent-ruleset-installer/1.0",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {exc.code}: {detail}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
