import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_ruleset_requires_pr_and_both_ci_jobs():
    payload = json.loads(
        (ROOT / ".github" / "rulesets" / "main.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["enforcement"] == "active"
    assert payload["conditions"]["ref_name"]["include"] == ["refs/heads/main"]
    rules = {item["type"]: item for item in payload["rules"]}
    assert "pull_request" in rules
    assert "non_fast_forward" in rules
    assert "deletion" in rules
    checks = {
        item["context"]
        for item in rules["required_status_checks"]["parameters"]["status_checks"]
    }
    assert checks == {"pytest (3.11)", "pytest (3.12)"}


def test_ruleset_installer_and_manual_workflow_exist():
    script = ROOT / "scripts" / "apply_repository_ruleset.py"
    workflow = ROOT / ".github" / "workflows" / "apply-main-ruleset.yml"
    assert script.exists()
    assert workflow.exists()
    body = script.read_text(encoding="utf-8")
    assert "GITHUB_ADMIN_TOKEN" in body
    assert "Authorization" in body
    assert "print(token" not in body
