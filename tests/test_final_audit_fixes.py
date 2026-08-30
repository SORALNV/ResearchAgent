import hashlib
import json
import sys
from pathlib import Path

from harness.config import HarnessConfig
from harness.doctor import run_doctor
from harness.eval import run_golden_eval
from harness.papers import PaperStore


def _check(checks, name):
    return next(item for item in checks if item.name == name)


def test_doctor_does_not_require_bubblewrap_for_bare_codex(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.sandbox.shutil.which", lambda executable: None)
    config = HarnessConfig(
        project_root=tmp_path,
        main_agent_command="codex",
        agent_sandbox_backend="auto",
        agent_allow_unsandboxed_generic=False,
    )

    sandbox = _check(run_doctor(config), "agent_os_sandbox")

    assert sandbox.ok is True
    assert "Codex sandbox" in sandbox.detail


def test_doctor_rejects_unsandboxed_generic_command(tmp_path):
    config = HarnessConfig(
        project_root=tmp_path,
        main_agent_command=f"{sys.executable} -c pass",
        agent_sandbox_backend="none",
        agent_allow_unsandboxed_generic=False,
    )

    sandbox = _check(run_doctor(config), "agent_os_sandbox")

    assert sandbox.ok is False
    assert "generic command" in sandbox.detail


def test_evaluation_verifies_promoted_artifact_hash_and_size(tmp_path):
    research_dir = tmp_path / "run"
    final_dir = research_dir / "artifacts" / "final" / "R001"
    artifact = final_dir / "S1" / "result.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("verified", encoding="utf-8")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (final_dir / "promotion_manifest.json").write_text(
        json.dumps(
            {
                "promoted": [
                    {
                        "destination": str(artifact),
                        "sha256": expected,
                        "size_bytes": artifact.stat().st_size,
                    }
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text("", encoding="utf-8")
    store = PaperStore(tmp_path / "papers.jsonl")

    clean = run_golden_eval(golden, store, research_dir=research_dir)
    artifact.write_text("tampered", encoding="utf-8")
    tampered = run_golden_eval(golden, store, research_dir=research_dir)

    assert clean.artifact_integrity_ok is True
    assert tampered.artifact_integrity_ok is False
    assert tampered.artifact_error_count >= 1
