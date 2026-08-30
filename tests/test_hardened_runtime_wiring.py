import json
import sys
from pathlib import Path

from harness.agent_runner import MockAgentRunner
from harness.config import HarnessConfig
from harness.multi_agent_runner import MultiAgentRunner
from harness.state import ResearchSession


FAKE_AGENT = r'''
import json
import os
import sys
from pathlib import Path

prompt = sys.stdin.read()

def field(name, default=""):
    prefix = name + ":"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return default

stage = field("STAGE")
role = field("ROLE")
task_id = field("TASK_ID")
log_path = os.environ.get("FAKE_AGENT_LOG")
if log_path:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{role}|{stage}|{task_id}\n")

if stage == "main_plan":
    print(json.dumps({
        "summary": "split evidence and implementation",
        "subtasks": [
            {"id": "S1", "task": "collect evidence", "deliverable": "evidence file"},
            {"id": "S2", "task": "audit risks", "deliverable": "risk result"}
        ],
        "confidence": "high"
    }))
elif role == "sub":
    if task_id == "S1":
        Path("result.txt").write_text("verified artifact", encoding="utf-8")
    print(
        "environment "
        f"discord={bool(os.environ.get('DISCORD_BOT_TOKEN'))} "
        f"unrelated={bool(os.environ.get('UNRELATED_SECRET'))}"
    )
    if os.environ.get("TEST_MULTI_OPS") == "1":
        if task_id == "S1":
            print(
                "APPROVAL_REQUIRED: operation=sudo apt install graphviz; "
                "reason=figure; impact=system; dry_run_result=not executed"
            )
            print(
                "IMPORTANT_NOTICE: operation=long_running_command:benchmark; "
                "reason=benchmark; impact=time; dry_run_result=not executed"
            )
        if task_id == "S2":
            print(
                "APPROVAL_REQUIRED: operation=external_post:publish; "
                "reason=publish; impact=external; dry_run_result=not executed"
            )
elif stage == "review":
    if os.environ.get("TEST_BAD_REVIEW") == "1":
        print("review crashed without JSON")
    else:
        print(json.dumps({
            "verdict": "accept",
            "summary": "outputs are sufficient",
            "revisions": [],
            "confidence": "high"
        }))
elif stage == "fresh":
    print("fresh counter-hypothesis")
elif stage == "main_integrate":
    if os.environ.get("TEST_BAD_INTEGRATION") == "1":
        print("not-json")
    else:
        print(json.dumps({
            "summary": "integrated result",
            "decision": "completed",
            "confidence": "high",
            "next_action": "write final report",
            "accepted_ideas": ["evidence-backed result"],
            "rejected_ideas": [],
            "promote_artifacts": [{"task_id": "S1", "path": "result.txt"}],
            "round_status": "completed",
            "progress_score": 1.0,
            "new_evidence_ids": ["P-001"],
            "unresolved_blockers": []
        }))
else:
    print("unhandled")
'''


def write_agent(tmp_path: Path) -> str:
    script = tmp_path / "fake_hardened_agent.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    return f"{sys.executable} {script}"


def make_session(tmp_path: Path) -> ResearchSession:
    session = ResearchSession.new("hardened runtime integration")
    session.research_dir = str(tmp_path / "run")
    Path(session.research_dir).mkdir()
    return session


def make_config(tmp_path: Path, command: str, **kwargs) -> HarnessConfig:
    defaults = {
        "project_root": tmp_path,
        "main_agent_command": command,
        "sub_agent_command": command,
        "review_agent_command": command,
        "fresh_agent_command": command,
        "sub_agent_count": 2,
        "agent_parallelism": 2,
        "max_review_retries": 1,
        "max_protocol_retries": 0,
        "fresh_interval": 1,
        "max_command_seconds": 10,
        "agent_sandbox_backend": "none",
        "agent_allow_unsandboxed_generic": True,
    }
    defaults.update(kwargs)
    return HarnessConfig(**defaults)


def test_mock_facade_uses_hardened_runner_and_real_safety_path(tmp_path, monkeypatch):
    command = write_agent(tmp_path)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "do-not-leak")
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-leak")
    monkeypatch.setenv("TEST_MULTI_OPS", "1")
    config = make_config(
        tmp_path,
        command,
        agent_env_allowlist=("TEST_MULTI_OPS",),
    )
    facade = MockAgentRunner(config)

    assert isinstance(facade._real_runner, MultiAgentRunner)
    output = facade.run_round(make_session(tmp_path))

    assert output.round_status == "completed"
    assert output.round_number == 1
    assert "discord=False" in output.sub_agent_output
    assert "unrelated=False" in output.sub_agent_output
    assert len(output.proposed_operations) == 3
    assert any("sudo" in item.operation for item in output.proposed_operations)
    assert any("external_post" in item.operation for item in output.proposed_operations)
    assert any("long_running_command" in item.operation for item in output.proposed_operations)

    promoted = (
        tmp_path
        / "run"
        / "artifacts"
        / "final"
        / "R001"
        / "S1"
        / "result.txt"
    )
    assert promoted.read_text(encoding="utf-8") == "verified artifact"

    checkpoint = json.loads(
        (
            tmp_path
            / "run"
            / "artifacts"
            / "checkpoints"
            / "R001.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "completed"
    assert checkpoint["final_output"]["round_status"] == "completed"


def test_invalid_review_is_fail_closed_and_integration_is_not_called(tmp_path, monkeypatch):
    command = write_agent(tmp_path)
    log_path = tmp_path / "calls.log"
    monkeypatch.setenv("TEST_BAD_REVIEW", "1")
    monkeypatch.setenv("FAKE_AGENT_LOG", str(log_path))
    config = make_config(
        tmp_path,
        command,
        agent_env_allowlist=("TEST_BAD_REVIEW", "FAKE_AGENT_LOG"),
    )

    output = MultiAgentRunner(config).run_round(make_session(tmp_path))

    assert output.round_status == "blocked"
    assert output.decision == "blocked"
    assert output.protocol_errors
    assert any(
        "agent_protocol_failure" in item.operation
        for item in output.proposed_operations
    )
    calls = log_path.read_text(encoding="utf-8")
    assert "main|main_integrate" not in calls


def test_checkpoint_resume_reuses_completed_stages(tmp_path, monkeypatch):
    command = write_agent(tmp_path)
    log_path = tmp_path / "resume.log"
    monkeypatch.setenv("TEST_BAD_INTEGRATION", "1")
    monkeypatch.setenv("FAKE_AGENT_LOG", str(log_path))
    config = make_config(
        tmp_path,
        command,
        agent_env_allowlist=("TEST_BAD_INTEGRATION", "FAKE_AGENT_LOG"),
    )
    runner = MultiAgentRunner(config)
    session = make_session(tmp_path)

    first = runner.run_round(session)
    assert first.round_status == "blocked"
    first_calls = log_path.read_text(encoding="utf-8").splitlines()

    monkeypatch.delenv("TEST_BAD_INTEGRATION")
    second = runner.run_round(session)
    assert second.round_status == "completed"
    second_calls = log_path.read_text(encoding="utf-8").splitlines()

    assert sum("main_plan" in line for line in first_calls) == 1
    assert sum("main_plan" in line for line in second_calls) == 1
    assert sum("sub_execute" in line for line in second_calls) == 2
    assert sum(line.startswith("review|review|") for line in second_calls) == 1
    assert sum("main_integrate" in line for line in second_calls) == 2


def test_generic_command_is_rejected_when_os_sandbox_is_required(tmp_path):
    command = write_agent(tmp_path)
    config = make_config(
        tmp_path,
        command,
        agent_sandbox_backend="none",
        agent_allow_unsandboxed_generic=False,
    )
    output = MultiAgentRunner(config).run_round(make_session(tmp_path))

    assert output.round_status == "blocked"
    assert any("sandbox" in error.lower() for error in output.protocol_errors)
