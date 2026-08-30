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
import time

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
prior = field("PRIOR_OUTPUT", "なし")
log_path = os.environ.get("FAKE_AGENT_LOG")

def log(event):
    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{role}|{stage}|{task_id}|{event}|{time.time()}\n")

log("start")
if role == "sub":
    time.sleep(0.25)

if stage == "main_plan":
    print(json.dumps({
        "summary": "two independent tasks",
        "subtasks": [
            {"id": "S1", "task": "collect evidence", "deliverable": "evidence"},
            {"id": "S2", "task": "build implementation", "deliverable": "implementation"}
        ],
        "confidence": "high"
    }))
elif role == "sub":
    if os.environ.get("FAKE_APPROVAL") == "1" and task_id == "S2":
        print("APPROVAL_REQUIRED: operation=sudo apt install graphviz; reason=figure; impact=system; dry_run_result=not executed")
    elif prior == "なし":
        print(f"initial result for {task_id}")
    else:
        print(f"revised result for {task_id}")
elif stage == "review":
    attempt = int(field("REVIEW_ATTEMPT", "0"))
    if attempt == 0:
        print(json.dumps({
            "verdict": "revise",
            "summary": "S1 needs stronger evidence",
            "revisions": [{"task_id": "S1", "instructions": "add a direct check"}],
            "confidence": "high"
        }))
    else:
        print(json.dumps({"verdict": "accept", "summary": "sufficient", "revisions": [], "confidence": "high"}))
elif stage == "fresh":
    print("fresh alternative: test the opposite hypothesis")
elif stage == "main_integrate":
    print(json.dumps({
        "summary": "integrated real-agent result",
        "decision": "accept",
        "confidence": "high",
        "next_action": "persist verified result",
        "accepted_ideas": ["parallel evidence and implementation tracks"],
        "rejected_ideas": ["trust first draft without review"]
    }))
else:
    print(f"unhandled role={role} stage={stage}")
log("end")
'''


def write_fake_agent(tmp_path: Path) -> str:
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    return f"{sys.executable} {script}"


def make_session(tmp_path: Path) -> ResearchSession:
    session = ResearchSession.new("parallel research pipeline")
    session.research_dir = str(tmp_path / "run")
    Path(session.research_dir).mkdir()
    return session


def test_real_pipeline_parallel_review_retry_fresh_and_integration(tmp_path, monkeypatch):
    command = write_fake_agent(tmp_path)
    log_path = tmp_path / "agent.log"
    monkeypatch.setenv("FAKE_AGENT_LOG", str(log_path))
    monkeypatch.setenv("SUB_AGENT_COUNT", "2")
    monkeypatch.setenv("AGENT_PARALLELISM", "2")
    monkeypatch.setenv("MAX_REVIEW_RETRIES", "1")
    config = HarnessConfig(
        project_root=tmp_path,
        main_agent_command=command,
        sub_agent_command=command,
        review_agent_command=command,
        fresh_agent_command=command,
        fresh_interval=1,
        max_command_seconds=10,
    )
    session = make_session(tmp_path)

    output = MultiAgentRunner(config).run_round(session)

    assert output.decision == "accept"
    assert output.confidence == "high"
    assert "integrated real-agent result" in output.main_agent_summary
    assert "initial result for S1" in output.sub_agent_output
    assert "revised result for S1" in output.sub_agent_output
    assert "initial result for S2" in output.sub_agent_output
    assert "fresh alternative" in (output.fresh_agent_output or "")
    assert "verdict=revise" in output.review_output
    assert "verdict=accept" in output.review_output
    assert session.cost.agent_calls == 8

    workspace = Path(session.research_dir) / "artifacts" / "agent_workspaces" / "R001"
    assert (workspace / "S1" / "attempt-01").is_dir()
    assert (workspace / "S1" / "attempt-02").is_dir()
    assert (workspace / "S2" / "attempt-01").is_dir()
    assert not (workspace / "S2" / "attempt-02").exists()

    events = [line.split("|") for line in log_path.read_text(encoding="utf-8").splitlines()]
    starts = {}
    ends = {}
    for role, stage, task_id, event, timestamp in events:
        if role == "sub" and task_id in {"S1", "S2"} and stage == "sub_retry":
            target = starts if event == "start" else ends
            target.setdefault(task_id, float(timestamp))
    assert set(starts) == {"S1", "S2"}
    assert max(starts.values()) < min(ends.values()), "initial sub-agent calls did not overlap"


def test_single_configured_command_makes_all_roles_real(tmp_path, monkeypatch):
    command = write_fake_agent(tmp_path)
    monkeypatch.setenv("SUB_AGENT_COUNT", "2")
    monkeypatch.setenv("AGENT_PARALLELISM", "2")
    monkeypatch.setenv("MAX_REVIEW_RETRIES", "1")
    config = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command=command,
        fresh_interval=1,
        max_command_seconds=10,
    )
    output = MockAgentRunner(config).run_round(make_session(tmp_path))
    stages = [item["stage"] for item in output.conversation_sessions]
    assert "main_plan" in stages
    assert stages.count("sub_execute") == 2
    assert "review" in stages
    assert "fresh" in stages
    assert "main_integrate" in stages
    assert output.decision == "accept"


def test_approval_from_parallel_sub_is_propagated(tmp_path, monkeypatch):
    command = write_fake_agent(tmp_path)
    monkeypatch.setenv("SUB_AGENT_COUNT", "2")
    monkeypatch.setenv("AGENT_PARALLELISM", "2")
    monkeypatch.setenv("MAX_REVIEW_RETRIES", "0")
    monkeypatch.setenv("FAKE_APPROVAL", "1")
    config = HarnessConfig(project_root=tmp_path, sub_agent_command=command, fresh_interval=1, max_command_seconds=10)
    output = MultiAgentRunner(config).run_round(make_session(tmp_path))
    assert output.proposed_operation is not None
    assert output.proposed_operation.operation == "sudo apt install graphviz"
    assert output.proposed_operation.dry_run_result == "not executed"
