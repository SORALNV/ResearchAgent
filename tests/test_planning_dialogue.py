from harness.config import HarnessConfig
from harness.planning_dialogue import CodexRunResult, PlanningDialogueRunner
from harness.state import ResearchSession


class StubPlanningDialogueRunner(PlanningDialogueRunner):
    def __init__(self, config, runs):
        super().__init__(config)
        self.runs = list(runs)
        self.commands = []
        self.prompts = []

    def _run_codex(self, command, prompt, session, output_path):
        self.commands.append(command)
        self.prompts.append(prompt)
        if output_path is not None and output_path.exists():
            output_path.unlink()
        run = self.runs.pop(0)
        if run.stdout.startswith("FILE:") and output_path is not None:
            output_path.write_text(run.stdout.removeprefix("FILE:"), encoding="utf-8")
            return CodexRunResult(run.returncode, "", run.stderr)
        return run


def test_converse_runs_search_directive_and_uses_followup_text(tmp_path):
    session = ResearchSession.new("search dialogue")
    session.research_dir = str(tmp_path)
    runner = StubPlanningDialogueRunner(
        HarnessConfig(project_root=tmp_path, sub_agent_command="codex"),
        [
            CodexRunResult(
                0,
                '{"type":"thread.started","thread_id":"th-1"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"調べます。\\nSEARCH_QUERY: research planning agents"}}',
                "",
            ),
            CodexRunResult(
                0,
                '{"type":"item.completed","item":{"type":"agent_message","text":"検索結果を踏まえた最終返答です。\\nSEARCH_QUERY: ignored"}}',
                "",
            ),
        ],
    )
    searched = []

    result = runner.converse(session, "類似研究も見たい", lambda query: searched.append(query) or "[P-001] Paper")

    assert result.text == "検索結果を踏まえた最終返答です。"
    assert result.thread_id == "th-1"
    assert result.searches == ["research planning agents"]
    assert searched == ["research planning agents"]
    assert runner.commands[1][0:4] == ["codex", "exec", "resume", "th-1"]


def test_thread_started_jsonl_parse_gets_thread_id(tmp_path):
    session = ResearchSession.new("json parse")
    session.research_dir = str(tmp_path)
    runner = PlanningDialogueRunner(HarnessConfig(project_root=tmp_path))
    output_path = runner._last_message_path(session)
    run = CodexRunResult(0, '{"type":"thread.started","thread_id":"abc-123"}\n', "")

    thread_id, text = runner._extract_run_payload(run, output_path)

    assert thread_id == "abc-123"
    assert text == '{"type":"thread.started","thread_id":"abc-123"}'
