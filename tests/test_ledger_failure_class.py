from harness.agent_runner import RoundOutput
from harness.ledger import _failure_class


def round_output(text: str) -> RoundOutput:
    return RoundOutput(
        main_agent_summary="summary",
        subtask="task",
        sub_agent_output=text,
        review_output="機械チェック: 注意語を検出しました: 未確認, 失敗。",
        claude_consultation=None,
        fresh_agent_output=None,
        conversation_sessions=[],
        proposed_operation=None,
        accepted_ideas=[],
        rejected_ideas=[],
        decision="採用",
        confidence="high",
        next_action="next",
    )


def test_failure_class_ignores_empty_failure_section_and_review_warning():
    output = round_output(
        """結果:
- 最終レポートを作成しました。

失敗:
- なし。API呼び出しや新規実験は実行していません。

未確認事項:
- 自然動画複数本での検索品質
"""
    )

    assert _failure_class(output) is None


def test_failure_class_detects_environment_issue_from_failure_section():
    output = round_output(
        """結果:
- 実験準備をしました。

**失敗**
- `google-genai` は未導入でした。
- `gcloud services list` は sandbox 内で credentials.db を作れず失敗しました。

未確認事項:
- ADC が実API呼び出しで通るか。
"""
    )

    assert _failure_class(output) == "environment_issue"


def test_failure_class_detects_sub_agent_process_failure():
    output = round_output("Real sub-agent failed: returncode=1. stdout=なし stderr=boom")

    assert _failure_class(output) == "execution_failed"
