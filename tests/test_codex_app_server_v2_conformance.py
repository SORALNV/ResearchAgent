from __future__ import annotations

from harness import codex_app_server_v2 as implementation
from harness.codex_app_server import CodexAppServerRuntime


def test_v2_text_user_input_uses_generated_snake_case_field() -> None:
    payload = implementation._text_input("continue this turn")

    assert payload == {
        "type": "text",
        "text": "continue this turn",
        "text_elements": [],
    }
    assert "textElements" not in payload


def test_facade_keeps_the_existing_app_server_runtime_contract() -> None:
    assert issubclass(CodexAppServerRuntime, implementation.CodexAppServerRuntime)
