from harness.config import HarnessConfig
from harness.doctor import render_doctor, run_doctor


def test_doctor_reports_provider_readiness_without_leaking_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_ORDER", "openai_responses")
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    report = render_doctor(run_doctor(HarnessConfig(project_root=tmp_path)))

    assert "runtime_provider_order" in report
    assert "openai_responses" in report
    assert "api_key=configured" in report
    assert "test-model" in report
    assert "super-secret-openai-key" not in report


def test_doctor_rejects_incomplete_computer_use_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_ORDER", "openai_computer")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "text-model")
    monkeypatch.setenv("OPENAI_COMPUTER_ENABLED", "true")
    monkeypatch.setenv("OPENAI_COMPUTER_MODEL", "computer-model")
    monkeypatch.delenv("OPENAI_COMPUTER_BRIDGE_URL", raising=False)
    monkeypatch.setenv("OPENAI_COMPUTER_ALLOWED_STAGES", "manual_browser_task")

    report = render_doctor(run_doctor(HarnessConfig(project_root=tmp_path)))

    assert "NG openai_computer" in report
    assert "bridge=missing" in report
