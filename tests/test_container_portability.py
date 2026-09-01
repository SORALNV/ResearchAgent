from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_is_architecture_neutral_and_non_root():
    body = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG TARGETARCH" in body
    assert "chatgpt.com/codex/install.sh" in body
    assert "codex-x86_64" not in body
    assert "platform=linux/amd64" not in body
    assert "USER researchagent" in body
    assert "CODEX_HOME=/data/codex" in body
    assert "harness.container_health" in body


def test_compose_is_same_for_windows_docker_desktop_and_jetson():
    body = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "platform:" not in body
    assert "INSTALL_CODEX" in body
    assert "RA_RUNTIME_DIR" in body
    assert "RA_RESEARCH_DIR" in body
    assert "RA_CODEX_HOME_DIR" in body
    assert "cap_drop:" in body
    assert "no-new-privileges:true" in body
    assert "read_only: true" in body
    assert "/home/researchagent:rw,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=0700" in body


def test_codex_auth_state_is_excluded_from_container_build_context():
    body = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "codex-home" in body.splitlines()


def test_runtime_dependencies_and_provider_configuration_are_documented():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert 'openai = ["openai>=' in pyproject
    assert 'runtime = ["discord.py>=' in pyproject
    for key in (
        "AGENT_RUNTIME_ORDER",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_COMPUTER_ENABLED",
        "OPENAI_COMPUTER_BRIDGE_URL",
        "OPENAI_COMPUTER_REQUIRE_APPROVAL",
        "RA_CODEX_HOME_DIR",
    ):
        assert f"{key}=" in env_example


def test_portable_runner_is_the_public_real_agent_path():
    body = (ROOT / "harness" / "agent_runner.py").read_text(encoding="utf-8")
    portable = (ROOT / "harness" / "portable_multi_agent_runner.py").read_text(
        encoding="utf-8"
    )
    assert "harness.portable_multi_agent_runner" in body
    assert "ProviderAwareAgentCommandExecutor" in portable
    assert "harness.provider_runtime" in portable
