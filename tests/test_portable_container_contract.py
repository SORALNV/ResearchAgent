from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_container_defines_separate_core_edge_and_worker_targets():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python-base AS core" in dockerfile
    assert "FROM python-base AS edge" in dockerfile
    assert "FROM python-base AS worker" in dockerfile
    assert "@openai/codex" in dockerfile
    assert "harness.platform.asgi_portable:app" in dockerfile
    assert "harness.compute.worker_api" in dockerfile


def test_compose_keeps_credentials_separated_by_service():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "core:" in compose
    assert "edge:" in compose
    assert "worker:" in compose
    core_block = compose.split("  core:", 1)[1].split("  edge:", 1)[0]
    edge_block = compose.split("  edge:", 1)[1].split("  worker:", 1)[0]
    worker_block = compose.split("  worker:", 1)[1]
    assert "OPENAI_API_KEY" not in edge_block
    assert "KAGGLE_API_TOKEN" not in edge_block
    assert "DISCORD_BOT_TOKEN" not in core_block
    assert "DISCORD_BOT_TOKEN" not in worker_block
    assert "RESEARCH_WORKER_TOKEN" not in core_block
    assert "no-new-privileges:true" in compose


def test_portable_environment_template_is_secret_free():
    template = (ROOT / ".env.platform.example").read_text(encoding="utf-8")
    for key in [
        "RESEARCH_AGENT_CORE_TOKEN",
        "OPENAI_API_KEY",
        "KAGGLE_API_TOKEN",
        "DISCORD_BOT_TOKEN",
        "RESEARCH_WORKER_TOKEN",
    ]:
        assert f"{key}=\n" in template
    assert "sk-proj-" not in template
    assert "your-real-token" not in template


def test_sitecustomize_activates_portable_remote_and_worker_implementations():
    import harness.compute.worker_api as worker_api
    import harness.platform.application as application
    from harness.compute.remote_portable import PortableRemoteWorkerBackend
    from harness.compute.worker_api_portable import PortableWorkerJobManager

    assert application.RemoteWorkerBackend is PortableRemoteWorkerBackend
    assert worker_api.WorkerJobManager is PortableWorkerJobManager
