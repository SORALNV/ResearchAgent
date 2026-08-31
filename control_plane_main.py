from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from harness.application_runtime import ApplicationRuntime
from harness.compute import FakeComputeBackend
from harness.config import HarnessConfig
from harness.control_plane import JobSpec
from harness.control_plane_config import ControlPlaneConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.hardened_orchestrator import HardenedResearchOrchestrator
from harness.unified_discord_adapter import UnifiedDiscordBotAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResearchAgent unified research/Kaggle control plane"
    )
    parser.add_argument("--workdir", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bot")
    subparsers.add_parser("status")

    demo = subparsers.add_parser("demo")
    demo.add_argument("--domain", choices=("research", "kaggle"), default="research")
    demo.add_argument("--title", default="Control plane demo")

    args = parser.parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    harness_config = HarnessConfig.from_env(workdir)
    control_config = ControlPlaneConfig.from_env(workdir)

    if args.command == "demo":
        control_config = replace(
            control_config,
            enable_fake_backend=True,
            enable_kaggle_backend=False,
            enable_local_cpu_backend=False,
        )
        application = ApplicationRuntime.build(
            control_config,
            extra_backends=[FakeComputeBackend()],
        )
        try:
            project, session = application.work_sessions.create_session(
                domain=args.domain,
                title=args.title,
                project_root=control_config.projects_root / "demo",
            )
            job = application.work_sessions.queue_job(
                JobSpec.new(
                    project_id=project.project_id,
                    work_session_id=session.work_session_id,
                    domain=args.domain,
                    task_type="demo",
                    payload={
                        "fake_result": {
                            "summary": "portable control-plane demo completed"
                        }
                    },
                    backend_preferences=("fake",),
                    resources={"accelerator": "cpu"},
                )
            )
            import time

            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                current = application.registry.get_job(job.spec.job_id)
                if current and current.status in {
                    "completed",
                    "failed",
                    "cancelled",
                    "blocked",
                }:
                    print(
                        json.dumps(
                            {
                                "project": project.__dict__,
                                "work_session": session.__dict__,
                                "job": {
                                    "job_id": current.spec.job_id,
                                    "status": current.status,
                                    "backend": current.backend,
                                    "result": current.result,
                                    "error": current.error,
                                },
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    return
                time.sleep(0.05)
            raise SystemExit("demo job timed out")
        finally:
            application.close(cancel_running=True)

    application = ApplicationRuntime.build(control_config)
    if args.command == "status":
        try:
            print(json.dumps(application.status(), ensure_ascii=False, indent=2))
        finally:
            application.close()
        return

    token = harness_config.discord_bot_token
    if not token:
        application.close()
        raise SystemExit("DISCORD_BOT_TOKEN is required")

    UnifiedDiscordBotAdapter(
        orchestrator_factory=lambda discord: HardenedResearchOrchestrator(
            harness_config,
            discord=discord,
        ),
        application=application,
        harness_config=harness_config,
        token=token,
        channel_id=harness_config.discord_channel_id,
        important_channel_id=harness_config.discord_important_channel_id,
        log_channel_id=harness_config.discord_log_channel_id,
        worker_queue_size=harness_config.discord_worker_queue_size,
    ).run()


if __name__ == "__main__":
    main()
