from __future__ import annotations

import argparse
import os
from pathlib import Path

from harness.command_parser import parse_research_command
from harness.commands import Command, CommandContext
from harness.compute_models import BackendCapabilities
from harness.config import HarnessConfig
from harness.control_plane import ControlPlaneStore
from harness.discord_adapter import FakeDiscordAdapter
from harness.discord_thread_router import ChannelDomainMap, DiscordThreadRouter
from harness.final_actions import (
    CompleteRoutedDiscordService,
    build_complete_routed_service,
)
from harness.hardened_orchestrator import HardenedResearchOrchestrator
from harness.routed_discord_adapter import DomainRoutedDiscordBotAdapter
from harness.worker_discord_adapter import WorkerDiscordBotAdapter


def build_orchestrator(
    workdir: Path,
    research_archive_dir: Path | None = None,
) -> HardenedResearchOrchestrator:
    config = HarnessConfig.from_env(workdir, research_archive_dir)
    return HardenedResearchOrchestrator(
        config=config,
        discord=FakeDiscordAdapter(),
    )


def build_routed_discord_service(
    config: HarnessConfig,
) -> CompleteRoutedDiscordService:
    control_plane_dir = Path(
        os.getenv("CONTROL_PLANE_DIR", "control_plane")
    ).expanduser()
    if not control_plane_dir.is_absolute():
        control_plane_dir = config.project_root / control_plane_dir
    router = DiscordThreadRouter(
        ControlPlaneStore(control_plane_dir),
        ChannelDomainMap.from_environment(os.environ),
    )
    service = build_complete_routed_service(config, router)
    _configure_kaggle_backend_capabilities(service)
    if not _bool_env("LOCAL_PROCESS_COMPUTE_ENABLED", False):
        # AI-generated experiments must not share the Core process namespace by
        # default because Core owns Codex/OpenAI/Kaggle/Discord credentials.
        # A local GPU is provided safely through compose.local-gpu.yaml, which
        # runs the same Worker API in a secret-minimal sidecar.
        service.compute.broker.backends.pop("local_gpu", None)
        service.compute.broker.backends.pop("local_cpu", None)
    return service


def _configure_kaggle_backend_capabilities(
    service: CompleteRoutedDiscordService,
) -> None:
    backend = service.compute.broker.backends.get("kaggle_notebook")
    if backend is None:
        return
    current = backend.capabilities
    backend.capabilities = BackendCapabilities(
        accelerators=current.accelerators,
        domains=current.domains,
        gpu_count=_nonnegative_int_env("KAGGLE_GPU_COUNT", current.gpu_count),
        gpu_memory_mb=_optional_positive_int_env(
            "KAGGLE_GPU_MEMORY_MB",
            current.gpu_memory_mb,
        ),
        cpu_cores=_optional_positive_float_env(
            "KAGGLE_CPU_CORES",
            current.cpu_cores,
        ),
        memory_mb=_optional_positive_int_env(
            "KAGGLE_MEMORY_MB",
            current.memory_mb,
        ),
        ephemeral_storage_mb=_optional_positive_int_env(
            "KAGGLE_STORAGE_MB",
            current.ephemeral_storage_mb,
        ),
        network_available=current.network_available,
        labels=current.labels,
        supports_cancel=current.supports_cancel,
        recoverable=current.recoverable,
    )


def _print_result(orchestrator: HardenedResearchOrchestrator, command: Command) -> None:
    result = orchestrator.handle(command, CommandContext(actor="local", source="cli"))
    print(result.message)
    if not result.ok:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent harness")
    parser.add_argument(
        "--workdir",
        default=".",
        help="Workspace for state, journal, and brief files.",
    )
    parser.add_argument(
        "--research-archive-dir",
        default=None,
        help="Directory where new versioned research folders are created.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    re = sub.add_parser("re")
    re_sub = re.add_subparsers(dest="re_command", required=True)
    re_sub.add_parser("new")
    re_sub.add_parser("plan")
    re_sub.add_parser("start")
    re_sub.add_parser("status")
    re_sub.add_parser("pause")
    re_sub.add_parser("resume")
    re_redirect = re_sub.add_parser("redirect")
    re_redirect.add_argument("text")
    re_idea = re_sub.add_parser("idea")
    re_idea.add_argument("text")
    re_search = re_sub.add_parser("search")
    re_search.add_argument("query")
    re_sub.add_parser("papers")
    re_paper = re_sub.add_parser("paper")
    re_paper.add_argument("paper_id")
    re_sub.add_parser("eval")
    re_sub.add_parser("cost")
    re_sub.add_parser("doctor")
    re_sub.add_parser("runs")
    re_approve = re_sub.add_parser("approve")
    re_approve.add_argument("approval_id")
    re_accept = re_sub.add_parser("accept")
    re_accept.add_argument("gate_id")
    re_revise = re_sub.add_parser("revise")
    re_revise.add_argument("gate_id")
    re_revise.add_argument("reason")
    re_reject = re_sub.add_parser("reject")
    re_reject.add_argument("approval_id")
    re_reject.add_argument("reason")
    re_sub.add_parser("stop")

    goal = sub.add_parser("goal")
    goal.add_argument("text")
    sub.add_parser("plan")
    sub.add_parser("start")
    sub.add_parser("status")
    sub.add_parser("pause")
    sub.add_parser("resume")
    redirect = sub.add_parser("redirect")
    redirect.add_argument("text")
    idea = sub.add_parser("idea")
    idea.add_argument("text")
    approve = sub.add_parser("approve")
    approve.add_argument("approval_id")
    reject = sub.add_parser("reject")
    reject.add_argument("approval_id")
    reject.add_argument("reason")
    sub.add_parser("stop")
    demo = sub.add_parser("demo")
    demo.add_argument("--goal", default="研究ハーネスのE2Eを検証する")

    bot = sub.add_parser("bot")
    bot.add_argument("--token", default=None)
    bot.add_argument("--channel-id", default=None)

    args = parser.parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    research_archive_dir = (
        Path(args.research_archive_dir).expanduser().resolve()
        if args.research_archive_dir
        else None
    )

    if args.command == "bot":
        config = HarnessConfig.from_env(workdir, research_archive_dir)
        token = args.token or config.discord_bot_token
        if not token:
            raise SystemExit("DISCORD_BOT_TOKEN or --token is required for the real bot.")
        if _domain_routing_is_configured():
            service = build_routed_discord_service(config)
            service.start()
            try:
                DomainRoutedDiscordBotAdapter(
                    token=token,
                    service=service,
                    create_threads=_bool_env("DISCORD_CREATE_THREADS", True),
                    log_channel_id=config.discord_log_channel_id,
                ).run()
            finally:
                service.stop(wait=False)
        else:
            WorkerDiscordBotAdapter(
                orchestrator_factory=lambda discord: HardenedResearchOrchestrator(
                    config,
                    discord=discord,
                ),
                token=token,
                channel_id=args.channel_id or config.discord_channel_id,
                important_channel_id=config.discord_important_channel_id,
                log_channel_id=config.discord_log_channel_id,
                worker_queue_size=config.discord_worker_queue_size,
            ).run()
        return

    orchestrator = build_orchestrator(workdir, research_archive_dir)
    if args.command == "re":
        action = args.re_command
        if action in {"redirect", "idea"}:
            command = parse_research_command(f"/re {action} {args.text}")
        elif action == "search":
            command = parse_research_command(f"/re search {args.query}")
        elif action == "paper":
            command = parse_research_command(f"/re paper {args.paper_id}")
        elif action == "approve":
            command = parse_research_command(f"/re approve {args.approval_id}")
        elif action == "accept":
            command = parse_research_command(f"/re accept {args.gate_id}")
        elif action == "revise":
            command = parse_research_command(f"/re revise {args.gate_id} {args.reason}")
        elif action == "reject":
            command = parse_research_command(
                f"/re reject {args.approval_id} {args.reason}"
            )
        else:
            command = parse_research_command(f"/re {action}")
        _print_result(orchestrator, command)
    elif args.command == "goal":
        _print_result(orchestrator, Command("goal", {"text": args.text}))
    elif args.command == "plan":
        _print_result(orchestrator, Command("plan"))
    elif args.command == "start":
        _print_result(orchestrator, Command("start"))
    elif args.command == "status":
        _print_result(orchestrator, Command("status"))
    elif args.command == "pause":
        _print_result(orchestrator, Command("pause"))
    elif args.command == "resume":
        _print_result(orchestrator, Command("resume"))
    elif args.command == "redirect":
        _print_result(orchestrator, Command("redirect", {"text": args.text}))
    elif args.command == "idea":
        _print_result(orchestrator, Command("idea", {"text": args.text}))
    elif args.command == "approve":
        _print_result(orchestrator, Command("approve", {"approval_id": args.approval_id}))
    elif args.command == "reject":
        _print_result(
            orchestrator,
            Command(
                "reject",
                {"approval_id": args.approval_id, "reason": args.reason},
            ),
        )
    elif args.command == "stop":
        _print_result(orchestrator, Command("stop"))
    elif args.command == "demo":
        fake = orchestrator.discord
        if not isinstance(fake, FakeDiscordAdapter):
            raise SystemExit("demo requires FakeDiscordAdapter")
        for content in [
            "/re new",
            "/re plan",
            args.goal,
            "類似研究を少し調べて方向性を考えたい",
            "/re search research agent harness citation",
            "/re papers",
            "/re cost",
            "/re start",
            "/re approve AP-1",
            "/re eval",
            "/re stop",
        ]:
            result = (
                fake.inject(orchestrator, content)
                if content.startswith("/")
                else fake.inject_message(orchestrator, content)
            )
            if result:
                print(result.message)


def _domain_routing_is_configured() -> bool:
    return any(
        os.getenv(name, "").strip()
        for name in (
            "DISCORD_CHANNEL_DOMAIN_MAP",
            "DISCORD_RESEARCH_CHANNEL_IDS",
            "DISCORD_KAGGLE_CHANNEL_IDS",
        )
    )


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _nonnegative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in {None, ""}:
        return max(0, int(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _optional_positive_int_env(
    name: str,
    default: int | None,
) -> int | None:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _optional_positive_float_env(
    name: str,
    default: float | None,
) -> float | None:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be positive") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


if __name__ == "__main__":
    main()
