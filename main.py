from __future__ import annotations

import argparse
from pathlib import Path

from harness.command_parser import parse_research_command
from harness.commands import Command, CommandContext
from harness.config import HarnessConfig
from harness.discord_adapter import DiscordBotAdapter, FakeDiscordAdapter
from harness.orchestrator import ResearchOrchestrator


def build_orchestrator(workdir: Path, research_archive_dir: Path | None = None) -> ResearchOrchestrator:
    config = HarnessConfig.from_env(workdir, research_archive_dir)
    return ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())


def _print_result(orchestrator: ResearchOrchestrator, command: Command) -> None:
    result = orchestrator.handle(command, CommandContext(actor="local", source="cli"))
    print(result.message)
    if not result.ok:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent harness MVP")
    parser.add_argument("--workdir", default=".", help="Workspace for state, journal, and brief files.")
    parser.add_argument(
        "--research-archive-dir",
        default=None,
        help="Directory where new versioned research folders are created.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    re = sub.add_parser("re")
    re_sub = re.add_subparsers(dest="re_command", required=True)
    re_sub.add_parser("new")
    re_plan = re_sub.add_parser("plan")
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
    demo.add_argument("--goal", default="研究ハーネスMVPのE2Eを検証する")

    bot = sub.add_parser("bot")
    bot.add_argument("--token", default=None)
    bot.add_argument("--channel-id", default=None)

    args = parser.parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    research_archive_dir = Path(args.research_archive_dir).expanduser().resolve() if args.research_archive_dir else None

    if args.command == "bot":
        config = HarnessConfig.from_env(workdir, research_archive_dir)
        token = args.token or config.discord_bot_token
        if not token:
            raise SystemExit("DISCORD_BOT_TOKEN or --token is required for the real bot.")
        DiscordBotAdapter(
            orchestrator_factory=lambda discord: ResearchOrchestrator(config, discord=discord),
            token=token,
            channel_id=args.channel_id or config.discord_channel_id,
            important_channel_id=config.discord_important_channel_id,
            log_channel_id=config.discord_log_channel_id,
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
            command = parse_research_command(f"/re reject {args.approval_id} {args.reason}")
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
            Command("reject", {"approval_id": args.approval_id, "reason": args.reason}),
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
            if content.startswith("/"):
                result = fake.inject(orchestrator, content)
            else:
                result = fake.inject_message(orchestrator, content)
            if result:
                print(result.message)


if __name__ == "__main__":
    main()
