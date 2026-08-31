from __future__ import annotations

import argparse
from typing import Any, Mapping

from harness.platform.config import PlatformConfig
from harness.platform.discord_edge import DiscordEdgeBot


class PortableDiscordEdgeBot(DiscordEdgeBot):
    """Discord Edge with computer-use and Kaggle submission approval views."""

    def _computer_approval_view(
        self,
        discord,
        core,
        route,
        *,
        original_text: str,
        actor: str,
        pending: Any,
    ):
        edge = self
        pending_items = [dict(item) for item in pending if isinstance(item, Mapping)]
        submission = next(
            (
                item
                for item in pending_items
                if item.get("type") == "tool_approval"
                and item.get("tool") == "request_kaggle_submission"
            ),
            None,
        )
        computer_checks = [
            item
            for item in pending_items
            if item.get("type") == "computer_safety_checks"
        ]

        class ApprovalView(discord.ui.View):
            def __init__(self) -> None:
                super().__init__(timeout=3600)

            @discord.ui.button(
                label=("承認して提出" if submission else "Computer-useを承認"),
                style=discord.ButtonStyle.danger,
            )
            async def approve(self, interaction, _button) -> None:
                if str(interaction.user.id) not in edge.allowed_users:
                    await interaction.response.send_message(
                        "権限がありません。", ephemeral=True
                    )
                    return
                await interaction.response.defer(thinking=True)
                try:
                    if submission:
                        arguments = submission.get("arguments")
                        if isinstance(arguments, str):
                            import json

                            arguments = json.loads(arguments)
                        if not isinstance(arguments, Mapping) or not arguments.get(
                            "candidate_id"
                        ):
                            raise ValueError("submission candidate_id is missing")
                        candidate_id = str(arguments["candidate_id"])
                        approved = await core._request(
                            "POST",
                            f"/v1/kaggle/submission-candidates/{candidate_id}/approve",
                            {"approval_id": f"discord-{interaction.id}"},
                        )
                        submitted = await core._request(
                            "POST",
                            f"/v1/kaggle/submission-candidates/{candidate_id}/submit",
                            {},
                        )
                        await interaction.followup.send(
                            f"`{candidate_id}`を承認し、Kaggle Gatewayへ提出しました。\n"
                            f"状態: `{submitted.get('status')}`\n"
                            f"SHA-256: `{submitted.get('file_sha256')}`"
                        )
                    else:
                        acknowledged: list[str] = []
                        for item in computer_checks:
                            for check in item.get("checks", []):
                                if isinstance(check, Mapping):
                                    value = check.get("id")
                                else:
                                    value = check
                                if value:
                                    acknowledged.append(str(value))
                        result = await core.message(
                            route.work_session_id,
                            text=original_text,
                            actor=actor,
                            correlation_id=f"computer-approved-{interaction.id}",
                            mode="computer",
                            computer_use_allowed=True,
                            metadata={
                                "approval_source": str(interaction.id),
                                "acknowledged_safety_check_ids": acknowledged,
                                "previous_pending_actions": pending_items,
                            },
                        )
                        await interaction.followup.send(
                            str(result.get("message") or "Computer-use処理完了")
                        )
                    self.stop()
                except Exception as exc:
                    await interaction.followup.send(
                        f"承認後の処理に失敗しました: {type(exc).__name__}: {exc}"
                    )

            @discord.ui.button(label="却下", style=discord.ButtonStyle.secondary)
            async def reject(self, interaction, _button) -> None:
                if str(interaction.user.id) not in edge.allowed_users:
                    await interaction.response.send_message(
                        "権限がありません。", ephemeral=True
                    )
                    return
                await interaction.response.send_message("操作を却下しました。")
                self.stop()

        return ApprovalView()


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent portable Discord Edge")
    parser.add_argument("--workdir", default=".")
    args = parser.parse_args()
    PortableDiscordEdgeBot(PlatformConfig.from_env(args.workdir)).run()


if __name__ == "__main__":
    main()
