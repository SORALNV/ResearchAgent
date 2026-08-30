from __future__ import annotations

from harness.approval import ProposedOperation, render_approval_request
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator


class HardenedResearchOrchestrator(ResearchOrchestrator):
    """Adds cancellation, resumable execution, and multi-operation approval handling."""

    def cancel_active(self, reason: str = "control-plane cancellation") -> int:
        cancel = getattr(self.runner, "cancel_active", None)
        return int(cancel(reason)) if callable(cancel) else 0

    def reset_cancellation(self) -> None:
        reset = getattr(self.runner, "reset_cancellation", None)
        if callable(reset):
            reset()

    def start(self) -> str:
        self.reset_cancellation()
        return super().start()

    def pause(self) -> str:
        self.cancel_active("/re pause requested")
        return super().pause()

    def stop(self) -> str:
        self.cancel_active("/re stop requested")
        return super().stop()

    def resume(self) -> str:
        session = self._require_session()
        if session.mode != Mode.PAUSED:
            return f"PAUSEDではありません。現在: {session.mode.value}"
        target = session.paused_from or Mode.PLANNING
        session.paused_from = None
        session.transition_to(target)
        session.next_action = "再開後の状態を確認する"
        self._save_journal_and_brief(session, f"{target.value}へ復帰")
        if target == Mode.RESEARCH:
            self.reset_cancellation()
            return self._run_until_blocked_or_done(session)
        return self.status()

    def approve(self, approval_id: str) -> str:
        session = self._require_session()
        self.approval_gate.approve(session, approval_id)
        pending = [
            request_id
            for request_id, request in session.approval_requests.items()
            if request.status == "pending"
        ]
        self._record(
            session,
            event_type="approval_received",
            decision=f"{approval_id} approved",
            confidence="high",
            commands_run=[f"/re approve {approval_id}"],
        )
        if pending:
            session.next_action = "残りの承認依頼を確認する: " + ", ".join(pending)
            self.store.save(session)
            return self.status()
        if session.mode == Mode.APPROVAL_BLOCKED:
            session.transition_to(Mode.RESEARCH)
        session.next_action = "承認済み条件を記録し、次の研究ラウンドへ進む"
        self.store.save(session)
        self.reset_cancellation()
        return self._run_until_blocked_or_done(session)

    def _handle_round_output(self, session, output) -> str:
        operations = self._all_operations(output)
        if not operations:
            return super()._handle_round_output(session, output)

        operations.sort(key=self._operation_priority)
        output.proposed_operation = operations[0]
        report = super()._handle_round_output(session, output)
        extra_messages: list[str] = []

        for operation in operations[1:]:
            if self.approval_gate.requires_approval(operation):
                request = self.approval_gate.create_request(session, operation)
                if session.mode != Mode.APPROVAL_BLOCKED:
                    session.transition_to(Mode.APPROVAL_BLOCKED)
                rendered = render_approval_request(request)
                self.discord.send(rendered, channel="important")
                self._record(
                    session,
                    event_type="approval_requested",
                    decision="APPROVAL_BLOCKED",
                    discord_report=rendered,
                )
                extra_messages.append(rendered)
            elif self.approval_gate.requires_important_notice(operation):
                rendered = (
                    "⚠️ 重要通知: 承認不要ポリシーで許可する操作候補\n"
                    f"操作: {operation.operation}\n"
                    f"理由: {operation.reason}\n"
                    f"影響: {operation.impact}\n"
                    f"ドライラン結果: {operation.dry_run_result}"
                )
                self.discord.send(rendered, channel="important")
                self._record(
                    session,
                    event_type="important_notice_sent",
                    decision="allowed_after_notice",
                    discord_report=rendered,
                )
                extra_messages.append(rendered)

        self.store.save(session)
        if extra_messages:
            return report + "\n\n" + "\n\n".join(extra_messages)
        return report

    def _all_operations(self, output) -> list[ProposedOperation]:
        candidates = list(getattr(output, "proposed_operations", None) or [])
        single = getattr(output, "proposed_operation", None)
        if single is not None:
            candidates.insert(0, single)
        result: list[ProposedOperation] = []
        seen: set[tuple[str, str, str, str]] = set()
        for operation in candidates:
            key = (
                operation.operation,
                operation.reason,
                operation.impact,
                operation.dry_run_result,
            )
            if key not in seen:
                seen.add(key)
                result.append(operation)
        return result

    def _operation_priority(self, operation: ProposedOperation) -> int:
        if self.approval_gate.requires_approval(operation):
            return 0
        if self.approval_gate.requires_important_notice(operation):
            return 1
        return 2
