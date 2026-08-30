from __future__ import annotations

from pathlib import Path

from harness.approval import ProposedOperation, render_approval_request
from harness.convergence import ConvergenceTracker
from harness.eval import run_golden_eval
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.process_manager import AgentCancelledError


class HardenedResearchOrchestrator(ResearchOrchestrator):
    """Cancellation, recovery, convergence, and multi-operation orchestration."""

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

    def status(self) -> str:
        """Read-only status path safe to call outside the mutation worker."""
        session = self._require_session()
        pending = [
            approval_id
            for approval_id, request in session.approval_requests.items()
            if request.status == "pending"
        ]
        snapshot = getattr(self.runner, "runtime_snapshot", None)
        runtime = snapshot(session) if callable(snapshot) else {}
        convergence = ConvergenceTracker(session, self.config).snapshot()
        return "\n".join(
            [
                f"モード: {session.mode.value}",
                f"現在の問い: {session.current_question or '未設定'}",
                f"進捗: round {session.round_id}/{self.config.max_rounds}",
                f"実行ステージ: {runtime.get('current_stage', 'idle')}",
                f"checkpoint: {runtime.get('checkpoint_status', 'none')}",
                f"実行中Agent: {runtime.get('active_agents', 0)}",
                (
                    "subtask: "
                    f"completed={runtime.get('completed_subtasks', 0)} "
                    f"failed={runtime.get('failed_subtasks', 0)} "
                    f"total={runtime.get('total_subtasks', 0)}"
                ),
                (
                    "Agent呼び出し/推定token: "
                    f"{session.cost.agent_calls}/{session.cost.estimated_tokens}"
                ),
                (
                    "収束監視: "
                    f"stagnation={convergence.get('stagnation_rounds', 0)}, "
                    f"no_evidence={convergence.get('no_evidence_rounds', 0)}"
                ),
                f"次の一手: {session.next_action or '未設定'}",
                f"承認待ち: {', '.join(pending) if pending else 'なし'}",
                f"直近エラー: {runtime.get('last_error') or 'なし'}",
            ]
        )

    def eval(self) -> str:
        session = self._require_session()
        report_path = self.config.report_path(session.session_id)
        ledger_path = self.config.research_ledger_path(session.session_id)
        report_text = (
            report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        )
        ledger_text = (
            ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else ""
        )
        if not report_text:
            report_text = "\n".join(session.accepted_ideas[-30:])
        result = run_golden_eval(
            self.config.golden_questions_path,
            self._paper_store(session),
            report_text=report_text,
            ledger_text=ledger_text,
            journal_entries=self._journal(session).read_entries(),
            research_dir=Path(session.research_dir),
        )
        self._record(
            session,
            event_type="eval_completed",
            main_agent_summary=result.summary,
            confidence="high",
            evaluation={
                "overall_score": result.overall_score,
                "keyword_hits": result.expected_keyword_hits,
                "keyword_total": result.expected_keyword_total,
                "invalid_citations": list(result.invalid_citations),
                "execution_success_rate": result.execution_success_rate,
                "artifact_integrity_ok": result.artifact_integrity_ok,
            },
        )
        return result.summary

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
        session.next_action = "承認済み条件を記録し、中断地点から研究を再開する"
        self.store.save(session)
        self.reset_cancellation()
        return self._run_until_blocked_or_done(session)

    def _run_until_blocked_or_done(self, session) -> str:
        reports: list[str] = []
        tracker = ConvergenceTracker(session, self.config)
        while session.mode == Mode.RESEARCH and session.round_id < self.config.max_rounds:
            try:
                output = self.runner.run_round(session)
            except AgentCancelledError as exc:
                session.next_action = (
                    "Agent実行は中断済み。/re pause、/re resume、"
                    "または /re stop を選択してください。"
                )
                self.store.save(session)
                self._brief_writer(session).write(session)
                message = (
                    "⏸ Agent実行を中断しました\n"
                    f"理由: {exc}\ncheckpointは保存済みです。"
                )
                self.discord.send(message, channel="important")
                self._record(
                    session,
                    event_type="agent_execution_interrupted",
                    decision="interrupted",
                    confidence="high",
                    errors=[str(exc)],
                    discord_report=message,
                )
                reports.append(message)
                break

            executed_round = int(
                getattr(output, "round_number", 0) or (session.round_id + 1)
            )
            session.round_id = max(session.round_id, executed_round)
            session.current_question = f"R{executed_round}: {session.research_goal}"
            session.accepted_ideas.extend(output.accepted_ideas)
            session.rejected_ideas.extend(output.rejected_ideas)
            session.next_action = output.next_action
            report = self._handle_round_output(session, output)
            reports.append(report)
            if session.mode == Mode.APPROVAL_BLOCKED:
                break

            convergence = tracker.evaluate(output)
            self._record(
                session,
                event_type="convergence_evaluated",
                decision=convergence.action,
                confidence=output.confidence,
                convergence=convergence.__dict__,
            )
            if convergence.should_complete:
                session.transition_to(Mode.DONE)
                session.completed_reason = convergence.reason
                session.next_action = "研究セッション完了。/re stop で最終レポートを生成"
                done_report = self._format_report(
                    session,
                    purpose="収束・成功条件の確認",
                    did=convergence.reason,
                    result="main integrationが完了条件を満たしたと判定した。",
                    verification="review、checkpoint、convergence記録を保存済み。",
                    decision="DONE",
                    confidence=output.confidence,
                )
                self.discord.send(done_report, channel="important")
                self._record(
                    session,
                    event_type="research_completed_by_convergence",
                    decision="DONE",
                    confidence=output.confidence,
                    discord_report=done_report,
                )
                reports.append(done_report)
                break

            if convergence.needs_human_review:
                gate = self._ensure_phase_gate(
                    session,
                    phase="convergence_review",
                    reason=convergence.reason,
                )
                session.transition_to(Mode.PLANNING)
                session.phase = "convergence_review"
                session.next_action = (
                    f"/re accept {gate.gate_id} で続行、または "
                    f"/re revise {gate.gate_id} <理由> で方針修正"
                )
                self.store.save(session)
                message = (
                    "⏸ 収束監視で研究を停止しました\n"
                    f"理由: {convergence.reason}\ngate: {gate.gate_id}\n"
                    f"{session.next_action}"
                )
                self.discord.send(message, channel="important")
                self._record(
                    session,
                    event_type="convergence_gate_requested",
                    decision="PLANNING",
                    confidence="high",
                    phase_gate=gate.__dict__,
                    convergence=convergence.__dict__,
                    discord_report=message,
                )
                reports.append(message)
                break

        if session.mode == Mode.RESEARCH and session.round_id >= self.config.max_rounds:
            session.transition_to(Mode.DONE)
            session.completed_reason = "MAX_ROUNDS reached"
            session.next_action = "/re stop でjournal要約と最終レポートを生成する"
            done_report = self._format_report(
                session,
                purpose="停止条件の確認",
                did="MAX_ROUNDSに達した。",
                result="設定された研究ラウンドを完了した。",
                verification="journal.jsonlとcheckpointに各ラウンドを記録済み。",
                decision="DONE",
                confidence="high",
            )
            self.discord.send(done_report, channel="important")
            self._record(
                session,
                event_type="discord_report_sent",
                main_agent_summary="MAX_ROUNDS reached",
                discord_report=done_report,
            )
            reports.append(done_report)

        self.store.save(session)
        self._brief_writer(session).write(session)
        return "\n\n".join(reports)

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
                    f"操作: {operation.operation}\n理由: {operation.reason}\n"
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
        return report + ("\n\n" + "\n\n".join(extra_messages) if extra_messages else "")

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
