from __future__ import annotations

import json

from harness.agent_runner import MockAgentRunner, RoundOutput
from harness.approval import ApprovalGate, render_approval_request
from harness.commands import Command, CommandContext, CommandResult
from harness.config import HarnessConfig
from harness.config import effective_model_name
from harness.cost import CostManager
from harness.doctor import render_doctor, run_doctor
from harness.discord_adapter import DiscordAdapter, FakeDiscordAdapter
from harness.eval import run_golden_eval
from harness.journal import Journal
from harness.ledger import ResearchLedger
from harness.modes import Mode
from harness.paper_search import ArxivPaperSearchProvider, FakePaperSearchProvider, PaperSearchProvider
from harness.papers import Paper, PaperStore
from harness.planning_dialogue import PlanningDialogueRunner, parse_plan_update
from harness.planning import build_planning_scout, generate_planning_query, render_planning_scout
from harness.reporting import render_report, render_run_summary, review_report
from harness.research_brief import ResearchBriefWriter, render_research_brief
from harness.state import PhaseGate, ResearchSession, SessionStore, utc_timestamp


class ResearchOrchestrator:
    def __init__(
        self,
        config: HarnessConfig,
        discord: DiscordAdapter | None = None,
        runner: MockAgentRunner | None = None,
    ) -> None:
        self.config = config
        self.discord = discord or FakeDiscordAdapter()
        self.store = SessionStore(config.state_path)
        self.approval_gate = ApprovalGate()
        self.cost_manager = CostManager(config)
        self.paper_provider = self._build_paper_provider(config.paper_provider)
        self.runner = runner or MockAgentRunner(config)
        self.planning_dialogue = PlanningDialogueRunner(config)

    def handle(
        self,
        command: Command,
        context: CommandContext | None = None,
    ) -> CommandResult:
        context = context or CommandContext()
        handlers = {
            "goal": lambda: self.goal(str(command.args["text"])),
            "new_session": self.new_session,
            "enter_plan": self.enter_plan,
            "plan_text": lambda: self.plan_text(
                str(command.args["text"]),
                list(command.args.get("attachments", [])),
            ),
            "plan": self.plan,
            "start": self.start,
            "status": self.status,
            "pause": self.pause,
            "resume": self.resume,
            "redirect": lambda: self.redirect(str(command.args["text"])),
            "idea": lambda: self.idea(str(command.args["text"])),
            "accept": lambda: self.accept_phase_gate(str(command.args["gate_id"])),
            "revise": lambda: self.revise_phase_gate(
                str(command.args["gate_id"]),
                str(command.args["reason"]),
            ),
            "approve": lambda: self.approve(str(command.args["approval_id"])),
            "reject": lambda: self.reject(
                str(command.args["approval_id"]),
                str(command.args["reason"]),
            ),
            "search": lambda: self.search_papers(str(command.args["query"])),
            "papers": self.list_papers,
            "paper": lambda: self.paper_detail(str(command.args["paper_id"])),
            "exps": self.list_experiments,
            "task": lambda: self.run_task(str(command.args["text"])),
            "eval": self.eval,
            "cost": self.cost,
            "doctor": self.doctor,
            "runs": self.runs,
            "stop": self.stop,
        }
        if command.name not in handlers:
            return CommandResult(
                ok=False,
                mode=None,
                message=f"unknown command: {command.name}",
                data={"context": context.__dict__},
            )
        try:
            message = handlers[command.name]()
            session = self.store.load()
            return CommandResult(
                ok=True,
                mode=session.mode.value if session else None,
                message=message,
                data={"context": context.__dict__},
            )
        except Exception as exc:
            session = self.store.load()
            return CommandResult(
                ok=False,
                mode=session.mode.value if session else None,
                message=str(exc),
                data={"context": context.__dict__},
            )

    def goal(self, text: str) -> str:
        existing = self.store.load()
        if existing and existing.mode != Mode.DONE:
            return (
                "アクティブなセッションがあります。勝手に上書きしません。\n"
                f"現在のsession_id: {existing.session_id}\n"
                "先に /re stop するか、新規開始してよいか確認してください。"
            )
        session = ResearchSession.new(text)
        version_label, research_dir = self.config.allocate_research_dir(session.session_id, text)
        session.version_label = version_label
        session.research_dir = str(research_dir)
        self.store.save(session)
        self._record(
            session,
            event_type="discord_command_received",
            user_instruction=text,
            commands_run=["/re new <text>"],
        )
        search_callback = lambda query: self._planning_search_for_dialogue(session, query)
        result = self.planning_dialogue.converse(
            session,
            text,
            search_callback,
        )
        if result.thread_id:
            session.plan_thread_id = result.thread_id
            self.store.save(session)
        if result.error:
            message = f"⚠️ PLAN LLMの応答に失敗しました（{result.error}）。もう一度送ってください。"
            self._record(session, event_type="error", errors=[result.error])
            return message
        dialogue = result.text
        session.accepted_ideas.append(f"PLANNING dialogue: {dialogue[:500]}")
        brief = self._brief_writer(session).write(session)
        citation_footer = self._format_planning_citation_footer(session, dialogue, result.searches)
        report = self._format_report(
            session,
            purpose="研究開始前の要件定義",
            did="新しい研究テーマを登録し、LLM判断に応じた類似研究検索と初回のオーケストレーション方針整理を実行した。",
            result=dialogue,
            verification="壁打ち内容、検索判断、取得できた類似研究情報をresearch_brief.mdとjournal.jsonlへ保存済み。",
            decision="PLANNING継続: 通常メッセージで壁打ちを続ける",
            confidence="mid",
        )
        self.discord.send(report, channel="important")
        self._record(
            session,
            event_type="planning_questions_generated",
            main_agent_summary="PLANNING開始、LLM検索判断、LLM壁打ち初回応答",
            decision="PLANNING継続",
            confidence="mid",
        )
        self._record(
            session,
            event_type="planning_dialogue_completed",
            user_instruction=text,
            main_agent_summary=dialogue,
            decision="Soraの返答待ち",
            confidence="mid",
        )
        self._record(
            session,
            event_type="research_brief_updated",
            research_brief_snapshot=brief,
            files_changed=[
                str(self.config.research_brief_path(session.session_id)),
                str(self.config.session_state_path(session.session_id)),
            ],
        )
        self._record(
            session,
            event_type="discord_report_sent",
            discord_report=report,
        )
        return report + citation_footer

    def new_session(self) -> str:
        previous = self.store.load()
        previous_summary = ""
        if previous and previous.mode != Mode.DONE:
            previous_summary = self.stop(quiet=True)
        session = ResearchSession.new("未設定")
        session.current_question = "新しい研究テーマの提案待ち"
        session.next_action = "/re plan でplanモードに切り替える"
        version_label, research_dir = self.config.allocate_research_dir(session.session_id, "untitled-research")
        session.version_label = version_label
        session.research_dir = str(research_dir)
        self.config.experiments_dir(session.session_id).mkdir(parents=True, exist_ok=True)
        (self.config.paper_dir(session.session_id) / "figures").mkdir(parents=True, exist_ok=True)
        self.store.save(session)
        self._record(
            session,
            event_type="session_created",
            commands_run=["/re new"],
            main_agent_summary="既存テーマの後処理後、新しい対話型PLANNINGセッションを作成",
            decision="PLANNING_READY",
            confidence="high",
        )
        brief = self._brief_writer(session).write(session)
        self._record(
            session,
            event_type="research_brief_updated",
            research_brief_snapshot=brief,
            files_changed=[
                str(self.config.session_state_path(session.session_id)),
                str(self.config.research_brief_path(session.session_id)),
            ],
        )
        message = self._format_report(
            session,
            purpose="新規研究テーマの準備",
            did="既存テーマがあれば終了処理を行い、新しい研究用フォルダと対話型PLANNINGセッションを作成した。",
            result="/re plan でplanモードに切り替え、その後は通常メッセージで壁打ちを継続できます。",
            verification="state.json、journal.jsonl、research_brief.mdを保存済み。",
            decision="NEUTRAL",
            confidence="high",
        )
        if previous_summary:
            message = "## 前テーマの終了処理\n" + previous_summary + "\n\n" + message
        self.discord.send(message, channel="important")
        return message

    def enter_plan(self) -> str:
        session = self.store.load()
        if not session:
            self.new_session()
            session = self._require_session()
        if session.mode == Mode.DONE:
            return "DONE状態です。新しい研究に移るには /re new を実行してください。"
        if session.mode != Mode.PLANNING:
            session.transition_to(Mode.PLANNING)
        session.phase = "plan"
        session.current_question = "研究開始前の対話型要件定義"
        session.next_action = "通常メッセージでテーマ案や制約を送ってください。固まったら /re start"
        self.store.save(session)
        brief = self._brief_writer(session).write(session)
        self._record(
            session,
            event_type="mode_switched",
            commands_run=["/re plan"],
            decision="plan",
            confidence="high",
            research_brief_snapshot=brief,
        )
        return "planモードになりました。\n普通にテーマ案や気になる点を書いてください。"

    def plan_text(self, text: str, attachments: list[str] | None = None) -> str:
        session = self._require_session()
        if session.mode != Mode.PLANNING:
            return f"/re plan はPLANNING中だけ有効です。現在: {session.mode.value}"
        if session.research_goal == "未設定":
            session.research_goal = text
            session.current_question = "研究開始前の対話型要件定義"
            session.next_action = "通常メッセージで壁打ちを継続し、固まったら /re start"
            self.store.save(session)
            self._record(
                session,
                event_type="planning_dialogue_started",
                user_instruction=text,
                commands_run=["normal Discord message in plan mode"],
                decision="dialogue_mode_enabled",
                confidence="high",
            )
        return self.discuss(text, attachments=attachments)

    def plan(self) -> str:
        session = self._require_session()
        brief = self._brief_writer(session).write(session)
        self._record(
            session,
            event_type="research_brief_updated",
            research_brief_snapshot=brief,
            main_agent_summary="/plan requested",
            decision="現在の要件定義ドラフトを表示",
            confidence="high",
        )
        return brief

    def start(self) -> str:
        session = self._require_session()
        if session.mode != Mode.PLANNING:
            return f"/re start はPLANNING中だけ有効です。現在: {session.mode.value}"
        if self._pending_phase_gate(session):
            return self._block_for_phase_gate(session)
        if session.planning_scout.get("blocking") and not self._has_accepted_phase_gate(session, "literature_review"):
            return self._block_for_novelty_gate(session)
        if not session.goal_criteria and not session.start_warning_issued:
            session.start_warning_issued = True
            session.next_action = "ゴール条件を確認する。開始するなら再度 /re start"
            self.store.save(session)
            brief = self._brief_writer(session).write(session)
            message = (
                "⚠️ ゴール条件（何をもって完了/成功とみなすか）がまだ確定していません。"
                "壁打ちで『ゴール条件は◯◯にしよう』のように合意すると記録されます。"
                "このまま開始する場合はもう一度 /re start を送ってください。"
            )
            self._record(
                session,
                event_type="start_soft_blocked",
                decision="goal_criteria_missing",
                confidence="high",
                research_brief_snapshot=brief,
                discord_report=message,
            )
            return message
        plan_path = self._write_plan_md(session)
        session.transition_to(Mode.RESEARCH)
        session.phase = "research"
        session.current_question = "MVPのE2E研究ループをMockAgentRunnerで通す"
        session.next_action = "RESEARCHラウンドを開始する"
        self.store.save(session)
        self._record(
            session,
            event_type="discord_command_received",
            commands_run=["/re start"],
            main_agent_summary="/start accepted",
            decision="RESEARCHへ移行",
            files_changed=[str(plan_path)],
        )
        return self._run_until_blocked_or_done(session)

    def status(self) -> str:
        session = self._require_session()
        pending = [
            approval_id
            for approval_id, request in session.approval_requests.items()
            if request.status == "pending"
        ]
        self._record(
            session,
            event_type="status_requested",
            main_agent_summary="現在状態を表示",
            confidence="high",
        )
        return (
            f"モード: {session.mode.value}\n"
            f"現在の問い: {session.current_question or '未設定'}\n"
            f"進捗: round {session.round_id}/{self.config.max_rounds}\n"
            f"次の一手: {session.next_action or '未設定'}\n"
            f"承認待ち: {', '.join(pending) if pending else 'なし'}"
        )

    def pause(self) -> str:
        session = self._require_session()
        if session.mode == Mode.DONE:
            return "DONE状態のためpauseできません。"
        if session.mode != Mode.PAUSED:
            session.transition_to(Mode.PAUSED)
        session.next_action = "/resume を待つ"
        self._save_journal_and_brief(session, "PAUSEDへ移行")
        return self.status()

    def resume(self) -> str:
        session = self._require_session()
        if session.mode != Mode.PAUSED:
            return f"PAUSEDではありません。現在: {session.mode.value}"
        target = session.paused_from or Mode.PLANNING
        session.paused_from = None
        session.transition_to(target)
        session.next_action = "再開後の状態を確認する"
        self._save_journal_and_brief(session, f"{target.value}へ復帰")
        return self.status()

    def redirect(self, text: str) -> str:
        session = self._require_session()
        session.redirects.append(text)
        session.current_question = "Soraのredirectを反映した要件再確認"
        session.next_action = "必要なら /re plan で確認し、/re start で再承認する"
        if session.mode in {Mode.RESEARCH, Mode.APPROVAL_BLOCKED}:
            session.transition_to(Mode.PLANNING)
        self._save_journal_and_brief(session, "redirect反映", user_instruction=text)
        return self.plan()

    def idea(self, text: str) -> str:
        session = self._require_session()
        session.accepted_ideas.append(text)
        session.next_action = "追加アイデアを次ラウンドの文脈に含める"
        self._save_journal_and_brief(session, "idea反映", user_instruction=text)
        return f"ideaを記録しました。\n{text}"

    def discuss(self, text: str, attachments: list[str] | None = None) -> str:
        session = self._require_session()
        if session.mode != Mode.PLANNING:
            return f"plan中の通常メッセージはPLANNING中だけ有効です。現在: {session.mode.value}"
        session.accepted_ideas.append(f"Sora discuss: {text}")
        search_callback = lambda query: self._planning_search_for_dialogue(session, query)
        result = self.planning_dialogue.converse(
            session,
            text,
            search_callback,
            image_paths=attachments,
        )
        if result.thread_id:
            session.plan_thread_id = result.thread_id
        if result.error:
            self.store.save(session)
            self._record(session, event_type="error", user_instruction=text, errors=[result.error])
            return f"⚠️ PLAN LLMの応答に失敗しました（{result.error}）。もう一度送ってください。"
        dialogue = result.text
        display_text, plan_updates = parse_plan_update(dialogue)
        update_summary = self._apply_plan_update(session, plan_updates)
        response = display_text + self._format_planning_citation_footer(session, display_text, result.searches)
        if update_summary:
            response = response.rstrip() + f"\n\n📌 プラン更新: {update_summary}"
        session.accepted_ideas.append(f"PLANNING dialogue: {display_text[:500]}")
        session.next_action = "通常メッセージで継続し、方針が固まったら /re start"
        self.store.save(session)
        brief = self._brief_writer(session).write(session)
        if update_summary:
            self._record(
                session,
                event_type="plan_structure_updated",
                decision=update_summary,
                confidence="high",
                research_brief_snapshot=brief,
            )
        self._record(
            session,
            event_type="planning_dialogue_completed",
            user_instruction=text,
            main_agent_summary=display_text,
            decision="Soraの返答待ち",
            confidence="mid",
            research_brief_snapshot=brief,
        )
        self.discord.send(response, channel="important")
        self._record(session, event_type="discord_report_sent", discord_report=response)
        return response

    def accept_phase_gate(self, gate_id: str) -> str:
        session = self._require_session()
        gate = session.phase_gates.get(gate_id)
        if not gate:
            return f"phase gate not found: {gate_id}"
        gate.status = "accepted"
        gate.decision = "accept"
        gate.resolved_at = utc_timestamp()
        session.phase_decisions.append(
            {
                "gate_id": gate_id,
                "decision": "accept",
                "phase": gate.phase,
                "reason": gate.reason,
            }
        )
        session.next_action = "/re start で次フェーズへ進む"
        self.store.save(session)
        self._record(
            session,
            event_type="phase_gate_accepted",
            decision=f"{gate_id} accepted",
            confidence="high",
            phase_gate=gate.__dict__,
        )
        self._brief_writer(session).write(session)
        return f"{gate_id} をacceptしました。/re start で続行できます。"

    def revise_phase_gate(self, gate_id: str, reason: str) -> str:
        session = self._require_session()
        gate = session.phase_gates.get(gate_id)
        if not gate:
            return f"phase gate not found: {gate_id}"
        gate.status = "revised"
        gate.decision = "revise"
        gate.resolved_at = utc_timestamp()
        session.phase_decisions.append(
            {
                "gate_id": gate_id,
                "decision": "revise",
                "phase": gate.phase,
                "reason": reason,
            }
        )
        session.redirects.append(f"revise {gate_id}: {reason}")
        session.next_action = "差し戻し理由を踏まえ、通常メッセージで方針や追加調査観点を伝える"
        self.store.save(session)
        self._record(
            session,
            event_type="phase_revision_requested",
            user_instruction=reason,
            decision=f"{gate_id} revised",
            confidence="high",
            phase_gate=gate.__dict__,
        )
        self._brief_writer(session).write(session)
        return f"{gate_id} をreviseしました。\n理由: {reason}"

    def approve(self, approval_id: str) -> str:
        session = self._require_session()
        self.approval_gate.approve(session, approval_id)
        if session.mode == Mode.APPROVAL_BLOCKED:
            session.transition_to(Mode.RESEARCH)
        session.next_action = "承認済み操作はMVPドライランとして続行し、次ラウンドへ進む"
        self.store.save(session)
        self._record(
            session,
            event_type="approval_received",
            decision=f"{approval_id} approved",
            confidence="high",
            commands_run=[f"/re approve {approval_id}"],
        )
        return self._run_until_blocked_or_done(session)

    def reject(self, approval_id: str, reason: str) -> str:
        session = self._require_session()
        self.approval_gate.reject(session, approval_id, reason)
        if session.mode == Mode.APPROVAL_BLOCKED:
            session.transition_to(Mode.PLANNING)
        session.next_action = "却下理由を踏まえて /re redirect または /re start を待つ"
        self.store.save(session)
        self._record(
            session,
            event_type="approval_rejected",
            decision=f"{approval_id} rejected",
            confidence="high",
            user_instruction=reason,
            commands_run=[f"/re reject {approval_id} <reason>"],
        )
        self._save_journal_and_brief(session, f"{approval_id} rejected", user_instruction=reason)
        return self.status()

    def stop(self, quiet: bool = False) -> str:
        session = self._require_session()
        if session.mode != Mode.DONE:
            session.transition_to(Mode.DONE)
        session.completed_reason = session.completed_reason or "/stop requested"
        session.next_action = "研究セッション終了"
        brief = self._brief_writer(session).write(session)
        summary = self._journal_summary()
        report = self._format_report(
            session,
            purpose="研究セッションの終了",
            did="/stopを受け取り、journal要約を作成した。",
            result=summary,
            verification="journal.jsonl と research_brief.md を保存済み。",
            decision="DONE",
            confidence="high",
        )
        papers = self._paper_store(session).read_all()
        ledger_entries = self._research_ledger(session).read_entries()
        generated_report = render_report(
            session,
            papers,
            ledger_entries,
            model_or_command=self.config.sub_agent_command or "mock",
        )
        report_path = self.config.report_path(session.session_id)
        harness_report_path = report_path.parent / "harness_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        existing_report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        report_files_changed: list[str]
        if existing_report.strip():
            final_report = existing_report
            harness_report_path.write_text(generated_report + "\n", encoding="utf-8")
            report_files_changed = [str(harness_report_path)]
            self._record(
                session,
                event_type="report_preserved",
                decision="existing report.md preserved; harness_report.md generated",
                confidence="high",
                files_changed=[str(harness_report_path)],
            )
        else:
            final_report = generated_report
            report_path.write_text(generated_report + "\n", encoding="utf-8")
            report_files_changed = [str(report_path)]
        report_review = review_report(final_report, papers, session, report_path=report_path)
        run_summary = render_run_summary(session, papers, ledger_entries)
        self.config.run_summary_path(session.session_id).write_text(run_summary + "\n", encoding="utf-8")
        report_files_changed.append(str(self.config.run_summary_path(session.session_id)))
        if not quiet:
            self.discord.send(report, channel="important")
        self.store.save(session)
        self._record(
            session,
            event_type="report_generated",
            decision=(
                "report.md preserved; harness_report.md generated"
                if existing_report.strip()
                else "report.md generated"
            ),
            confidence="high",
            files_changed=report_files_changed,
        )
        self._record(
            session,
            event_type="report_review_completed",
            decision=f"warnings={report_review['warning_count']}",
            confidence="high",
            report_review=report_review,
        )
        if int(report_review["warning_count"]) > 0:
            review_gate = self._ensure_phase_gate(
                session,
                phase="review",
                reason="; ".join(report_review["warnings"]) or "report review warning",
            )
            self.store.save(session)
            self._record(
                session,
                event_type="phase_gate_requested",
                decision="review_warnings",
                confidence="high",
                phase_gate=review_gate.__dict__,
                report_review=report_review,
            )
        self._record(
            session,
            event_type="session_ended",
            research_brief_snapshot=brief,
            main_agent_summary=summary,
            decision="DONE",
            confidence="high",
            discord_report=report,
            files_changed=[
                str(self.config.journal_path(session.session_id)),
                str(self.config.research_brief_path(session.session_id)),
                str(self.config.report_path(session.session_id)),
                str(self.config.run_summary_path(session.session_id)),
            ],
        )
        return report

    def search_papers(self, query: str) -> str:
        session = self._require_session()
        if session.mode in {Mode.DONE, Mode.APPROVAL_BLOCKED}:
            return f"文献検索できません。現在: {session.mode.value}"
        self._record(
            session,
            event_type="literature_search_started",
            user_instruction=query,
            commands_run=[f"/re search {query}"],
        )
        try:
            candidates = self.paper_provider.search(query, max_results=5)
        except Exception as exc:
            self._record(session, event_type="error", errors=[str(exc)])
            return f"文献検索に失敗しました: {exc}"
        store = self._paper_store(session)
        inserted, updated = store.upsert_many(candidates)
        all_papers = store.read_all()
        changed_papers = inserted + updated
        result_text = "\n".join(_format_paper_research_note(paper) for paper in changed_papers)
        citation_note = self._citation_note(all_papers[: min(3, len(all_papers))])
        if citation_note not in session.accepted_ideas:
            session.accepted_ideas.append(citation_note)
        brief = self._brief_writer(session).write(session)
        check = self.cost_manager.record_literature_search(session, query, result_text)
        self.store.save(session)
        self._record(
            session,
            event_type="literature_search_completed",
            user_instruction=query,
            main_agent_summary=f"{len(candidates)}件取得、{len(inserted)}件追加、{len(updated)}件統合",
            files_changed=[str(self.config.papers_path(session.session_id))],
        )
        self._record(
            session,
            event_type="paper_summaries_created",
            main_agent_summary="出典ID付き要約を作成",
            sub_agent_output="\n".join(f"{paper.paper_id}: {paper.summary}" for paper in inserted + updated),
        )
        self._record(
            session,
            event_type="research_brief_updated",
            research_brief_snapshot=brief,
            files_changed=[str(self.config.research_brief_path(session.session_id))],
        )
        if not check.ok:
            return self._block_for_cost_limit(session, check)
        if not changed_papers:
            return "文献候補は見つかりませんでした。未確認です。"
        detailed_notes = "\n\n".join(_format_paper_research_note(paper) for paper in changed_papers[:5])
        report = (
            f"文献検索: {query}\n"
            f"取得: {len(candidates)} / 追加: {len(inserted)} / 統合: {len(updated)}\n"
            f"保存先: {self.config.papers_path(session.session_id)}\n"
            f"\n見つかった研究の中身:\n{detailed_notes}\n\n"
            f"出典付き短縮メモ:\n{citation_note}"
        )
        self.discord.send(report, channel="important")
        self._record(session, event_type="discord_report_sent", discord_report=report)
        return report

    def scout_planning(self, query: str = "") -> str:
        session = self._require_session()
        if session.mode != Mode.PLANNING:
            return f"内部の類似研究調査はPLANNING中だけ有効です。現在: {session.mode.value}"
        query = query.strip() or generate_planning_query(session.research_goal)
        return self._run_planning_scout(session, query, send_discord=True)

    def _planning_search_for_dialogue(self, session: ResearchSession, query: str) -> str:
        self._run_planning_scout(session, query, send_discord=False)
        similar = (session.planning_scout or {}).get("similar_research") or []
        lines: list[str] = []
        for item in similar[:5]:
            paper_id = item.get("paper_id") or "P-???"
            title = item.get("title") or "Untitled"
            year = item.get("year") or "年不明"
            url = item.get("url") or "URL未確認"
            summary = " ".join(str(item.get("summary") or "要約未確認").split())[:200]
            abstract = " ".join(str(item.get("abstract_excerpt") or item.get("abstract") or "").split())[:300]
            lines.append(f"[{paper_id}] {title} ({year}) | {url} | {summary} | {abstract}")
        return "\n".join(lines) if lines else "検索結果はありませんでした。"

    def _format_planning_citation_footer(
        self,
        session: ResearchSession,
        display_text: str,
        searches: list[str],
    ) -> str:
        if not searches:
            return ""
        referenced: list[str] = []
        for token in display_text.replace("\n", " ").split("[P-"):
            if len(token) >= 4 and token[:3].isdigit() and token[3:4] == "]":
                paper_id = f"P-{token[:3]}"
                if paper_id not in referenced:
                    referenced.append(paper_id)
        if not referenced:
            return ""
        introduced = {
            paper_id
            for paper_id in (session.planning_answers.get("introduced_paper_ids") or "").split(",")
            if paper_id
        }
        paper_rows: dict[str, dict[str, object]] = {}
        for paper in self._paper_store(session).read_all():
            data = paper if isinstance(paper, dict) else getattr(paper, "__dict__", {})
            paper_id = str(data.get("paper_id") or "")
            if paper_id:
                paper_rows[paper_id] = data
        for item in (session.planning_scout or {}).get("similar_research") or []:
            paper_id = str(item.get("paper_id") or "")
            if paper_id and paper_id not in paper_rows:
                paper_rows[paper_id] = item
        lines: list[str] = []
        for paper_id in referenced:
            if paper_id in introduced:
                continue
            paper = paper_rows.get(paper_id)
            if not paper:
                continue
            title = paper.get("title") or "Untitled"
            year = paper.get("year") or "年不明"
            url = paper.get("url") or "URL未確認"
            lines.append(f"{paper_id}: {title} ({year}) {url}")
            introduced.add(paper_id)
            if len(lines) >= 3:
                break
        if not lines:
            return ""
        session.planning_answers["introduced_paper_ids"] = ",".join(sorted(introduced))
        return "\n\n" + "\n".join(lines)

    def _run_planning_scout(self, session: ResearchSession, query: str, send_discord: bool) -> str:
        self._record(
            session,
            event_type="planning_scout_started",
            user_instruction=query,
            commands_run=[f"internal planning paper search: {query}".strip()],
        )
        try:
            candidates = self.paper_provider.search(query, max_results=5)
        except Exception as exc:
            self._record(session, event_type="error", errors=[str(exc)])
            return f"類似研究スカウトに失敗しました: {exc}"
        store = self._paper_store(session)
        inserted, updated = store.upsert_many(candidates)
        papers = store.read_all()
        scout = build_planning_scout(session.research_goal, query, papers)
        session.planning_scout = scout
        session.current_question = "類似研究を踏まえた研究スコープ決定"
        if scout.get("blocking"):
            session.next_action = "novelty gateの判断が必要。通常メッセージで追加調査観点または方針転換を伝える"
            self._ensure_phase_gate(
                session,
                phase="literature_review",
                reason=scout.get("rationale") or "novelty gate blocking",
            )
        else:
            session.phase = "literature_review"
            session.next_action = "Soraが差分・成果物・比較対象を確認し、/re start で承認する"
        rendered_scout = render_planning_scout(scout)
        brief = self._brief_writer(session).write(session)
        result_text = "\n".join(_format_paper_line(paper) for paper in inserted + updated)
        check = self.cost_manager.record_literature_search(session, query, result_text)
        self.store.save(session)
        self._record(
            session,
            event_type="novelty_gate_evaluated",
            planning_scout=scout,
            decision=scout.get("novelty_status", "unknown"),
            confidence="mid",
        )
        if scout.get("blocking"):
            self._record(
                session,
                event_type="phase_gate_requested",
                planning_scout=scout,
                decision="novelty_gate_blocking",
                confidence="high",
            )
        self._record(
            session,
            event_type="planning_scout_completed",
            planning_scout=scout,
            main_agent_summary=f"{len(candidates)}件取得、{len(inserted)}件追加、{len(updated)}件統合",
            files_changed=[
                str(self.config.papers_path(session.session_id)),
                str(self.config.research_brief_path(session.session_id)),
            ],
        )
        self._record(
            session,
            event_type="research_brief_updated",
            research_brief_snapshot=brief,
            files_changed=[str(self.config.research_brief_path(session.session_id))],
        )
        if not check.ok:
            return self._block_for_cost_limit(session, check)
        report = (
            "類似研究スカウト完了\n"
            f"query: {query}\n"
            f"取得: {len(candidates)} / 追加: {len(inserted)} / 統合: {len(updated)}\n\n"
            f"{rendered_scout}"
        )
        if send_discord:
            self.discord.send(report, channel="important")
            self._record(session, event_type="discord_report_sent", discord_report=report)
        return report

    def list_papers(self) -> str:
        session = self._require_session()
        papers = self._paper_store(session).read_all()
        self._record(
            session,
            event_type="papers_list_requested",
            main_agent_summary=f"{len(papers)}件の文献を表示",
            confidence="high",
        )
        if not papers:
            return "papers.jsonl はまだ空です。/re search <query> を実行してください。"
        return "\n".join(_format_paper_line(paper) for paper in papers)

    def paper_detail(self, paper_id: str) -> str:
        session = self._require_session()
        paper = self._paper_store(session).get(paper_id)
        self._record(
            session,
            event_type="paper_detail_requested",
            user_instruction=paper_id,
            main_agent_summary="文献詳細を表示" if paper else "文献詳細が見つからない",
            confidence="high" if paper else "low",
        )
        if not paper:
            return f"paper not found: {paper_id}"
        authors = ", ".join(paper.authors) or "未確認"
        return (
            f"{paper.paper_id}: {paper.title}\n"
            f"authors: {authors}\n"
            f"year: {paper.year or '未確認'}\n"
            f"venue: {paper.venue or '未確認'}\n"
            f"source: {paper.source}\n"
            f"url: {paper.url or '未確認'}\n"
            f"doi: {paper.doi or '未確認'}\n"
            f"arxiv_id: {paper.arxiv_id or '未確認'}\n"
            f"summary: {paper.summary}\n"
            f"abstract: {paper.abstract or '未確認'}\n"
            f"research_use: {_paper_research_use(paper)}\n"
            f"unverified: abstractとメタデータベース。本文の手法詳細、実験設定、限界は未確認。\n"
            f"used_in_rounds: {paper.used_in_rounds}"
        )

    def cost(self) -> str:
        session = self._require_session()
        check = self.cost_manager.check(session)
        status = "ok" if check.ok else f"blocked: {check.reason}"
        self._record(
            session,
            event_type="cost_status_requested",
            main_agent_summary=status,
            confidence="high",
        )
        return (
            f"session: {session.session_id}\n"
            f"api_calls: {session.cost.api_calls}"
            f"{f' / {self.config.max_api_calls}' if self.config.max_api_calls else ''}\n"
            f"estimated_tokens: {session.cost.estimated_tokens}"
            f"{f' / {self.config.max_total_tokens}' if self.config.max_total_tokens else ''}\n"
            f"literature_searches: {session.cost.literature_searches}\n"
            f"status: {status}"
        )

    def eval(self) -> str:
        session = self._require_session()
        result = run_golden_eval(self.config.golden_questions_path, self._paper_store(session))
        self._record(
            session,
            event_type="eval_completed",
            main_agent_summary=result.summary,
            confidence="mid",
        )
        return result.summary

    def doctor(self) -> str:
        report = render_doctor(run_doctor(self.config))
        session = self.store.load()
        if session:
            self._record(
                session,
                event_type="doctor_requested",
                main_agent_summary=report,
                confidence="high",
            )
        return report

    def runs(self) -> str:
        rows = self._list_runs()
        if not rows:
            return f"研究フォルダはまだありません: {self.config.research_archive_path}"
        lines = ["Research runs:"]
        for row in rows:
            lines.append(
                f"- {row['version']} {row['session_id']} mode={row['mode']} "
                f"round={row['round_id']} goal={row['goal']} path={row['path']}"
            )
        session = self.store.load()
        if session:
            self._record(
                session,
                event_type="runs_list_requested",
                main_agent_summary=f"{len(rows)}件の研究フォルダを表示",
                confidence="high",
            )
        return "\n".join(lines)

    def list_experiments(self) -> str:
        from harness.experiments import ExperimentStore

        session = self._require_session()
        entries = ExperimentStore(self.config.experiments_path(session.session_id)).read_all()
        self._record(
            session,
            event_type="experiments_list_requested",
            main_agent_summary=f"{len(entries)}件の実験を表示",
            confidence="high",
        )
        if not entries:
            return "実験はまだ登録されていません。"
        lines = []
        for entry in entries:
            hypothesis = entry.hypothesis[:60]
            metrics = ", ".join(f"{key}={value}" for key, value in entry.metrics.items())
            if not metrics:
                metrics = "{}"
            lines.append(
                f"{entry.exp_id} [{entry.status}] {entry.name} | {hypothesis} | {metrics} | {entry.dir}"
            )
        return "\n".join(lines)

    def run_task(self, text: str) -> str:
        from harness.agent_runner import SubAgentCommandRunner, parse_approval_required

        session = self.store.load()
        if not session:
            return "Run /re new first"
        if not self.config.sub_agent_command:
            return "実サブエージェント未設定のため /re task は使えません。"
        if session.mode == Mode.PLANNING:
            return "PLANNING中は /re task を使えません。/re start で研究を開始するか、壁打ちで計画を詰めてください。"
        if session.mode not in {Mode.RESEARCH, Mode.DONE}:
            return "現在のモードでは /re task を使えません。RESEARCHまたはDONEで実行してください。"

        round_number = session.round_id + 1
        runner = SubAgentCommandRunner(self.config)
        sub_output = runner.run(session, round_number=round_number, task=text)
        output = RoundOutput(
            main_agent_summary=f"/re task: {text[:80]}",
            subtask=text,
            sub_agent_output=sub_output,
            review_output=MockAgentRunner(self.config)._review_sub_agent_output(sub_output),
            claude_consultation=None,
            fresh_agent_output=None,
            conversation_sessions=[],
            proposed_operation=parse_approval_required(sub_output),
            accepted_ideas=[f"task: {text[:120]}"],
            rejected_ideas=[],
            decision="採用: /re taskの単発実行",
            confidence="mid" if "未確認" in sub_output else "high",
            next_action="結果を確認し、必要なら次の /re task または /re paper",
        )
        previous_mode = session.mode
        session.round_id = round_number
        session.current_question = f"R{session.round_id}: /re task"
        session.accepted_ideas.extend(output.accepted_ideas)
        session.rejected_ideas.extend(output.rejected_ideas)
        session.next_action = output.next_action
        report = self._handle_round_output(session, output)
        if (
            previous_mode == Mode.DONE
            and session.mode not in {Mode.APPROVAL_BLOCKED, Mode.DONE}
        ):
            session.transition_to(Mode.DONE)
        self.store.save(session)
        self._brief_writer(session).write(session)
        preview = sub_output[:1200]
        return (
            f"{preview}\n\n"
            f"保存先: {self.config.session_dir(session.session_id)}\n"
            f"研究台帳: {self.config.research_ledger_path(session.session_id)}\n"
            f"詳細レポートはjournal.jsonlとresearch_ledger.jsonlに記録済み。"
        )

    def _run_until_blocked_or_done(self, session: ResearchSession) -> str:
        reports: list[str] = []
        while session.mode == Mode.RESEARCH and session.round_id < self.config.max_rounds:
            output = self.runner.run_round(session)
            session.round_id += 1
            session.current_question = f"R{session.round_id}: {session.research_goal}"
            session.accepted_ideas.extend(output.accepted_ideas)
            session.rejected_ideas.extend(output.rejected_ideas)
            session.next_action = output.next_action
            report = self._handle_round_output(session, output)
            reports.append(report)
            if session.mode == Mode.APPROVAL_BLOCKED:
                break
        if session.mode == Mode.RESEARCH and session.round_id >= self.config.max_rounds:
            session.transition_to(Mode.DONE)
            session.completed_reason = "MAX_ROUNDS reached"
            session.next_action = "/re stop でjournal要約を表示する"
            done_report = self._format_report(
                session,
                purpose="停止条件の確認",
                did="MAX_ROUNDSに達した。",
                result="MockAgentRunnerの研究ラウンドを完了した。",
                verification="journal.jsonlに各ラウンドを記録済み。",
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

    def _handle_round_output(self, session: ResearchSession, output: RoundOutput) -> str:
        approval_text = "なし"
        notice_text = ""
        if output.proposed_operation and self.approval_gate.requires_approval(output.proposed_operation):
            request = self.approval_gate.create_request(session, output.proposed_operation)
            session.transition_to(Mode.APPROVAL_BLOCKED)
            approval_text = f"@Sora {request.approval_id}: {request.reason}"
        elif output.proposed_operation and self.approval_gate.requires_important_notice(output.proposed_operation):
            notice_text = (
                "⚠️ 重要通知: 承認不要ポリシーで許可する操作候補\n"
                f"操作: {output.proposed_operation.operation}\n"
                f"理由: {output.proposed_operation.reason}\n"
                f"影響: {output.proposed_operation.impact}\n"
                f"ドライラン結果: {output.proposed_operation.dry_run_result}"
            )
        report = self._format_report(
            session,
            purpose=session.current_question,
            did=output.main_agent_summary,
            result=output.sub_agent_output,
            fresh=output.fresh_agent_output or "なし",
            verification=output.review_output
            + (f"\nClaude: {output.claude_consultation}" if output.claude_consultation else ""),
            decision=output.decision,
            confidence=output.confidence,
            approval=approval_text,
        )
        if output.proposed_operation and approval_text != "なし":
            request = session.approval_requests[f"AP-{len(session.approval_requests)}"]
            report = report + "\n\n" + render_approval_request(request)
        if notice_text:
            report = report + "\n\n" + notice_text
        self.discord.send(report, channel="important")
        self._record(
            session,
            event_type="research_round_completed",
            main_agent_summary=output.main_agent_summary,
            subtask=output.subtask,
            conversation_sessions=output.conversation_sessions,
            sub_agent_output=output.sub_agent_output,
            review_output=output.review_output,
            claude_consultation=output.claude_consultation,
            fresh_agent_output=output.fresh_agent_output,
            decision=output.decision,
            confidence=output.confidence,
            files_changed=[str(self.config.journal_path(session.session_id))],
        )
        ledger_entry = self._research_ledger(session).append_round(session, output)
        self._record(
            session,
            event_type="ledger_entry_appended",
            decision=ledger_entry.node_id,
            confidence="high",
            node_id=ledger_entry.node_id,
            parent_node_id=ledger_entry.parent_node_id,
            selected_as_best=ledger_entry.selected_as_best,
            files_changed=[str(self.config.research_ledger_path(session.session_id))],
        )
        if output.fresh_agent_output:
            self._record(
                session,
                event_type="fresh_agent_output",
                fresh_agent_output=output.fresh_agent_output,
            )
        if output.proposed_operation and approval_text != "なし":
            self._record(
                session,
                event_type="approval_requested",
                decision="APPROVAL_BLOCKED",
                discord_report=render_approval_request(request),
            )
        if notice_text:
            self._record(
                session,
                event_type="important_notice_sent",
                decision="allowed_after_notice",
                discord_report=notice_text,
            )
        self._record(
            session,
            event_type="discord_report_sent",
            discord_report=report,
        )
        return report

    def _format_report(
        self,
        session: ResearchSession,
        purpose: str,
        did: str,
        result: str,
        verification: str,
        decision: str,
        confidence: str,
        fresh: str = "なし",
        approval: str = "なし",
    ) -> str:
        lines = [f"{self._display_mode(session)}モードになりました。", str(result).strip()]
        if fresh != "なし":
            lines.append(f"新規アイデア: {fresh}")
        if approval != "なし":
            lines.append(f"承認待ち: {approval}")
        elif session.next_action:
            lines.append(f"次: {session.next_action}")
        return "\n".join(line for line in lines if line)

    def _display_mode(self, session: ResearchSession) -> str:
        if session.mode == Mode.APPROVAL_BLOCKED:
            return "blocked"
        if session.mode == Mode.RESEARCH:
            return "researching"
        if session.mode == Mode.PLANNING and session.phase == "plan":
            return "plan"
        if session.mode == Mode.DONE:
            return "Neutral"
        return "Neutral"

    def _planning_prompt(self, session: ResearchSession) -> str:
        questions = "\n".join(f"- {item}" for item in session.planning_questions)
        return f"実装前確認事項:\n{questions}"

    def _save_journal_and_brief(
        self,
        session: ResearchSession,
        decision: str,
        user_instruction: str | None = None,
    ) -> None:
        session.updated_at = utc_timestamp()
        self.store.save(session)
        brief = self._brief_writer(session).write(session)
        self._record(
            session,
            event_type="research_brief_updated",
            user_instruction=user_instruction,
            research_brief_snapshot=brief,
            decision=decision,
            confidence="high",
            files_changed=[
                str(self.config.session_state_path(session.session_id)),
                str(self.config.research_brief_path(session.session_id)),
            ],
        )

    def _journal_summary(self) -> str:
        session = self._require_session()
        entries = self._journal(session).read_entries()
        modes = sorted({entry.get("mode") for entry in entries if entry.get("mode")})
        return f"journal entries: {len(entries)}; modes: {', '.join(modes) if modes else 'なし'}"

    def _require_session(self) -> ResearchSession:
        session = self.store.load()
        if not session:
            raise ValueError("active session not found. Run /re new first.")
        return session

    def _record(self, session: ResearchSession, event_type: str, **fields):
        entry = self._journal(session).append(session, event_type=event_type, **fields)
        self.discord.send(_format_log_entry(entry), channel="log")
        return entry

    def _journal(self, session: ResearchSession) -> Journal:
        return Journal(self.config.journal_path(session.session_id))

    def _brief_writer(self, session: ResearchSession) -> ResearchBriefWriter:
        return ResearchBriefWriter(self.config.research_brief_path(session.session_id))

    def _paper_store(self, session: ResearchSession) -> PaperStore:
        return PaperStore(self.config.papers_path(session.session_id))

    def _research_ledger(self, session: ResearchSession) -> ResearchLedger:
        return ResearchLedger(self.config.research_ledger_path(session.session_id))

    def _write_plan_md(self, session: ResearchSession) -> Path:
        path = self.config.plan_path(session.session_id)
        path.write_text(self._render_plan_md(session) + "\n", encoding="utf-8")
        self._record(
            session,
            event_type="plan_finalized",
            decision="plan.md finalized for research handoff",
            confidence="high",
            files_changed=[str(path)],
        )
        return path

    def _render_plan_md(self, session: ResearchSession) -> str:
        scout = session.planning_scout or {}
        primary = scout.get("primary_comparison") or {}
        accepted = "\n".join(f"- {item}" for item in session.accepted_ideas[-12:]) or "- なし"
        goal_criteria = "\n".join(f"- {item}" for item in session.goal_criteria) or "- 未確定"
        research_methods = "\n".join(f"- {item}" for item in session.research_methods) or "- 未確定"
        open_questions = "\n".join(f"- {item}" for item in session.open_questions) or "- なし"
        redirects = "\n".join(f"- {item}" for item in session.redirects) or "- なし"
        return f"""# Research Plan

## Handoff
- plan_model: {self.config.plan_agent_model}
- effective_plan_model: {effective_model_name(self.config.plan_agent_model)}
- research_model: {self.config.research_agent_model}
- effective_research_model: {effective_model_name(self.config.research_agent_model)}
- session_id: {session.session_id}
- version: {session.version_label}
- research_dir: {session.research_dir}

## Goal
{session.research_goal}

## Current Direction
{session.current_question or "未設定"}

## Similar Research
- novelty_status: {scout.get("novelty_status", "未評価")}
- primary_comparison: {primary.get("title", "未確認")} [{primary.get("paper_id", "未確認")}]
- rationale: {scout.get("rationale", "未確認")}

## Conversation Notes
{accepted}

## Goal Criteria (何をもって完了/成功とみなすか)
{goal_criteria}

## Research Methods
{research_methods}

## Open Questions
{open_questions}

## Redirects And Constraints
{redirects}

## Next Step For Researching
{session.next_action or "未設定"}
"""

    def _build_paper_provider(self, provider_name: str) -> PaperSearchProvider:
        if provider_name == "arxiv":
            return ArxivPaperSearchProvider()
        return FakePaperSearchProvider()

    def _list_runs(self) -> list[dict[str, object]]:
        if not self.config.research_archive_path.exists():
            return []
        rows = []
        for path in sorted(self.config.research_archive_path.iterdir(), reverse=True):
            if not path.is_dir() or not path.name.startswith("V"):
                continue
            state_path = path / "state.json"
            if not state_path.exists():
                rows.append(
                    {
                        "version": path.name.split("_", 1)[0],
                        "session_id": "unknown",
                        "mode": "unknown",
                        "round_id": "?",
                        "goal": "state.jsonなし",
                        "path": path,
                    }
                )
                continue
            try:
                session = ResearchSession.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
            except Exception:
                rows.append(
                    {
                        "version": path.name.split("_", 1)[0],
                        "session_id": "unknown",
                        "mode": "broken",
                        "round_id": "?",
                        "goal": "state.json読込失敗",
                        "path": path,
                    }
                )
                continue
            rows.append(
                {
                    "version": session.version_label or path.name.split("_", 1)[0],
                    "session_id": session.session_id,
                    "mode": session.mode.value,
                    "round_id": session.round_id,
                    "goal": session.research_goal,
                    "path": path,
                }
            )
        return rows

    def _citation_note(self, papers: list[Paper]) -> str:
        if not papers:
            return "未確認: 出典候補がありません。"
        return "\n".join(
            f"- {paper.title}: {paper.summary} [{paper.paper_id}]"
            for paper in papers
        )

    def _block_for_cost_limit(self, session: ResearchSession, check) -> str:
        request = self.approval_gate.create_request(
            session,
            self.cost_manager.make_limit_operation(check),
        )
        if session.mode != Mode.APPROVAL_BLOCKED:
            session.transition_to(Mode.APPROVAL_BLOCKED)
        self.store.save(session)
        message = (
            "⏸ コスト上限に到達しました\n"
            f"session: {session.session_id}\n"
            f"reason: {check.reason}\n"
            f"used: {check.used} / {check.limit}\n"
            f"次に進むには /re approve {request.approval_id} または設定変更が必要です。"
        )
        self.discord.send(message, channel="important")
        self._record(
            session,
            event_type="cost_limit_reached",
            decision="APPROVAL_BLOCKED",
            discord_report=message,
        )
        return message

    def _block_for_novelty_gate(self, session: ResearchSession) -> str:
        scout = session.planning_scout
        gate = self._ensure_phase_gate(
            session,
            phase="literature_review",
            reason=scout.get("rationale") or "novelty gate blocking",
        )
        session.next_action = "novelty gateで停止中。通常メッセージで追加調査観点または方針転換を伝えてください。"
        self.store.save(session)
        self._brief_writer(session).write(session)
        message = (
            "⏸ Novelty gateで研究開始を保留しました\n"
            f"session: {session.session_id}\n"
            f"gate: {gate.gate_id}\n"
            f"status: {scout.get('novelty_status') or 'unknown'}\n"
            f"reason: {scout.get('rationale') or '未確認'}\n"
            "次の候補:\n"
            f"- /re accept {gate.gate_id} でこのリスクを理解して進める\n"
            "- 通常メッセージで追加調査観点を伝える\n"
            "- 通常メッセージで研究スコープを変える\n"
            "- 十分な根拠が揃ったら再度 /re start"
        )
        self.discord.send(message, channel="important")
        self._record(
            session,
            event_type="phase_gate_requested",
            decision="novelty_gate_blocked_start",
            confidence="high",
            discord_report=message,
            planning_scout=scout,
        )
        self._record(
            session,
            event_type="discord_report_sent",
            discord_report=message,
        )
        return message

    def _block_for_phase_gate(self, session: ResearchSession) -> str:
        gate = self._pending_phase_gate(session)
        assert gate is not None
        message = (
            "⏸ Phase gateがpendingです\n"
            f"session: {session.session_id}\n"
            f"gate: {gate.gate_id}\n"
            f"phase: {gate.phase}\n"
            f"reason: {gate.reason}\n"
            f"続行: /re accept {gate.gate_id}\n"
            f"差し戻し: /re revise {gate.gate_id} <reason>"
        )
        self.discord.send(message, channel="important")
        self._record(
            session,
            event_type="phase_gate_requested",
            decision="phase_gate_pending",
            confidence="high",
            phase_gate=gate.__dict__,
            discord_report=message,
        )
        return message

    def _ensure_phase_gate(self, session: ResearchSession, phase: str, reason: str) -> PhaseGate:
        for gate in session.phase_gates.values():
            if gate.phase == phase and gate.status == "pending":
                return gate
        gate_id = f"PG-{len(session.phase_gates) + 1}"
        gate = PhaseGate(gate_id=gate_id, phase=phase, reason=reason)
        session.phase_gates[gate_id] = gate
        return gate

    def _pending_phase_gate(self, session: ResearchSession) -> PhaseGate | None:
        for gate in session.phase_gates.values():
            if gate.status == "pending":
                return gate
        return None

    def _has_accepted_phase_gate(self, session: ResearchSession, phase: str) -> bool:
        return any(
            gate.phase == phase and gate.status == "accepted"
            for gate in session.phase_gates.values()
        )

    def _apply_plan_update(self, session: ResearchSession, updates: dict[str, list[str]]) -> str:
        if not updates:
            return ""
        counts = {
            "ゴール条件": 0,
            "方法": 0,
            "確認事項": 0,
            "解決": 0,
        }
        for item in updates.get("goal_criteria", []):
            if item not in session.goal_criteria:
                session.goal_criteria.append(item)
                counts["ゴール条件"] += 1
        for item in updates.get("research_methods", []):
            if item not in session.research_methods:
                session.research_methods.append(item)
                counts["方法"] += 1
        for item in updates.get("open_questions", []):
            if item not in session.open_questions:
                session.open_questions.append(item)
                counts["確認事項"] += 1
        for resolved in updates.get("resolved_questions", []):
            before = len(session.open_questions)
            session.open_questions = [
                item
                for item in session.open_questions
                if not (item == resolved or item.startswith(resolved) or resolved.startswith(item))
            ]
            counts["解決"] += before - len(session.open_questions)
        labels = [
            f"{label}+{count}"
            for label, count in counts.items()
            if count > 0 and label != "解決"
        ]
        if counts["解決"] > 0:
            labels.append(f"確認事項-{counts['解決']}")
        return ", ".join(labels)


def _format_paper_line(paper: Paper) -> str:
    year = paper.year if paper.year is not None else "未確認"
    return f"{paper.paper_id} [{paper.source}] {paper.title} ({year}) score={paper.relevance_score:.2f}"


def _format_paper_research_note(paper: Paper) -> str:
    authors = ", ".join(paper.authors[:5]) or "未確認"
    abstract_excerpt = _paper_excerpt(paper.abstract, 900)
    return (
        f"[{paper.paper_id}] {paper.title}\n"
        f"- authors: {authors}\n"
        f"- year/venue/source: {paper.year or '未確認'} / {paper.venue or '未確認'} / {paper.source}\n"
        f"- url: {paper.url or '未確認'}\n"
        f"- relevance/confidence: {paper.relevance_score:.2f} / {paper.confidence}\n"
        f"- 内容要約: {paper.summary or '要約なし'}\n"
        f"- abstract抜粋: {abstract_excerpt}\n"
        f"- 今回の使いどころ: {_paper_research_use(paper)}\n"
        f"- 未確認点: abstractとメタデータベース。本文の手法詳細、実験設定、限界、再現条件は未確認。"
    )


def _paper_research_use(paper: Paper) -> str:
    title = paper.title or "この文献"
    if paper.source == "fake":
        return f"{title} はテスト用候補なので、実在根拠ではなく表示・保存フローの検証材料として扱う。"
    return (
        f"{title} を類似研究候補として、研究テーマとの重複点、差分仮説、比較すべき評価指標を決める材料にする。"
        f"この文献に基づく主張は必ず [{paper.paper_id}] を付ける。"
    )


def _paper_excerpt(text: str, max_chars: int) -> str:
    normalized = " ".join((text or "").split())
    if not normalized:
        return "未確認"
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "..."


def _format_log_entry(entry: dict[str, object]) -> str:
    parts = [
        f"[{entry.get('timestamp')}]",
        str(entry.get("event_type")),
        f"session={entry.get('session_id')}",
        f"mode={entry.get('mode')}",
        f"round={entry.get('round_id')}",
    ]
    if entry.get("decision"):
        parts.append(f"decision={entry.get('decision')}")
    if entry.get("files_changed"):
        parts.append(f"files={len(entry.get('files_changed') or [])}")
    if entry.get("errors"):
        parts.append(f"errors={entry.get('errors')}")
    return " ".join(parts)
