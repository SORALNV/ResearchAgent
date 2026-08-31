from __future__ import annotations

from dataclasses import dataclass

from harness.state import ApprovalRequest, ResearchSession


@dataclass(frozen=True)
class ProposedOperation:
    operation: str
    reason: str = ""
    impact: str = "可逆性は未確認。MVPではドライランのみ。"
    dry_run_result: str = "実操作は行わず、承認ゲートのみ検証した。"


class ApprovalGate:
    DANGEROUS_KEYWORDS = (
        "delete",
        "delete_file",
        "delete_folder",
        "remove",
        "rm ",
        "rm -",
        "rmdir",
        "folder_delete",
        "directory_delete",
        "overwrite",
        "outside_research_archive",
        "push",
        "pull_request",
        "pr_create",
        "release",
        "external_post",
        "paid_api",
        "untrusted_network",
        "secret_transfer",
        "env_transfer",
        "sudo",
        "chmod",
        "chown",
        # GUI automation can cross trust boundaries and must always be explicit.
        "openai_computer_use",
        "openai_computer_safety",
        # Harness-internal fail-closed conditions.
        "agent_protocol_failure",
        "review_unresolved",
        "artifact_promotion_failure",
        "sub_agent_failure",
    )
    IMPORTANT_NOTICE_KEYWORDS = (
        "long_running_job",
        "long_running_command",
        "mass_file_generation",
        "large_file_generation",
        "many_files",
    )

    def requires_approval(self, operation: ProposedOperation) -> bool:
        normalized = operation.operation.lower()
        return any(keyword in normalized for keyword in self.DANGEROUS_KEYWORDS)

    def requires_important_notice(self, operation: ProposedOperation) -> bool:
        normalized = operation.operation.lower()
        return (
            not self.requires_approval(operation)
            and any(keyword in normalized for keyword in self.IMPORTANT_NOTICE_KEYWORDS)
        )

    def classify(self, operation: ProposedOperation) -> str:
        if self.requires_approval(operation):
            return "approval_required"
        if self.requires_important_notice(operation):
            return "important_notice"
        return "allowed"

    def create_request(self, session: ResearchSession, operation: ProposedOperation) -> ApprovalRequest:
        approval_id = f"AP-{len(session.approval_requests) + 1}"
        request = ApprovalRequest(
            approval_id=approval_id,
            operation=operation.operation,
            reason=operation.reason,
            impact=operation.impact,
            dry_run_result=operation.dry_run_result,
        )
        session.approval_requests[approval_id] = request
        return request

    def approve(self, session: ResearchSession, approval_id: str) -> ApprovalRequest:
        request = self._get_pending(session, approval_id)
        request.status = "approved"
        session.approvals_received.append(approval_id)
        return request

    def reject(self, session: ResearchSession, approval_id: str, reason: str) -> ApprovalRequest:
        request = self._get_pending(session, approval_id)
        request.status = f"rejected: {reason}"
        session.rejected_ideas.append(f"{approval_id}: {reason}")
        return request

    def _get_pending(self, session: ResearchSession, approval_id: str) -> ApprovalRequest:
        if approval_id not in session.approval_requests:
            raise ValueError(f"unknown approval id: {approval_id}")
        request = session.approval_requests[approval_id]
        if request.status != "pending":
            raise ValueError(f"approval is not pending: {approval_id}")
        return request


def render_approval_request(request: ApprovalRequest) -> str:
    return f"""⏳ 承認依頼 [id: {request.approval_id}]
操作: {request.operation}
理由: {request.reason}
影響: {request.impact}
ドライラン結果: {request.dry_run_result}
承認方法: /re approve {request.approval_id}
却下方法: /re reject {request.approval_id} <理由>
"""
