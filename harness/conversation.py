from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    participant: str
    message: str


@dataclass
class ConversationSession:
    topic: str
    participants: list[str]
    max_turns: int = 4
    timeout_seconds: int = 60
    session_id: str = field(default_factory=lambda: f"CS-{uuid.uuid4().hex[:10]}")
    scratchpad: list[ConversationTurn] = field(default_factory=list)
    stop_condition: str = ""
    final_summary: dict[str, str] = field(default_factory=dict)

    def run_scripted(self, responses: list[str]) -> "ConversationSession":
        start = time.monotonic()
        stagnant_count = 0
        previous = None
        for index, response in enumerate(responses):
            if index >= self.max_turns:
                self.stop_condition = "max_turns"
                break
            if time.monotonic() - start > self.timeout_seconds:
                self.stop_condition = "timeout"
                break
            participant = self.participants[index % len(self.participants)]
            self.scratchpad.append(ConversationTurn(participant=participant, message=response))
            if response == previous:
                stagnant_count += 1
            else:
                stagnant_count = 0
            previous = response
            if stagnant_count >= 2:
                self.stop_condition = "stagnation"
                break
        if not self.stop_condition:
            self.stop_condition = "completed"
        self.final_summary = self._summarize()
        return self

    def _summarize(self) -> dict[str, str]:
        conclusion = self.scratchpad[-1].message if self.scratchpad else "結論なし"
        evidence = "; ".join(turn.message for turn in self.scratchpad[:2]) or "根拠なし"
        return {
            "問い": self.topic,
            "結論": conclusion,
            "根拠": evidence,
            "採用する判断": "MVPではfinal_summaryだけをメイン文脈へ戻す",
            "未解決点": "実エージェント接続はMVP OUT",
            "次アクション": "MockAgentRunnerの次ラウンドへ進む",
            "人間に聞くべきこと": "設計思想の未確定事項があればSoraへ確認する",
            "stop_condition": self.stop_condition,
        }

    def to_journal_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "participants": self.participants,
            "max_turns": self.max_turns,
            "timeout_seconds": self.timeout_seconds,
            "scratchpad": [turn.__dict__ for turn in self.scratchpad],
            "stop_condition": self.stop_condition,
            "final_summary": self.final_summary,
        }

