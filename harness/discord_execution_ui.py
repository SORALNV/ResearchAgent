from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from harness.discord_channel_map import DiscordLocation
from harness.state import utc_timestamp


_NARRATION_RULES = """

実行中の進捗共有ルール:
- 実際にツール、command、file変更、外部確認を行う前に、利用者へ見せる短いcommentaryを1〜2文で出す。
- commentaryには「現在何を確認したか」「次に何をするか」「なぜその順序か」「既存の何を保護するか」のうち必要なものだけを書く。
- 例: `C:\\VSCode\\ResearchAgent の現在状態を確認してから、origin/main の最新版を --ff-only で取り込みます。既存のfix branchや未マージ変更は削除しません。`
- 非公開のchain-of-thought、内部推論全文、reasoning token、秘密情報は出さない。観測可能な作業方針、操作、判断根拠、結果だけをcommentaryとして共有する。
- commandやfile変更を複数回行う長い作業では、意味のある区切りごとにcommentaryを更新する。細かなtoken単位の実況はしない。
"""


@dataclass(frozen=True)
class ExecutionThreadRecord:
    thread_id: str
    guild_id: str
    parent_channel_id: str
    parent_conversation_id: str
    work_session_id: str
    source_message_id: str
    action_kind: str
    subject: str
    status: str = "active"
    job_ids: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["job_ids"] = list(self.job_ids)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionThreadRecord":
        now = utc_timestamp()
        return cls(
            thread_id=_snowflake(data.get("thread_id"), "thread_id"),
            guild_id=_snowflake(data.get("guild_id"), "guild_id"),
            parent_channel_id=_snowflake(
                data.get("parent_channel_id"),
                "parent_channel_id",
            ),
            parent_conversation_id=_snowflake(
                data.get("parent_conversation_id") or data.get("parent_channel_id"),
                "parent_conversation_id",
            ),
            work_session_id=str(data.get("work_session_id") or "").strip(),
            source_message_id=_snowflake(
                data.get("source_message_id"),
                "source_message_id",
            ),
            action_kind=str(data.get("action_kind") or "execution").strip(),
            subject=" ".join(str(data.get("subject") or "execution").split()),
            status=str(data.get("status") or "active").strip(),
            job_ids=tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in data.get("job_ids", [])
                    if str(item).strip()
                )
            ),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
        )

    def parent_location(self) -> DiscordLocation:
        return DiscordLocation(
            guild_id=self.guild_id,
            channel_id=self.parent_channel_id,
            parent_channel_id=None,
            thread_id=None,
        )


class ExecutionThreadRegistry:
    """Persistent mapping from a progress Thread to its parent WorkSession."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "discord_execution_threads.json"
        self._lock = threading.RLock()

    def get(self, thread_id: str) -> ExecutionThreadRecord | None:
        key = _snowflake(thread_id, "thread_id")
        with self._lock:
            return self._read_all().get(key)

    def list(self) -> tuple[ExecutionThreadRecord, ...]:
        with self._lock:
            values = list(self._read_all().values())
        return tuple(sorted(values, key=lambda item: (item.created_at, item.thread_id)))

    def latest_for_session(self, work_session_id: str) -> ExecutionThreadRecord | None:
        candidates = [
            item
            for item in self.list()
            if item.work_session_id == str(work_session_id)
        ]
        return candidates[-1] if candidates else None

    def bind(
        self,
        *,
        thread_id: str,
        location: DiscordLocation,
        work_session_id: str,
        source_message_id: str,
        action_kind: str,
        subject: str,
    ) -> ExecutionThreadRecord:
        now = utc_timestamp()
        record = ExecutionThreadRecord(
            thread_id=_snowflake(thread_id, "thread_id"),
            guild_id=_snowflake(location.guild_id, "guild_id"),
            parent_channel_id=_snowflake(location.channel_id, "parent_channel_id"),
            parent_conversation_id=_snowflake(
                location.conversation_id,
                "parent_conversation_id",
            ),
            work_session_id=str(work_session_id).strip(),
            source_message_id=_snowflake(source_message_id, "source_message_id"),
            action_kind=str(action_kind or "execution").strip(),
            subject=" ".join(str(subject or "execution").split()),
            created_at=now,
            updated_at=now,
        )
        if not record.work_session_id:
            raise ValueError("work_session_id must be non-empty")
        with self._lock:
            values = self._read_all()
            existing = values.get(record.thread_id)
            if existing is not None:
                if existing.work_session_id != record.work_session_id:
                    raise ValueError("execution Thread is already bound to another WorkSession")
                return existing
            values[record.thread_id] = record
            self._write_all(values)
        return record

    def bind_jobs(
        self,
        thread_id: str,
        job_ids: list[str] | tuple[str, ...],
    ) -> ExecutionThreadRecord:
        key = _snowflake(thread_id, "thread_id")
        with self._lock:
            values = self._read_all()
            current = values.get(key)
            if current is None:
                raise KeyError(f"unknown execution Thread: {key}")
            updated = replace(
                current,
                job_ids=tuple(
                    dict.fromkeys(
                        [*current.job_ids]
                        + [str(item).strip() for item in job_ids if str(item).strip()]
                    )
                ),
                updated_at=utc_timestamp(),
            )
            values[key] = updated
            self._write_all(values)
            return updated

    def set_status(self, thread_id: str, status: str) -> ExecutionThreadRecord:
        key = _snowflake(thread_id, "thread_id")
        with self._lock:
            values = self._read_all()
            current = values.get(key)
            if current is None:
                raise KeyError(f"unknown execution Thread: {key}")
            updated = replace(
                current,
                status=str(status or "active").strip(),
                updated_at=utc_timestamp(),
            )
            values[key] = updated
            self._write_all(values)
            return updated

    def _read_all(self) -> dict[str, ExecutionThreadRecord]:
        if not self.path.is_file():
            return {}
        try:
            decoded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        raw = decoded.get("threads") if isinstance(decoded, Mapping) else None
        if not isinstance(raw, Mapping):
            return {}
        result: dict[str, ExecutionThreadRecord] = {}
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                continue
            try:
                result[str(key)] = ExecutionThreadRecord.from_dict(value)
            except (TypeError, ValueError):
                continue
        return result

    def _write_all(self, values: Mapping[str, ExecutionThreadRecord]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "threads": {
                key: value.to_dict()
                for key, value in sorted(values.items(), key=lambda item: item[0])
            },
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        temporary.replace(self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


def attach_execution_narration_prompt(service: Any) -> Any:
    """Ask Codex for safe user-facing commentary, never private reasoning."""

    dispatcher = getattr(service, "dispatcher", None)
    handlers = getattr(dispatcher, "handlers", {})
    if not isinstance(handlers, Mapping):
        return service
    for handler in handlers.values():
        original = getattr(handler, "_build_prompt", None)
        if not callable(original) or bool(
            getattr(handler, "_execution_narration_prompt_installed", False)
        ):
            continue

        def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> str:
            return str(_original(*args, **kwargs)) + _NARRATION_RULES

        handler._build_prompt = wrapped
        handler._execution_narration_prompt_installed = True
    return service


def execution_thread_name(subject: str, action_kind: str, source_message_id: str) -> str:
    label = {
        "experiment": "run",
        "submission": "submit",
        "paper": "paper",
    }.get(str(action_kind), "work")
    compact = re.sub(r"[^0-9A-Za-zぁ-んァ-ヶ一-龠_-]+", "-", str(subject)).strip("-")
    compact = compact[:55] or "ResearchAgent"
    suffix = str(source_message_id)[-6:]
    return f"{label}-{compact}-{suffix}"[:95]


def execution_opening_message(
    *,
    subject: str,
    action_kind: str,
    request_text: str,
) -> str:
    action = {
        "experiment": "実装・検証・実験",
        "submission": "提出前検証と提出",
        "paper": "証拠整理と論文生成",
    }.get(str(action_kind), "依頼された作業")
    request = " ".join(str(request_text).split())[:500]
    return (
        f"**実行スレッドを開始しました。** **{subject}** の現在状態と既存の作業・Jobを確認してから、{action}を進めます。"
        "既存のbranch、未マージ変更、成果物は削除しません。\n"
        f"**依頼:** {request}\n"
        "ここには非公開の思考そのものではなく、次の操作、理由、確認結果、command/file変更などの作業ログを順次送ります。"
    )


def build_help_message(service: Any, location: DiscordLocation) -> str:
    channel = getattr(service, "registry", None)
    config = channel.get(location) if channel is not None else None
    context = (
        f"**{config.subject}**（`{config.domain.value}`）"
        if config is not None
        else "未設定チャンネル"
    )
    return (
        f"**ResearchAgent Help — {context}**\n"
        "**通常会話:** 仮説相談、`これを実装して試して`、結果解釈、`このCSVで提出しよう`、`この結果を論文にまとめて`。実行依頼は専用Threadを作り、途中経過を流します。\n"
        "**案件:** `/agent setup` · `/agent channel` · `/agent status` · `/agent readiness` · `/agent finish`\n"
        "**Job:** `/agent job list` · `/agent compute_backends` · `/agent approve_compute` · `/agent cancel_job`\n"
        "**Codex:** `/agent codex_status` · `/agent steer` · `/agent interrupt` · `/agent codex_approvals` · `/agent codex_approval`"
    )


def build_readiness_message(service: Any, location: DiscordLocation) -> str:
    checks: list[tuple[str, str, str]] = []
    registry = getattr(service, "registry", None)
    channel = registry.get(location) if registry is not None else None
    if channel is None:
        checks.append(("blocked", "Channel", "未設定。`/agent setup`が必要"))
    elif str(getattr(channel.status, "value", channel.status)) != "active":
        checks.append(("blocked", "Channel", "案件は終了済み"))
    else:
        checks.append(("ready", "Channel", f"{channel.subject} / {channel.domain.value}"))

    policy = getattr(service, "discord_access_policy", None)
    if policy is None:
        checks.append(("warning", "Access", "ACL hardeningを確認できません"))
    elif bool(getattr(policy, "required", False)):
        allowed = len(getattr(policy, "global_user_ids", ()) or ())
        owner = str(getattr(channel, "created_by", "") or "") if channel else ""
        if allowed or owner.isdigit():
            checks.append(("ready", "Access", "owner/allowlist制限が有効"))
        else:
            checks.append(("blocked", "Access", "許可ユーザーが未設定"))
    else:
        checks.append(("warning", "Access", "open-server設定"))

    if channel is not None and str(getattr(channel.status, "value", channel.status)) == "active":
        try:
            state = service.codex_status(location, title=channel.subject)
            if bool(state.get("running")):
                checks.append(("ready", "Codex", "App Server running"))
            else:
                checks.append(("blocked", "Codex", "App Server not running"))
        except Exception as exc:
            checks.append(("blocked", "Codex", f"{type(exc).__name__}: {exc}"[:300]))
    else:
        checks.append(("blocked", "Codex", "Channel setup待ち"))

    broker = getattr(getattr(service, "compute", None), "broker", None)
    if broker is None:
        checks.append(("blocked", "Compute", "Broker未接続"))
    else:
        try:
            snapshot = broker.snapshot()
            available = [
                name
                for name, state in snapshot.items()
                if bool(state.get("available"))
            ]
            if available:
                checks.append(("ready", "Compute", ", ".join(available[:5])))
            else:
                checks.append(("warning", "Compute", "利用可能Backendなし"))
        except Exception as exc:
            checks.append(("warning", "Compute", f"{type(exc).__name__}: {exc}"[:300]))

    root = Path(getattr(getattr(service, "config", None), "project_root", "."))
    if root.exists() and os.access(root, os.W_OK):
        checks.append(("ready", "Storage", str(root)))
    else:
        checks.append(("blocked", "Storage", f"書込不可: {root}"))

    if channel is not None and getattr(channel, "domain", None) is not None:
        domain = str(getattr(channel.domain, "value", channel.domain))
        if domain == "kaggle":
            submission = getattr(getattr(service, "final_actions", None), "submission", None)
            if submission is None:
                checks.append(("blocked", "Kaggle", "submission pipeline未接続"))
            elif _kaggle_credentials_present():
                checks.append(("ready", "Kaggle", "CLI認証候補あり"))
            else:
                checks.append(("warning", "Kaggle", "認証情報を確認してください"))
        elif domain == "research":
            final_actions = getattr(service, "final_actions", None)
            checks.append(
                (
                    "ready" if final_actions is not None else "warning",
                    "Paper",
                    "生成pipeline接続" if final_actions is not None else "pipeline未確認",
                )
            )

    rank = {"ready": 0, "warning": 1, "blocked": 2}
    worst = max((rank[state] for state, _, _ in checks), default=2)
    overall = ("READY", "WARNING", "BLOCKED")[worst]
    icon = {"ready": "✅", "warning": "⚠️", "blocked": "❌"}
    lines = [f"**Readiness: {overall}**"]
    lines.extend(
        f"{icon[state]} **{name}:** {detail}"
        for state, name, detail in checks
    )
    return "\n".join(lines)


def build_job_list_message(
    service: Any,
    location: DiscordLocation,
    *,
    limit: int = 20,
) -> str:
    registry = getattr(service, "registry", None)
    channel = registry.get(location) if registry is not None else None
    if channel is None or not getattr(channel, "work_session_id", ""):
        return "**Jobs:** このチャンネルは未設定です。"
    jobs = service.router.store.list_jobs(
        work_session_id=channel.work_session_id
    )
    if not jobs:
        return f"**Jobs — {channel.subject}:** まだありません。"
    lines = [f"**Jobs — {channel.subject}:** 最新{min(len(jobs), limit)}件"]
    for job in jobs[-max(1, int(limit)):]:
        status = str(getattr(getattr(job, "status", None), "value", getattr(job, "status", "unknown")))
        backend = str(getattr(job, "backend_id", None) or "-")
        payload = getattr(getattr(job, "spec", None), "payload", {}) or {}
        title = " ".join(
            str(
                payload.get("title")
                or payload.get("hypothesis")
                or getattr(getattr(job, "spec", None), "experiment_id", "")
                or "experiment"
            ).split()
        )[:120]
        result_ref = str(getattr(job, "checkpoint_ref", None) or "")
        suffix = f" · result `{result_ref}`" if result_ref else ""
        lines.append(
            f"- `{job.job_id}` — **{status}** · backend `{backend}` · {title}{suffix}"
        )
    return "\n".join(lines)


def format_job_progress(job: Any) -> str:
    status = str(getattr(getattr(job, "status", None), "value", getattr(job, "status", "unknown")))
    backend = str(getattr(job, "backend_id", None) or "-")
    error = " ".join(str(getattr(job, "error", "") or "").split())[:600]
    result_ref = str(getattr(job, "checkpoint_ref", None) or "")
    message = f"**Job更新:** `{job.job_id}` → **{status}** · backend `{backend}`"
    if result_ref:
        message += f" · result `{result_ref}`"
    if error:
        message += f"\n**理由:** {error}"
    return message


def job_progress_key(job: Any) -> tuple[str, str, str, str]:
    return (
        str(getattr(getattr(job, "status", None), "value", getattr(job, "status", ""))),
        str(getattr(job, "backend_id", None) or ""),
        str(getattr(job, "checkpoint_ref", None) or ""),
        str(getattr(job, "error", None) or ""),
    )


def job_is_terminal(job: Any) -> bool:
    return job_progress_key(job)[0] in {"succeeded", "failed", "cancelled"}


def _kaggle_credentials_present() -> bool:
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    root = Path(os.getenv("KAGGLE_CONFIG_DIR") or (Path.home() / ".kaggle"))
    return (root / "kaggle.json").is_file()


def _snowflake(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text.isdigit():
        raise ValueError(f"{name} must be a Discord snowflake")
    return text
