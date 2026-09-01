from __future__ import annotations

import inspect
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise RuntimeError(f"unable to patch {label}")
    return text.replace(old, new, 1)


def main() -> None:
    # Make the public module a stable facade while keeping the implementation
    # independent from private helpers in discord_channel_map.py.
    channel_facade = ROOT / "harness" / "channel_sessions.py"
    channel_facade.write_text(
        '"""Public channel-session API."""\n\n'
        'from harness.channel_sessions_compat import *  # noqa: F401,F403\n',
        encoding="utf-8",
    )

    service_path = ROOT / "harness" / "natural_channel_service.py"
    service = service_path.read_text(encoding="utf-8")
    service = service.replace(
        "from harness.channel_sessions import (",
        "from harness.channel_sessions_compat import (",
    )
    service = service.replace("import hashlib\n", "import hashlib\nimport inspect\n")
    service = service.replace("    ControlledAction,\n", "")
    service = service.replace(
        "        event = self.registry_event_store(ingress).append_event(\n",
        "        event = self._store.append_event(\n",
    )
    service = re.sub(
        r"\n    def registry_event_store\(self, ingress: DiscordIngressResult\):.*?"
        r"\n    @property\n    def _store\(self\):",
        "\n    @property\n    def _store(self):",
        service,
        flags=re.S,
    )
    service = service.replace(
        "                proposal = normalize_hypothesis_proposal(\n"
        "                    value,\n"
        "                    domain=self.domain,\n"
        "                    parent_job_id=(\n"
        "                        str(value[\"parent_job_id\"])\n"
        "                        if value.get(\"parent_job_id\")\n"
        "                        else None\n"
        "                    ),\n"
        "                    parent_result_ref=(\n"
        "                        str(value[\"parent_result_ref\"])\n"
        "                        if value.get(\"parent_result_ref\")\n"
        "                        else None\n"
        "                    ),\n"
        "                    seed=f\"natural:{ingress.event.event_id}:{index}\",\n"
        "                )",
        "                proposal = _normalize_proposal(\n"
        "                    value,\n"
        "                    domain=self.domain,\n"
        "                    parent_job_id=(\n"
        "                        str(value[\"parent_job_id\"])\n"
        "                        if value.get(\"parent_job_id\")\n"
        "                        else None\n"
        "                    ),\n"
        "                    parent_result_ref=(\n"
        "                        str(value[\"parent_result_ref\"])\n"
        "                        if value.get(\"parent_result_ref\")\n"
        "                        else None\n"
        "                    ),\n"
        "                    seed=f\"natural:{ingress.event.event_id}:{index}\",\n"
        "                )",
    )
    service = re.sub(
        r"        if proposal\.parent_result_ref:\n"
        r"            gate = self\.router\.check_human_gate\(.*?\n"
        r"                    project_id=route\.project\.project_id,\n"
        r"                \)\n",
        "        if proposal.parent_result_ref:\n"
        "            self.base_service.record_decision(\n"
        "                location,\n"
        "                title=route.work_session.title,\n"
        "                kind=HumanDecisionKind.RESULT_INTERPRETATION,\n"
        "                verdict=HumanDecisionVerdict.ACCEPT,\n"
        "                subject_ref=proposal.parent_result_ref,\n"
        "                note=(\n"
        "                    \"通常会話で次実験を明示選択したため、親結果を次の検証に使う\"\n"
        "                    f\"という人間判断として記録: {text}\"\n"
        "                ),\n"
        "                actor_id=actor_id,\n"
        "                message_id=message_id,\n"
        "                actor_is_human=True,\n"
        "                project_id=route.project.project_id,\n"
        "            )\n",
        service,
        flags=re.S,
    )
    service = re.sub(
        r"        gate = self\.router\.check_human_gate\(\n"
        r"            route,\n"
        r"            action=ControlledAction\.CONTINUE_FROM_RESULT,\n"
        r"            subject_ref=result_ref,\n"
        r"        \)\n"
        r"        if not gate\.allowed:\n"
        r"            self\.base_service\.record_decision\((.*?)\n"
        r"            \)\n",
        "        self.base_service.record_decision(\\1\n            )\n",
        service,
        flags=re.S,
    )
    service = service.replace(
        "                if str(job.spec.payload.get(\"hypothesis_subject_ref\") or \"\")\n"
        "                == proposal.subject_ref\n",
        "                if str(\n"
        "                    (getattr(getattr(job, \"spec\", None), \"payload\", {}) or {}).get(\n"
        "                        \"hypothesis_subject_ref\"\n"
        "                    )\n"
        "                    or \"\"\n"
        "                ) == proposal.subject_ref\n",
    )

    helper = '''\n\ndef _normalize_proposal(\n    value: Mapping[str, Any],\n    *,\n    domain: Domain,\n    parent_job_id: str | None,\n    parent_result_ref: str | None,\n    seed: str,\n) -> HypothesisProposal:\n    """Call the repository normalizer without depending on optional parameters."""\n\n    candidates = {\n        "domain": domain,\n        "parent_job_id": parent_job_id,\n        "parent_result_ref": parent_result_ref,\n        "seed": seed,\n    }\n    parameters = inspect.signature(normalize_hypothesis_proposal).parameters\n    kwargs = {key: item for key, item in candidates.items() if key in parameters}\n    return normalize_hypothesis_proposal(value, **kwargs)\n'''
    marker = "\ndef _extract_protocol(text: str) -> dict[str, Any]:\n"
    if helper.strip() not in service:
        service = replace_once(
            service,
            marker,
            helper + marker,
            label="proposal compatibility helper",
        )
    service_path.write_text(service, encoding="utf-8")

    env_path = ROOT / ".env.example"
    env_text = env_path.read_text(encoding="utf-8")
    if "DISCORD_CHANNEL_SESSIONS_JSON=" not in env_text:
        anchor = "DISCORD_KAGGLE_CHANNEL_IDS=\n"
        addition = (
            "DISCORD_KAGGLE_CHANNEL_IDS=\n"
            "# Preferred channel-native registry bootstrap. Each key is one Discord\n"
            "# channel and each value contains mode, subject, and optional target.\n"
            "DISCORD_CHANNEL_SESSIONS_JSON=\n"
        )
        env_text = replace_once(env_text, anchor, addition, label="channel JSON env")
    env_text = env_text.replace("DISCORD_CREATE_THREADS=true", "DISCORD_CREATE_THREADS=false")
    env_path.write_text(env_text, encoding="utf-8")

    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    heading = "## Discordチャンネル単位の運用"
    if heading not in readme:
        section = '''\n## Discordチャンネル単位の運用\n\n現在のDiscord Edgeでは、**1チャンネルを1件のKaggleコンペまたは研究テーマ**として扱います。`/agent setup`で`kaggle`または`research`、案件名、対象を登録すると、Project・WorkSession・Codex threadが永続的に紐付きます。会話中に「試して」「実装して」「この案で進めて」と指示すれば実装からJob実行へ進み、「このCSVで提出しよう」「この結果を論文にまとめて」で既存の安全ゲート付き最終処理へ進みます。Discord上に戦略モード／実行モードの切替はありません。\n\n案件終了時は`/agent finish`で内部状態をアーカイブし、Discordチャンネル自体はユーザーが整理します。詳細は[`docs/natural_channel_workflow.md`](docs/natural_channel_workflow.md)を参照してください。\n'''
        insert_at = readme.find("\n## ")
        if insert_at < 0:
            readme += section
        else:
            readme = readme[:insert_at] + section + readme[insert_at:]
    readme_path.write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
