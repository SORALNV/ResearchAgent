# Discord channel-native workflow

## Product model

The active Discord product does not expose separate planning, strategy, research,
or execution modes. One Discord channel (or an explicitly configured Discord
thread) is one persistent work context:

```text
Discord channel
  -> ChannelSessionConfig
  -> one Project
  -> one WorkSession
  -> one durable Codex thread
  -> experiment Jobs and final actions for that subject only
```

Create a separate channel for each Kaggle competition or research subject. When
the work is finished, run `/agent finish`; this closes the WorkSession and
archives the Project. A regular Discord text/forum channel remains for the user
to archive through Discord. A Discord thread is archived by the bot after the
command succeeds.

An archived channel cannot be silently repurposed. Create a new Discord channel
for a new subject so history, files, experiments, decisions, and Codex context do
not mix.

## Interactive setup

Run the command in a new channel:

```text
/agent setup mode:kaggle subject:House Prices target:house-prices-advanced-regression-techniques
```

or:

```text
/agent setup mode:research subject:軽量画像認識モデルの知識蒸留 target:CIFAR-100
```

Setup writes `CONTROL_PLANE_DIR/channel_sessions.json`, creates the corresponding
Project and WorkSession, and initializes a durable Codex thread. The JSON record
contains the Discord IDs, domain, subject, target, Project/WorkSession IDs,
Codex thread ID, status, creator, and timestamps.

Example persisted record:

```json
{
  "schema_version": 1,
  "channels": {
    "123456789012345678": {
      "conversation_id": "123456789012345678",
      "guild_id": "111111111111111111",
      "channel_id": "123456789012345678",
      "parent_channel_id": null,
      "mode": "kaggle",
      "domain": "kaggle",
      "subject": "House Prices",
      "target_ref": "house-prices-advanced-regression-techniques",
      "project_id": "PRJ-CHANNEL-KAGGLE-...",
      "work_session_id": "WS-...",
      "codex_thread_id": "...",
      "status": "active"
    }
  }
}
```

## Environment bootstrap

Interactive setup is the normal path. Multiple channels can also be seeded with
one JSON environment variable:

```env
DISCORD_CHANNEL_SESSIONS_JSON={"123456789012345678":{"mode":"kaggle","subject":"House Prices","target":"house-prices-advanced-regression-techniques"},"234567890123456789":{"mode":"research","subject":"軽量画像認識モデルの知識蒸留","target":"CIFAR-100"}}
```

The historical variables remain accepted as migration input:

```env
DISCORD_CHANNEL_DOMAIN_MAP=123456789012345678=kaggle,234567890123456789=research
DISCORD_RESEARCH_CHANNEL_IDS=
DISCORD_KAGGLE_CHANNEL_IDS=
```

They do not contain a meaningful subject, so `/agent setup` or
`DISCORD_CHANNEL_SESSIONS_JSON` is preferred.

`DISCORD_CREATE_THREADS` defaults to `false`. The Edge never creates a new
Discord thread merely because a user sends a message. Creating and organizing
channels is a Discord-side operation controlled by the user.

## Ordinary conversation and execution

The same channel handles consultation and action:

```text
User: 次に試す候補を考えて
Agent: compact proposal with expected effect, change, risk, cost and success condition

User: P-021を実装して試して
Agent: implements/repairs the experiment, performs a smoke test, creates a durable Job and reports its Job ID

User: 結果を比較して。次はその案で進めて
Agent: records the human interpretation implied by the message, accepts the selected child hypothesis and creates the next Job
```

There is no `/plan`, `/start`, strategy mode, or execution mode in the active
Discord adapter. Internally, the user's ordinary message is still recorded as an
immutable human decision event before a Job is created. This retains auditability
without making the user operate a state machine.

Question-like messages do not start work merely because they mention an
experiment. Execution requires an explicit expression such as `試して`,
`実装して`, `回して`, `実行して`, or `この案で進めて`, plus a valid structured
Job proposal.

## Natural final actions

Kaggle submission is requested in ordinary chat:

```text
User: このCSVで提出しよう
```

The Core discovers validated submission candidates. A single unambiguous
candidate is bound to its exact SHA-256 and passed to the existing submission
pipeline. If multiple candidates exist, the Agent asks for a filename or SHA
prefix rather than guessing. Competition-rule acknowledgement, credential
isolation, duplicate suppression, and submission-history reconciliation remain
unchanged.

Research paper generation is also requested in ordinary chat:

```text
User: この結果を論文にまとめて
```

The latest or explicitly named result is recorded as the human-selected result,
then passed to the evidence-grounded paper pipeline. This generates local
artifacts and review history; it does not publish externally.

Paid compute, destructive operations, Codex command/file approvals, credential
changes, and external publication remain operational safety boundaries. They are
not bypassed by natural-language routing.

## Discord commands

The active command surface is deliberately small:

```text
/agent setup
/agent channel
/agent status
/agent finish
/agent codex_status
/agent steer
/agent interrupt
/agent codex_approvals
/agent codex_approval
/agent compute_backends
/agent approve_compute
/agent cancel_job
```

Hypothesis selection, result interpretation, Kaggle submission selection, and
paper preparation are expressed in normal messages instead of separate slash
commands.

## Response format

Visible responses are compact Discord Markdown. The Agent should normally use
2–4 short paragraphs, inline **bold** labels, and at most one short list. It must
not render every field as a separate label/value block.

Preferred:

```text
**次は P-021（CatBoost native categorical）を優先します。** LightGBMをCatBoostへ置き換え、CV-001は固定します。カテゴリ変数をnative処理できるため改善余地がありますが、高カーディナリティ列への過依存と過学習は要監視です。

**採用条件:** 現在best `0.8398`に対してCV `0.8430`以上。Kaggle Notebookで25〜40分を見込みます。進めるなら「これを試して」と返してください。
```

Rejected:

```text
期待:
カテゴリ変数の処理改善

変更:
LightGBM → CatBoost
```

A formatter also joins simple label/value pairs and removes repeated blank lines
outside fenced code blocks, so provider output remains readable in Discord.
