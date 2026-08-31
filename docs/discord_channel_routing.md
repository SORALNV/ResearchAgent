# Discord channel routing and human decision boundary

## Purpose

ResearchAgent uses one Discord-facing control plane for both domains. The
selected domain is determined by the Discord channel ID, not by guessing from
message text:

```text
Discord channel / thread
        ↓ strict ChannelDomainMap
Research domain       Kaggle domain
        \               /
          Project / WorkSession / Event / Steering
```

A parent text/forum channel can be mapped once and its child threads inherit
that domain. An exact thread ID mapping overrides its parent. Unmapped channels
fail closed and are not silently treated as Research.

## Configuration

Use either the compact map, the separate channel lists, or both:

```env
# Compact syntax
DISCORD_CHANNEL_DOMAIN_MAP=111111111111111111=research,222222222222222222=kaggle

# Equivalent split lists; comma-separated IDs are accepted.
DISCORD_RESEARCH_CHANNEL_IDS=111111111111111111
DISCORD_KAGGLE_CHANNEL_IDS=222222222222222222

# Durable router state. Keep this on persistent storage.
CONTROL_PLANE_DIR=runtime/control_plane
```

JSON is also accepted:

```env
DISCORD_CHANNEL_DOMAIN_MAP={"111111111111111111":"research","222222222222222222":"kaggle"}
```

Rules:

1. An exact channel/thread mapping wins.
2. Otherwise a child thread inherits its parent channel mapping.
3. A channel cannot be mapped to both domains.
4. `hybrid` is valid for a long-lived Project but not for Discord channel
   routing. A Discord conversation must have one active domain.
5. `DISCORD_CHANNEL_ID` remains a backward-compatible Research mapping only
   when no explicit domain mapping exists.
6. Unknown channels raise `UnmappedDiscordChannelError`.

## WorkSession binding

`DiscordThreadRouter` creates a deterministic, domain-scoped Project for each
mapped parent channel unless the caller supplies a Project ID. It then creates
one WorkSession per Discord thread/forum post/conversation and persists the full
Discord route in `external_ref`.

Incoming messages are stored as immutable `discord.message.received` control
events. Discord retries reuse the original event through an idempotency key.
Domain handlers receive that event and must use its `event_id` as their own
correlation/idempotency key.

```python
from harness.control_plane import Domain
from harness.discord_thread_router import (
    DiscordChannelDispatcher,
    DiscordLocation,
    DiscordThreadRouter,
)

router = DiscordThreadRouter.from_environment()
dispatcher = DiscordChannelDispatcher(
    router,
    {
        Domain.RESEARCH: research_handler,
        Domain.KAGGLE: kaggle_handler,
    },
)

result = dispatcher.dispatch_message(
    DiscordLocation(
        guild_id="999",
        channel_id="333",        # thread ID
        parent_channel_id="222", # mapped Kaggle parent
        thread_id="333",
    ),
    message_id="444",
    actor_id="555",
    text="この特徴量仮説を試したい",
    title="Kaggle: feature hypothesis",
)
assert result.domain == Domain.KAGGLE
```

The dispatcher does not fall back from a missing Kaggle handler to a Research
handler. That would mix state and is rejected with `MissingDomainHandlerError`.

## Human responsibility boundary

There are exactly three research-direction decisions per domain.

| Domain | Human-owned decision 1 | Human-owned decision 2 | Human-owned decision 3 |
|---|---|---|---|
| Research | AIと相談して何を試すか選ぶ | 実験結果を解釈する | 論文としてまとめるか決める |
| Kaggle | AIと相談して何を試すか選ぶ | 実験結果を解釈する | そのsubmissionを提出してよいか決める |

The Agent owns the remaining research work: public-information investigation,
implementation, tests, error repair, smoke tests, experiment execution, logs,
comparison, counterarguments, artifact/checkpoint management, and proposing the
next hypothesis. In Kaggle mode it may validate and present a submission
candidate but may not submit it. In Research mode it may prepare a paper draft
only after the human paper decision.

This table describes **research-direction ownership**. Existing operational
safety gates remain separate: destructive operations, credential exposure,
paid compute, external publication, and Computer Use must still obey their
security/approval policy. Removing those gates would make unattended operation
unsafe.

## Enforced gates

`HumanResponsibilityPolicy` maps controlled actions to immutable human decision
events:

```text
start_experiment      -> hypothesis
continue_from_result  -> result_interpretation
submit_kaggle         -> kaggle_submission
start_paper_draft     -> research_paper
```

A Bot or Agent cannot satisfy these gates. Decisions are scoped to one
WorkSession and one `subject_ref`, so approval for one hypothesis/result cannot
be reused for another.

Kaggle submission approval is additionally bound to the exact file SHA-256:

```text
subject_ref = sha256:<64 lowercase hex characters>
```

Changing even one byte produces another hash and invalidates the old approval.
The latest human verdict for the same subject wins, so a later `reject` or
`defer` withdraws an earlier `accept`.

## Integration contract for Discord Edge

The Discord Edge should perform these steps for each message:

1. Build `DiscordLocation` from guild, channel/thread, and parent IDs.
2. Call `DiscordChannelDispatcher.dispatch_message`.
3. Pass `result.ingress.route` to the selected Research or Kaggle handler.
4. Use `result.correlation_id` for handler-side idempotency.
5. Record hypothesis, interpretation, final submission, or paper decisions with
   `record_human_decision` only when `message.author.bot` is false.
6. Before a controlled action, call `check_human_gate` with the exact subject.
7. Continue to apply the existing safety, checkpoint, review, cancellation, and
   artifact-promotion mechanisms.

No Discord token, Kaggle token, OpenAI key, or other credential is stored in the
channel map, Project, WorkSession, or decision events.
