# Codex App Server v2 conformance

ResearchAgent sends text input with the generated v2 `UserInput` field
`text_elements`. Discord continues to use the existing App Server thread,
turn, steering, interruption, approval, and event-forwarding implementation.
Codex collaboration/subagent items remain owned by Codex Core/Harness and are
forwarded as App Server events without a ResearchAgent-specific subagent
protocol.
