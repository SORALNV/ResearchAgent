# Codex App Server protocol conformance note

ResearchAgent uses the generated Codex App Server v2 request shapes.  The
runtime sends text inputs as `text_elements` and does not send removed legacy
fields such as `persistExtendedHistory` on `thread/start` or `thread/resume`, or
`ephemeral` on `thread/resume`.

This file exists as a review-visible guardrail for the schema-conformance fix.
The authoritative protocol remains the generated types in `openai/codex`.
