# Sandboxed Claude Code

Run one Claude Code turn in a sandbox supplied by an Environment or Resources Server. The adapter reconnects through `SandboxAccess`, copies an already provisioned executable into a private session directory, verifies its version, and supervises its descendants until cleanup is confirmed.

Import `configs/claude_code_sandboxed.yaml` and set `model_server` to a Gym model server exposing Anthropic Messages. Supply `executable_path` and `python_executable` in the borrowed sandbox; optional `executable_sha256` checks the supplied binary before copying it. The example pins Claude Code 2.1.108 with 50 turns and uses `/opt/gym-runtime`. Provision the complete Python installation before seeding the agent session.

Seed `/v1/agent_sessions`, call `/v1/responses`, then close `/v1/agent_sessions/close`. Input may be a text string or one user message preceded by system/developer text instructions. Request `instructions` remain system instructions. Unsupported history, media, external tool policies, and per-request model controls fail validation; configure model and execution settings on the agent or model server. Identical response retries share one execution; a different second request conflicts.

`metadata.timeout_seconds` can reduce the configured process timeout. `metadata.deadline_unix_seconds` bounds execution by an absolute deadline, including input staging. The seed request's `setup_deadline_unix_seconds` bounds runtime checks. Shared supervision confirms descendant cleanup before the close receipt; failed cleanup retains the borrowed session for retry.

Close returns `agent_observations` using the existing Claude native transcript parser, including native invocation identity, tools, conversation records, and compaction evidence. Missing transcripts produce an explicit observation gap. The raw CLI log and checked runtime version remain response metadata.
