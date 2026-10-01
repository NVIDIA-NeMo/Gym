# OpenCode Agent

The canonical OpenCode package supports native EnvironmentServer sessions, local guest
runners, and an explicit compatibility mode for legacy Resources lifecycles.

Native sessions use ordinary sandbox `exec`, not a PTY. The supervisor enforces the
runner deadline and confirms descendant cleanup before verification; uncertain
cleanup remains retryable and blocks verification. The task image needs Linux,
Bash and Python 3.8+; the installer selects a glibc or musl OpenCode release binary.

`context_window` and `max_output_tokens` are independent native settings. For a
bounded 262144-context smoke, use `max_output_tokens: 32768`; this is a per-call
budget, not a full-benchmark recommendation. A full-context output setting requires
a matching model-server policy that leaves room for the prompt, as in the historical
vLLM sampling overrides. Do not copy it unchanged to a generic OpenAI endpoint.

See the [OpenCode wiring and migration guide](../../fern/versions/latest/pages/evaluation/opencode.mdx)
for configuration, runtime requirements, and lifecycle guarantees.
