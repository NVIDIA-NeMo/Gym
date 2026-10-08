# OpenCode Agent

The canonical OpenCode package supports native EnvironmentServer sessions, local guest
runners, and an explicit compatibility mode for legacy Resources lifecycles.

Native sessions use ordinary sandbox `exec`, not a PTY. The supervisor enforces the
runner deadline and confirms descendant cleanup before verification; uncertain
cleanup remains retryable and blocks verification. The task image needs Linux.
Session setup installs missing Python and Bash with apt-get or apk when running
as root; otherwise preinstall Python 3.8+ and Bash. Existing tools are preserved.
The installer selects a glibc or musl OpenCode release binary.

`context_window` and `max_output_tokens` are independent native settings. For a
bounded 262144-context smoke, use `max_output_tokens: 32768`; this is a per-call
budget, not a full-benchmark recommendation. A full-context output setting requires
a matching model-server policy that leaves room for the prompt, as in the historical
vLLM sampling overrides. Do not copy it unchanged to a generic OpenAI endpoint.

See the [OpenCode wiring and migration guide](../../fern/versions/latest/pages/evaluation/opencode.mdx)
for configuration, runtime requirements, and lifecycle guarantees.

Without `sandbox_access`, configure `sandbox_provider` and `sandbox_config`
(`SandboxSpec` fields such as `image`, `workdir`, and `ttl_s`) on the agent.
It creates a sandbox, runs there, and destroys it on close; the default workdir is
`/app`. A supplied access always wins, including its workdir; connection failure
never triggers a replacement or host execution. With neither access nor a usable
provider, setup fails. Verifiers that inspect task files must keep using a
Resources-owned sandbox, since agent-owned sandboxes are gone before verification.
Native sessions are selected by the session cookie, regardless of `execution_mode`.
Without an agent session, requests run the local CLI, including with the default config.
Invalid or closed sessions are rejected, never retried on the host. The older `local`
and `sandbox` mode values are compatibility aliases for this session-based routing;
only explicit `legacy_sandbox` selects the old Resources bridge. Local `/run` still
requires a Resources binding; direct `/v1/responses` does not.
