# OpenCode Agent

The canonical OpenCode package supports native EnvironmentServer sessions, local guest
runners, and an explicit compatibility mode for legacy Resources lifecycles.

Interactive environments can seed this adapter with `continuation.mode: native_conversation`
and call `/v1/agent_sessions/activate` with ordered activation IDs starting at zero.
Each request contains only the next user-message delta. OpenCode resumes the same
native session ID, database, HOME, and caches; the adapter returns new output and
usage for that activation. Identical retries replay their receipt. Close returns
cumulative conversation evidence after all agent execution has stopped.

Use `configs/opencode_interactive.yaml` as the independent minimal definition; the
composition supplies the exact runtime, model profile, budgets, and permissions.
The harness settings remain independent of benchmark prompts and stopping policy:

- Pin `opencode_version` to the required cohort; continuation does not change the existing default.
- Set `native_model_id` to the actual provider model ID to preserve model-specific harness behavior.
- Preserve the provider's catalog identity with `native_provider_id` where required.
- Use `native_env` for explicit harness flags such as a virtual VCS selection. Adapter-owned
  storage, route configuration, update policy and output limits cannot be overridden this way.
- `native_session_title: null` retains native title generation; `native_auxiliary_model: native_default`
  retains native catalog selection for title/summary calls. A fixed-model gateway must explicitly
  document any substitution of those auxiliary models. `native_load_project_config` controls
  native project configuration discovery. The minimal definition preserves these native defaults.
- Select `native_provider_npm` for the configured Gym model route, and use `native_model_options`
  for reasoning capability, interleaved reasoning, variants, or provider options.
- The minimal definition uses `native_model_catalog: native` to inherit catalog limits and
  capabilities from the pinned executable. `configured` retains explicit `context_window` /
  `max_output_tokens` registration and the compatibility reasoning-content field.
- `native_output_token_max` independently controls OpenCode's experimental per-call output
  cap. `null` preserves the runtime default, an integer sets an explicit cap, and `model_limit`
  retains the existing behavior of copying `max_output_tokens`. The minimal definition uses
  `null`: a catalog output limit does not imply that native requests use that entire limit.
- `reasoning_effort` selects an explicit variant. Configure `reasoning: true` and the matching
  entry in `native_model_options.variants`; unsupported profiles fail during setup.
- Session seeds may carry `runtime_policy` with format `harbor.agent-kwargs.v1` and
  `settings.disallowed_tools` as the original comma-separated string. The adapter preserves
  that format's upstream translation into nested `permission.tools`; this is compatibility
  metadata, not a claim that every native version enforces those nested entries. Unknown
  formats/settings fail before setup. The composition's sandbox egress policy remains independent.
- For offline runtime setup, supply `local_opencode_binary_path` and its required
  `local_opencode_binary_sha256`. The adapter uploads and verifies the binary before
  checking its exact version inside the task sandbox. Alternatively provide a keyless
  `prefetched_opencode_binary_url` serving gzip bytes and `prefetched_opencode_binary_sha256`
  for the decompressed executable. That artifact host must be allowed by the sandbox's
  network policy. Both paths verify the installed runtime and keep setup outside the task repository.
- On images without Python 3.8+, configure `python_runtime_url` and `python_runtime_sha256`
  with a pinned standalone Python install-only archive. The adapter independently prepares
  its supervisor prerequisite outside the task repository; no Resources installation is assumed.
- For restricted network environments, set `prefetched_ripgrep_url`, `prefetched_ripgrep_sha256`
  (the release tar.gz digest), and `ripgrep_version` to the utility release required by the
  pinned OpenCode runtime. Setup preserves a system `rg` when present; otherwise it verifies
  and places the executable in OpenCode's persistent private cache before any activation.

`timeout` limits each supervised activation. Optional `session_execution_timeout_seconds`
also caps wall time from the first activation, including waits between user turns;
expired sessions retain their checkpoint and refuse to launch more work. The
interactive environment can also require an immutable UTC interaction deadline. It is checked
before native dispatch, after waiting for capacity, and after staging input and the supervisor.
It covers simulator waits as well as native work; clocks across components must be synchronized.
The outer episode deadline also covers setup and verification. New activations cannot start until the previous process and its
tools are confirmed stopped. Per-activation native event ordering, raw logs,
visible text, reasoning, tool evidence, and native step counts are preserved.
The raw-log fallback excludes dedicated reasoning events, which remain in typed evidence.
Provider-call counts remain a separate captured-model measurement.
Only native `length` termination is a model budget limit. An explicit tool-permission rejection
with a valid checkpoint and confirmed cleanup returns an incomplete `permission_denied` activation,
allowing the caller to supply feedback and resume the same session. The failed tool remains visible
and permissions stay unchanged. Other unfinished tool turns or content-filter results fail the
activation; a zero process exit code alone does not make it a healthy turn.
Failed activations return a bounded structured HTTP 502 detail with the native session,
finish reason, process outcome, and visible event previews. Transport metadata and dedicated
reasoning events are omitted; close still retains cumulative failed invocation/tool evidence.

Native sessions use ordinary sandbox `exec`, not a PTY. The supervisor enforces the
runner deadline and confirms descendant cleanup before verification; uncertain
cleanup remains retryable and blocks verification. The task image needs Linux,
Bash and Python 3.8+ (preinstalled or bootstrapped as above); the installer selects a
glibc or musl OpenCode release binary.

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
That setting still controls unseeded local/legacy calls; `sandbox` rejects unseeded calls.

With `observability_enabled: true` and a Gym `model_server`, the collector retains captured provider prompts and answers. Local execution can build canonical turns from invocation-owned captures; interactive execution retains native SQLite provider-turn boundaries. HTTP failures remain invocation-owned attempts when session identity is available. Bash exit codes determine execution success even when OpenCode records the tool lifecycle as completed.

Interactive close receipts retain cumulative native provider-turn boundaries, including failed activations with readable checkpoints. Turns preserve native session/message IDs; a replay does not duplicate them. Model capture can attribute official OpenCode requests by its `x-session-affinity` header when transport proxies preserve it. Conflicting session-header aliases remain unattributed. Official releases do not send the patched binary's assistant-message header, so per-assistant HTTP joins and full provider questions stay unavailable rather than being inferred from timing or text.
