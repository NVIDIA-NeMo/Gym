# Stirrup Agent

A NeMo Gym responses API agent that runs the [Stirrup](https://github.com/ArtificialAnalysis/Stirrup) agent loop.
It is benchmark-agnostic: the task prompt, the task tools and the sandbox come from the environment. GDPVal
(`benchmarks/gdpval/`) is the benchmark that uses it.

## How it works

Stirrup runs only in agent sessions, seeded by an environment server with these grants:

- `sandbox_access`: the task sandbox, owned by the resources server. Without it, the agent starts its own sandbox from
  `sandbox_provider` and `sandbox_config`. At seed, the agent uploads a Python runtime with the server's Stirrup
  version, built on the agent's host, to a directory outside the working directory, and stages `sandbox_runner.py`
  with the modules it imports. The sandbox needs no network access or Python of its own.
- At most one `direct_http` tool access: the resources server's routes and the session cookies. A request with `tools`
  needs it.

The sandbox must be x86_64 or aarch64 Linux with glibc 2.28 or newer, `bash`, and `tar` with gzip. musl images, such as
Alpine, and older glibc are not supported: `imageio-ffmpeg` publishes no musllinux wheel, and the Pillow version in
`pyproject.toml` publishes only manylinux_2_28 wheels. The agent's host builds the runtime with `uv` and caches it under
`sandbox_python` in Gym's `cache_dir`. Building needs the Python package index and the python-build-standalone
releases on GitHub; `HTTPS_PROXY` and `UV_PYTHON_INSTALL_MIRROR` apply, and a host that cannot download the Python
build can be given the file in that cache directory.

The activation (`/ng-rollout/<id>/v1/responses`) runs `sandbox_runner.py` in the sandbox under Gym's process
supervisor. The runner builds Stirrup's messages from `responses_create_params.input` and gives the model:

- `code_exec` (unless the config sets `code_exec: false`), which runs each command in a fresh shell in the working
  directory (`cd <workdir> && ( timeout -k 10 300 bash -c ... )`), so files persist and shell state does not;
- the request's `tools`, each posted to `{base_url}/{name}` with the session cookies. A 400 or 422 answer goes back to
  the model; any other error fails the episode. The tools named in `finish_tool_names` end the episode.

The model server is reached directly from the sandbox, so its address must be reachable from there. The trajectory
comes back as the response's output items; verification belongs to the resources server.

An HTTP 400 or 413 with the structured error code `context_length_exceeded` reaches Stirrup's existing context
recovery. Other invalid requests (HTTP 400, 413 or 422) end the episode there. The response carries the turns
completed so far, with `status: failed`, an `error` and
`metadata.stop_reason: model_error`, and the resources server verifies it. Any other model server error fails the
episode.

## Configuration

`configs/stirrup_agent.yaml` is the stand-alone config. Keys:

| Key | Default | Meaning |
|-----|---------|---------|
| `model_server` | required | The policy model. |
| `agent_max_turns` | `250` | Turn cap for the agent loop. |
| `concurrency` | `32` | Maximum concurrent episodes. |
| `temperature` | `0.6` | Sampling temperature when the request sets none. |
| `finish_tool_names` | `null` | Request tools that end the episode when they succeed; `null` uses Stirrup's own finish tool. |
| `code_exec` | `true` | Whether the model gets the `code_exec` shell tool. Off when the request's tools are the only tools the task allows (EnterpriseOps-Gym). |
| `sandbox_install_timeout_seconds` | `900` | Bound on building, uploading and installing the runtime at seed. |
| `sandbox_runner_timeout_seconds` | `12600` | Bound on one episode. At the limit the response carries the turns completed so far, with `status: incomplete` and `metadata.stop_reason: wall_time`. |
| `session_close_timeout_seconds` | `30` | Bound on each cleanup phase at close. |
| `completion_token_buffer` | `1000` | Safety margin (in tokens) reserved when sizing `max_completion_tokens` per call. |

## Dynamic `max_completion_tokens` sizing

Stirrup's `ChatCompletionsClient` sends a static
`max_completion_tokens = self._max_tokens` on every call.  For long-context
models (Ultra V3, Qwen3-Coder-30B's 131K, etc.), this can exceed
`max_model_len − prompt_tokens` once the prompt grows, and the server
returns an HTTP 400 (or `finish_reason=length` with zero output) that the
agent cannot recover from.

The wrapper ships a `DynamicMaxTokensChatCompletionsClient`
(`nemo_client.py`) that, on every request:

1. Estimates the input tokens from a character count of the whole serialized history and tool schemas, tool
   calls included.
2. Computes `max_completion_tokens = context_window − input_tokens − completion_token_buffer`.
3. Retains partial content from successful responses with `finish_reason=length`; that finish reason alone does
   not discard history or raise `ContextOverflowError`.

`completion_token_buffer` absorbs the residual gap between our estimate and the exact prompt the server renders
(chat-template wrappers, tool-schema injection). The default `1000` works in practice; raise it (e.g. 2000–5000) if
you see sporadic HTTP 400 responses at the vLLM proxy.

The character estimate can substantially overcount retained reasoning, so the client still dispatches the configured
completion floor; the hard completion cap always applies.

When a model request is refused with HTTP 400 or 413 and the structured code `context_length_exceeded`, the client
raises Stirrup's `ContextOverflowError`. It accepts the OpenAI error envelope and FastAPI's `detail` envelope;
error-message text alone is insufficient. Other errors, including invalid output-token parameters, retain their
original exception.

Stirrup's existing recovery removes the latest completed turn and its metadata, then retries the model call or
compaction. It preserves the original task and never crosses a prior summary boundary: at least one completed turn
must remain after that boundary, otherwise recovery raises and the episode fails. This can add model calls and
change the retained history. It does not change the configured output budget, thinking, or sampling settings, and
is separate from the opt-in `truncation_recovery` behavior below. If recovery cannot continue, the runner retains
its saved partial trajectory and observations and reports `status: failed`, `metadata.stop_reason: model_error`,
and a normalized 400 model-request outcome with a context-exhaustion message. It does not turn exhaustion into a runner
crash or a successful completion.

If all three compaction attempts return unusable summary text, the runner preserves the partial trajectory,
including the turn that triggered compaction, and reports `status: failed` with
`metadata.stop_reason: compaction_summary_exhausted`. This is an `agent_run_error` outcome with no upstream HTTP
status. The environment's default outcome policy leaves it unscored and does not retry it automatically. An explicit
environment policy can override that default. Unrelated implementation errors still fail the runner.

The response's `metadata.compaction_events` contains a JSON-encoded list of compaction attempt and overflow-recovery
events on both successful and failed episodes. Events record accepted/rejected status, fixed rejection categories,
response-field lengths, tool-call counts, token usage, and available response diagnostics. They exclude prompts,
answers, reasoning text, tool arguments, and provider error bodies. These diagnostics expose what the summarizer
returned without changing its configured output budget, thinking, or sampling settings.

### Client and agent options

These are constructor arguments of `DynamicMaxTokensChatCompletionsClient`
(`nemo_client.py`) and `NeMoAgent` (`nemo_agent.py`).

| Argument | Default | Description |
|---|---|---|
| `min_completion_tokens` (client) | `1024` | Target per-call completion floor. The hard cap always applies, and the approximate estimate keeps this floor even when it exceeds the context. Agents that write a whole script or deliverable in one tool call may need a higher floor: a 1024-token completion cannot reliably hold one. |
| `prompt_estimator_truncate_history_thinking` (client) | `None` | Estimator-only. When `True`, the character estimate drops completed reasoning from assistant turns before the last user turn, as Nemotron's template does. It is never sent to the provider and never changes the recorded history. |
| `truncation_recovery` (client) | `False` | When `True`, a turn that uses its whole completion budget without a schema-valid tool call is followed by one turn that requests thinking off and appends a one-time instruction to act now. The instruction is sent to the model but not recorded in the trajectory. |
| `min_compaction_summary_words` (agent) | `1` | Minimum word count for a context-compaction summary, counted after a leading `<think>` block is stripped. The agent makes up to 3 summary attempts (the last one without tools). Exhaustion preserves the history and returns an explicitly failed, unscored outcome under the default environment policy. |

Thinking control (`enable_thinking`, and the thinking-off turn of
`truncation_recovery`) is sent as `extra_body.chat_template_kwargs`. It only
takes effect if the model server passes request `chat_template_kwargs` through
to vLLM.

## Licensing

- **Code**: Apache License 2.0 (see repository `LICENSE`).
- **Dependencies**: `stirrup` (Apache 2.0). See `pyproject.toml` and the top-level `pyproject.toml` for full
  attribution.
