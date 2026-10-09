# Stirrup Agent

A NeMo Gym responses API agent that runs the [Stirrup](https://github.com/ArtificialAnalysis/Stirrup) agent loop.
It is benchmark-agnostic: the task prompt, the task tools and the sandbox come from the environment. GDPVal
(`benchmarks/gdpval/`) is the benchmark that uses it.

## How it works

Stirrup runs only in agent sessions, seeded by an environment server with two grants:

- `sandbox_access`: the task sandbox, owned by the resources server. At seed, the agent installs the server's Stirrup
  version in a venv outside the working directory and stages `sandbox_runner.py` with the modules it imports.
- One `direct_http` tool access: the resources server's routes and the session cookies.

The activation (`/ng-rollout/<id>/v1/responses`) runs `sandbox_runner.py` in the sandbox under Gym's process
supervisor. The runner builds Stirrup's messages from `responses_create_params.input` and gives the model:

- `code_exec`, which runs each command in a fresh shell in the working directory (`cd <workdir> && ( timeout -k 10
  300 bash -c ... )`), so files persist and shell state does not;
- the request's `tools`, each posted to `{base_url}/{name}` with the session cookies. A 400 or 422 answer goes back to
  the model; any other error fails the episode. The tools named in `finish_tool_names` end the episode.

The model server is reached directly from the sandbox, so its address must be reachable from there. The trajectory
comes back as the response's output items; verification belongs to the resources server.

## Configuration

`configs/stirrup_agent.yaml` is the stand-alone config. Keys:

| Key | Default | Meaning |
|-----|---------|---------|
| `model_server` | required | The policy model. |
| `agent_max_turns` | `250` | Turn cap for the agent loop. |
| `concurrency` | `32` | Maximum concurrent episodes. |
| `temperature` | `0.6` | Sampling temperature when the request sets none. |
| `finish_tool_names` | `null` | Request tools that end the episode when they succeed; `null` uses Stirrup's own finish tool. |
| `sandbox_install_timeout_seconds` | `900` | Bound on installing the runtime in the sandbox. |
| `sandbox_runner_timeout_seconds` | `12600` | Bound on one episode. |
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
3. Replicates upstream's response parsing but does **not** raise
   `ContextOverflowError` on `finish_reason=length`; the agent loop
   terminates normally via the `finish` tool or `max_turns`.

`completion_token_buffer` absorbs the residual gap between our estimate and the exact prompt the server renders
(chat-template wrappers, tool-schema injection). The default `1000` works in practice; raise it (e.g. 2000–5000) if
you see sporadic HTTP 400 responses at the vLLM proxy.

The character estimate can substantially overcount retained reasoning, so the client still dispatches the configured
completion floor; the hard completion cap always applies.

### Client and agent options

These are constructor arguments of `DynamicMaxTokensChatCompletionsClient`
(`nemo_client.py`) and `NeMoAgent` (`nemo_agent.py`).

| Argument | Default | Description |
|---|---|---|
| `min_completion_tokens` (client) | `1024` | Target per-call completion floor. The hard cap always applies, and the approximate estimate keeps this floor even when it exceeds the context. Agents that write a whole script or deliverable in one tool call may need a higher floor: a 1024-token completion cannot reliably hold one. |
| `prompt_estimator_truncate_history_thinking` (client) | `None` | Estimator-only. When `True`, the character estimate drops completed reasoning from assistant turns before the last user turn, as Nemotron's template does. It is never sent to the provider and never changes the recorded history. |
| `truncation_recovery` (client) | `False` | When `True`, a turn that uses its whole completion budget without a schema-valid tool call is followed by one turn that requests thinking off and appends a one-time instruction to act now. The instruction is sent to the model but not recorded in the trajectory. |
| `min_compaction_summary_words` (agent) | `1` | Minimum word count for a context-compaction summary, counted after a leading `<think>` block is stripped. The agent makes up to 3 summary attempts (the last one without tools) and raises instead of replacing the history with an unusable summary. |

Thinking control (`enable_thinking`, and the thinking-off turn of
`truncation_recovery`) is sent as `extra_body.chat_template_kwargs`. It only
takes effect if the model server passes request `chat_template_kwargs` through
to vLLM.

## Licensing

- **Code**: Apache License 2.0 (see repository `LICENSE`).
- **Dependencies**: `stirrup` (Apache 2.0). See `requirements.txt` and the top-level `pyproject.toml` for full
  attribution.
