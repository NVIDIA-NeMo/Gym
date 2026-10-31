# Agentic Turn Simulation Agent

A `simple_agent` that plays one whole agent turn for the
[`agentic_turn_simulation`](../../resources_servers/agentic_turn_simulation/README.md) resources server, which answers
the tool calls and scores the turn. It inherits `simple_agent`'s `/run` and `/v1/responses` and replaces only the
episode loop.

## What the episode does differently

- **The row's harness.** At the start of the turn the agent asks the resources server's `/harness`, with the session
  the row was seeded with, for the row's stop tools and tool protocol (native function calls, or Cline's XML). A bare
  `/v1/responses` call without a seeded session uses the config's `stop_tools` and native calls. A 404 for a seeded
  session (the call reached another resources-server worker under `num_workers` > 1, or the server restarted) fails
  the rollout instead of running a Cline row as native; so does any other `/harness` error or an unknown protocol.
- **Every tool call goes to `/simulate_tool_call`**, with its `call_id`, `name`, raw `arguments` (parseable or not)
  and the turn so far without reasoning: the simulator needs that context. A successful reply's output is the tool's
  result, including an error the simulated tool reports (a missing file, a failed command), which is judged as usual.
  An error reply from the resources server (not 2xx or 3xx) is retried, `simulate_max_attempts` times in all, after
  1, 2, ... seconds. If every attempt fails, the tool result is the resources server's simulator-failure output, so
  `/verify` masks the turn as `simulator_failed` instead of judging a turn that got an error page as a tool result.
- **What ends the turn** (`response.metadata.stop_reason`, with `steps`, `elapsed_seconds` and the per-call
  [token usage](#token-usage) `model_calls`):

  | `stop_reason` | When | How the server scores it (defaults) |
  |---|---|---|
  | `final_message` | Native rows: a reply without a tool call | judged |
  | `empty_output` | Native rows: a reply with neither a tool call nor an assistant message (reasoning only) | 0 (`unfinished_turn`); judged when it is a failed request's ([which](../../resources_servers/agentic_turn_simulation/docs/design.md#masking)) |
  | the stop tool's name | A call to one of the row's stop tools. It never runs; the reply's other calls run first and their results are kept (Cline rows: only the calls before it, as Cline runs them in order) | judged |
  | `mistake_limit` | Cline rows: `cline_max_consecutive_mistakes` replies in a row without a tool call | judged |
  | `max_steps` | The step cap, checked before each model call | 0 (`unfinished_turn`) |
  | `time_cap` | The wall-clock cap, checked before each model call | masked |
  | `context_overflow` | The history no longer fits the policy's context | masked at `steps` 0 (the row does not fit), else 0 |
  | `max_output_tokens` | A cut-off reply with a tool call or nothing visible, once the budget cannot grow | 0 (`unfinished_turn`) |

- **Output budget.** A reply cut off by its output budget (the request's `max_output_tokens`) is discarded and retried
  with double the budget, up to `max_output_tokens_cap`: its arguments may be cut, and running them would score the
  budget, not the policy. A reply is cut off when the model server marks it incomplete or its output tokens reach the
  budget: vLLM reports a reply cut inside a tool call as `tool_calls`, not `length`, after its parser drops the
  unfinished argument. A reply that ended on exactly its last allowed token looks the same and is retried too. Once
  the budget cannot grow (at the cap, after it shrank, or with no `max_output_tokens` set), a cut-off reply that shows
  only text is kept as a step (on Cline rows the XML calls it completed run); one with a tool call, or with nothing
  visible, ends the turn as `max_output_tokens`, its calls unrun. When the prompt fits but prompt plus budget does
  not, the budget shrinks to what fits (and never grows again). Retries are not steps. Shrinking needs the overflow
  error from the model server (`propagate_context_overflow_errors: true` on a `vllm_model` policy; with Gym's default,
  false, an overflow comes back as an empty cut-off reply). Any other model-server error fails the rollout.

## Token usage

`response.usage` sums the usage of the turn's kept replies (its steps), `null` when no reply was kept. These are the
model calls the rollout's trajectory references, so the usage matches Gym's captured calls (the
`rollout_token_count_mismatch` health check compares the two). Discarded replies spent tokens too:
`response.metadata.model_calls` lists every reply, discarded ones included, one by one, in order, as a JSON string
(Responses metadata values are strings):

```json
[{"input_tokens": 9120, "output_tokens": 4096, "cut_off": true, "kept": false},
 {"input_tokens": 9120, "output_tokens": 5211, "cut_off": false, "kept": true}]
```

- `input_tokens`, `output_tokens`: the reply's usage, `null` when the model server reported none.
- `cut_off`: the output budget cut the reply off: the model server marked it incomplete, or its `output_tokens`
  reached the requested `max_output_tokens`.
- `kept`: the reply became a step. A cut-off reply that was retried with a larger budget, or that ended the turn as
  `max_output_tokens`, was not.

A model-server error returns no reply and is not listed (a context overflow that shrinks the budget, for one).

## Cline's XML protocol

For `protocol: cline_xml` rows the model writes tool calls as XML in its reply text
(`<read_file><path>src/app.ts</path></read_file>`), parsed with the resources server's `cline_xml` rules. Each call goes
to `/simulate_tool_call` in order as `xml_<step>_<k>`, and the results return together as the next user message, as
Cline sends them. A reply without a tool call gets Cline's own "no tools used" message and the latest
`<environment_details>`, and the model tries again.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `resources_server` | required | The `agentic_turn_simulation` resources server. |
| `model_server` | required | The policy model server. |
| `max_steps` | `50` | Model steps per turn. `0` disables. |
| `max_time_seconds` | `7200` | Wall-clock cap per turn. `null` or `0` disables. Lower it for training. |
| `stop_tools` | `[ask_user]` | Stop tools when the resources server has no session for the request. |
| `max_output_tokens_cap` | `65536` | Ceiling for doubling the output budget. `0` disables doubling. |
| `min_output_tokens` | `2048` | Smallest shrunk budget worth retrying with. |
| `context_margin_tokens` | `256` | Tokens kept free when shrinking the budget to fit the context. |
| `cline_max_consecutive_mistakes` | `3` | Cline rows: replies in a row without a tool call before the turn ends. |
| `simulate_max_attempts` | `3` | `/simulate_tool_call` attempts per tool call before its result becomes a simulator failure (masked). |

Inherited: `skip_verification`, `skip_verification_reward`, `token_id_capture` (set `true` for training).

Run it through this environment's config (`gym env start --resources-server agentic_turn_simulation`), not alone:
every episode needs the session `/seed_session` opens on the resources server, so Gym agent sessions are refused.

## Tests

```bash
gym env test +entrypoint=responses_api_agents/agentic_turn_simulation_agent
```

## Licensing information

Code: Apache 2.0

Dependencies

- nemo_gym: Apache 2.0
