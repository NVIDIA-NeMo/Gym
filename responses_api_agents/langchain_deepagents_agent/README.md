# LangChain DeepAgents Agent

Runs a [LangChain DeepAgents](https://python.langchain.com/docs/experimental/deep_agents/) agent as a
native NeMo Gym agent server.

`app.py` is the shared entrypoint for both tool profiles of `DeepAgentsAgent`:

- `tool_profile: builtin` runs deepagents' built-in file and shell tools inside a task sandbox.
- `tool_profile: tavily_search` preserves the in-process Tavily search graph used by the bundled reasoning_gym
  and tavily_search configurations. It requires `tavily_api_key`.

Both profiles use the same session endpoints. `sandbox.py` installs and manages the sandbox session, and
`sandbox_runner.py` runs the graph inside the sandbox. `reasoning_search_agent.py` preserves the old search class
names and entrypoint for existing callers; all bundled configs now select `app.py`. Custom in-process graphs can
still override `build_agent(model)` using the search-profile configuration.

The adapter runs deepagents 0.6.12 (`deepagents<0.7.0` in `pyproject.toml`, resolved in `uv.lock`) at the Evaluation
capability level: it returns responses for verification, and its close bundles record the missing invocation
and tool-call evidence as gaps.

## Quick start

### env.yaml

Create `env.yaml` in the repo root if you don't already have one (see [Quickstart](https://docs.nvidia.com/nemo/gym/main/get-started/quickstart)),
and add the following keys. `policy_*` drives the deepagents agent itself — required for both benchmarks below:

```yaml
policy_api_key: sk-or-...
policy_model_name: nvidia/nemotron-3-ultra-550b-a55b:free
tavily_api_key: <YOUR_KEY>
```

**Running the `tavily_search` benchmark only** (skip this for `reasoning_gym`): add these keys to the
same `env.yaml` too. They configure an LLM judge that the *resources server* uses to grade the agent's
final answer against ground truth at verification time — a separate model from `policy_*` above, which
the agent itself never talks to:

```yaml
exclude_domains_file_path: resources_servers/tavily_search/tests/dummy_exclude_domains_file.json
search_judge_model_base_url: https://openrouter.ai/api/v1
search_judge_model_api_key: sk-or-...
search_judge_model_name: anthropic/claude-sonnet-4.5
```

`exclude_domains_file_path` points at a JSON list of domains to exclude from search results; the path
above is a test fixture usable as-is for a quick run — swap in your own list for real evals.

### Launch

This agent always composes with `--model-type`, since it has no provider credentials of its own — it
reaches `model_server`, not a provider directly. Two ready-to-run combo configs bundle a resources server
with this agent:

```bash
# reasoning_gym
gym env start --resources-server reasoning_gym/reasoning_gym_langchain_deepagents_agent_model_server \
    --model-type inference_provider/openrouter

# tavily_search
gym env start --resources-server tavily_search/tavily_search_langchain_deepagents_agent_model_server \
    --model-type inference_provider/openrouter
```

To use this agent with an existing resources-server config, select the generic
[`configs/langchain_deepagents_agent.yaml`](configs/langchain_deepagents_agent.yaml) via `--agent-type`.
For example:

```bash
gym env start --resources-server reasoning_gym \
    --model-type inference_provider/openrouter \
    --agent-type langchain_deepagents_agent
```

Replace `reasoning_gym` with your resources-server config. Gym replaces its bundled agent with DeepAgents
and updates the existing environment server's agent reference. Leave `resources_server.name` in the
generic agent config unset (`???`): overriding it disables this automatic composition and can leave an
extra agent without an environment server, causing configuration validation to fail.

### Run the agent

For the prebundled `reasoning_gym` combo above, use the command below. If you launched with the generic
`--agent-type` example instead, use `--agent reasoning_gym_langchain_deepagents_agent`.

```bash
gym eval run --no-serve --agent reasoning_gym_langchain_deepagents_agent_model_server \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output /tmp/langchain_deepagents_agent_rg.jsonl
```

Expect `mean/reward: 1.0` over those 5 rows — `reasoning_gym` scores deterministically by extracting the
`<answer>` tag, so there's no judge noise in the number.

## Environment Server and agent sessions

Both bundled search configs select `app.py` with `tool_profile: tavily_search` behind `single_agent_turn_legacy`,
so the commands above are unchanged.
The Environment Server seeds the resources server, opens an agent session, runs the graph once, closes the
session, and then calls `/verify`. The agent's side of that contract:

- **Seed** refuses any *required* tool grant with HTTP 422, because the graph runs only its own tools; an
  optional grant is skipped. Keep `resources_tool_transports: []`, including on a `single_agent_turn_legacy`
  Environment Server you swap this agent into with `--agent-type`.
- **Task sandbox.** Seed refuses `sandbox_access` with HTTP 422: the graph runs in the agent server, its files
  live in graph state, and it has no `execute` tool. Use the [builtin profile](#sandboxed-execution-tool_profile-builtin)
  for a benchmark that lends its task sandbox.
- **Activation** runs the graph once per session. An identical retry joins that run, and a different request or
  a request for another rollout answers 409. A client that disconnects does not cancel the run.
- **Output budget.** Both profiles reject a non-null request `max_output_tokens` with HTTP 422 before starting or
  binding an activation: the graph cannot enforce a total across its model calls. Remove that request field and
  configure the Model Server's per-call output limit instead; a corrected request can use the same session.
  The legacy search `/run` route rejects this field before seeding resources as well.
- **Close** cancels a graph that is still running. Close returns no resources cookie jar, so `/verify` receives
  the seed's cookies unchanged. It returns an observation bundle with the gaps `agent_transcript_unavailable` and
  `model_call_ownership_unavailable` (see [Open gaps](#open-gapslimitations)).
- **Request fields.** `input`, `instructions` (as a system message), and `temperature`, `top_p` and `reasoning`
  apply to every model call, on legacy `/run` as well. The request's `tools` are ignored: the graph calls only its
  own Tavily search. `concurrency` bounds only the builtin profile's sandbox runs.
- **Model errors.** A Model Server status that another episode attempt can change (408, 425, 429 or 5xx), or a
  connection failure, fails the activation, and the Environment Server runs the episode again. Any other HTTP
  error ends the session's run with a reply that has no output: `incomplete` with `metadata.stop_reason:
  context_limit` for the error code `context_length_exceeded`, which is verified, and otherwise `failed` with
  `model_request_error` and the HTTP status in `ng_agent_outcome`. Legacy `/run` raises every model error.
- The agent does not enforce a task's agent timeout (`metadata.agent_timeout_sec`), and it does not support
  rollout stops (`/rollouts/<id>/stop` answers 501).
- **`execute_tools: false`** is rejected when the config loads: the graph always runs its own tools.
- **Checkpointing** restarts an unfinished rollout from its input. The graph's state lives inside one
  `ainvoke()` call and cannot be exported.
- **`skip_verification: true`** skips `/verify` on the session path, including the Tavily judge. Set it on the
  Environment Server, or on the agent when the Environment Server leaves it unset.

## Sandboxed execution (`tool_profile: builtin`)

`DeepAgentsAgent` with `tool_profile: builtin` runs deepagents' own graph, `create_deep_agent()` with its built-in
tools (`write_todos`, `ls`,
`read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute` and `task` subagents), inside a task sandbox. It
adds no tools of its own, so it has no Tavily search. It runs only in agent sessions:
[`configs/langchain_deepagents_sandbox_agent.yaml`](configs/langchain_deepagents_sandbox_agent.yaml) selects `app.py` with
`tool_profile: builtin` and `resources_server: null`, and an Environment Server composes it with a benchmark.
For Terminal-Bench 2.1, follow
[Configure a run](../../benchmarks/terminal_bench_2_1/README.md#configure-a-run) with this config import, the agent
name `langchain_deepagents_sandbox_agent` and `resources_tool_transports: []`. The benchmark's `allowed_agents` does
not list this agent yet, so start its servers with `--allow-unsupported-pairing`.

- **Sandbox.** The session runs in the sandbox the resources server lends (`sandbox_access`, direct connections), or
  in one the agent starts from `sandbox_provider` with an absolute `sandbox_config.workdir` and stops at close. Seed
  refuses with HTTP 422 a session with neither, a `sandbox_access.user` override, a relative `workdir`, a *required*
  tool grant (an optional one is skipped; keep `resources_tool_transports: []`), and a loopback Model Server URL
  for a provider whose sandboxes cannot reach the host's loopback.
- **Runtime.** Seed installs a private Python with deepagents and its dependencies at this server's versions
  (`SandboxPythonRuntime`) under `/tmp/nemo-gym-deepagents-runtime-<key>`, and reuses one that an earlier session
  left there. The sandbox needs x86_64 or aarch64 Linux (glibc 2.17 or newer, or musl), `bash` and `tar`, but no
  Python and no network access to a package index. The agent server needs `uv` and access to the package index and
  the python-build-standalone releases; `sandbox_install_timeout_seconds` bounds the install.
- **Model calls** go from the sandbox to the Model Server at the rollout's URL, so the sandbox must reach the Model
  Server: set `use_absolute_ip: true` for a sandbox outside the host's network. An HTTP 408, 425, 429 or 5xx, or a
  connection failure, is retried twice after the server's `Retry-After` or an exponential backoff, at most 30 s; any
  other 4xx is not retried. A call that still fails ends the run: the response is `failed` and reports the cause in `ng_agent_outcome`
  (`model_request_error` with the HTTP status, or `model_transport_error`), so the Environment Server's outcome
  policy decides whether to verify or retry the episode.
- **Request fields.** `input`, `instructions` (as a system message), and `temperature`, `top_p` and `reasoning`
  apply to every model call. Request `max_output_tokens` is refused with HTTP 422 because no response-wide
  budget is enforced; configure a per-call limit on the Model Server instead. A request that declares `tools`
  is refused with HTTP 422, because the graph runs only its own tools.
  Other fields, such as `tool_choice` and `parallel_tool_calls`, are ignored.
- **Tools** work on the sandbox's filesystem. Commands run in `workdir` with the sandbox's environment plus the
  owner's `sandbox_access.env`; `HOME` is the task's. A command's default time limit is `shell_timeout_seconds`
  (120 s), and the model may pass its own. Each command runs in a session of its own with its output in files, so
  a background process it starts does not hold the command open, and survives as a task service unless the
  command times out, which kills its process group. deepagents' own files (offloaded conversation history, evicted
  large tool results) stay in graph state under `/.deepagents/`; the runtime and session files live under `/tmp`,
  outside `workdir`.
- **Time limit and stop.** The run's limit is `sandbox_runner_timeout_seconds`, or a task's shorter
  `metadata.agent_timeout_sec`. At the limit Gym's process supervisor stops the runner, which reports the messages
  finished so far: the response is `incomplete` with `metadata.stop_reason: wall_time`, and it is still verified. If
  no model reply had arrived by then, the Model Server never answered: the response is `failed` with
  `model_transport_error`, which the Environment Server retries. The runner writes its report before it waits for
  any tool thread, and a stop during its startup still leaves an empty report. A
  rollout stop (`POST /rollouts/<id>/stop`) ends the run as at the limit and adds `ng_rollout_stop`; while the graph
  runs, it needs a provider whose uploads never wait for running commands (Local, or OpenSandbox with background
  exec). A stop that the supervisor's receipt does not show as the limit, such as a task command `pkill python3`
  that reaches the runner, ends the run `failed` with `agent_run_error`. Reaching deepagents' step limit also
  returns `incomplete`, with `stop_reason: max_iterations`.
- **Activation** runs the graph once per session. An identical retry joins that run, and a different request or a
  request for another rollout answers 409. A client that disconnects does not cancel the run.
- **Close** stops the graph through the supervisor, then disconnects from a borrowed sandbox without stopping it:
  the resources server that lent it stops it after `/verify`. The observation bundle has a `sandbox` record whose
  outcome says how the run ended (`completed`, `failed`, `timeout` or `cancelled`; `oom` or `unknown` for a run that
  left no report after it launched), and the same two gaps as the in-process agent.
- **Usage** is the sum over the run's model calls when every call returned usage; otherwise it is unknown.

## Tool ownership: agent vs. resources server

Tools belong to the harness, not the resources server. The search profile declares its tools by
passing them into `create_deep_agent(tools=[...])` inside `build_agent()`. It wires `TavilySearch` this way,
calling the Tavily API directly with `tavily_api_key`. The builtin profile creates its graph in the sandbox runner.
The `tavily_search`
resources server it can pair with does define its own `web_search` endpoint, but this agent never calls it. Only its `verify()` endpoint is used for verification. In short, the agent works with any resource-server for
task data and verification functionality, but ignores other functionality of the resource server like tools. 

This is the default shape because tool definitions (schemas, call handling, framework-specific bindings
like `langchain_tavily.TavilySearch`) are inherently agent-harness concerns. They travel with the
framework you're wrapping. Keeping them in the agent server also lets the same
resources server pair with agent servers that expose the same capability differently (e.g. a different
search tool implementation, or a framework with its own native tool-calling conventions).

## LangChain ↔ Gym Responses conversion contract

Gym uses the OpenAI Responses API to describe what the agent server exchanges with the resources server and the model server. 
For this implementation there are 2 complexities:
1. `deepagents`/LangGraph knows nothing about that format: it runs on its own `BaseMessage` types (`HumanMessage`, `AIMessage`, `ToolMessage`, ...) held
as graph state
2. `deepagents` agent server is implemented with the DeepAgent "owning" the agent loop. In short, it drives its own internal tool/summarization/subagent etc loop. One `/v1/responses` call into this agent server
can trigger many internal LLM calls before the graph is done. But at the same time, all these calls need to be logged in the Gym rollout! 

Wrapping an unmodified `deepagents` graph
inside a Gym agent server means converting between these two message shapes every time one side talks to
the other. And since so much occurs in the internal agentic loop of the DeepAgent, it's not acceptable to merely track information once per request. Every time there is an internal model turn it must be tracked, too. The functions below describe this conversion layer.

One rollout crosses these representations:

```text
NeMoGym Responses input
  |
  |  to_langchain()             once per /v1/responses call
  v
LangChain message state
  |
  |  to_gym_input()             every internal model turn
  v
NeMoGym Responses request to the Model Server
  |
  |  to_langchain_ai_message()  every internal model turn
  v
LangChain AIMessage / ToolMessage state
  |
  |  to_responses()             once, after the graph completes
  v
Final NeMoGymResponse
```

Why each representation exists:

- `to_langchain()`: deepagents' graph only operates on LangChain's own `BaseMessage` types. So the incoming conversation is converted once, up front. Now the query can be sent to Langchain `.ainvoke` function.
- `to_gym_input()`: `GymResponsesChatModel` (deepagents' plugged-in "model") talks to the Model Server
  exclusively over the Responses API, and deepagents calls its model repeatedly inside one `/v1/responses`
  call as it runs its own internal tool loop — so the *current* message list is re-serialized on every one
  of those internal calls, not just the outer one.
- `to_langchain_ai_message()`: the graph can only keep looping on LangChain messages. It has no way to
  interpret a raw Responses response, so each Model Server reply is converted back before deepagents can
  decide its next tool call.
- `to_responses()`: the outer Gym protocol expects this agent server to hand back one Responses-shaped
  result, not a LangChain state object, so the whole accumulated message history is flattened back into a
  single `NeMoGymResponse` at the end.

### Data stored in representation shifts

For each seam, "preserved" means the data survives the hop unchanged; "lost" means it's dropped or replaced
with a synthesized value.

This occurs because there is not a perfect 1:1 mapping between Langchain message format and Responses API.

| seam | preserved | lost |
| --- | --- | --- |
| `to_langchain()` | `message`, `function_call`, `function_call_output`, `call_id` | unrecognized item types (silently); per-item `id`; `developer` vs `system` distinction |
| `to_gym_input()` | text content, tool calls, reasoning items (round-tripped) | `AIMessage.id`, `response_metadata`, non-text content parts |
| `to_langchain_ai_message()` | text, tool calls, reasoning items (full item dict), response `id` | `usage_metadata` |
| `to_responses()` | full `function_call`/`function_call_output` trace, reasoning | per-turn response ids; response-level fields are synthesized |

### Known lossy points - will need to fixed/hardened in follow-up PRs
1. to_langchain() drops unknown item types silently, while to_langchain_ai_message() raises on them. Asymmetric and the input side should fail loudly too.
2. usage_metadata is not populated, so LangChain-side token accounting sees nothing. Forces deepagents onto its approximate counter, which is not the best solution. Note that rollouts do get total token usage, but
`usage_metadata` not being populated/lost can be a bug for deepagents and/or Langsmith. 
3. developer and system both map to SystemMessage and both come back as system. Responses API differentiates between `developer` from `system` but LangChain doesn't have this differentiation, so it 
gets lost. 
4. Response-level fields on the final NeMoGymResponse (id, tools, tool_choice) are synthesized, not
carried from the model server responses because they're not relevant since the schema assumes one request 
to response from the agent server. But each internal loop in the agent server generates these so there
is nothing to report at the end. This is a bug that should be handled more elegantly. 


## Context summarization and offloaded history
DeepAgents have summarization middleware that is automatically built in with `create_deep_agent()` function.
Evicted history goes to `/conversation_history/{thread_id}.md` and `thread_id` is the same as `rollout_id`.
Without explicitly providing a `thread_id`, deepagents creates a fresh session per event so the 
conversation is considered a fresh thread each time. Worth noting the agent can and does read_file that path mid-rollout to recover context.

### Max input tokens
Max input tokens is set by using the DeepAgents model profile and is left to be implemented by 
the DeepAgent you subclass. 

### Token accounting is approximate
The missing `usage_metadata` bug makes it so LangChain deepagents summarizationmiddleware uses 
`count_tokens_approximately()` which only looks at `messages`. This can drastically undercount token
usage, making summarization trigger far too late according to what `max_input_tokens` is set to. 

### Logical vs. effective history
The final `NeMoGymResponse` shows the full conversation becasue it has the message hsitory in the
graph state. However, when summarization is called the model server likely did not have the same
request - it gets the compacted version. 

## Open gaps/limitations

- **No trajectory/observability capture (yet).** This agent does not build a `TrajectoryRecord`
  (per-tool-call/model-call observability) — `responses()` always takes the plain `graph.ainvoke()` path.
  Reward/pass-fail scoring is unaffected either way (`verify` only reads `body.response.output_text`);
  what's missing is only the rich per-call detail Gym's own rollout-collection pipeline can attach to a
  trajectory. On the session path, close reports this as observation gaps instead of returning nothing.

- **Context-overflow errors don't reach deepagents' own retry path.** When the Model Server rejects a
  request for exceeding the model's real context window, `raise_for_status()` raises an aiohttp
  `ClientResponseError` — not LangChain's `ContextOverflowError` — so `deepagents`' built-in overflow-retry
  fallback (which catches that specific exception and retries with more aggressive compaction) never fires;
  a raw HTTP error propagates instead. In practice this is now mostly avoided in the first place:
  `max_input_tokens` is a required config field, and `deepagents`' `SummarizationMiddleware` triggers
  compaction at 85% of that threshold, so a well-configured rollout should compact well before hitting the
  provider's real limit — `max_input_tokens` wasn't defined for this agent at all when this gap was first
  reported, so that alone accounts for most of the risk. That's considered safe enough to ship as-is; the
  exception-type mismatch itself is still unfixed, so if compaction ever lags (e.g. `max_input_tokens` set
  too high relative to the real limit, compounded by the approximate token counting above), the failure
  mode is still an aiohttp `ClientResponseError` rather than a graceful retry. A session's run then ends
  `incomplete` with no output (see the search profile's **Model errors** above); the builtin profile keeps the
  messages finished before the error.

- **Known lossy points in the LangChain ↔ Responses translation layer** — see the table and list above:
  unrecognized input item types are dropped silently on one side but raise on the other, `usage_metadata`
  isn't populated, `developer` and `system` roles collapse into one, and several fields on the final
  `NeMoGymResponse` are synthesized rather than carried from any real model call. These are tracked as
  follow-up hardening, not blockers.
