# Two ways to run the same agent in NeMo Gym: `remote_agent` vs. in-tree

This compares two working implementations of the same LangChain `deepagents` graph — same model, same
TavilySearch tool, same `<answer>`-tag system prompt — wired into Gym two different ways:

- **`examples/langchain_deepagent`** — a standalone external FastAPI service, reached via `remote_agent`.
  Eval-only by design (no `model_server` reference; brings its own model, talking to OpenRouter directly).
- **`responses_api_agents/langchain_deepagents_agent`** — the same graph, run as a native in-tree Gym agent
  server, calling Gym's own `model_server` (on-policy, training-eligible in principle).

The goal is to document, concretely, what the in-tree pattern costs an engineer coming from the
agent-framework world — the kind of finding that generalizes to wiring a *different* framework in-tree,
not specific to `deepagents`.

## The numbers

| Benchmark | `remote_agent` (`examples/langchain_deepagent`) | In-tree `langchain_deepagents_agent` |
|---|---|---|
| `reasoning_gym` (5 rows) | `mean/reward: 1.0` | `mean/reward: 1.0` — same free-tier model, apples-to-apples |
| `tavily_search` (5 rows) | `mean/reward: 0.4` | `mean/reward: 0.8` |

`reasoning_gym` is the clean comparison: identical reward on the identical free-tier model
(`nvidia/nemotron-3-ultra-550b-a55b:free`), deterministic `<answer>`-tag scoring with no judge noise. The
architecture change — off-policy external process vs. on-policy native agent server — didn't change
task-solving quality at all.

`tavily_search` isn't a clean comparison: the in-tree number was captured on a different model tier than
the `remote_agent` baseline, after the free-tier model hit a transient OpenRouter overload mid-run. The
`0.8` isn't evidence the in-tree agent scores higher on this benchmark; a same-tier rerun hasn't been done.
Treat the `reasoning_gym` row as the one that isolates the architecture change.

## What "in-tree" actually costs, concretely

### `deepagents` cannot be installed alongside `nemo_gym[dev]` at its latest version

A plain dependency conflict, unrelated to Gym's agent-server design. `nemo_gym`'s `pyproject.toml` pins
`anthropic<=0.109.2` (type-only use in `nemo_gym/anthropic_converter.py` — the real client is never
constructed, for the same aiohttp-over-httpx reason as below). `deepagents>=0.7.0` unconditionally depends
on `langchain-anthropic`, which requires `anthropic>=0.120.0`. Those ranges don't overlap; `uv` reports it
unsatisfiable. Fixed by pinning `deepagents<0.7.0` (resolves to `0.6.12`) in
`responses_api_agents/langchain_deepagents_agent/requirements.txt` — meaning the in-tree agent runs a
different `deepagents` version than `examples/langchain_deepagent` (`>=0.7.6,<0.8`, in its own isolated
dependency tree with no `nemo-gym[dev]`, so it never hits this conflict). The
`create_deep_agent(model=, tools=, system_prompt=)` surface used here is stable across that gap, but it's a
real, permanent version delta between "the same agent" run two ways — not a rounding error. (Gym's own
`anthropic` pin isn't a hard ceiling — it's a routine periodic bump per its own comment — so a repo-wide
bump is plausible, but was out of scope for one agent.)

### A framework that owns its own model-calling loop can't use its native model client

`deepagents`/LangChain agents are built around `langchain_openai.ChatOpenAI`, which runs on the `openai`
SDK, which runs on `httpx`. CLAUDE.md requires all async HTTP inside a Gym server process to go through
Gym's own aiohttp client instead (`httpx`/`httpcore`'s connection pooling degrades badly at high
concurrency). `langgraph_agent` sidesteps this entirely — it never constructs a real LangChain
`BaseChatModel`; its hand-rolled graph nodes call `server_client.post()` directly. `deepagents` doesn't
give this agent that option: its internal `create_agent()` graph calls
`model.bind_tools(...).ainvoke(...)` on whatever `BaseChatModel` it's handed, and that loop isn't the
agent author's to rewrite. So this build required `GymResponsesChatModel`
(`responses_api_agents/langchain_deepagents_agent/app.py`) — a from-scratch `BaseChatModel` subclass
reimplementing LangChain-message-to-Gym-Responses-API translation, tool-schema conversion, and per-request
correlation. This generalizes: *any* third-party agent framework that owns its own model-calling loop (not
just its graph structure) hits this same wall the moment it's wired in-tree.

### Rebuilding the graph once (not per-request) needs a mechanism unrelated to agent-building

`langgraph_agent` builds its graph exactly once, in `__init__`. Doing the same here — `self.agent =
self.build_agent(GymResponsesChatModel(agent=self))` — required a Python `contextvars.ContextVar`,
because the shared `GymResponsesChatModel` needs per-request info (a rollout-correlation id, cookies) that
doesn't exist until a request arrives. It's set once per request in `responses()` and read inside
`_agenerate()`; safe under concurrent in-flight requests because `ContextVar` values are asyncio-task-local
(`langgraph_agent`, by contrast, just threads a plain dict through `graph.ainvoke(initial_state)`, since
nothing framework-owned reaches for state out-of-band). An agent builder coming from the LangChain world
has no reason to expect "the model" needs anything beyond `ChatOpenAI(...)`, let alone a concurrency-safe
context-propagation pattern.

### The agent stops being a self-contained, separate thing

`examples/langchain_deepagent` is fully standalone — its own `.env`, its own `pyproject.toml`, runnable
with nothing but `uv run`. In-tree, config comes from Gym's own `env.yaml` + Hydra `${...}` interpolation
instead of a local `.env`, and dependencies move from an installable package to a `requirements.txt` that
editable-installs the whole monorepo (`-e nemo-gym[dev] @ ../../`) — so, per the point above, its
dependency resolution has to coexist with the rest of Gym's dependency tree, not just its own.

### Config composition is its own small DSL

`gym env start --resources-server reasoning_gym/reasoning_gym_langchain_deepagents_agent_model_server
--model-type inference_provider/openrouter` only works because the `_model_server` filename suffix means
"this config expects `--model-type` to supply the model" — a convention discoverable only by reading other
agents' filenames (e.g. `..._claude_code_agent.yaml` vs. `..._claude_code_agent_model_server.yaml`), not
communicated by the CLI's flags or `--help` text.

### Relative paths in resources-server configs resolve against that server's own process cwd

`tavily_search_resources_server`'s `exclude_domains_file_path` has to be relative to
`resources_servers/tavily_search/` (that server's own working directory when Gym spins it up), not the
repo root and not wherever `gym env start` was invoked from. The existing `remote_agent` tutorial documents
this exact gotcha in a comment, but it's easy to get wrong the first time regardless — worth knowing before
wiring config for any new in-tree server.

### Gym's tool schema requires a field LangChain's own conversion doesn't provide

`langchain_core.utils.function_calling.convert_to_openai_tool()` — the standard helper every real
provider's `bind_tools()` uses — produces `{"type": "function", "function": {"name":..., "description":...,
"parameters":...}}`. Gym's `FunctionToolParam` type (a passthrough of the real OpenAI Responses API type)
additionally requires a present-but-nullable `"strict"` key. Omitting it doesn't fail at the LangChain
layer — it fails downstream as a 422 from Gym's model server. `GymResponsesChatModel.bind_tools()` patches
`"strict": False` into each converted tool schema to work around this; a test
(`test_bind_tools_unnests_chat_completions_shape_into_responses_shape`) locks it in.

### Trajectory observability is opt-in work, not something the base class gives you for free

The in-tree agent currently does **not** capture a `TrajectoryRecord` at all — `responses()` always takes
the plain `self.agent.ainvoke()` path, regardless of whether model-call capture is enabled. Reward
scoring is unaffected (`verify` only reads `body.response.output_text`), but per-tool-call/per-model-call
detail for `gym eval profile` is missing. An event-streaming path that reconstructs tool calls and
populates `AgentInvocation.conversation` (matching `SimpleAgent`'s own pattern) was built and then
deliberately stripped before merging — it's a clean, self-contained addition to reinstate later, not an
architectural rework.

Separately, and true of Gym's own reference implementation independent of anything built here: even
`SimpleAgent`'s `_create_episode` doesn't pass `model_calls=` on the top-level `TrajectoryRecord` it
constructs — only on the nested `AgentInvocation`. The top-level `TrajectoryRecord.model_calls` field gets
filled in later, by `rollout_collection.py`'s `_build_trajectory_record`, from raw per-rollout
model-call-capture JSONL matched by `model_call_id` — not assembled by the agent itself. Any agent wanting
full trajectory detail needs both: an agent-side capture path (for tool calls and conversation shape) *and*
`model_call_capture_dir` enabled at the model-server level (for the model-call detail merged in downstream).
Neither is optional if the goal is a complete trajectory, and this isn't obvious from
`+observability_enabled=true` alone — `model_call_capture_dir` (an absolute path) is a hard requirement the
moment observability is on; omitting it crashes the model server at startup with a Pydantic validation
error, not a "you forgot a flag" message.

## What it buys you

- **On-policy, training-eligible in principle** — the whole reason `remote_agent` is explicitly documented
  as eval-only.
- **Trajectory/tool-call observability is possible in principle**, unlike `remote_agent`: its own
  `service.py` docstring calls out that its bare-minimum response means deepagents' internal tool calls are
  invisible to `gym eval profile` entirely, by design. The in-tree agent doesn't currently capture that
  detail either (see above), but nothing about its architecture rules it out — reinstating the stripped
  event-streaming path is the only work required.
- **Reachable from inside Gym's own network**, without the "how does an externally-hosted `remote_agent`
  process reach `model_server`" problem that prompted this whole comparison in the first place.

## The abstraction-level point

`SimpleResponsesAPIAgent` is a genuinely thin, correct interface wrapper — any framework can sit behind it.
What this build shows concretely is that closing the gap between *"runs"* and *"is properly observed,
on-policy, and trainable"* requires understanding a substantial amount of Gym-specific machinery that has
nothing to do with the wrapped framework itself: a packaging/dependency conflict invisible until install
time, a concurrency-safe context-propagation pattern invisible until you try to match an existing agent's
build-once shape, a silently-required schema field invisible until a live 422, a path-resolution convention
invisible until a `FileNotFoundError`, and a trajectory-observability system with real gaps even in Gym's
own reference agent. None of these are hard to fix once found; none of them are about deepagents,
LangChain, or agent frameworks in general — they're about Gym's own internals.

## Explicitly out of scope

No recommendation on *how* to fix any of this. That's a decision for engineers actually working on Gym's
architecture, informed by this document — not something to resolve here.
