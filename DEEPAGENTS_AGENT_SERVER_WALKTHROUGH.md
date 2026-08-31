# langchain_deepagents_agent: implementation walkthrough and comparisons

This is a teaching walkthrough of how `responses_api_agents/langchain_deepagents_agent/` was built, how it
compares to `langgraph_agent`, `claude_code_agent`, and `remote_agent`, and what resource-server-side
config was added to support it. It's the code-level companion to `AGENT_SERVER_ARCHITECTURE_COMPARISON.md`
(repo root) — that doc has the original cost narrative in prose; this one grounds each claim in the
actual code and adds the `langgraph_agent`/`claude_code_agent` comparisons it doesn't cover.

## The shape of the problem

`SimpleResponsesAPIAgent` is a thin interface: implement `responses()` (handle one turn) and you inherit
`run()` (the seed→loop→verify rollout driver) from `SimpleAgent`. The four agents below all sit behind
that same interface, but each pays a wildly different tax to get there, depending on what kind of thing
they're wrapping:

| Agent | What it wraps | Where the model call happens |
|---|---|---|
| `langgraph_agent` | LangGraph as a bare graph-execution library | Hand-written `server_client.post()` calls inside graph nodes |
| `langchain_deepagents_agent` | `deepagents`, a framework that **owns its own model-calling loop** | Inside `create_agent()`'s internals, via `model.bind_tools().ainvoke()` |
| `claude_code_agent` | The Claude Code CLI, a separate OS process | Entirely outside Python, inside the Node.js `claude` binary |
| `remote_agent` (+ `examples/langchain_deepagent`) | An arbitrary external HTTP service | Entirely outside Gym, in someone else's process |

Everything else follows from one axis: **does Gym's Python process need to intercept the model call, and
if so, whose calling convention does it have to satisfy?**

---

## Part 1: `langchain_deepagents_agent`, piece by piece

Files (`responses_api_agents/langchain_deepagents_agent/`):
```
app.py                    # generic base: config, GymResponsesChatModel, DeepAgentsAgent
reasoning_search_agent.py # the one concrete instance actually run
requirements.txt          # deepagents<0.7.0 pin
README.md
tests/test_app.py
```

**`DeepAgentsAgent`** (`app.py:224-357`) subclasses `SimpleAgent`. `__init__` builds the graph exactly once:

```python
self.graph = self.build_agent(GymResponsesChatModel(agent=self))
```

`build_agent` is abstract — `reasoning_search_agent.py`'s ~50-line implementation is just:

```python
create_deep_agent(model=model, tools=[TavilySearch(...)], system_prompt=SYSTEM_PROMPT)
```

That's the whole agent definition. Everything else in `app.py` exists to make `model` — a Gym-shaped
object — satisfy LangChain's `BaseChatModel` contract.

**`GymResponsesChatModel`** (`app.py:178-221`) is the actual translation shim, and it's the most
substantial thing this build produced. `_agenerate` (async-only; `_generate` deliberately raises
`NotImplementedError`) posts to Gym's model server via `self.agent.server_client` — Gym's aiohttp client,
never `httpx`/`ChatOpenAI` — because `deepagents` internally calls `model.bind_tools(...).ainvoke(...)`
and that loop isn't something the agent author gets to rewrite. `bind_tools()` converts LangChain tool
objects via `convert_to_openai_tool()`, then explicitly patches in `"strict": False` — Gym's
`FunctionToolParam` requires that key present-but-nullable, and LangChain's own converter doesn't produce
it. Miss that and you get a 422 from the model server, not a helpful error at the LangChain layer. A test
(`test_bind_tools_unnests_chat_completions_shape_into_responses_shape`) locks this in.

**The `ContextVar`** (`app.py:69`, module-level `_request_context`): because the graph and its chat model
are built once in `__init__`, there's no natural place to stash per-request data (rollout correlation id,
cookies) — the model object is a singleton shared across concurrent requests. `responses()` sets it at the
top of each request and resets it in a `finally`; `_agenerate()` reads it. It's task-local, so it's safe
under concurrent in-flight requests — a dedicated test spins up two staggered concurrent calls and asserts
neither leaks the other's rollout id.

**Observability**: `_invoke_with_trajectory` drives `self.graph.astream_events(..., version="v2")`,
reconstructing `ModelCallRef`s and `TrajectoryToolCall`s from `on_chat_model_end`/`on_tool_start`/
`on_tool_end` events, and — matching `SimpleAgent`'s own pattern exactly — populates
`AgentInvocation.conversation`. This path only runs when model-call capture is actually enabled;
otherwise `responses()` takes a cheap `graph.ainvoke()` path with no event-streaming overhead.

**The dependency pin** (`requirements.txt`): `deepagents<0.7.0`, resolving to `0.6.12`. `deepagents>=0.7.0`
unconditionally pulls in `langchain-anthropic`, which requires `anthropic>=0.120.0` — but `nemo_gym[dev]`
itself pins `anthropic<=0.109.2` (for type-only use, never a live client, for the same aiohttp-over-httpx
reason). Those ranges don't overlap; `uv` reports it as flatly unsatisfiable. This means the in-tree agent
runs a strictly older `deepagents` than `examples/langchain_deepagent` (which pins `>=0.7.6,<0.8` in its
own isolated dependency tree). The `create_deep_agent(model=, tools=, system_prompt=)` surface is stable
across that gap, but it's a real, permanent asterisk on "the same agent," not a rounding error.

---

## Part 2: vs. `langgraph_agent`

This is the cleanest contrast, because `langgraph_agent` sidesteps almost everything above — not through
cleverness, but by never adopting an agent framework in the first place. It uses LangGraph purely as a
control-flow library:

- Real `StateGraph`/`TypedDict` state, `add_conditional_edges`, `.compile()` — genuine LangGraph.
- But every node is a hand-written async function that calls
  `self.server_client.post(url_path="/v1/responses", ...)` directly and hand-parses the raw response
  back into `AIMessage`/`HumanMessage` objects. Grep across the whole directory: zero hits for
  `bind_tools`, `ChatOpenAI`, `BaseChatModel`.
- No `GymResponsesChatModel`-equivalent exists, because there's nothing to adapt into — the nodes already
  speak Gym's Responses API natively.
- No `ContextVar` either — per-request state (rollout id, cookies) is just a plain dict, explicitly
  threaded through `graph.ainvoke(initial_state)`, because nothing framework-owned is reaching for it
  out-of-band.
- No dependency conflict — `requirements.txt` has `langchain-core` + `langgraph` only, no
  `langchain-openai`/`langchain-anthropic` transitive tree to fight with `nemo_gym`'s pins.
- Tool calls aren't interpreted by the graph at all — `function_call` output items just round-trip
  through state back into the next `/v1/responses` call, pushing execution down to the resources/model
  server.

**Takeaway**: `langchain_deepagents_agent`'s extra machinery isn't the cost of "using LangChain" in general —
`langgraph_agent` uses LangChain-adjacent tooling with none of that cost. It's specifically the cost of
adopting a framework whose `create_agent()` **owns the model-calling loop internally**, which
`langgraph_agent` never does.

---

## Part 3: vs. `claude_code_agent`

Structurally the opposite kind of problem. `claude_code_agent` wraps a finished product shipped as a CLI
binary, not a Python framework:

- It shells out: `asyncio.create_subprocess_exec("claude", "-p", "--output-format", "stream-json", ...)`.
  Claude Code itself runs the whole agent loop — tool execution, permissioning, everything — inside the
  Node.js process.
- Because the model call happens in a *separate OS process*, the Python side never constructs an HTTP
  client for it at all — the httpx-vs-aiohttp constraint that forced `GymResponsesChatModel` into
  existence simply doesn't apply here. Python only sets `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` env
  vars pointing the subprocess at Gym's model server.
- The translation problem doesn't disappear, it moves: a ~450-line `observability.py` stream-parses
  Claude Code's own JSONL transcript format (`stream-json` events, plus on-disk transcripts under
  `CLAUDE_CONFIG_DIR`) to reconstruct tool calls, subagent trees, and context-compaction events — the same
  *kind* of work as `GymResponsesChatModel`'s message conversion, just against a CLI's wire format instead
  of a Python object model.
- It has its own, richer observability schema: `AgentObservationBundle` (subagent trees, tool timing,
  compaction, explicit `ObservationGap`s for anything it can't establish exactly) feeding
  `join_model_call_observations()`. This is a **separate schema from `TrajectoryRecord`**, which is what
  `SimpleAgent` (and therefore `langchain_deepagents_agent`) actually populates. Confirmed directly: nothing in the
  repo calls `join_model_call_observations()` on a `TrajectoryRecord`, including `SimpleAgent` itself —
  its own `_create_episode` never passes `model_calls=`. So the "observability comes for free" assumption
  doesn't hold even for Gym's own reference agent, independent of anything `langchain_deepagents_agent` did.
- No sandbox: `_create_episode` always appends `ObservationGap(code="no_sandbox_runtime")` — isolation
  comes only from a fresh per-request `CLAUDE_CONFIG_DIR`, cleaned up after.

**One-line summary**: wrapping a CLI product is an **IPC/parsing problem** (opaque transcript → structured
data, but total insulation from Gym's transport rules); wrapping a Python framework with its own model
loop is an **in-process API-impedance problem** (direct function-call interception, but no transcript to
parse since everything's already a Python object).

---

## Part 4: vs. `remote_agent`

`remote_agent` and `langchain_deepagents_agent` are two different deployment models for **the literal same
`create_deep_agent(model=, tools=[TavilySearch], system_prompt=...)` call** —
`examples/langchain_deepagent` is that call, standalone.

How `remote_agent` works generically (`responses_api_agents/remote_agent/app.py`): `run()` does
`seed_session` → POST to `{agent_base_url}/v1/responses` → `verify`, with any failure anywhere becoming a
`reward: 0.0` row tagged `remote_agent_error` rather than crashing the collector. The loop distinguishes
*paired* `function_call`/`function_call_output` items (the remote service's own internal tool calls,
passed through untouched) from *unpaired* ones (which Gym executes itself against its resources server) —
that pairing is the entire contract.

`examples/langchain_deepagent/service.py` is the translation shim on the *other* side — and it's
deliberately minimal, by its own docstring:

```python
input_messages = to_langchain(params["input"])
result = await agent.ainvoke({"messages": input_messages})
return to_responses(result["messages"][len(input_messages):])
```

Only the last `AIMessage` gets reported back as a single `output_text` — deepagents' internal Tavily
calls are never surfaced as paired records, so they're genuinely invisible to `gym eval profile`.
`params["tools"]` is ignored on purpose too (deepagents binds its toolset once at import time, so it can't
honor a per-task dynamic tool set). Model sourcing confirms the "off-policy" label concretely:
`ChatOpenAI(base_url="https://openrouter.ai/api/v1", ...)` talks straight to OpenRouter over real
`httpx`, outside Gym entirely — `RemoteAgentConfig` has no `model_server` field at all, so this path is
architecturally incapable of using Gym's model server even if you wanted it to.

| | `remote_agent` (`examples/langchain_deepagent`) | in-tree `langchain_deepagents_agent` |
|---|---|---|
| Process | External FastAPI service, own venv, own `uv run` | Inside Gym's server process |
| Model | Real `ChatOpenAI` → OpenRouter directly | Gym's `model_server`, via `GymResponsesChatModel` |
| On-policy / trainable | No — no token IDs, explicit doc callout | Yes in principle |
| Tool-call observability | Opaque by this example's design — one call in, one message out | Real trajectory records (with the `TrajectoryRecord` caveats above) |
| Translation code | ~40 lines (`to_langchain`/`to_responses`) | `GymResponsesChatModel` + `ContextVar` + trajectory-building (~180 lines) |
| Dependency isolation | Fully separate tree, no conflict | Shares `nemo_gym[dev]`'s tree → the `anthropic` pin conflict |

The one thing that stays constant across both is the `create_deep_agent(...)` call shape itself —
everything else (model plumbing, dependency tree, observability, on/off-policy status) is unique to each
deployment choice.

---

## Part 5: the resource-server side

Purely additive — confirmed via `git diff main...HEAD -- resources_servers/` returning empty; the only new
things are two untracked YAML configs, no server/verifier code touched:

- `resources_servers/reasoning_gym/configs/reasoning_gym_langchain_deepagents_agent_model_server.yaml`
- `resources_servers/tavily_search/configs/tavily_search_langchain_deepagents_agent_model_server.yaml`

Both follow the existing `_model_server`-suffix naming convention (the suffix means "expects
`--model-type` at `gym env start` time to supply `policy_model`") and are structurally near-identical to
the existing `..._claude_code_agent_model_server.yaml` siblings — just swapping the agent block to point
at `responses_api_agents/langchain_deepagents_agent` with `entrypoint: reasoning_search_agent.py`.

Why these two benchmarks specifically — they stress different things:

- **`reasoning_gym`** — thin wrapper around the `reasoning_gym` pip package, deterministic no-LLM-judge
  scoring (`<answer>` tag extraction → `get_score_answer_fn`), no tools of its own. This is what makes it
  the "clean" comparison in the numbers table: no judge noise, so `mean/reward: 1.0` on both `remote_agent`
  and in-tree really does isolate the architecture change from everything else.
- **`tavily_search`** — exposes its own Gym-hosted tools (`web_search`/`find_in_page`/`scroll_page`) *and*
  LLM-judges the final answer against ground truth, agnostic to which search tool actually produced it —
  which is exactly why `langchain_deepagents_agent`'s own internal LangChain `TavilySearch` tool (not this server's
  tools) still gets scored validly. This is the "messier" comparison, since it's sensitive to model tier
  (the free-tier OpenRouter overload issue documented in the comparison doc).

`examples/langchain_deepagent/configs/*.yaml` wire the *same* resources-server blocks to a `remote_agent`
block instead (`agent_base_url`, `max_steps: 1` — commented as "deepagents runs its own tool loop
internally, one step is the whole rollout" — and no `model_server` field at all).

---

## Where this all lives

Everything above is grounded in code that's currently **untracked**
(`responses_api_agents/langchain_deepagents_agent/`, both new resource-server YAMLs) — not yet committed on
`hwolff/deepagents-agent-server`. `AGENT_SERVER_ARCHITECTURE_COMPARISON.md` at the repo root has the
original cost narrative in prose form; this file is the concrete code-level grounding for each claim in
it, plus the `langgraph_agent`/`claude_code_agent` comparisons it doesn't cover.
