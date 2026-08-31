# Two ways to run the same agent in NeMo Gym: `remote_agent` vs. in-tree

This compares two real, working implementations of the exact same LangChain `deepagents` graph — same
model, same TavilySearch tool, same `<answer>`-tag system prompt — wired into Gym two different ways:

- **`examples/langchain_deepagent`** — a standalone external FastAPI service, reached via `remote_agent`.
  Eval-only by design (no `model_server` reference; brings its own model).
- **`responses_api_agents/langchain_deepagents_agent`** — the same graph, run as a native Gym agent server, calling
  Gym's own `model_server` (on-policy, training-eligible in principle).

The goal isn't to recommend one over the other — it's to document, concretely, what "just use the
in-tree pattern" actually costs an engineer coming from the agent-framework world, since that cost is
easy to underestimate from the outside. Everything below either happened during this build or was
confirmed by reading the actual source; nothing here is speculative.

## The numbers

| Benchmark | `remote_agent` (documented baseline) | In-tree `langchain_deepagents_agent` |
|---|---|---|
| `reasoning_gym` (5 rows) | `mean/reward: 1.0` | `mean/reward: 1.0` — **same free-tier model, apples-to-apples** |
| `tavily_search` (5 rows) | `mean/reward: 0.4` | `mean/reward: 0.8` — **different model tier, not a clean comparison** (see below) |

`reasoning_gym` is the clean result: identical reward on the identical free-tier model. The architecture
change — off-policy external process vs. on-policy native agent server — did not change task-solving
quality at all. That's the headline finding, and it's a real, positive one for the in-tree pattern.

`tavily_search` isn't a clean comparison. The free-tier model (`nvidia/nemotron-3-ultra-550b-a55b:free`)
hit a genuine, reproducible upstream overload on OpenRouter partway through this build (confirmed via a
direct diagnostic — see "OpenRouter's free tier fails in a way retries don't catch" below), so this one
result used the paid tier instead. The `0.8` isn't evidence the in-tree agent scores higher; it's evidence
of nothing about the architecture at all, since a different backend model produced it. Worth re-running
on the free tier once it's not overloaded for a real number.

## What "in-tree" actually costs, concretely

Ordered roughly by how surprising each one was, not by file order.

### 1. `deepagents` cannot be installed alongside `nemo_gym[dev]` at all

This is the single biggest, most concrete surprise of the whole build, and it has nothing to do with Gym's
agent-server design — it's a plain dependency conflict, only discoverable by actually running `uv pip
install`. `nemo_gym`'s own `pyproject.toml` pins `anthropic<=0.109.2` (for type-only use in
`nemo_gym/anthropic_converter.py` — the real client is deliberately never used, for the exact same
httpx-vs-aiohttp reason as point 2 below). But `deepagents>=0.7.0` unconditionally depends on
`langchain-anthropic`, which has required `anthropic>=0.120.0` since that version — even though this
agent never touches an Anthropic model. Those ranges don't overlap; `uv` reports it as flatly
unsatisfiable. The fix here was pinning `deepagents<0.7.0` (the newest version that still resolves,
`0.6.12`), which means **the in-tree agent runs a different `deepagents` version than the example**
(`examples/langchain_deepagent` uses `>=0.7.6,<0.8` — a fully separate dependency tree with no
`nemo-gym[dev]`, so it never hits this). The core `create_deep_agent(model=, tools=, system_prompt=)`
surface used here is basic and stable across that gap, but it's a real, permanent asterisk on "the exact
same agent," not a rounding error. (Gym's own `anthropic` pin is confirmed *not* a hard technical ceiling
— its comment says it's just a routine periodic bump, mirroring how `openai` is pinned the same way — so a
repo-wide bump is plausible, but out of scope for one agent.)

### 2. The framework's own model client can't be used at all

`deepagents`/LangChain agents are built around `langchain_openai.ChatOpenAI`, which runs on the `openai`
SDK, which runs on `httpx`. This repo requires all async HTTP inside a Gym server process to go through
Gym's own aiohttp client instead — CLAUDE.md states this outright (`httpx`/`httpcore`'s connection pooling
degrades badly at high concurrency). `langgraph_agent` hit this same wall earlier and sidestepped it
entirely: it never constructs a real LangChain `BaseChatModel` — its hand-rolled graph nodes call
`server_client.post()` directly. `deepagents` doesn't give this agent that option — its internal
`create_agent()` graph calls `model.bind_tools(...).ainvoke(...)` on whatever `BaseChatModel` it's handed,
and that loop isn't ours to rewrite. So this build had to write `GymResponsesChatModel`, a from-scratch
`BaseChatModel` subclass reimplementing LangChain-message-to-Gym-Responses-API translation, tool-schema
conversion, and per-request correlation — real, substantial code whose only purpose is working around a
transport-layer constraint that has nothing to do with deepagents itself. This generalizes: *any*
third-party agent framework that owns its own model-calling loop (not just its graph structure) hits this
identical wall the moment it's wired in-tree.

### 3. Getting back to "build once" needed a mechanism unrelated to agent-building

`langgraph_agent` builds its graph exactly once, in `__init__` (`self.graph = self.build_graph()`) — not
per request. The first draft of this agent rebuilt the whole `deepagents` graph fresh on every single
incoming request, because the model object needed per-request info (a rollout-correlation id, cookies)
that doesn't exist until a request arrives. Matching `langgraph_agent`'s "build once" shape meant reaching
for a Python `contextvars.ContextVar` — set once per request in `responses()`, read inside
`GymResponsesChatModel._agenerate()` — safe under concurrent in-flight requests only because `ContextVar`
values are asyncio-task-local. `examples/langchain_deepagent` never needs any of this: its model talks
straight to OpenRouter, with no per-rollout correlation concept at all. An agent builder coming from the
LangChain world has no reason to expect "the model" to need anything beyond `ChatOpenAI(...)`, let alone a
concurrency-safe context-propagation mechanism.

### 4. You can't keep your agent as a nice, self-contained, separate thing

`examples/langchain_deepagent` is fully standalone — its own `.env`, its own `pyproject.toml`, runnable
with nothing but `uv run`. In-tree, secrets and config stop coming from a local `.env` file and instead
flow through Gym's own `env.yaml` + Hydra `${...}` interpolation, and dependencies move from a real
installable package to a `requirements.txt` that editable-installs the whole monorepo (`-e nemo-gym[dev] @
../../`). The agent stops being a thing you can hand someone as an isolated unit — and, per point 1 above,
its dependency resolution now has to coexist with the *entire rest of Gym's* dependency tree, not just its
own.

### 5. Config composition is its own small DSL

`gym env start --resources-server reasoning_gym/reasoning_gym_langchain_deepagents_agent_model_server --model-type
inference_provider/openrouter` only works because you already know the `_model_server` filename suffix
means "this config expects `--model-type` to supply the model" — a convention discoverable only by reading
other agents' filenames (`reasoning_gym_claude_code_agent.yaml` vs.
`reasoning_gym_claude_code_agent_model_server.yaml`), not something the CLI flags or `--help` text
communicate.

### 6. Relative paths in Gym configs resolve against a resources server's *own* process cwd

Discovered the hard way: `tavily_search_resources_server`'s `exclude_domains_file_path` config value has
to be relative to `resources_servers/tavily_search/` (that server's own working directory when Gym spins
it up), not the repo root and not wherever `gym env start` itself was invoked from. Got a `FileNotFoundError`
on first attempt with a repo-root-relative path; the existing `remote_agent` tutorial actually documents
this exact gotcha in a comment, but it's easy to miss and easy to get wrong the first time regardless.

### 7. Gym's tool schema silently requires a field LangChain's own conversion doesn't provide

`langchain_core.utils.function_calling.convert_to_openai_tool()` — the standard helper every real
provider's `bind_tools()` uses — produces `{"type": "function", "function": {"name":..., "description":...,
"parameters":...}}`. Gym's own `FunctionToolParam` type (a plain passthrough of the real OpenAI Responses
API type) additionally requires a present-but-nullable `"strict"` key. Omitting it doesn't fail loudly at
the LangChain layer — it fails downstream, as a 422 from Gym's own model server, discoverable only by
running a live request and reading a Pydantic validation trace back through several layers of FastAPI/
Starlette wrapping. Easy to get right once you know; invisible until you hit it.

### 8. Observability's "for free" join isn't free, and doesn't even work for the reference implementation

This was the most surprising finding of the whole build, and it directly corrects something asserted
earlier in this same effort's own planning: `join_model_call_observations()` — the function that was
supposed to merge captured model-call data into a trajectory "for free" — operates on a completely
different schema (`AgentObservationBundle`, `claude_code_agent`'s own richer observability system), not on
`TrajectoryRecord` (the schema `SimpleAgent`, and therefore this agent, actually use via the `_ng_trajectory`
convention). No agent in this repo calls `join_model_call_observations()` on a `TrajectoryRecord` — including
`SimpleAgent` itself. Read directly: `SimpleAgent._create_episode`'s own `TrajectoryRecord(...)` construction
never passes `model_calls=`, leaving Gym's own reference agent implementation with an identical
`ObservationGap(code="model_calls_unavailable")` to what this build produced initially. The *actual* merge
mechanism lives inside `nemo_gym/rollout_collection.py`, reading raw captured JSONL from
`model_call_capture_dir` and matching by `model_call_id` against whatever the agent's own top-level
`TrajectoryRecord.model_calls` field already contains — and since no shipped agent populates that field, the
merge has nothing to attach to. This means "full trajectory observability," even done in good faith and even
matching the one existing precedent (`SimpleAgent`) as closely as possible, is *still* structurally
incomplete in this codebase today — not a gap unique to this build, but a gap in Gym's observability story
itself, invisible until you actually try to get a complete trajectory and go looking for why one field stays
empty. (One real, achievable fix *was* found and applied here: `SimpleAgent` populates
`AgentInvocation.conversation`; this build's first draft didn't, and now does, matching the reference
implementation exactly.)

There's also a separate required knob that isn't obvious from `+observability_enabled=true` alone:
`model_call_capture_dir` (an absolute path) is a hard requirement the moment observability is on — omitting
it crashes the model server at startup with a Pydantic validation error, not a helpful "you forgot a flag"
message.

### 9. OpenRouter's free tier fails in a way retries don't catch

Not a Gym design issue, but a concrete illustration of how thin the abstraction between "in-tree" and "the
actual upstream API" really is. `inference_provider`'s retry logic (`RETRY_ERROR_CODES`) only retries on
real non-2xx HTTP statuses. OpenRouter, at least for this free-tier model under load, returns HTTP `200`
with an embedded error body instead (`{"error": {"message": "Upstream error from Nvidia: Service
temporarily overloaded", "code": 502}}`) — confirmed via a direct diagnostic print at the failure site, not
guessed. That slips straight past the retry logic and surfaces as a bare `KeyError: 'choices'` several
layers up. This isn't necessarily unique to the in-tree path — `remote_agent`'s own `ChatOpenAI` talks to
the exact same OpenRouter endpoint and might hit the identical failure mode via the real `openai` SDK's own
response validation; that wasn't tested here, so this is flagged as an open question, not a claimed
difference between the two architectures.

## What it buys you

- **On-policy, training-eligible in principle** — the whole reason `remote_agent` is explicitly
  documented as eval-only.
- **Real trajectory/tool-call observability** (with the caveats above) — `remote_agent`'s own `service.py`
  docstring calls out that its bare-minimum response means deepagents' internal tool calls are invisible
  to `gym eval profile` entirely, by design.
- **Reachable from inside Gym's own network** without the "how does an externally-hosted `remote_agent`
  process reach `model_server`" problem that prompted this whole comparison in the first place.

## The abstraction-level point

This whole exercise started from a PR review comment: `SimpleResponsesAPIAgent` is a genuinely thin,
correct interface wrapper — any framework can sit behind it. What this build shows concretely is that
closing the gap between *"runs"* and *"is properly observed, on-policy, and trainable"* requires
understanding a substantial amount of Gym-specific machinery that has nothing to do with the wrapped
framework itself: a packaging/dependency conflict invisible until install time, a concurrency-safe
context-propagation pattern invisible until you try to match an existing agent's construction shape, a
silently-required schema field invisible until a live 422, a path-resolution convention invisible until a
`FileNotFoundError`, and an observability system that turns out to be incomplete even for its own reference
implementation. None of these are hard to fix once found. All of them were invisible from the outside, and
none of them are about deepagents, LangChain, or agent frameworks in general — they're about Gym's own
internals.

## Explicitly out of scope

No recommendation on *how* to fix any of this. That's a decision for engineers actually working on Gym's
architecture, informed by this document — not something to resolve here.
