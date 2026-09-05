# LangChain DeepAgents Agent

Runs a [LangChain DeepAgents](https://python.langchain.com/docs/experimental/deep_agents/) agent as a
native NeMo Gym agent server — model calls go through Gym's own `model_server` (on-policy).

`app.py`'s `DeepAgentsAgent` is the generic base: it owns the Gym Responses API bridge and the request/response
plumbing, but does not itself define a `deepagents` graph. `reasoning_search_agent.py` is one concrete
subclass — a sample usage wiring `TavilySearch` in via `build_agent()` — not the thing being wrapped. To
run a different `deepagents` graph, subclass `DeepAgentsAgent` the same way rather than modifying
`reasoning_search_agent.py`; this mirrors `langgraph_agent/`, whose `app.py` base likewise ships several
concrete subclasses (`orchestrator_agent.py`, `reflection_agent.py`, etc.) side by side in one directory.

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

To attach this agent to a different resources server, add the generic
[`configs/langchain_deepagents_agent.yaml`](configs/langchain_deepagents_agent.yaml) alongside your
resources server's config and override its `resources_server.name` (`???` by default):

```bash
gym env start --resources-server <your_server> --model-type inference_provider/openrouter \
    --config responses_api_agents/langchain_deepagents_agent/configs/langchain_deepagents_agent.yaml \
    +langchain_deepagents_agent.responses_api_agents.langchain_deepagents_agent.resources_server.name=<your_server>_resources_server
```

### Run the agent

```bash
gym eval run --no-serve --agent reasoning_gym_langchain_deepagents_agent_model_server \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output /tmp/langchain_deepagents_agent_rg.jsonl
```

Expect a `mean/reward` comparable to `examples/langchain_deepagent`'s documented baseline (1.0 over the
same 5 rows) — the underlying agent and model are unchanged, only the calling architecture differs.

## Tool ownership: agent vs. resources server

Tools belong on the agent by design. `reasoning_search_agent.py` above wires `TavilySearch` straight into
`create_deep_agent(tools=[...])` inside the agent server, calling the Tavily API directly with
`tavily_api_key`. The `tavily_search` resources server it can pair with does define its own `web_search`
endpoint, but this agent never calls it — only its `verify()` endpoint is used, for grading.

This is the default shape because tool definitions (schemas, call handling, framework-specific bindings
like `langchain_tavily.TavilySearch`) are inherently agent-harness concerns — they travel with the
framework you're wrapping, not with task verification. Keeping them in the agent server also lets the same
resources server pair with agent servers that expose the same capability differently (e.g. a different
search tool implementation, or a framework with its own native tool-calling conventions).

If you do want a tool centralized in the resources server instead — e.g. so it's shared identically across
multiple agent harnesses, or so its implementation can be versioned/tested alongside the verifier — that's
a supported alternative, not the default. See other resources servers (e.g. `tavily_search`'s own
tool-call endpoints when used with non-deepagents agents) for that pattern.

## Config fields

- `resources_server`: the resources server this agent interacts with for tools, state, and verification
- `model_server`: the Gym model server driving the deepagents graph (composed via `--model-type`)
- `tavily_api_key`: Tavily API key for the `TavilySearch` tool
- `max_search_results`: max results returned per `TavilySearch` call
- `max_steps` (inherited, unused): deepagents runs its own internal tool loop and answers in one call

## Known limitation: no trajectory/observability capture (yet)

This agent does not build a `TrajectoryRecord` (per-tool-call/model-call observability) — `responses()`
always takes the plain `graph.ainvoke()` path. Reward/pass-fail scoring is unaffected either way (`verify`
only reads `body.response.output_text`); what's missing is only the rich per-call detail Gym's own
rollout-collection pipeline can attach to a trajectory.
