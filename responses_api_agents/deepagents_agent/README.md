# deepagents Agent

Runs a LangChain `deepagents` graph as a native NeMo Gym agent server — the in-tree counterpart to
[`examples/langchain_deepagent`](../../examples/langchain_deepagent), which wraps the same graph as a
`remote_agent` instead. Both point at the same OpenRouter/Nemotron model and run the same TavilySearch
tool with the same `<answer>`-tag system prompt; only the calling architecture differs. See
[`/AGENT_SERVER_ARCHITECTURE_COMPARISON.md`](../../AGENT_SERVER_ARCHITECTURE_COMPARISON.md) at the repo
root for what that difference actually costs and buys.

## Structure

- `app.py` — the generic base (`DeepAgentsAgent`, `GymResponsesChatModel`). Reusable by any future
  deepagents-based agent, not just this one; has no TavilySearch/reasoning-gym-specific knowledge.
- `reasoning_search_agent.py` — the concrete instance this repo actually runs: TavilySearch tool, same
  system prompt as the `remote_agent` version, reused across both benchmark configs below. Building a
  *different* deepagents-based agent means writing a new sibling file here (subclass `DeepAgentsAgent`,
  implement `build_agent(model)`), not editing `app.py`.

## Why model calls don't use `ChatOpenAI`

`langchain_openai.ChatOpenAI` runs on the `openai` SDK, which runs on `httpx`. This repo requires async
HTTP inside a Gym server process to go through Gym's own aiohttp-backed `server_client` instead
(`httpx`/`httpcore`'s connection pooling degrades badly at high concurrency — see CLAUDE.md). So model
calls go through `GymResponsesChatModel`, a custom `BaseChatModel` backed by `server_client.post()`
directly. See `app.py`'s module docstring for the full reasoning.

## Quick start

### env.yaml

```yaml
policy_api_key: <your OPENROUTER_API_KEY>
policy_model_name: nvidia/nemotron-3-ultra-550b-a55b:free
tavily_api_key: <your TAVILY_API_KEY>
```

For the `tavily_search` benchmark, also add:

```yaml
exclude_domains_file_path: resources_servers/tavily_search/tests/dummy_exclude_domains_file.json
search_judge_model_base_url: https://openrouter.ai/api/v1
search_judge_model_api_key: <your OPENROUTER_API_KEY>
search_judge_model_name: anthropic/claude-sonnet-4.5
```

### Launch

This agent has no self-contained config — it always composes with `--model-type`, since it has no
provider credentials of its own (it reaches `model_server`, not a provider directly):

```bash
# reasoning_gym
gym env start --resources-server reasoning_gym/reasoning_gym_deepagents_agent_model_server \
    --model-type inference_provider/openrouter

# tavily_search
gym env start --resources-server tavily_search/tavily_search_deepagents_agent_model_server \
    --model-type inference_provider/openrouter
```

`inference_provider/openrouter` proxies to OpenRouter — the same backend `examples/langchain_deepagent`
talks to directly, so reward numbers are comparable. Any other `responses_api_models` implementation
(`vllm_model`, `local_vllm_model`, ...) works too; just swap `--model-type`.

### Run the agent

```bash
gym eval run --no-serve --agent reasoning_gym_deepagents_agent_model_server \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output /tmp/deepagents_agent_rg.jsonl
```

Expect a `mean/reward` comparable to `examples/langchain_deepagent`'s documented baseline (1.0 over the
same 5 rows) — the underlying agent and model are unchanged, only the calling architecture differs.

## Known limitation: no trajectory/observability capture (yet)

This agent does not build a `TrajectoryRecord` (per-tool-call/model-call observability) — `responses()`
always takes the plain `graph.ainvoke()` path. Reward/pass-fail scoring is unaffected either way (`verify`
only reads `body.response.output_text`); what's missing is only the rich per-call detail Gym's own
rollout-collection pipeline can attach to a trajectory.

This was implemented and then deliberately stripped before merging, because it wasn't needed to get a
working agent server, and it's **not** the same thing as LangSmith tracing — LangSmith is LangChain's own
env-var-driven instrumentation (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, ...), completely orthogonal to
this agent's code either way. It's safe to re-add later: it only ever touched `responses()` via one
`self.graph.astream_events(...)` branch, with no architectural rework required elsewhere.

**Flag this explicitly as deferred follow-up work in the MR description — not an oversight.**
