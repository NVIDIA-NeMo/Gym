# Using Remote Agent with LangChain Deepagents

LangChain [deepagents](https://github.com/langchain-ai/deepagents) invoked as a remote agent
for NeMo Gym.

[`langchain_agent/`](langchain_agent/) is the agent itself — plain LangChain/deepagents, no NeMo Gym
awareness at all, independently installable and runnable on its own. Everything else in this
directory (`service.py`, `configs/`) is NeMo Gym wiring that depends on `langchain_agent/` as a local
package.

## Running the agent

This can be any agent, but we are using LangChain deepagents as a tutorial.
We will start with running the agent outside of NeMo Gym.
```bash
cd examples/langchain_deepagent
cp .env.example .env   # fill in OPENROUTER_API_KEY and TAVILY_API_KEY

cd langchain_agent
uv run run_agent.py "When is the Valkyries next game?"
```

`run_agent.py` prints the agent's final answer.

```python
result = agent.invoke({"messages": [{"role": "user", "content": sys.argv[1]}]})
```
The output, `result` is LangChain's message format.

## Importing into NeMo Gym

As noted by the
[remote agent documentation](../../fern/versions/latest/pages/agent-server/remote-agent/index.mdx), we are given input from Gym task execution
and then every response we return is a single Responses API object.

This means we must convert from Gym's input task messages to LangChain's message format and
then back to the format NeMo Gym requires.

You can see this conversion in [`service.py`](service.py). It imports `langchain_agent` as a local
package — see [`pyproject.toml`](pyproject.toml)'s `[tool.uv.sources]`.


### Start up the agent with uvicorn
```bash
# Terminal 1 — the agent service with format conversion between the APIs
cd examples/langchain_deepagent
uv run uvicorn service:app --host 0.0.0.0 --port 9000
```

You can see sending a request directly to the agent.
```bash
curl -s http://localhost:9000/v1/responses \
    -H "Content-Type: application/json" \
    -d '{"input": [{"role": "user", "content": "What year was the Eiffel Tower completed, and who designed it?"}]}'
```
This is basically what we will do each time in the remote agent!

### Wire the agent into NeMo Gym Remote Agent Server

[`configs/config_reasoning_gym.yaml`](configs/config_reasoning_gym.yaml) points Gym's `remote_agent` at the service we just started:

```yaml
remote_agent:                 # server_id — your name for this instance
  responses_api_agents:
    remote_agent:                # implementation — must match responses_api_agents/remote_agent/
      entrypoint: app.py
      agent_base_url: http://localhost:9000
```

We use the `reasoning_gym` resource server in this example, but this is configurable — swap the
`resources_server` ref in `config_reasoning_gym.yaml` to point at a different benchmark; `service.py` doesn't change.

```bash
# Terminal 2 — Gym: reasoning_gym + remote_agent
gym env start "+config_paths=[examples/langchain_deepagent/configs/config_reasoning_gym.yaml]"
```
```bash
# Terminal 3 — once terminal 2 is up
gym eval run --no-serve +agent_name=remote_agent \
    +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=/tmp/deepagents_rollouts.jsonl
```

The output should be something like:
```bash
Collecting rollouts: 100%|███████████████████████████████████████████████████| 5/5 [00:11<00:00,  2.26s/it]
Sorting results to ensure consistent ordering
Computing aggregate metrics

Key metrics for remote_agent:
{
    "mean/reward": 1.0,
    "mean/score": 1.0
}
Finished rollout collection! View results at
```

Expect 5 rows in the output JSONL and non-zero `mean/reward`.

#### Trying a different benchmark: tavily_search

Let's see how to use a different benchmark via another resources server
in NeMo Gym.

[`resources_servers/tavily_search`](../../resources_servers/tavily_search/) is a web-research
benchmark, judged by an LLM on whether the final answer text matches `ground_truth` — it doesn't
execute or inspect tool calls at all (`num_tool_calls` is reported as a metric only, not part of
the reward), so it doesn't matter that our agent uses its own internal `TavilySearch` instead of
this resources server's Gym-hosted `web_search`/`find_in_page`/`scroll_page` tools. This is *not*
true of every environment. Sometimes tools live in the
resources server itself rather than or in addition to the agent; see
[Tools in Agent vs. Resources Server](https://docs.nvidia.com/nemo/gym/agent-server/integrate-existing-agents/#tools-in-agent-vs-resources-server)
for that distinction.

[`configs/config_tavily_search.yaml`](configs/config_tavily_search.yaml) wires this up:
`remote_agent` pointed at `tavily_search_resources_server`, plus a `judge_model` block (the judge
needs its own model — this reuses OpenRouter and `OPENROUTER_API_KEY` for it, via
`SEARCH_JUDGE_MODEL` env var). A judge generally shouldn't be the same model it's judging, so it
defaults to `anthropic/claude-sonnet-4.5` — stronger than, and distinct from, the agent's own free
default (`nvidia/nemotron-3-ultra-550b-a55b:free`, see `.env.example`).

```bash
# Terminal 1 — the agent service (same as above)
cd examples/langchain_deepagent
uv run uvicorn service:app --host 0.0.0.0 --port 9000
```
```bash
# Terminal 2 — Gym: tavily_search + remote_agent + judge_model. Needs TAVILY_API_KEY (this
# resources server's own key, for its Gym-hosted search tools — separate from the agent's) and
# OPENROUTER_API_KEY (reused as the judge's key) in the environment.
set -a; source examples/langchain_deepagent/.env; set +a
gym env start "+config_paths=[examples/langchain_deepagent/configs/config_tavily_search.yaml]"
```
```bash
# Terminal 3 — once terminal 2 is up
gym eval run --no-serve +agent_name=remote_agent \
    +input_jsonl_fpath=resources_servers/tavily_search/data/example.jsonl \
    +output_jsonl_fpath=/tmp/deepagents_tavily_search_rollouts.jsonl
```

Verified end to end: 5 rows, no failures, `mean/reward: 0.4` (2/5 correct per judge) — these are
hard, multi-hop research questions and our agent does a single shallow search per query, so a
middling score is expected, not a broken pipeline.

**Why this benchmark works but not every one will:** `single_step_tool_use_with_argument_comparison`
(a different resources server) expects the agent to pick the correct *next* tool call from a
dynamic, per-task tool set declared in `responses_create_params.tools`, comparing the call's
arguments directly — it doesn't work with this agent as-is, because deepagents binds a fixed
toolset once at import time and never looks at `params["tools"]` (`service.py` ignores it on
purpose). Before pointing this agent at a new resources server, read its `verify()` function to
see what it actually inspects — see [`service.py`](service.py)'s docstring and the
[remote agent documentation](../../fern/versions/latest/pages/agent-server/remote-agent/index.mdx).
