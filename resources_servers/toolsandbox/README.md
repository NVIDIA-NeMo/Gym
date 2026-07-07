# ToolSandbox (gym)

Multi-turn, stateful, tool-using benchmark ported from
[apple/ToolSandbox](https://github.com/apple/ToolSandbox). An agent-under-test
(the gym policy model) converses with a simulated user while issuing Python tool
calls against a stateful sandbox (contacts, messaging, reminders, device
settings). Each scenario is scored against milestone / minefield snapshots; the
reward is the milestone **similarity** in `[0, 1]` (0 if a minefield is hit).

## Architecture

This is a native gym multi-turn benchmark built on the aviary env pattern — it
does **not** shell out to a CLI. The vendored `tool_sandbox/` package (scoring,
scenarios, tools, execution environment, user role) is unchanged; only the
conversation *driver* is re-wired:

| Component | Where it runs |
|-----------|---------------|
| Agent-under-test | gym **policy model**, driven by `toolsandbox_agent` |
| User simulator | inside the resources server, calling `user_model_server` |
| Python execution env | inside the resources server |
| Milestone scoring | resources server `/verify` (pure scoring) |

Flow (mirrors aviary): `seed_session -> obs + tools`, `/step(action) -> obs,
done`, `/close` (computes + caches the reward), `/verify` (returns it).

- `app.py` — `ToolSandboxResourcesServer`
- `schemas.py` — request/response + config schemas
- `tool_sandbox/` — vendored apple/ToolSandbox (self-contained; no `benchmarks/` deps)
- `../../responses_api_agents/toolsandbox_agent/` — the driving agent harness

## Install

ToolSandbox's dependencies are **not** part of the base gym install — they live
in this server's `requirements.txt` and are installed into an isolated
per-server `.venv` only when the benchmark is used. `gym env start` / `gym env
test` build that venv automatically. To set it up manually for development:

```bash
cd gym/resources_servers/toolsandbox
uv venv --seed --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

(scipy is intentionally not required — the one assignment-solver call is served
by a vendored, dependency-free implementation; see
`tool_sandbox/common/_linear_assignment.py`.)

## Required env vars

The user simulator and the agent both need a model endpoint. When
`user_model_server` points at `policy_model` (the default), only the policy
model credentials are needed (e.g. `OPEN_ROUTER_KEY` for OpenRouter, or a vLLM
endpoint on a cluster).

## Run locally

Start the resources server and the driving agent together (the config wires
both), backed by an OpenAI-compatible model server:

```bash
gym env start \
    --config resources_servers/toolsandbox/configs/toolsandbox.yaml \
    --model-type openai_model \
    ++policy_base_url=https://openrouter.ai/api/v1 \
    ++policy_api_key=$OPEN_ROUTER_KEY \
    ++policy_model_name=qwen/qwen3.5-9b
```

Then collect rollouts over the smoke set:

```bash
gym eval run --no-serve \
    --agent toolsandbox_agent \
    --input resources_servers/toolsandbox/data/example.jsonl \
    --output results/toolsandbox_rollouts.jsonl \
    --num-repeats 1
```

## Datasets

`data/example.jsonl` is a 5-row smoke set (`{"task_idx": 0..4}`). `task_idx`
indexes into the sorted list of scenario names, so the full set is one row per
scenario index. Regenerate it with:

```bash
python resources_servers/toolsandbox/prepare_toolsandbox.py \
  --output resources_servers/toolsandbox/data/test.jsonl
```

## nemo-evaluator

To drive the multi-turn agent from nemo-evaluator via the `gym://` adapter, use
`configs/toolsandbox_serve.yaml`: unlike a deterministic scorer, ToolSandbox is
agentic, so the gym side launches **both** the resources server and the
`toolsandbox_agent` harness. nemo-evaluator's `gym_delegation` solver discovers
the agent, calls its `/run` per dataset row, and (with `trust_reward: true`)
uses the reward returned from `/verify`.

## Tests

```bash
gym env test --resources-server toolsandbox
```
