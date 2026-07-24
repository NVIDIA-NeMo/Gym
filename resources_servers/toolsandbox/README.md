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

## Scoring caveat: Responses API vs Chat Completions (read before comparing numbers)

**gym ToolSandbox scores are ~0.08 lower than Apple's published baseline for the
same model, and this is expected — it is a property of the API surface, not a bug
in this port.** The port is faithful: same vendored scoring, scenarios, tools,
system prompt, and user simulator as upstream. The difference is *how the
agent-under-test is called*.

- **Upstream apple/ToolSandbox** drives the agent over the **Chat Completions**
  API (`/v1/chat/completions`).
- **This gym benchmark** drives the agent over the **Responses** API
  (`/v1/responses`) — because gym is Responses-API-native (see `gym/CLAUDE.md`:
  the Responses format represents multi-turn, tool-calling trajectories without
  custom serialization, and carries the typed items / token IDs that training
  needs). The `toolsandbox_agent` harness posts to `/v1/responses`; the
  `openai_model` server forwards to the provider's native Responses endpoint.

For `gpt-4o-2024-05-13`, these two OpenAI endpoints **behave differently**: on the
Responses API the model is roughly **2× more likely to call a tool with a guessed
argument** instead of asking for clarification. Measured directly by firing the same
refusal turn (identical system prompt + tools) 30× at each endpoint and counting how
often the model wrongly guesses-and-calls instead of asking:

| Endpoint | Wrong "guess-and-call" rate |
|----------|-----------------------------|
| Chat Completions (upstream) | 27% |
| Responses, system as input message (gym today) | 53% |
| Responses, system via `instructions` field | 43% |

This over-eagerness only hurts the **restraint** scenario categories —
`INSUFFICIENT_INFORMATION`, `ARG_TYPE_SCRAMBLED`, and the many-distraction-tools
variants — where the correct behavior is to decline or ask. On a like-for-like
30-scenario run, gym scored **0.672** vs upstream **0.761**; gym **tied or beat**
upstream on 20 of 30 scenarios, and the entire gap came from that restraint cluster.
Things ruled out along the way: model identity (both pinned to `gpt-4o-2024-05-13`,
verified), the system prompt (it *is* seeded, as a `role:"system"` input message),
the user simulator (vendored verbatim), and the `additionalProperties:False`
tool-schema injection (A/B'd — no effect; gated by
`tool_schema_additional_properties`).

### What this means for you

- **Running gym evals to compare models against each other:** no action needed.
  All models are scored through the same Responses-API harness, so the comparison
  is apples-to-apples. Just don't read the absolute number as Apple's.
- **Training / RL on this environment:** the Responses-API number is the *correct*
  signal — it measures the model on the exact API surface you train and deploy
  against. This is the intended use.
- **Reproducing Apple's published ~0.73 (or any Chat-Completions baseline):** you
  cannot get there with this harness; the endpoint difference is intrinsic to the
  model, and moving the system prompt to `instructions` does not close it. Reaching
  the reference number requires driving the agent over Chat Completions instead —
  a harness change that is deliberately *not* made here, to keep the benchmark
  gym-native. If you need that, add a Chat-Completions agent path
  (`openai_model` already exposes `/v1/chat/completions`).

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
    --model-url https://openrouter.ai/api/v1 \
    --model-api-key $OPEN_ROUTER_KEY \
    --model qwen/qwen3.5-9b
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
  --output resources_servers/toolsandbox/data/test_toolsandbox.jsonl
```

## Serve-only mode

To serve ToolSandbox for external orchestration, use
`configs/toolsandbox_serve.yaml`: unlike a deterministic scorer, ToolSandbox is
agentic, so the gym side launches **both** the resources server and the
`toolsandbox_agent` harness. The orchestrator discovers the agent via
`/server_instances`, calls its `/run` per dataset row, and (with
`trust_reward: true`) uses the reward returned from `/verify`.

## Tests

```bash
gym env test --resources-server toolsandbox
```
