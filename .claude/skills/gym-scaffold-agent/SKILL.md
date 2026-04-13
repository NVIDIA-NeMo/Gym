---
name: gym-scaffold-agent
description: >
  Create a custom agent server for NeMo Gym. Use when the default simple_agent is
  insufficient — for multi-turn interaction, external library wrapping, custom tool
  orchestration, or non-standard interaction patterns (model assimilation). Covers
  agent server scaffolding, cookie/token propagation, httpx replacement, and async
  patterns for high-concurrency operation.
license: Apache-2.0
compatibility: Requires Python 3.12+, NeMo Gym installed.
metadata:
  author: nvidia-nemo-gym
  version: "1.0"
allowed-tools: Bash(python:*) Bash(ng_*) Bash(git:*) Read Write Edit Grep Glob
---

# Scaffold a Custom Agent Server

## When you need a custom agent

The built-in agents cover most cases:
- **`simple_agent`** — single-turn: sends prompt to model, gets response, calls verify. Works for most benchmarks.
- **`proof_refinement_agent`** — multi-turn correction: model gets error feedback and retries.

Build a custom agent when:
- The interaction pattern doesn't fit single-turn or simple correction loops
- You're wrapping an external library that has its own orchestration
- The benchmark requires custom tool-call sequencing or state management
- You need to teach the model a specific interaction protocol (assimilation)

## Step 1: Create the directory

```
responses_api_agents/my_agent/
├── app.py              # Server class extending SimpleResponsesAPIAgent
├── configs/my_agent.yaml
├── tests/__init__.py
├── tests/test_app.py
└── requirements.txt    # just: -e nemo-gym[dev] @ ../../
```

## Step 2: Implement the agent

Your agent extends `SimpleResponsesAPIAgent` and implements `responses()` and `run()`.

```python
from nemo_gym.server import SimpleResponsesAPIAgent

class MyAgent(SimpleResponsesAPIAgent):
    async def responses(self, request):
        # Single response from model
        ...

    async def run(self, request):
        # Full orchestration loop
        ...
```

### The `/run` endpoint

This is where orchestration happens. The general pattern:

1. Receive the input (system prompt + user message + verifier_metadata)
2. Call the model server (`/v1/responses` or `/v1/chat/completions`)
3. If the model returns tool calls, execute them against the resources server
4. Optionally loop (multi-turn)
5. Call `/verify` on the resources server
6. Return the verify response (includes reward)

The `/run` endpoint **must be async**.

## Step 3: Cookie propagation (critical for stateful environments)

Every downstream request must forward cookies from the incoming request:

```python
async def run(self, request):
    cookies = request.cookies  # Capture from incoming request

    # Every downstream call passes cookies
    model_response = await self.server_client.post(
        model_url, json=payload, cookies=cookies
    )
    verify_response = await self.server_client.post(
        verify_url, json=payload, cookies=cookies
    )
```

Missing cookies break stateful environments where the resources server tracks session state.

## Step 4: Token ID propagation (critical for RL training)

Multi-turn agents must propagate token IDs from model responses into subsequent turns:

```python
# After receiving model response
prompt_token_ids = model_response.get("prompt_token_ids", [])
generation_token_ids = model_response.get("generation_token_ids", [])
generation_log_probs = model_response.get("generation_log_probs", [])

# Accumulate across turns and include in final response
```

Without these, the RL training framework can't compute policy gradients for multi-turn interactions.

## Step 5: Wrapping external libraries

When integrating a 3rd-party benchmark library:

1. **Replace httpx transport**: If the library uses httpx internally, replace its HTTP transport with an aiohttp adapter. See `resources_servers/tavily_search/app.py` (`TavilySearchAIOHTTPClient`) for the pattern.

2. **Pre-process input**: Convert from Gym schema (`responses_create_params.input` + `verifier_metadata`) to the library's expected input format.

3. **Post-process output**: Convert the library's results back to `BaseVerifyResponse` (must include `reward` field).

4. **Reproduce published numbers**: Run the original library standalone first and record scores. Then run through your Gym wrapper and verify scores match.

```python
async def run(self, request):
    # Pre-process: Gym schema -> library input
    lib_input = self.convert_to_library_format(request)

    # Run library (may need asyncio.Semaphore for concurrency control)
    async with self.semaphore:
        lib_result = await self.run_library(lib_input)

    # Post-process: library output -> Gym response
    return self.convert_to_gym_response(lib_result)
```

## Step 6: Concurrency control

The agent must handle 4k-65k concurrent requests. Use `asyncio.Semaphore` for any blocking or resource-intensive operations:

```python
class MyAgent(SimpleResponsesAPIAgent):
    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
```

## Step 7: Wire YAML config

```yaml
my_agent_instance:
  responses_api_agents:
    my_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources_server    # Must match resources server instance name
      model_server:
        type: responses_api_models
        name: policy_model           # Must match model server instance name
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl
```

## Step 8: Test

Write tests covering:
- Happy path (model produces correct output, gets reward 1.0)
- Model failure (bad output, gets reward 0.0)
- Multi-turn logic (if applicable — verify correct number of turns, proper accumulation)
- Cookie propagation (verify cookies are forwarded)
- Concurrency (verify semaphore bounds are respected)

Coverage must be >= 95%.
