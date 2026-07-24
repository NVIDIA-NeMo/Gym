# Conversational Tool-Use Scenario Generation Agent

This `SimpleResponsesAPIAgent` runs one customer-service domain per Gym rollout. The `/run` coordinator:

- assigns 20 independent outside-policy-scope values with unseeded `random() < 0.1`
- launches exactly 20 concurrent `/v1/chat/completions` calls
- asks for 80 scenarios in every call using package-owned prompt and schema assets
- sends only `messages` in each child request
- performs fence removal and Pydantic JSON parsing without semantic retries
- records individual failures and continues
- consumes successful collections in completion order and deduplicates on the five scenario-content fields

Every normally completed rollout has reward `1.0`, including partial and empty results. `generation_trace.calls` is
the typed per-call trace and `result.scenarios` is the accepted completion-ordered scenario list. A scenario that omits
`unknown_info` is excluded; an explicit JSON `null` is retained. The top-level `response` is converted from the last
successful completion that produced scenarios; an empty valid Responses object is returned when none did. Coordinator
failures are not converted into successful results.

`POST /v1/responses` is a direct model-server bridge and preserves all caller-supplied Responses parameters. This route
is separate from `/run`, whose 20 child Chat Completions requests contain only `messages`.

## Input

[`data/example.jsonl`](data/example.jsonl) shows the one-domain input shape. `domain_name`, `policy`, and
`responses_create_params` are required. The materialized rows retain the domain policy and raw simulator tools.

## Configuration

Compose [`configs/conversational_tool_use_scenario_generation.yaml`](configs/conversational_tool_use_scenario_generation.yaml)
with a `responses_api_models` server named `scenario_generation_model`. Sampling is owned by that model server. The
agent does not send model, temperature, top-p, token-limit, or seed fields in child requests.

## Materialization

Convert rollout JSONL to the existing `conversational_tool_use_agent` input-row format with explicit paths:

```bash
uv run python -m responses_api_agents.conversational_tool_use_scenario_generation.materialize \
  --input /path/to/scenario_generation_rollouts.jsonl \
  --output /path/to/conversational_tool_use_inputs.jsonl
```

The materializer preserves rollout order and `result.scenarios` order. It emits the policy system prompt, Responses
API tools, raw simulator tools, seven-field customer scenario, stable row ID, and `agent_ref` expected by
`conversational_tool_use_agent`. It does not emit temperature or any other sampling parameter. Model and collection
configuration own downstream sampling.

## Tests

```bash
uv run pytest responses_api_agents/conversational_tool_use_scenario_generation/tests
```
