# Conversational Tool-Use Policy and Tool Generation Agent

This `SimpleResponsesAPIAgent` generates one accepted policy/tool artifact for one domain and one profile per Gym
rollout. Supported profiles are `general` and `proactive`.

The `/run` coordinator performs this sequence for every attempt:

1. Sample a 2025 US timestamp and shuffle all eight golden pairs.
2. Generate policy v1 and tools v1.
3. Shuffle again and refine the policy. The proactive profile consumes this shuffle but omits the references.
4. Consume a third, unused shuffle and refine the tools.
5. Apply permissive Tau2-compatible tool and JSON Schema validation.
6. Run three concurrent cohesion judgments.
7. Shuffle a fourth time and run four concurrent golden comparisons.

The golden comparisons intentionally repeat the first two comparison prompts while changing only their expected
labels. A failure fraction greater than `0.5` rejects an attempt; exactly `0.5` passes. The default permits 20 retries
after the first attempt, for at most 21 attempts. Exhaustion raises an error.

All random operations use Python's module-global `random` state. Independent Gym tasks therefore share normal
process-level interleaving rather than receiving per-item seeds.

## Requests and Results

[`data/example.jsonl`](data/example.jsonl) shows the input contract. `domain.applications` is retained as arbitrary
JSON and is not interpreted. Domain names are rendered by removing parentheses, replacing `/` with `-`, replacing
spaces with `_`, and removing `&`.

Internal generation and judgment requests use `/v1/chat/completions` with only a `messages` field. Model and sampling
settings belong to the configured model servers. The public `/v1/responses` endpoint transparently forwards the
caller's request to the policy model server.

Accepted `/run` responses have reward `1.0`. `result` contains the policy Markdown, parsed tools, and exact JSONL
artifact. `generation_trace` contains the exact messages, raw chat completions, parsing outcomes, random reference
order, judgment fractions, and retry outcomes. The top-level Responses object is converted from the tools-refinement
completion; judge completions remain in `generation_trace`.

## Configuration

Compose
[`configs/conversational_tool_use_policy_tool_generation.yaml`](configs/conversational_tool_use_policy_tool_generation.yaml)
with model servers named `policy_generation_model` and `policy_tool_judge_model`.

## Materialization

Convert accepted rollout JSONL into scenario-generation Gym input JSONL with explicit paths:

```bash
uv run python -m responses_api_agents.conversational_tool_use_policy_tool_generation.materialize \
  --input-path /path/to/policy_tool_rollouts.jsonl \
  --output-path /path/to/scenario_generation_inputs.jsonl
```

Rows retain accepted input order and contain only the next agent's request fields: empty Responses input, agent
reference, sanitized domain name, policy, and parsed tools. The rollout JSONL remains the generation record.

## Tests

```bash
uv run pytest responses_api_agents/conversational_tool_use_policy_tool_generation/tests
```
