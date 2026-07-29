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

The config creates `policy_generation_model` and `policy_tool_judge_model` as independent copies of Gym's standard
`policy_model`. They use one model by default but can be overridden independently.

Start the agent and its model servers:

```bash
gym env start \
  --config responses_api_agents/conversational_tool_use/policy_tool_generation/configs/policy_tool_generation.yaml \
  --model-type openai_model \
  --model-url "$MODEL_BASE_URL" \
  --model "$MODEL_NAME" \
  '+policy_api_key=${oc.env:MODEL_API_KEY}'
```

`gym env start` stays in the foreground. In a separate terminal, collect from the materialized input JSONL:

```bash
gym eval run --no-serve \
  --agent conversational_tool_use_policy_tool_generation \
  --input /tmp/conversational_tool_use/policy_tool_inputs.jsonl \
  --output /tmp/conversational_tool_use/policy_tool_rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```

## Materialization

Convert accepted rollout JSONL into scenario-generation Gym input JSONL with explicit paths:

```bash
python -m responses_api_agents.conversational_tool_use.policy_tool_generation.materialize \
  --input-path /path/to/policy_tool_rollouts.jsonl \
  --output-path /path/to/scenario_generation_inputs.jsonl
```

Rows retain accepted input order and contain the next agent's request fields: stable ID, empty Responses input, agent
reference, generation profile, sanitized domain name, policy, parsed tools, and source lineage. The rollout JSONL
remains the complete generation trace. IDs and lineage include available Gym task, rollout, and retry-attempt
coordinates.

## Tests

```bash
gym env test \
  +entrypoint=responses_api_agents/conversational_tool_use/policy_tool_generation \
  +should_validate_data=false
```
