# Conversational Tool-Use Policy and Tool Generation Agent

This `SimpleResponsesAPIAgent` generates one accepted policy/tool artifact for one domain and one profile per Gym
rollout. Supported profiles are `general` and `proactive`.

With the checked-in defaults, the `/run` coordinator performs this sequence for every attempt:

1. Sample a 2025 US timestamp and shuffle all eight golden pairs.
2. Generate policy v1 and tools v1.
3. Shuffle again and refine the policy. The proactive profile consumes this shuffle but omits the references.
4. Consume a third, unused shuffle and refine the tools.
5. Apply permissive Tau2-compatible tool and JSON Schema validation.
6. Run three concurrent cohesion judgments.
7. Shuffle a fourth time and run four concurrent golden comparisons.

The golden comparisons intentionally make two calls per selected reference using the same comparison prompt while
changing only the expected label. A failure fraction greater than the configured maximum rejects an attempt; equality
passes. The default permits 20 retries after the first attempt, for at most 21 complete attempts. Exhaustion raises an
error.

With `random_seed: null`, random operations use Python's module-global `random` state. When an integer seed is set, the
agent creates a rollout-local RNG from that base seed and stable row identity: ID, task and rollout indices, profile,
and domain. Infrastructure retry indices and unrelated metadata do not change the sequence.

## Requests and Results

[`data/example.jsonl`](data/example.jsonl) shows the input contract. `domain.applications` is retained as arbitrary
JSON and is not interpreted. Domain names are rendered by removing parentheses, replacing `/` with `-`, replacing
spaces with `_`, and removing `&`.

Internal generation and judgment requests use `/v1/chat/completions` with only a `messages` field. Model and sampling
settings belong to the configured model servers. The public `/v1/responses` endpoint transparently forwards the
caller's request to the policy model server.

Accepted `/run` responses have reward `1.0`. `result` contains the policy Markdown, parsed tools, and exact JSONL
artifact. `generation_trace` contains the exact messages, raw chat completions, parsing outcomes, random reference
order, judgment fractions, and retry outcomes. The top-level Responses object is converted from the final tools
completion: tools refinement by default, or tools v1 when refinement is disabled. Judge completions remain in
`generation_trace`.

## Configuration

The config creates `policy_generation_model` and `policy_tool_judge_model` as independent copies of Gym's standard
`policy_model`. They use one model by default but can be overridden independently.

Agent controls:

| Setting | Default | Meaning |
|---|---:|---|
| `policy_model_server` | `policy_generation_model` | Model-server instance used for generation and refinement |
| `judge_model_server` | `policy_tool_judge_model` | Model-server instance used for quality judgments |
| `max_retries` | `20` | Full-pipeline retries after the first attempt |
| `use_refinement` | `true` | Run policy and tool refinement after initial generation |
| `initial_reference_count` | `8` | Shuffled packaged policy/tool references included in the initial prompts |
| `policy_refine_reference_count` | `8` | Shuffled policy references used for general-profile policy refinement |
| `minimum_tool_count` | `0` | Reject an attempt producing fewer tools; `0` preserves permissive validation |
| `cohesion_judge_count` | `3` | Cohesion calls per attempt; `0` disables the cohesion gate |
| `cohesion_max_failure_fraction` | `0.5` | Maximum accepted cohesion failure fraction |
| `golden_reference_count` | `2` | Golden references per attempt; each creates two judge calls and `0` disables the gate |
| `golden_max_failure_fraction` | `0.5` | Maximum accepted golden-comparison failure fraction |
| `max_judge_concurrency` | `null` | Per-rollout cap for concurrent calls within a judge phase; `null` means no agent cap |
| `random_seed` | `null` | Optional base seed for rollout-local timestamp and reference sampling |

Counts for packaged references cannot exceed the eight checked-in reference pairs. The proactive profile still omits
references from policy refinement. Because each golden prompt is evaluated against both target labels, deterministic
judge output commonly produces a `0.5` golden failure fraction; lowering that maximum changes the existing gate
substantially.

Internal requests continue to contain only `messages`. Configure temperature, top-p, output limits, and global endpoint
concurrency on the corresponding model-server copies. Provider retry behavior remains model-server-specific.
`gym eval run --concurrency` separately controls concurrent policy/tool rollouts. Unknown agent settings are rejected
so misspelled controls cannot silently fall back to defaults.

Start the agent and its model servers:

```bash
gym env start \
  --config responses_api_agents/conversational_tool_use/policy_tool_generation/configs/conversational_tool_use_policy_tool_generation.yaml \
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
