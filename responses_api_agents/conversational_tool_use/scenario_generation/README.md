# Conversational Tool-Use Scenario Generation Agent

This `SimpleResponsesAPIAgent` runs one customer-service domain per Gym rollout. With the checked-in defaults, the
`/run` coordinator:

- assigns 20 independent outside-policy-scope values with `random() < 0.1`
- launches 20 `/v1/chat/completions` calls with at most 20 in flight
- asks for 80 scenarios in every call using prepared prompts and checked-in JSON schemas
- sends only `messages` in each child request
- performs fence removal and Pydantic JSON parsing without semantic retries
- records individual failures and continues
- consumes successful collections in completion order and deduplicates case-insensitively on the five scenario-content
  fields

Every normally completed rollout has reward `1.0`, including partial and empty results. `generation_trace.calls` is
the typed per-call trace and `result.scenarios` is the accepted completion-ordered scenario list. A scenario that omits
`unknown_info` is excluded; an explicit JSON `null` is retained. The top-level `response` is converted from the last
successful completion that produced scenarios; an empty valid Responses object is returned when none did. Coordinator
failures are not converted into successful results.

`POST /v1/responses` is a direct model-server bridge and preserves all caller-supplied Responses parameters. This route
is separate from `/run`, whose configured child Chat Completions requests contain only `messages`.

## Input

[`data/example.jsonl`](data/example.jsonl) shows the one-domain input shape. `domain_name`, `policy`, and
`responses_create_params` are required. Pipeline-materialized rows also retain the generation profile, source lineage,
domain policy, and raw simulator tools.

## Prepare Assets

Prepare the shared assets before running this agent:

```bash
python -m resources_servers.conversational_tool_use_simulation.prepare
```

This downloads the file-backed scenario prompts and the other generation assets from
[`nvidia/NeMo-Gym-Conversational-Tool-Use-Assets`](https://huggingface.co/datasets/nvidia/NeMo-Gym-Conversational-Tool-Use-Assets).
Add `--include-prompt-history` for the optional prompt history. JSON schemas and example JSONL files remain in Git.

## Configuration

The config creates `scenario_generation_model` as an independent copy of Gym's standard `policy_model`. Sampling is
owned by that model server. The agent does not send model, temperature, top-p, token-limit, or seed fields in child
requests.

Agent controls:

| Setting | Default | Meaning |
|---|---:|---|
| `model_server` | `scenario_generation_model` | Model-server instance used for child chat completions |
| `request_count` | `20` | Number of scenario-generation calls made for one input domain |
| `max_concurrency` | `20` | Maximum child calls in flight within one rollout |
| `scenarios_per_request` | `80` | Scenario count requested in each prompt; returned count is not enforced |
| `outside_policy_scope_fraction` | `0.1` | Independent probability that one call requests outside-policy scenarios |
| `random_seed` | `null` | Optional base seed for the rollout-local outside-policy schedule |

When `random_seed` is an integer, the schedule is derived from the seed and stable row identity: ID, task and rollout
indices, profile, and domain. Infrastructure retry indices and unrelated metadata do not change the sequence. With
`null`, the existing module-global random behavior is retained.

Model sampling and global provider concurrency remain model-server settings. `gym eval run --concurrency` controls how
many domain rollouts run concurrently in addition to each rollout's `max_concurrency`. Unknown agent settings are
rejected so misspelled controls cannot silently fall back to defaults.

Start the agent and its model server:

```bash
gym env start \
  --config responses_api_agents/conversational_tool_use/scenario_generation/configs/conversational_tool_use_scenario_generation.yaml \
  --model-type openai_model \
  --model-url "$MODEL_BASE_URL" \
  --model "$MODEL_NAME" \
  '+policy_api_key=${oc.env:MODEL_API_KEY}'
```

`gym env start` stays in the foreground. In a separate terminal, collect from the materialized input JSONL:

```bash
gym eval run --no-serve \
  --agent conversational_tool_use_scenario_generation \
  --input /tmp/conversational_tool_use/scenario_generation_inputs.jsonl \
  --output /tmp/conversational_tool_use/scenario_generation_rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```

With the checked-in defaults, one scenario-generation rollout makes 20 upstream model requests.

## Materialization

Convert rollout JSONL to the existing `conversational_tool_use_agent` input-row format with explicit paths:

```bash
python -m responses_api_agents.conversational_tool_use.scenario_generation.materialize \
  --input /path/to/scenario_generation_rollouts.jsonl \
  --output /path/to/conversational_tool_use_inputs.jsonl
```

The materializer preserves rollout order and `result.scenarios` order. It emits the canonical policy system prompt,
Responses API tools, raw simulator tools, seven-field customer scenario, domain name, generation profile, source
lineage, stable row ID, and `agent_ref` expected by `conversational_tool_use_agent`. IDs and lineage include available
Gym task, rollout, and retry-attempt coordinates. It does not emit temperature or any other sampling parameter. Model
and collection configuration own downstream sampling.

## Tests

```bash
gym env test \
  +entrypoint=responses_api_agents/conversational_tool_use/scenario_generation \
  +should_validate_data=false
```
