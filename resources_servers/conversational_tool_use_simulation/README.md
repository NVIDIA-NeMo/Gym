# Conversational Tool-Use Simulation

This package contains the Gym environment for generated conversational tool-use tasks. Domain, policy/tool, and
scenario artifacts are produced by independent Responses API agents.

## Components

| Component | Implementation | Responsibility |
| --- | --- | --- |
| Rollout simulation | `app.py` | Simulate users and tools, maintain session state, and verify trajectories |
| Domain generation | `responses_api_agents/conversational_tool_use/domain_generation` | Generate domain candidates |
| Policy/tool generation | `responses_api_agents/conversational_tool_use/policy_tool_generation` | Generate, refine, validate, and judge policies and tools |
| Scenario generation | `responses_api_agents/conversational_tool_use/scenario_generation` | Generate customer scenarios |
| Policy agent | `responses_api_agents/conversational_tool_use/simulation` | Run the policy-model conversation loop |

The generation agents are connected through standard Gym rollout JSONL and explicit materialization steps:

```text
domain-generation rollouts
  -> policy/tool-generation inputs and rollouts
  -> scenario-generation inputs and rollouts
  -> conversation inputs
  -> policy agent + rollout simulation
```

Each generation agent README documents its input, output, and materializer. See
[generation workflow](docs/generation.md) for runnable end-to-end commands and
[rollout behavior](docs/rollout.md) for the session and verification contract.

## Prepare Assets

Prepare the standalone generation and reference assets before running the generation pipeline:

```bash
python -m resources_servers.conversational_tool_use_simulation.prepare
```

This downloads the prompt and policy/tool reference assets from
[`nvidia/NeMo-Gym-Conversational-Tool-Use-Assets`](https://huggingface.co/datasets/nvidia/NeMo-Gym-Conversational-Tool-Use-Assets).
Add `--include-prompt-history` to materialize the optional prompt history. Prompts implemented directly in the rollout
server or policy agent remain in Python and require no preparation. JSON schemas and example JSONL files remain in Git.

## Rollout Resource Server

It owns per-session domain state:

- policy markdown
- generated tool signatures
- customer scenario
- simulated trajectory messages
- user simulator, tool-result simulator, and judge configuration

The user simulator and tool-result simulator are separate logical roles, but they can both point at the same `simulator_model_server` in practice. They keep separate prompts, output parsing, retries, and metrics so either role can be swapped later.

## Main Routes

- `/seed_session`: loads one generated domain/scenario row into session state.
- `/session_tools`: returns generated tools in Responses API function-tool format.
- `/record_agent_outputs`: records one policy response's selected messages and function calls in provider order.
- `/record_agent_message`: compatibility route for appending one assistant message.
- `/record_agent_step_limit`: records assistant-step exhaustion after final response items have been stored.
- `/next_user_message`: generates or deterministically returns the next user turn.
- `/execute_agent_tool_call`: executes a previously recorded policy function call.
- `/discard_session`: idempotently removes abandoned session state.
- `/{tool_name}`: catch-all generated tool route that simulates a tool result.
- `/verify`: scores the completed trajectory with the configured message-verification pipeline.

The complete rollout row is the portable artifact. Its top-level `responses_create_params` and `response` fields are
the standard Gym-facing policy transcript, while `result` stores generation profile and lineage and
`result.trajectory` stores simulator-native user, agent, tool, terminal, and verification state that the Responses API
representation cannot carry.

One policy response can become several logical trajectory messages when it contains multiple output items. The complete
raw response is stored only on the first such message. Every decomposed message records the same `response_id` and its
own `response_output_index`, so consumers can recover the source item without duplicating the response payload.

Verification is message-level:

- user and environment/tool-result messages are judged first when termination is enabled
- agent text and tool-call messages are judged after user/tool-result messages pass
- a final agent-conversation judge runs only when the message checks leave positive agent reward
- user failures are labeled `user_failure`
- tool-result failures are labeled `tool_failure`
- agent failures are labeled `agent_failure`

Final verification state is synchronized before the response is built. The top-level reward, invalid reasons, failure
labels, judge diagnostics, and judge-generation error therefore agree with the verification objects stored in
`result.trajectory`. After a successful `/verify`, the server removes that session from process memory. Provider errors
leave it intact so direct resource-server callers can retry verification. The policy agent discards the session after
rollout or verification failure.

Judge retries distinguish provider availability from semantic output quality. `judge_provider_attempts` retries only
`408`, `409`, `425`, `429`, `5xx`, connection failures, and timeouts, using exponential backoff bounded by
`judge_provider_retry_max_backoff_seconds`. `generation_attempts` independently controls retries for judge responses
that arrive successfully but cannot be parsed or validated. Valid judge decisions, including reward zero, are never
retried.

An exhausted transient judge request makes `/verify` return `503`. The conversational tool-use agent converts that into
Gym's `_ng_failure_class: transient` result, so the collector writes the attempt to the failure sidecar and applies
the row-level `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS` policy instead of writing a scored zero. Nonretryable judge-provider
HTTP errors are returned without provider retry and do not become semantic rewards.

Rows normally omit conversation history and let the resource server generate the first user turn. When a row contains
prefilled Responses API history, the resource server hydrates the corresponding user, assistant, function-call, and
function-call-output events into simulator state and resumes from the next actor. Reasoning and system/developer items
remain policy-only context. The nested trajectory records both `prefill_message_count` and
`continuation_start_index` in simulator-message coordinates.

Before seeding, the policy agent verifies that the materialized system prompt is the exact prompt rendered from the
top-level policy and that the model-visible function tools are the exact rendering of the top-level simulator tools.
This prevents the policy model and verifier from operating on different task contracts. If a provider emits assistant
text and function calls in one response, both are recorded in order in the simulator trajectory before tool execution,
matching the policy-visible transcript. The items retain their shared `response_id` and distinct
`response_output_index` values.

When Gym observability is enabled, the policy, user-simulator, tool-simulator, and judge calls all inherit the rollout
correlation prefix. Standard model-call captures therefore remain attached to the rollout that caused them.

This is intended to be used with `responses_api_agents/conversational_tool_use/simulation`.

The rollout cap is owned by that agent as `max_agent_steps` and counts policy-model responses. The resource server no
longer applies a competing message-count limit. If the final policy response contains tool calls,
`/record_agent_outputs` first stores and validates the calls; `/record_agent_step_limit` then marks the trajectory
incomplete with `excessive_length` without invoking the tool simulator.

For executed calls, the route response keeps validation and terminal-state fields for orchestration, while its
`output` field is the exact text generated by the tool simulator. The agent extracts only that field for the
policy-visible tool message.

## Configuration Reference

[`conversational_tool_use_simulation.yaml`](configs/conversational_tool_use_simulation.yaml) is the canonical runnable
stack. User simulation, tool simulation, and judging share `simulator_model`. Requests for all three roles use
temperature `1.0`, top-p `1.0`, at most `8192` output tokens, and no parallel tool calls. The policy runtime owns
policy sampling and context length. Deterministic transfer-ground-truth enforcement is enabled.

The verification and retry controls are:

| Field | Canonical value | Behavior |
| --- | ---: | --- |
| `generation_attempts` | `3` | Maximum semantic attempts for generating a valid user message, valid tool result, or parseable judge decision. It does not retry provider transport failures. |
| `judge_provider_attempts` | `3` | Maximum judge requests after retryable provider or transport failures. Valid semantic reward-zero decisions are never retried. |
| `judge_provider_retry_initial_backoff_seconds` | `0.5` | Delay before the first judge-provider retry; the delay doubles after each failed attempt. |
| `judge_provider_retry_max_backoff_seconds` | `8.0` | Upper bound on the exponential judge-provider retry delay. |
| `enable_llm_judge` | `true` | Runs model-based verification. When `false`, a complete trajectory with no structural or transfer-gate failure receives reward `1.0` without judge calls. |
| `enable_termination` | `true` | In `message` verification, checks user and tool-result messages first and stops later judge stages after a failure. When `false`, every message and the complete agent conversation are judged. It does not affect combined verification. |
| `verification_type` | `message` | `message` performs staged per-message checks followed by an agent-conversation check. `complete_trajectory_combined_evaluation` sends the full trajectory through one combined user/agent/environment evaluation. |
| `enforce_transfer_ground_truth` | `true` | Compares observed transfer behavior with `customer_scenario.outside_policy_scope`. A mismatch deterministically receives reward `0` and skips the LLM judge; a match continues through normal verification. |

`user_responses_create_params`, `tool_simulator_responses_create_params`, and `judge_responses_create_params` own the
sampling parameters sent to those three logical roles. Their model-server references may point to one shared model, as
in the canonical config, or to independently configured model instances.

## Run the Environment

The config defines `simulator_model` as a copy of Gym's standard `policy_model`. The user simulator, tool simulator,
and judge share that instance, so a single-model run requires only one model configuration. Start the server stack:

```bash
gym env start \
  --config resources_servers/conversational_tool_use_simulation/configs/conversational_tool_use_simulation.yaml \
  --model-type openai_model \
  --model-url "$MODEL_BASE_URL" \
  --model "$MODEL_NAME" \
  '+policy_api_key=${oc.env:MODEL_API_KEY}'
```

`gym env start` stays in the foreground. In a separate terminal, collect from an explicit input JSONL:

```bash
gym eval run --no-serve \
  --agent conversational_tool_use_agent \
  --input resources_servers/conversational_tool_use_simulation/data/example.jsonl \
  --output /tmp/conversational_tool_use/conversation_rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```

Override the shared simulator instance when the environment roles should use a different model from the policy:

```bash
gym env start \
  --config resources_servers/conversational_tool_use_simulation/configs/conversational_tool_use_simulation.yaml \
  --model-type openai_model \
  --model-url "$MODEL_BASE_URL" \
  --model "$POLICY_MODEL_NAME" \
  '+policy_api_key=${oc.env:MODEL_API_KEY}' \
  "++simulator_model.responses_api_models.openai_model.openai_model=$SIMULATOR_MODEL_NAME"
```

The copied instance inherits the policy model's endpoint and key. Override its `openai_base_url` and `openai_api_key`
fields as well when it uses another provider. To use a separate judge, define another model-server instance in a
derived config and point `judge_model_server` to it.

## Offline Dataset Preparation

The checked-in five-row examples are `data/example.jsonl` and `data/example_parallel_tool_calls.jsonl`. They are
generated from raw per-domain seed artifacts, not from chat-training JSONLs that may contain literal DSML tool-call
leakage.

These commands transform local artifacts and do not call a model, so they are ordinary Python data-preparation
commands rather than `gym eval run` commands. Regenerate the examples after setting `CONVERSATIONAL_TOOL_USE_GENERAL_SOURCE_DIR` and
`CONVERSATIONAL_TOOL_USE_PROACTIVE_SOURCE_DIR`:

```bash
python resources_servers/conversational_tool_use_simulation/scripts/build_conversational_tool_use_dataset.py \
  --dataset-name example \
  --output-path resources_servers/conversational_tool_use_simulation/data/example.jsonl \
  --report-path /tmp/conversational_tool_use_example.report.json \
  --max-rows 5

python resources_servers/conversational_tool_use_simulation/scripts/build_conversational_tool_use_dataset.py \
  --dataset-name example_parallel_tool_calls \
  --output-path resources_servers/conversational_tool_use_simulation/data/example_parallel_tool_calls.jsonl \
  --report-path /tmp/conversational_tool_use_example_parallel_tool_calls.report.json \
  --max-rows 5 \
  --parallel-tool-calls
```

The builder strictly rejects incomplete tools, malformed scenarios, duplicate tool names, non-schema `params` or
`returns`, and DeepSeek V3.2 template leakage such as DSML blocks, fullwidth role sentinels, thinking tags, and
function-result wrappers. By default it samples at most one scenario per domain and scans the first 100 numeric domains
per source. Build reports are generated beside the requested output unless `--report-path` points elsewhere; reports
and full datasets are not checked in.

Rows include a top-level `metadata` dict with the source dataset name, source domain index, domain name, representative
domain, number of tools, tool names, scenario index, scenario file, scenario line, and generator model metadata:

- domain generator: `Qwen3-235B-A22B-Thinking-2507`
- policy/tools model: `DeepSeek-R1-0528`
- scenario generator: `Qwen3-235B-A22B-Thinking-2507`

Build the full raw-seed datasets with unbounded row/domain limits. Set
`CONVERSATIONAL_TOOL_USE_GENERAL_SOURCE_DIR` and `CONVERSATIONAL_TOOL_USE_PROACTIVE_SOURCE_DIR` to the two raw
seed policy directories, then select any subset of the `general`, `proactive`, `combined`, `general_parallel`,
`proactive_parallel`, and `combined_parallel` jobs:

```bash
python resources_servers/conversational_tool_use_simulation/scripts/build_full_conversational_tool_use_datasets.py \
  --jobs general proactive combined
```

The full builder writes `conversational_tool_use_<job>.jsonl` plus a contract report beside each dataset. Add
`general_parallel`, `proactive_parallel`, or `combined_parallel` to build variants whose row metadata and
`responses_create_params.parallel_tool_calls` fields are both `true`. `--skip-existing` skips only outputs whose reports
match the current dataset schema, source names, profiles, build limits, source-content fingerprints, validated row
count, file size, and SHA-256 digest.

For custom source paths, pass one `--source-dir`, `--source-name`, and `--source-profile general|proactive` per source to
`build_conversational_tool_use_dataset.py`. Explicit profiles take precedence over name-based defaults and must agree
with profile metadata present in the source artifacts.

## Tests

Run the resource server's isolated Gym test, including checked-in data validation:

```bash
gym env test --resources-server conversational_tool_use_simulation
```
