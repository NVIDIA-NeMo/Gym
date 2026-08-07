# Conversational Tool-Use Generation Workflow

The three generation stages run as self-contained Gym Responses API agents. Their rollout JSONL files are the
generation records. Pure Python materializers convert each stage's typed result into the next stage's Gym input JSONL.

The commands below assume the repository virtual environment is active:

```bash
source .venv/bin/activate
mkdir -p /tmp/conversational_tool_use

export MODEL_BASE_URL="https://inference-api.nvidia.com/v1"
export MODEL_API_KEY="<api-key>"
export MODEL_NAME="<model-id>"
```

Prepare the policy/tool generation references once per checkout:

```bash
python -m responses_api_agents.conversational_tool_use.policy_tool_generation.prepare
```

## Start the Pipeline Servers

Start the model, generation agents, conversation agent, and simulation resource server once:

```bash
gym env start \
  --config responses_api_agents/conversational_tool_use/domain_generation/configs/conversational_tool_use_domain_generation.yaml \
  --config responses_api_agents/conversational_tool_use/policy_tool_generation/configs/conversational_tool_use_policy_tool_generation.yaml \
  --config responses_api_agents/conversational_tool_use/scenario_generation/configs/conversational_tool_use_scenario_generation.yaml \
  --config resources_servers/conversational_tool_use_simulation/configs/conversational_tool_use_simulation.yaml \
  --model-type openai_model \
  --model-url "$MODEL_BASE_URL" \
  --model "$MODEL_NAME" \
  '+policy_api_key=${oc.env:MODEL_API_KEY}'
```

`gym env start` stays in the foreground. Run the remaining commands in a separate terminal. They use `--no-serve`
because they collect against that running stack and consume explicit JSONL paths.

The conversation policy uses the base `policy_model`. Generation uses independently overridable copies, while user
simulation, tool simulation, and conversation judging share `simulator_model`:

```text
domain_generation_model
policy_generation_model
policy_tool_judge_model
scenario_generation_model
simulator_model
```

`simulator_model` serves both the logical user-simulator and tool-simulator roles; those roles keep separate prompts
and parsers, and also serves the conversation judge. All copies inherit the base endpoint, key, and model by default.
Override any copied instance when a role should use a different model. For example, add these arguments to
`gym env start`:

```bash
"++domain_generation_model.responses_api_models.openai_model.openai_model=$DOMAIN_MODEL_NAME" \
"++policy_generation_model.responses_api_models.openai_model.openai_model=$POLICY_GENERATION_MODEL_NAME" \
"++policy_tool_judge_model.responses_api_models.openai_model.openai_model=$POLICY_TOOL_JUDGE_MODEL_NAME" \
"++scenario_generation_model.responses_api_models.openai_model.openai_model=$SCENARIO_MODEL_NAME" \
"++simulator_model.responses_api_models.openai_model.openai_model=$SIMULATOR_MODEL_NAME"
```

Override a copy's `openai_base_url` and `openai_api_key` too when it uses another provider.

The three layers of generation configuration are separate:

- Each generation agent's checked-in YAML controls its orchestration, including follow-up rounds, reference and judge
  counts, retry budget, scenario workload size, and per-rollout concurrency.
- The copied model-server instances own model sampling, output limits, provider transport behavior, and global endpoint
  concurrency. For `openai_model`, request defaults can be supplied through the model server's `extra_body`.
- `gym eval run --num-repeats` and `--concurrency` control dataset repetition and the number of independent Gym
  rollouts in flight.

Each generation agent README lists every supported agent setting and its checked-in default.

## 1. Generate Domains

```bash
gym eval run --no-serve \
  --agent conversational_tool_use_domain_generation \
  --input responses_api_agents/conversational_tool_use/domain_generation/data/example.jsonl \
  --output /tmp/conversational_tool_use/domain_rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```

## 2. Materialize Policy and Tool Inputs

Choose exactly one profile:

```bash
python -m responses_api_agents.conversational_tool_use.domain_generation.materialize \
  --input-file /tmp/conversational_tool_use/domain_rollouts.jsonl \
  --output-file /tmp/conversational_tool_use/policy_tool_inputs.jsonl \
  --profile general
```

## 3. Generate Policies and Tools

The config creates separate policy-generation and judge model instances by copying `policy_model`. Override either copy
when those roles should use different model settings.

```bash
gym eval run --no-serve \
  --agent conversational_tool_use_policy_tool_generation \
  --input /tmp/conversational_tool_use/policy_tool_inputs.jsonl \
  --output /tmp/conversational_tool_use/policy_tool_rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```

## 4. Materialize Scenario Inputs

```bash
python -m responses_api_agents.conversational_tool_use.policy_tool_generation.materialize \
  --input-path /tmp/conversational_tool_use/policy_tool_rollouts.jsonl \
  --output-path /tmp/conversational_tool_use/scenario_generation_inputs.jsonl
```

## 5. Generate Scenarios

By default, one input domain launches 20 upstream requests, allows all 20 in flight, and asks each request for 80
scenarios. These values are `request_count`, `max_concurrency`, and `scenarios_per_request` in the scenario-agent YAML.

```bash
gym eval run --no-serve \
  --agent conversational_tool_use_scenario_generation \
  --input /tmp/conversational_tool_use/scenario_generation_inputs.jsonl \
  --output /tmp/conversational_tool_use/scenario_generation_rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```

## 6. Materialize Conversation Inputs

```bash
python -m responses_api_agents.conversational_tool_use.scenario_generation.materialize \
  --input /tmp/conversational_tool_use/scenario_generation_rollouts.jsonl \
  --output /tmp/conversational_tool_use/conversation_inputs.jsonl
```

The materializer writes the policy system prompt and tools but intentionally omits an initial customer message. With
the default `seed_first_user_message=true` setting, the simulator generates that first turn. Callers may instead add
`initial_user_message` or prefilled Responses history as described in
[rollout behavior](rollout.md#initial-user-turn-and-prefilled-history).

## 7. Run Conversation Simulation

```bash
gym eval run --no-serve \
  --agent conversational_tool_use_agent \
  --input /tmp/conversational_tool_use/conversation_inputs.jsonl \
  --output /tmp/conversational_tool_use/conversation_rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```

The materializers are offline typed JSONL transforms, so they use Python rather than starting another Gym server. Each
transform preserves `profile` and deep-copies `source_artifacts`, then appends the current stage's `id`,
`_ng_task_index`, `_ng_rollout_index`, `_ng_attempt_index` when present, and stage-local candidate, generation-attempt,
or scenario index. Derived row IDs include available Gym task, rollout, and retry-attempt coordinates so repeated
collection cannot collide. Local filesystem paths are not embedded in the lineage.
