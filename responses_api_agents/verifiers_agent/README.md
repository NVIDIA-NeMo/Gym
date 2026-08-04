# Verifiers V1 agent

This agent runs a single-agent [Verifiers V1](https://github.com/PrimeIntellect-ai/verifiers)
taskset and harness inside its NeMo Gym server process. Verifiers owns the harness, tools,
task lifecycle, trace, and scoring; NeMo Gym owns the policy model and rollout collection.

The component has its own virtual environment, so its Verifiers and OpenAI versions do not
change the Gym head server's dependencies.

## Example

The included taskset asks the model to round-trip a word through a stateful Verifiers tool and
scores the exact answer. Configure the policy model in `env.yaml`, then run:

```bash
gym env start \
  --config responses_api_agents/verifiers_agent/configs/example.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --agent verifiers_agent \
  --input responses_api_agents/verifiers_agent/data/example.jsonl \
  --output responses_api_agents/verifiers_agent/data/example-rollouts.jsonl \
  --limit 1
```

The `verifiers` block in `configs/example.yaml` is a normal single-agent V1 environment config.
Change its `taskset` and `agent.harness` fields to use another installed V1 taskset or harness.
The agent loads the taskset once, keeps the V1 serving resources alive with the Gym server,
and runs the task selected by each row's `task_idx`.

## Export a taskset

Generate Gym rows in the same order as the configured V1 taskset:

```bash
cd responses_api_agents/verifiers_agent
uv run --python .venv/bin/python --no-project python3 scripts/create_dataset.py \
  --taskset scratchpad_taskset \
  --size 1 \
  --output data/tasks.jsonl
```

Gym creates this component's `.venv` when it starts the agent. Install any taskset-specific
dependency in this component's `requirements.txt`, then use the same taskset config in the export
command and the agent YAML. Hub tasksets can be addressed by their pinned `org/name@version`
identifier. Rows refer to tasks by index, so export and rollout collection must load the same
taskset version, config, and order.

The integration intentionally runs one V1 rollout per Gym row. V1 multi-agent environments and
group rewards are not supported. Responses harnesses preserve Gym's native output items,
including token IDs, log probabilities, and routed-expert payloads. Chat Completions and
Anthropic Messages harnesses remain useful for evaluation, but V1 does not retain their native
training fields in the trace.

## Licensing

Code: Apache 2.0

Data: N/A
