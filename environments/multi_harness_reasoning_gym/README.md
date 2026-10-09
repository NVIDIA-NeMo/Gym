# Multi-Harness Reasoning Gym

This profile routes one shared Reasoning Gym dataset across four native NeMo
Gym agent harnesses:

- Hermes (`hermes_reasoning_gym_agent`)
- OpenClaw (`openclaw_reasoning_gym_agent`)
- OpenCode (`opencode_reasoning_gym_agent`)
- Pi (`pi_reasoning_gym_agent`)

The pool is deterministic and ordered. Consecutive source tasks select Hermes,
OpenClaw, OpenCode, and Pi, then repeat. Every repeated generation of one task
stays on the selected harness, including retries and resumed runs.

## Prepare data

The profile reuses the Reasoning Gym generator and stores the generated file
beside the shared example data:

```bash
python environments/hermes_reasoning_gym/prepare.py \
  --task knights_knaves \
  --size 1000 \
  --output environments/hermes_reasoning_gym/data/train_knights_knaves.jsonl

gym dataset collate \
  --config environments/multi_harness_reasoning_gym/config.yaml \
  --output-dir data/multi_harness_reasoning_gym \
  --mode train_preparation
```

The composite config declares the dataset only once, on the shared
`reasoning_gym` resources server. Collation therefore writes one copy of each
task and stamps the stable source ordinal used for harness selection.

## Evaluate the routing

Start all four harnesses against an OpenAI-compatible policy model:

```bash
gym env start \
  --config environments/multi_harness_reasoning_gym/config.yaml \
  --model-type openai_model
```

Then collect repeated rollouts. No `--agent` is needed because `agent_pool`
routes every `task_source=reasoning_gym` row:

```bash
gym eval run --no-serve \
  --config environments/multi_harness_reasoning_gym/config.yaml \
  --input data/multi_harness_reasoning_gym/train.jsonl \
  --output results/multi_harness_reasoning_gym.jsonl \
  +num_repeats=2
```

For policy training, include the same config in the trainer's Gym
`config_paths`. Enable run-wide token capture for external harnesses in the
trainer config:

```yaml
env:
  nemo_gym:
    config_paths:
      - environments/multi_harness_reasoning_gym/config.yaml
    token_id_capture:
      enabled: true
      all_agents: true
```

For sharded NeMo RL jobs, place each agent on one shard and repeat the shared
`reasoning_gym` resources-server route on all four target shards.
