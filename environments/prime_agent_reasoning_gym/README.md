# Prime Agent Reasoning Gym

Runs reasoning gym tasks through Prime Agent.

```bash
gym env start --environment prime_agent_reasoning_gym --model-type openai_model

gym eval run --no-serve \
  --agent prime_agent_reasoning_gym_agent \
  --input environments/prime_agent_reasoning_gym/data/example.jsonl \
  --output environments/prime_agent_reasoning_gym/data/example_rollouts.jsonl \
  --limit 5
```

`data/example_rollouts.jsonl` contains five Prime Agent 0.7.0 rollouts generated with a deterministic
local policy and real IPython execution.

```bash
python environments/prime_agent_reasoning_gym/prepare.py --task knights_knaves --size 1000
```
