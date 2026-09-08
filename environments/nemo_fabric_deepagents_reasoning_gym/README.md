# NeMo Fabric DeepAgents Reasoning Gym

Runs five Reasoning Gym examples with DeepAgents through NeMo Fabric.

```bash
gym env start --environment nemo_fabric_deepagents_reasoning_gym --model-type openai_model

gym eval run --no-serve \
  --agent nemo_fabric_deepagents_reasoning_gym_agent \
  --input environments/nemo_fabric_deepagents_reasoning_gym/data/example.jsonl \
  --output environments/nemo_fabric_deepagents_reasoning_gym/data/example_rollouts.jsonl \
  --limit 5
```
