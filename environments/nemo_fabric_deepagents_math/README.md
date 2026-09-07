# NeMo Fabric DeepAgents Math

Runs five math examples with DeepAgents through NeMo Fabric and grades them with `math-verify`.

```bash
gym env start --environment nemo_fabric_deepagents_math --model-type openai_model

gym eval run --no-serve \
  --agent nemo_fabric_deepagents_math_agent \
  --input environments/nemo_fabric_deepagents_math/data/example.jsonl \
  --output environments/nemo_fabric_deepagents_math/data/example_rollouts.jsonl \
  --limit 5
```
