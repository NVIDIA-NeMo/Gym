# Prime Agent Math

Runs math tasks with Prime Agent and verifies boxed answers with `math-verify`.

```bash
gym env start --environment prime_agent_math --model-type openai_model

gym eval run --no-serve \
  --agent prime_agent_math_agent \
  --input environments/prime_agent_math/data/example.jsonl \
  --output environments/prime_agent_math/data/example_rollouts.jsonl \
  --limit 5
```

`data/example_rollouts.jsonl` contains five Prime Agent 0.7.0 rollouts generated with a deterministic
local policy and real IPython execution.

```bash
python environments/prime_agent_math/prepare.py --split train
```
