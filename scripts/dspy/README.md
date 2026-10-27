# DSPy recipe

In this example we use GEPA or MIPRO from DSPy in `optimize.py` to optimize a NeMo Gym agent system prompt. It uses a running agent's
`/run` endpoint, so no changes to NeMo Gym are needed. Supports `--optimizer gepa` and `--optimizer mipro`. In the future, this can 
be expanded to integrate with NeMo Gym rollout collection helpers to automate server launching, port discovery, and CLI integration.

## 1. Set up Gym

```bash
git clone https://github.com/NVIDIA-NeMo/Gym
cd Gym
uv venv ; source .venv/bin/activate; uv sync
uv pip install "dspy[optuna]>=2.5" gepa requests matplotlib
```

## 2. Set env.yaml

Set `env.yaml` in the repo root to an OpenAI-compatible endpoint, for example a vLLM server:

```yaml
policy_base_url: http://<host>:8000/v1
policy_api_key: EMPTY
policy_model_name: <served-model-name>
```

## 3. Prepare data

Optimize on training data, never on a benchmark's test set. This example uses the math_with_judge environment.
Its 5-row `example.jsonl` is enough for a smoke run. For a real run, download its training split:

```bash
gym dataset download --repo-id nvidia/Nemotron-RL-math-OpenMathReasoning --artifact train.jsonl \
  --output resources_servers/math_with_judge/data/OpenMathReasoning_train.jsonl
```

## 4. Launch a Gym agent

```bash
gym env start --resources-server math_with_judge --model-type vllm_model
```

From the logs, copy the agent endpoint URL (`math_with_judge_simple_agent`), e.g. `http://127.0.0.1:17349`. The
port is assigned per run, so use the one in your own logs, not this example. If optimize.py reports the agent is
not reachable, the URL is wrong.

## 5. Optimize

```bash
python scripts/dspy/optimize.py \
  --agent-url http://127.0.0.1:12185 \
  --data resources_servers/math_with_judge/data/OpenMathReasoning_train.jsonl \
  --seed "Solve the problem." \
  --reflection-model <served-model-name> \
  --reflection-base-url http://<host>:8000/v1 \
  --reflection-api-key EMPTY \
  --max-calls 300
```

Notes:
- `--reflection-model` is the LLM that writes better prompts (use a strong one). It is used through DSPy's OpenAI-compatible client against `--reflection-base-url`. Raise `--reflection-max-tokens` for reasoning models, which can spend the default budget on thinking.
- `--optimizer gepa` (default) or `--optimizer mipro`.
- `--max-calls` is the rollout budget. More calls means more iterations.
- `--seed` is the starting prompt. A short generic seed leaves room to improve for a demo.

## 6. Output

It prints the val accuracy before and after, and the final best prompt. It also writes results and the
per-iteration curve to `gepa_results.json` (set with `--out`). Plot the curve (iteration vs accuracy):

```bash
python scripts/dspy/plot.py gepa_results.json -o gepa_curve.png
```
