# DSPy recipe

In this example we use GEPA or MIPRO from DSPy in `optimize.py` to optimize a NeMo Gym agent system prompt. It uses a running agent's
`/run` endpoint, so no changes to NeMo Gym are needed. Supports `--optimizer gepa` and `--optimizer mipro`. In the future, this can 
be expanded to integrate with NeMo Gym rollout collection helpers to automate server launching, port discovery, and CLI integration.

## 1. Set up Gym

```bash
git clone https://github.com/NVIDIA-NeMo/Gym
cd Gym
uv venv ; source .venv/bin/activate; uv sync
uv pip install "dspy>=2.5" gepa requests "openai<=2.7.2"
```

## 2. Set env.yaml

Set `env.yaml` in the repo root, for example:

```yaml
hf_token: <your-hf-token>
policy_base_url: https://inference-api.nvidia.com/v1
policy_api_key: <your-api-key>
policy_model_name: nvidia/meta/llama-3.1-70b-instruct
```

## 3. Prepare data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/gpqa/config.yaml]"
```

## 4. Launch a Gym agent

```bash
ng_run "+config_paths=[benchmarks/gpqa/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

From the logs, copy the agent endpoint URL (`gpqa_mcqa_simple_agent`), e.g. `http://127.0.0.1:17349`. The
port is assigned per run, so use the one in your own logs, not this example. If optimize.py reports the agent is
not reachable, the URL is wrong.

## 5. Optimize

```bash
python scripts/dspy/optimize.py \
  --agent-url http://127.0.0.1:12185 \
  --data benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl \
  --seed "Answer the question." \
  --reflection-model aws/anthropic/bedrock-claude-opus-4-7 \
  --reflection-base-url https://inference-api.nvidia.com/v1 \
  --reflection-api-key sk-cYILjzbwwF57PMufDEvmdw \
  --max-calls 300
```

Notes:
- `--reflection-model` is the LLM that writes better prompts (use a strong one). It is used through DSPy's OpenAI-compatible client against `--reflection-base-url`.
- `--optimizer gepa` (default) or `--optimizer mipro`.
- `--max-calls` is the rollout budget. More calls means more iterations.
- `--seed` is the starting prompt. A short generic seed leaves room to improve for a demo.

## 6. Output

It prints the val accuracy before and after, and the final best prompt. It also writes results and the
per-iteration curve to `gepa_results.json` (set with `--out`). Plot the curve (iteration vs accuracy):

```bash
python scripts/dspy/plot.py gepa_results.json -o gepa_curve.png
```