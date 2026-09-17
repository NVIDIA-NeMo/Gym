# NeMo RL environment

A Gym environment for training models to train models.

1. The model runs in Claude Code to edit a recipe and pinned NeMo RL/Gym sources in a CPU sandbox
   (60 minutes, 200 turns, 256K context). Core changes such as loss fn, advantage estimation or
   data-processing changes are accepted.
2. A GPU sandbox applies the patch and trains Qwen2.5-1.5B-Instruct with
   NeMo RL + NeMo Gym GRPO: 8 prompts × 8 responses, with Gym's math verifier.
3. A fresh, unpatched GPU sandbox serves model with vLLM and runs
   `gym eval run` on AIME25 with a 32K context budget, including the prompt.

Reward is AIME25 avg@8: mean accuracy over eight sampled answers per problem
(30 problems, 240 answers; temperature 0.7).

## Setup

Configure `policy_model` and the sandbox connections in `config.yaml`. CPU/GPU
domains and API keys, author image, and registry credentials are declared there
using environment-variable references. The author image must provide Python at
`/agent_deps_mount/bin/python`, uv, and Git.

Generate datasets:

```bash
python responses_api_agents/nemorl_env/prepare.py
gym eval prepare --benchmark aime25
```

`app.py` orchestrates the sandboxes, `author_worker.py` runs Claude Code.
`task_environment/` contains the recipe, training data, runner, and evaluator.
`recipe.yaml` explicitly inherits NeMo RL's full `grpo_math_1B.yaml`, it is an
editable override, not a standalone config. The launcher writes the complete
`/testbed/resolved_grpo.yaml` before training. Data/environment edits are retained,
model, batch size, GPU/time budget, and held-out evaluation remain fixed.
`data/train.jsonl` contains one stable author prompt; `data/example.jsonl` contains
five independently identified copies. Use repeated rollouts for exploration.
`eval_concurrency` in `config.yaml` controls simultaneous AIME25 answers (default 30).

With the author model configured, record real example rollouts with:

```bash
gym env start --config responses_api_agents/nemorl_env/config.yaml --model-type vllm_model
gym eval run --no-serve --agent nemorl_env --input responses_api_agents/nemorl_env/data/example.jsonl --output responses_api_agents/nemorl_env/data/example_rollouts.jsonl
```

Math data: the first 512 training rows of NVIDIA's
[OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
(CC-BY-4.0)
