# MRCR resources server

OpenAI's Multi-Round Coreference Resolution ([MRCR](https://huggingface.co/datasets/openai/mrcr))
benchmark. Each task is a multi-turn conversation where the model has
produced several outputs of the same kind (e.g. multiple poems); the final
turn asks the model to reproduce the Nth occurrence exactly, prefixed by a
random token.

## Scoring

1. The response must start with a `random_string_to_prepend` prefix
   (reward = 0.0 if missing).
2. Otherwise the prefix is stripped from both response and expected
   answer, and `difflib.SequenceMatcher(...).ratio()` becomes the reward
   (continuous similarity in [0, 1]).

Grader ported from
https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/mrcr.py
which follows the official
[openai/mrcr grading function](https://huggingface.co/datasets/openai/mrcr).

### Launch local vllm server
```bash
pip install -U "vllm>=0.12.0"

wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py

vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 10240 \
  --trust-remote-code \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```

The `--reasoning-parser` flag is essential: MRCR's prefix gate expects the
response to start with `random_string_to_prepend`, and Nemotron-3 emits a
`<think>...</think>` preamble before its final answer. Without a reasoning
parser configured, the preamble leaks into `message.content` and every
sample scores 0.

### Set `env.yaml` in `Gym/`
```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

### Install Gym
```bash
cd Gym/
uv venv
source .venv/bin/activate
uv sync
```

### Start MRCR environment
```bash
ng_run "+config_paths=[resources_servers/mrcr/configs/mrcr.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

### Collect example rollouts
```bash
ng_collect_rollouts \
    +agent_name=mrcr_simple_agent \
    +input_jsonl_fpath=resources_servers/mrcr/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/mrcr/data/example_rollouts.jsonl
```

### Run the full MRCR benchmark
See [`benchmarks/mrcr/README.md`](../../benchmarks/mrcr/README.md) for the
benchmark config (data preparation + 4×rollouts with per-needle-count
subset metrics).

For training, see the
[docs](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/nemo-rl-grpo/index.html).

## Licensing

- Code: Apache 2.0
- Data ([openai/mrcr](https://huggingface.co/datasets/openai/mrcr)): see upstream license
