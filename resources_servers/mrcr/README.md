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
following the official
[openai/mrcr grading function](https://huggingface.co/datasets/openai/mrcr).

For reasoning models the vLLM server must be started with a reasoning
parser (e.g. `--reasoning-parser nano_v3` for Nemotron-3-Nano). Without
it, `<think>...</think>` leaks into `message.content` and the prefix gate
always fails.

## Start environment

```bash
gym env start \
    --resources-server mrcr \
    --model-type vllm_model
```

## Collect example rollouts

```bash
gym eval run --no-serve \
    --agent mrcr_simple_agent \
    --input resources_servers/mrcr/data/example.jsonl \
    --output resources_servers/mrcr/data/example_rollouts.jsonl
```

For the full benchmark run see
[`benchmarks/mrcr/README.md`](../../benchmarks/mrcr/README.md).

## Licensing

- Code: Apache 2.0
- Data ([openai/mrcr](https://huggingface.co/datasets/openai/mrcr)): see upstream license
