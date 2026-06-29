# LongCodeBench (LongCodeQA)

[LongCodeBench](https://huggingface.co/datasets/Steefano/LCB) is a multi-choice
question-answering benchmark over long code contexts. Each row presents a
long code prompt with options A/B/C/D and asks the model to pick the correct
letter; the prompt postfix instructs the model to emit `Answer: \boxed{X}`.

This benchmark reuses the existing `mcqa` resource server with
`grading_mode=strict_single_letter_boxed`. Each row's `question` field carries
the long code prompt plus the postfix; the shared
`benchmarks/prompts/generic/default.yaml` template (`user: "{question}"`)
wraps it as a single user message, mirroring NeMo Skills' `prompt_format=openai`
behaviour.

## Variants

| Variant | Config | Prepare script | Tokenizer | Max tokens | Output |
|---|---|---|---|---|---|
| Default | `config.yaml` | `prepare.py` | `o200k_base` (tiktoken) | none (no filter) | `data/longcodebench_benchmark.jsonl` |
| N3 1M | `config_n3_1m.yaml` | `prepare_n3_1m.py` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (HF) | `1048576` | `data/longcodebench_n3_1m_benchmark.jsonl` |

The N3 1M variant requires HF auth for the gated NVIDIA repo
(`HF_TOKEN` env or `huggingface-cli login`).

For one-off custom builds (different tokenizer / cap / output path),
invoke `prepare.py` directly:

```bash
python benchmarks/longcodebench/prepare.py \
    --tokenizer_name cl100k_base \
    --max_context_tokens 131072 \
    --output_fpath benchmarks/longcodebench/data/longcodebench_cl100k_128k_benchmark.jsonl
```

## Example usage

```bash
# Prepare benchmark data (default)
gym eval prepare --benchmark longcodebench

# Prepare benchmark data (N3 1M variant)
gym eval prepare --benchmark longcodebench/config_n3_1m

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark longcodebench

# Collecting rollouts — default
gym eval run --no-serve \
    --agent longcodebench_mcqa_simple_agent \
    --input benchmarks/longcodebench/data/longcodebench_benchmark.jsonl \
    --output results/longcodebench_rollouts.jsonl \
    --num-repeats 4 \
    --prompt-config benchmarks/prompts/generic/default.yaml

# Collecting rollouts — N3 1M
gym eval run --no-serve \
    --agent longcodebench_n3_1m_mcqa_simple_agent \
    --input benchmarks/longcodebench/data/longcodebench_n3_1m_benchmark.jsonl \
    --output results/longcodebench_n3_1m_rollouts.jsonl \
    --num-repeats 4 \
    --prompt-config benchmarks/prompts/generic/default.yaml
```
