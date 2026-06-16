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
ng_prepare_benchmark "+config_paths=[benchmarks/longcodebench/config.yaml]"

# Prepare benchmark data (N3 1M variant)
ng_prepare_benchmark "+config_paths=[benchmarks/longcodebench/config_n3_1m.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/longcodebench/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts — default
ng_collect_rollouts \
    +agent_name=longcodebench_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/longcodebench/data/longcodebench_benchmark.jsonl \
    +output_jsonl_fpath=results/longcodebench_rollouts.jsonl \
    +prompt_config=benchmarks/prompts/generic/default.yaml \
    +num_repeats=4

# Collecting rollouts — N3 1M
ng_collect_rollouts \
    +agent_name=longcodebench_n3_1m_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/longcodebench/data/longcodebench_n3_1m_benchmark.jsonl \
    +output_jsonl_fpath=results/longcodebench_n3_1m_rollouts.jsonl \
    +prompt_config=benchmarks/prompts/generic/default.yaml \
    +num_repeats=4
```
