# LongBench-v2

[LongBench v2](https://arxiv.org/abs/2412.15204) is a multiple-choice
long-context benchmark (4 choices: A/B/C/D, 503 questions) covering 6
domains over 8k-2M-word contexts: single-doc QA, multi-doc QA, long
in-context learning, long-dialogue history, code-repo understanding,
and long structured data.

Mirrors nemo-skills' `nemo_skills/dataset/longbench-v2`. Reuses the
existing [`mcqa`](../../resources_servers/mcqa) resource server for
grading; this directory adds only the dataset and prompt.

Data source: HuggingFace `THUDM/LongBench-v2` (single "train" split,
which is the full eval set). `prepare.py` preserves every Skills
field (`index`, `context`, `question`, `choice_A..D`, `expected_answer`,
`domain`, `sub_domain`, `difficulty`, `length`, `context_tokens`) and
additionally emits `options` and `grading_mode` for the mcqa server.

## Variants

| Variant | Config | Prepare script | Tokenizer | Max tokens | Output |
|---|---|---|---|---|---|
| Default | `config.yaml` | `prepare.py` | `o200k_base` (tiktoken) | none (no filter) | `data/longbench_v2_benchmark.jsonl` |
| N3 1M | `config_n3_1m.yaml` | `prepare_n3_1m.py` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (HF) | `1048576` | `data/longbench_v2_n3_1m_benchmark.jsonl` |

The N3 1M variant requires HF auth for the gated NVIDIA repo
(`HF_TOKEN` env or `huggingface-cli login`). LongBench-v2 contexts
span 8k-2M words, so the long-bucket rows above 1M tokens are filtered
out under the N3 1M cap.

For one-off custom builds (different tokenizer / cap / output path),
invoke `prepare.py` directly:

```bash
python benchmarks/longbench_v2/prepare.py \
    --tokenizer_name cl100k_base \
    --max_context_tokens 131072 \
    --output_fpath benchmarks/longbench_v2/data/longbench_v2_cl100k_128k_benchmark.jsonl
```

## Example usage

```bash
# Prepare benchmark data (default)
ng_prepare_benchmark "+config_paths=[benchmarks/longbench_v2/config.yaml]"

# Prepare benchmark data (N3 1M variant)
ng_prepare_benchmark "+config_paths=[benchmarks/longbench_v2/config_n3_1m.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/longbench_v2/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts — default
ng_collect_rollouts \
    +agent_name=longbench_v2_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/longbench_v2/data/longbench_v2_benchmark.jsonl \
    +output_jsonl_fpath=results/longbench_v2_rollouts.jsonl \
    +prompt_config=benchmarks/longbench_v2/prompts/default.yaml \
    +num_repeats=4

# Collecting rollouts — N3 1M
ng_collect_rollouts \
    +agent_name=longbench_v2_n3_1m_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/longbench_v2/data/longbench_v2_n3_1m_benchmark.jsonl \
    +output_jsonl_fpath=results/longbench_v2_n3_1m_rollouts.jsonl \
    +prompt_config=benchmarks/longbench_v2/prompts/default.yaml \
    +num_repeats=4
```
