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
`domain`, `sub_domain`, `difficulty`, `length`, `context_tokens` via
tiktoken `cl100k_base`) and additionally emits `options` and
`grading_mode` for the mcqa server.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/longbench_v2/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/longbench_v2/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=longbench_v2_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/longbench_v2/data/longbench_v2_benchmark.jsonl \
    +output_jsonl_fpath=results/longbench_v2_rollouts.jsonl \
    +prompt_config=benchmarks/longbench_v2/prompts/default.yaml \
    +num_repeats=4
```
