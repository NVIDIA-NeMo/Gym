# comp-math-24-25

AIME 2024, AIME 2025, and HMMT February 2024-2025 problems combined
(256 problems total: 30 AIME24 + 30 AIME25 + 196 HMMT24-25). Mirrors
nemo-skills' `nemo_skills/dataset/comp-math-24-25`.

Source data is a static JSONL snapshot (`data/test.txt`, 198KB) copied
verbatim from the Skills repository. `prepare.py` converts it to Gym's
benchmark format (`problem` -> `question`, all other fields preserved).

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/comp_math_24_25/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/comp_math_24_25/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=comp_math_24_25_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/comp_math_24_25/data/comp_math_24_25_benchmark.jsonl \
    +output_jsonl_fpath=results/comp_math_24_25_rollouts.jsonl \
    +num_repeats=4
```
