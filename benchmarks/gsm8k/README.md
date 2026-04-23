# GSM8K

Grade school math word problems (test split, 1319 problems). Mirrors
nemo-skills' `nemo_skills/dataset/gsm8k`.

Data is fetched from the upstream openai/grade-school-math repo at
prepare time. `prepare.py` applies the same Skills transforms
(hardcoded answer fixes, `<<...>>` calc-string stripping, int-cast when
the expected answer is integer-valued), then renames `problem` ->
`question` for Gym.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/gsm8k/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/gsm8k/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=gsm8k_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/gsm8k/data/gsm8k_benchmark.jsonl \
    +output_jsonl_fpath=results/gsm8k_rollouts.jsonl \
    +num_repeats=4
```
