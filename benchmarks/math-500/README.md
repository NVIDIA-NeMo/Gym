# Math-500

Adds the `math-500` benchmark to Gym on top of the shared `math_with_judge`
resource server.

## Details

- Data source: OpenAI `prm800k/math_splits/test.jsonl`
- Evaluation: free-form math answer with symbolic verification
- Prompt: shared `generic_math` prompt mirroring Skills' `generic/math`
- Verification: symbolic-only (`should_use_judge: false`) to match Skills'
  default `eval_type=math`

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/math-500/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/math-500/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=math_500_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/math-500/data/math-500_benchmark.jsonl \
    +output_jsonl_fpath=results/math-500/rollouts.jsonl \
    +prompt_config=benchmarks/prompts/generic_math.yaml
```
