# Global-PIQA

Adds the `global_piqa` benchmark to Gym on top of the shared `mcqa` resource
server.

## Details

- Data source: `mrlbenchmarks/global-piqa-nonparallel`
- Evaluation: 2-choice multiple choice
- Prompt uses the benchmark-local template matching the original Global-PIQA
  format
- Each row carries the original Skills regex list via
  `template_metadata.output_regex`

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/global-piqa/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/global-piqa/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=global_piqa_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/global-piqa/data/global-piqa_benchmark.jsonl \
    +output_jsonl_fpath=results/global-piqa/rollouts.jsonl \
    +prompt_config=benchmarks/global-piqa/prompts/default.yaml
```
