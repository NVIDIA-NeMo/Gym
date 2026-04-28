# bigcodebench

Port of NeMo-Skills' [`bigcodebench`](https://github.com/bigcode-project/bigcodebench)
benchmark. The dataset, prompt template, calibration, and code-extraction
logic mirror Skills' implementation byte-for-byte. Verification is
delegated to the [`bigcodebench`](../../resources_servers/bigcodebench/)
resource server.

The `hard` split (148 problems, default) is `bigcode/bigcodebench-hard@v0.1.4`;
the `full` split (~1140 problems) is `bigcode/bigcodebench@v0.1.4`.

## Example usage

```bash
# Prepare benchmark data (hard split, ~148 problems)
ng_prepare_benchmark "+config_paths=[benchmarks/bigcodebench/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/bigcodebench/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=bigcodebench_simple_agent \
    +input_jsonl_fpath=benchmarks/bigcodebench/data/bigcodebench_benchmark.jsonl \
    +output_jsonl_fpath=results/bigcodebench_rollouts.jsonl \
    +num_repeats=4 \
    +num_repeats_add_seed=true
```

`prepare.py` exposes a `--split` flag (`hard` or `full`); the config
defaults to `hard` to match the recipe's parity-comparison run.
