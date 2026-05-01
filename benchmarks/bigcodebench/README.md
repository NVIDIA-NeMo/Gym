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

# Collecting rollouts (5-row smoke test against the baked example set)
ng_collect_rollouts \
    +agent_name=bigcodebench_benchmark_simple_agent \
    +input_jsonl_fpath=resources_servers/bigcodebench/data/example.jsonl \
    +output_jsonl_fpath=results/bigcodebench_rollouts.jsonl \
    +num_repeats=1
```

The benchmark JSONL written by ``ng_prepare_benchmark`` is unbaked
(rows have ``question`` + ``verifier_metadata``; the prompt template is
applied by the agent at ``/run`` time). Standalone ``ng_collect_rollouts``
expects pre-baked ``responses_create_params.input``, so for full-dataset
runs use the production orchestrator (``nemo_gym_rollouts`` from
NeMo-Skills) rather than ``ng_collect_rollouts`` directly.

`prepare.py` exposes a `--split` flag (`hard` or `full`); the config
defaults to `hard` to match the recipe's parity-comparison run.
