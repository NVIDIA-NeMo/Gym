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
gym eval prepare --benchmark bigcodebench

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark bigcodebench

# Collecting rollouts (5-row smoke test against the baked example set)
gym eval run --no-serve \
    --agent bigcodebench_benchmark_simple_agent \
    --input resources_servers/bigcodebench/data/example.jsonl \
    --output results/bigcodebench_rollouts.jsonl \
    --num-repeats 1
```

The benchmark JSONL written by ``gym eval prepare`` is unbaked
(rows have ``question`` + ``verifier_metadata``; the prompt template is
applied by the agent at ``/run`` time). Standalone ``gym eval run --no-serve``
expects pre-baked ``responses_create_params.input``, so for full-dataset
runs use the production orchestrator (``nemo_gym_rollouts`` from
NeMo-Skills) rather than ``gym eval run --no-serve`` directly.

`prepare.py` exposes a `--split` flag (`hard` or `full`); the config
defaults to `hard` to match the recipe's parity-comparison run.
