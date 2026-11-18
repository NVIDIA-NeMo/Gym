# Run an evaluation on your machine against an endpoint

For a first run, a smoke, or a model behind an API (a hosted provider, or a vLLM server you already run). No Slurm, no submit config. Every command here is documented in the Gym [Installation](https://docs.nvidia.com/nemo/gym/latest/get-started/installation) and [Quickstart](https://docs.nvidia.com/nemo/gym/latest/get-started/quickstart) pages. Run `gym <command> --help` for the options of the version you installed.

## 1. Install

Python 3.13.14 or newer is required.

```bash
uv venv --python 3.13.14 && source .venv/bin/activate
uv pip install nemo-gym        # or: python3.13 -m pip install nemo-gym
gym --help
```

The package carries the built-in environments and benchmarks. Config and data paths resolve against the install. Pin a release with `nemo-gym==<version>` when the result must be repeatable. Ray is not included: environments that need it (`local_vllm_model`, the SWE agents) need `nemo-gym[ray]`.

## 2. Say where the model is

Either an `env.yaml` in the working directory:

```yaml
policy_base_url: https://api.openai.com/v1     # any OpenAI-compatible endpoint
policy_api_key: <key>
policy_model_name: <model name the endpoint serves>
```

or the same on the command line: `--model-type`, `--model-url`, `--model`, `--model-api-key`. Use `openai_model` for a plain OpenAI-compatible endpoint. Use `vllm_model` for a vLLM server when you need its reasoning output or `chat_template_kwargs` ([model-config.md](model-config.md), section 3). Never put a key in a file you commit.

## 3. Smoke it first

The two-terminal form, with the built-in multiple-choice example:

```bash
gym env start --resources-server mcqa --model-type openai_model    # terminal 1: leave it running

gym eval run --no-serve \                                          # terminal 2
    --agent mcqa_simple_agent \
    --input resources_servers/mcqa/data/example.jsonl \
    --output results/mcqa_rollouts.jsonl \
    --limit 5 --num-repeats 1
```

The run prints aggregate metrics and writes `results/mcqa_rollouts.jsonl`, `..._materialized_inputs.jsonl` and `..._aggregate_metrics.json`. `--limit` counts tasks. Rows are tasks times repeats.

## 4. A named benchmark

```bash
gym list benchmarks                  # the names you can pass to --benchmark
gym list benchmarks <name>           # its agent harness and datasets
gym eval prepare --benchmark <name>  # builds its dataset when it has a prepare step
gym eval run --benchmark <name> --model-type openai_model \
    --model <model> --model-url <url> --model-api-key "$KEY" \
    --split benchmark --output results/<name>/rollouts.jsonl --limit 5 --num-repeats 1
```

Read the benchmark's README (linked from `gym list benchmarks <name>`) first: some need a judge model, a search key or a sandbox provider of their own, and some need data you stage yourself. Task-folder benchmarks (Harbor format) run in a sandbox: pass `--sandbox <provider>` and the provider's credentials ([Harbor tasks](https://docs.nvidia.com/nemo/gym/latest/agent-interactions/harbor)). Remove `--limit` and `--num-repeats` for a full run: a score counts as the benchmark's only with all tasks at no fewer than the benchmark's own `num_repeats` (its `manifest.yaml`).

## 5. Read the result

- `gym eval profile` gives pass@1, pass@k and per-task variance from the rollouts file.
- A run can be healthy and score low: that is the model, not the evaluation. Check answers that stop mid-thought (output budget), reasoning that lands in the answer (parser) and tool calls that arrive as plain text (tool-call parser) before you trust a number. The health checks and their thresholds are in [slurm-run-and-recover.md](slurm-run-and-recover.md), section 6.
- To compare two runs, use the `nemo-gym-analyzing-results` skill.

## 6. When it fails

| You see | Cause and fix |
| --- | --- |
| `500 Internal Server Error` on `/run` | Usually a wrong or missing `policy_api_key` or `policy_base_url`. Check them. |
| `gym env start` stops with `model endpoint(s) never answered` | Nothing listens at the endpoint it names. Start the server or fix the URL ([Model Endpoint Unreachable](https://docs.nvidia.com/nemo/gym/latest/troubleshooting/configuration#model-endpoint-unreachable)). |
| `required config value(s) are unset ('???')` | Add the named value to `env.yaml` or pass it as `++name=value`. |
| `Server Reference Not Found` | A typo in a server name, and the error suggests the closest one. |
| The benchmark needs a sandbox and none starts | Name a provider with `--sandbox` and give its credentials. Check the provider is reachable before you retry. |
| Every answer is cut off or empty | The output budget or `--max-model-len` of your server is too small, or the reasoning parser is wrong ([model-config.md](model-config.md), sections 3 and 6). |
| It runs for hours or needs many GPUs | Use the Slurm path ([slurm-run-and-recover.md](slurm-run-and-recover.md)). |
