# LabBench2 VLM

[LabBench2](https://huggingface.co/datasets/EdisonScientific/labbench2) is a
scientific figure and table question answering benchmark for vision-language
models. This config bundles all four subtasks (`figqa2-img`, `figqa2-pdf`,
`tableqa2-img`, `tableqa2-pdf`) into a single benchmark run. Per-tag metrics
are emitted alongside the overall score via `verifier_metadata.tag`.

## Configuration

Uses the `labbench2_vlm` resource server with the `labbench2_vlm_agent` custom
agent. Scoring is LLM-as-judge (`[[A=B]]` / `[[A!=B]]`). Media files (images
and PDFs) are referenced by path in the JSONL and embedded at rollout time by
the agent.

The judge is wired up via `resources_servers/labbench2_vlm/configs/judge_model_openai.yaml`,
which this benchmark chains in automatically. It targets an OpenAI-compatible
hosted endpoint — swap it for your own `responses_api_models` instance named
`judge_model` (e.g. a local vLLM) if you want a different judge.

Set policy and judge model credentials in `env.yaml`:

```yaml
policy_base_url: https://inference-api.nvidia.com/v1
policy_api_key: <your-api-key>
policy_model_name: openai/openai/gpt-5.2
judge_base_url: https://inference-api.nvidia.com/v1
judge_api_key: <your-api-key>
judge_model_name: openai/openai/gpt-5-mini
```

## Usage

```bash
# Prepare data (downloads questions from HF + media files from GCS)
ng_prepare_benchmark "+config_paths=[benchmarks/labbench2_vlm/config.yaml]"

# Start servers
ng_run "+config_paths=[benchmarks/labbench2_vlm/config.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

# Collect rollouts
ng_collect_rollouts \
    "+config_paths=[benchmarks/labbench2_vlm/config.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
    +agent_name=labbench2_vlm_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/labbench2_vlm/data/labbench2_vlm_benchmark.jsonl \
    +output_jsonl_fpath=results/labbench2_vlm.jsonl
```

`+agent_name` and `+input_jsonl_fpath` are both required — rows in the
prepared JSONL don't carry an `agent_ref`, and `ng_collect_rollouts` doesn't
read the path from the benchmark config.

### One-shot alternative

`ng_e2e_collect_rollouts` starts the server stack, preprocesses, and
collects rollouts in a single command (don't run `ng_run` separately).
Input path and agent ref are auto-derived from the `type: benchmark`
dataset entry in the chained config:

```bash
ng_e2e_collect_rollouts \
    "+config_paths=[benchmarks/labbench2_vlm/config.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
    ++split=benchmark \
    ++output_jsonl_fpath=results/labbench2_vlm.jsonl \
    +num_samples_in_parallel=16
```

For a fast smoke test, add `+limit=10 +num_repeats=1`.

Media files are downloaded into `resources_servers/labbench2_vlm/data/media/`
(gitignored) so they resolve against the agent's inherited `media_base_dir`.
The combined benchmark JSONL is written to
`benchmarks/labbench2_vlm/data/labbench2_vlm_benchmark.jsonl` (gitignored).

`num_repeats` defaults to 3. Bump higher for tighter variance on the
judge-based reward.

## Throttling

Each in-flight sample fans out to one policy call + one judge call, so the
endpoints see roughly `2 × num_samples_in_parallel` concurrent requests.
On a hosted endpoint you'll likely hit rate limits or socket errors
(`Hit N global ClientOSError`) well before saturating your machine.

Cap concurrency with `+num_samples_in_parallel=<N>`:

```bash
ng_collect_rollouts ... +num_samples_in_parallel=16
```

Start around 16 and bump up if it holds.
