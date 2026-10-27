# SPEED-Bench

SPEED-Bench measures speculative-decoding (SD) throughput — primarily
**acceptance length (AL)** and **acceptance rate (AR)** — over a curated
mixture of multi-turn prompts drawn from 14 external benchmarks (BAMBOO,
HLE, LiveCodeBench, MMLU-Pro, MT-Bench-101, OPUS-100, …). Source:
[nvidia/SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench).

This benchmark is governed by the
[NVIDIA Evaluation Dataset License Agreement](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/main/License.pdf).

## Configs

Gym supports preparing and running all six configs:

- `qualitative` — single fixed mixture of qualitative prompts.
- `throughput_{1k,2k,8k,16k,32k}` — token-budgeted mixtures used to
  measure SD throughput at varying prompt lengths.

Select `--benchmark speed-bench/config_<name>` for any of these configs.
Each has its own dataset, agent, resources server, and environment server.
Each selects a small preparation wrapper that delegates to shared Speed Bench
logic. All splits can share one policy model.

The default Hugging Face source is `nvidia/SPEED-Bench`. Override it for preparation:

```bash
gym eval prepare \
    --benchmark speed-bench/config_qualitative \
    --benchmark speed-bench/config_throughput_32k \
    +prepare_script_args.dataset_repo=<org>/<private-SPEED-Bench-repo>
```

Optionally add `+prepare_script_args.dataset_revision=<commit-sha>` for a pinned
source. Private repositories use your existing Hugging Face login or `HF_TOKEN`.
The replacement repository must provide the
selected configs' `test` split with compatible `turns` and metadata fields.

Each dataset's `prepare_script` selects its split. Gym's existing global
`prepare_script_args` pass repository and revision overrides to every selected
wrapper without changing its split selection. Run `gym eval prepare` explicitly
before evaluation; `gym eval run` does not invoke these preparation scripts when
the input files are missing.
The standalone equivalent also supports preparing all six:

```bash
# Install preparation dependencies into the active Gym Python environment.
uv pip install numpy pandas tiktoken
python benchmarks/speed-bench/prepare.py --config all \
    --dataset-repo <org>/<private-SPEED-Bench-repo> \
    --dataset-revision <commit-sha>
```


## Multi-turn shape

Each row's `responses_create_params.input` is a list of
`{"role": "user", "content": "<turn>"}` messages with **no interspersed
assistant messages**. The `speed_bench_agent` replays these one turn at
a time at rollout time, mirroring Skills'
`SpecdecGenerationTask.process_single_datapoint`.

## Verification

Response-based metrics give each model call (each turn) equal weight. Acceptance
rates and lengths are computed per request before averaging; histograms are
normalized per request before averaging. Draft/accepted counters are means per
request. Raw provider counters remain in the rollout output.

Verification is server-side: the `speed_bench` resources server aggregates
per-request acceptance counters returned through the model wrapper and agent,
and reports `spec_acceptance_length` / `spec_acceptance_rate`. It does not
require the underlying vLLM URL or `/metrics` endpoint. There is no notion of answer correctness;
`verify()` always returns `reward = 0.0`.

vLLM must be launched with speculative decoding and per-request metrics enabled.
For Gym-managed local vLLM, load the shared
[`configs/local_vllm_metrics.yaml`](configs/local_vllm_metrics.yaml) overlay
alongside any split config, as shown below. It sets
`policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.per_request_spec_decode_metrics`
to `summary`. Selecting a benchmark alone does not enable metrics. Externally launched
servers need `--per-request-spec-decode-metrics summary` (supported by the pinned
vLLM 0.29.0).
Two common speculative-decoding configurations:

- **ngram (model-agnostic, no draft model)**:
  `--speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'`
- **Eagle3 / MTP** (when the target model has a paired draft):
  `--speculative-config '{"method": "eagle3", "num_speculative_tokens": 3, "model": "<draft model id>"}'`

## Example usage

```bash
# Prepare benchmark data (downloads the upstream HF dataset
# nvidia/SPEED-Bench plus the 14 source datasets it interpolates from).
# Run on a host that has internet access — see prepare.py for details.
gym eval prepare --benchmark speed-bench/config_qualitative

# Running servers — uses the local_vllm_model demo config that bakes
# ngram speculative decoding into vllm_serve_kwargs.speculative_config.
# To use a different target model, swap this for any local_vllm_model
# config with `speculative_config:`. The shared overlay enables response metrics.
gym env start \
    --model-type local_vllm_model/Qwen/Qwen3-30B-A3B-Instruct-2507-ngram-specdec \
    --benchmark speed-bench/config_qualitative \
    --config benchmarks/speed-bench/configs/local_vllm_metrics.yaml \
    +policy_model._inherit_from=Qwen3-30B-A3B-Instruct-2507-ngram-specdec \
    ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4 \
    ++policy_model.responses_api_models.local_vllm_model.sampling_overrides.max_tokens=8192

# Collecting rollouts
gym eval run --no-serve \
    --agent speed_bench_qualitative_simple_agent \
    --input benchmarks/speed-bench/data/speed_bench_qualitative_benchmark.jsonl \
    --output results/speed_bench_qualitative_rollouts.jsonl \
    --num-repeats 1
```

For another split, replace `qualitative` in the benchmark selector, agent name,
input filename, and output filename. The generation cap applies to each model
call, not the whole multi-turn task. Set the model's context window large enough
for the prompt, accumulated conversation, and output; a split name does not set
the context window. The example allocates four GPUs; adjust tensor parallelism
to your hardware.

The same overlay works with every split or multiple
`--benchmark` selections; load it once. It targets the standard `policy_model`
binding and is only for Gym-managed `local_vllm_model` servers. To request
`detailed` metrics, add
`++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.per_request_spec_decode_metrics=detailed`.

If you're using the lighter-weight `vllm_model` config (external vLLM
endpoint), make sure to launch `vllm serve` with
`--speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'`
and `--per-request-spec-decode-metrics summary`, and omit the local-vLLM overlay.
Missing per-request metrics
produce `spec_decode_unavailable: true` and null acceptance metrics.

## Deferred items

- **Per-position acceptance rates**: recorded per-row but not surfaced in
  `get_key_metrics()` — they're only useful for deeper SD analysis.
- **SGLang per-request metrics file**: the Prometheus-delta SGLang path is
  available in legacy mode (set `metrics_source: prometheus`,
  `server_type_for_metrics: sglang`, and an explicit `vllm_base_url` in the
  resources server config). Skills' per-request metrics-file fallback
  (`--export-metrics-to-file`) gives finer per-task attribution; not
  ported.
