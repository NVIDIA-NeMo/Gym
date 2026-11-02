# speed_bench

Speculative-decoding throughput resources server. Reads vLLM's
[per-request acceptance metrics](https://docs.vllm.ai/en/stable/features/speculative_decoding/acceptance_metrics/)
from the agent response to compute **acceptance length (AL)** and
**acceptance rate (AR)**. No model URL or `/metrics` access is needed in
the default response mode. The legacy Prometheus implementation was ported from
[`nemo_skills/inference/eval/specdec.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/eval/specdec.py).

## What it measures

This server does **not** check answer correctness. Each task's `verify()`
records:

- `num_generated_tokens` — total tokens emitted by the model for this task
  (sum across multi-turn replies)
- `gen_seconds` — elapsed time for the agent's generation sequence, including
  request overhead, plus `generation_started_at` / `generation_finished_at`
- `num_requests` — number of model calls (each turn is one request)
- `acceptance_length` — mean of each request's `1 + accepted / steps`
- `acceptance_rate` — mean of each request's `100 * accepted / drafted`
  (percent, whereas vLLM's `draft_acceptance_rate` is a fraction)
- `num_drafts`, `draft_tokens`, `accepted_tokens` — mean verification steps,
  proposed draft tokens, and accepted draft tokens per request; these may be fractional
- `acceptance_histogram` — mean of each request's probability distribution of
  accepting exactly 0, 1, ... draft tokens. Each histogram is divided by its
  own step count before averaging; shorter histograms are zero-padded
- `per_position_acceptance_rates` — mean of each request's histogram tail probabilities

`compute_metrics()` reads the original request counters from every row and
averages individual requests, so grouping calls into multi-turn tasks does not
change their weight. Headline fields use the `spec_` prefix, including
`spec_acceptance_histogram` and `spec_num_requests`. `avg_tokens` is total output
tokens divided by the number of model requests. Aggregate `gen_seconds` is the
window from the earliest generation start to the latest finish, so overlapping
requests are not double-counted. Distributed agent hosts need synced clocks.

For example, a request with 100 steps accepting 3 tokens each and a request with
one step accepting no tokens produce acceptance length `(4 + 1) / 2 = 2.5` and
histogram `[0.5, 0, 0, 0.5]`, regardless of their different generation lengths.

For multi-rollout runs (e.g. `+num_repeats=N` on the rollout CLI),
`compute_metrics()` also emits variance estimators for
each spec score key (`acceptance_length`, `acceptance_rate`,
`num_drafts`, `draft_tokens`, `accepted_tokens`):

- `spec_<key>_avg` — mean across individual model requests
- `spec_<key>_std_dev_across_runs` — std-dev of per-run averages
  (run *i* = take rollout *i* from each task and average its individual requests)
- `spec_<key>_std_err_across_runs` — `std_dev / sqrt(num_repeats)`
- `spec_<key>_avg_sample_std_dev` — mean of per-task std-devs across
  that task's `num_repeats` rollouts

These use request averages for overall and per-run means, and per-task
request averages for sample deviations. They use the common repeat count and
tasks with defined values in every included repeat. Single-seed runs
(default for parity with Skills' default speed-bench eval) skip this
block — Skills also doesn't emit them at `max_k=1`.

## Verification

The server's `verify()` always returns `reward = 0.0`. Spec-decode metrics
live in the per-row response fields and the aggregate `compute_metrics()`
output. The model server must be vLLM with speculative decoding enabled
(`--speculative-config '{"method": "ngram", "num_speculative_tokens": ...}'`
or an Eagle/MTP method), with per-request metrics enabled.

The model wrapper preserves `metrics` when converting Chat Completions to
Responses. `speed_bench_agent` retains every turn's metrics in order under
`response.metrics.requests`, including `null` for missing metrics. These raw
metrics are also preserved when forwarding rows for aggregation.

Missing metrics on any turn mark that task `spec_decode_unavailable: true`.
If any task is unavailable, headline acceptance metrics and request-based
`avg_tokens` are null rather than reporting a partial or differently weighted
result. Zero-step requests contribute zero to mean counts, but their undefined
acceptance length, histogram, and per-position rates are excluded from those
averages. Requests with no drafted tokens are excluded from acceptance-rate
averages. If no request has a defined value, that metric is null.

Existing response-mode rollouts can be reaggregated from their saved request
metrics without rerunning inference. Old cumulative Prometheus rollouts still
require legacy mode, which retains before/after snapshots and selects the task
with the largest cumulative `draft_tokens` for headline metrics.

SGLang is supported in legacy `metrics_source: prometheus` mode via delta on
`sglang:spec_accept_length` / `sglang:spec_accept_rate` (running-average
gauges) combined with `sglang:num_requests_total` and
`sglang:generation_tokens_total` counters; the benchmark-only average is
recovered via a weighted delta (matches Skills'
`compute_sglang_spec_decode_delta`). Set `server_type_for_metrics: sglang`
in `configs/speed_bench.yaml` and launch the model server with
`server_type=sglang` on the rollout side. Skills' SGLang per-request
metrics-file fallback (`--export-metrics-to-file`) is not ported — the
Prometheus delta is sufficient for parity comparisons.

## Configuration

Top-level config fields (set in `configs/speed_bench.yaml`):

- `metrics_source` — `response` (default) or `prometheus` for legacy scraping.
  The following settings apply only to Prometheus mode; provide either
  `vllm_base_url` or `vllm_metrics_url` explicitly when using it.
- `vllm_base_url` (optional) — model server's OpenAI
  base URL. The `/v1` suffix is stripped when deriving `<base>/metrics`.
- `vllm_metrics_url` (optional) — explicit override; takes precedence over
  `vllm_base_url`.
- `server_type_for_metrics` — `vllm` (default) or `sglang`.
- `snapshot_at_init` — if true, take the "before" snapshot at server init
  time. Default false (lazy on first `verify()` call), which more
  precisely brackets the benchmark window when the server warmed up
  earlier.

## Starting the model server with spec-decode enabled

By default vLLM does not enable speculative decoding — you must pass a
`--speculative-config` flag (or set `vllm_serve_kwargs.speculative_config`
on a `local_vllm_model` config). Without this the resources server
records `spec_decode_unavailable: true` on every row and
`spec_acceptance_length` / `spec_acceptance_rate` come back null.

For `metrics_source: response` with a Gym-managed `local_vllm_model`, load
[`benchmarks/speed-bench/configs/local_vllm_metrics.yaml`](../../benchmarks/speed-bench/configs/local_vllm_metrics.yaml)
using `--config`, as shown below. This shared overlay sets
`per_request_spec_decode_metrics: summary` on `policy_model`; selecting Speed Bench
alone does not enable metrics. The setting appears in `gym env resolve`.
To override it, add
`++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.per_request_spec_decode_metrics=detailed`
(or `none` to disable reporting). Omit the overlay for other model adapters,
external endpoints, and legacy Prometheus mode. External endpoints (including
local wrappers with `base_url` set) need `--per-request-spec-decode-metrics summary`
on their own launch command.

Use a vLLM version supporting this experimental response schema (the local
model pins 0.29.0). Metrics are available for single-sequence requests (`n=1`).
`detailed` also works; its per-step arrays are preserved in the rollout output.

A ready-to-use demo config that bakes ngram speculative decoding into a
`local_vllm_model` lives at
[`responses_api_models/local_vllm_model/configs/Qwen/Qwen3-30B-A3B-Instruct-2507-ngram-specdec.yaml`](../../responses_api_models/local_vllm_model/configs/Qwen/Qwen3-30B-A3B-Instruct-2507-ngram-specdec.yaml).
The relevant block to copy into your own model config is:

```yaml
vllm_serve_kwargs:
  speculative_config:
    method: ngram                # model-agnostic; no draft model required
    num_speculative_tokens: 3
    prompt_lookup_max: 5
    prompt_lookup_min: 2
```

For an EAGLE3 / MTP setup with a paired draft model, see
`nemotron_3_ultra_dev_nemorl_gb200.yaml` for an MTP example or vLLM's
[spec-decode docs](https://docs.vllm.ai/en/latest/features/spec_decode.html).

## Example usage

```bash
# Running servers — uses the demo local_vllm_model config above (drop in
# your own model config to swap targets; just keep the speculative_config block).
gym env start \
    --model-type local_vllm_model/Qwen/Qwen3-30B-A3B-Instruct-2507-ngram-specdec \
    --resources-server speed_bench \
    --config responses_api_agents/speed_bench_agent/configs/speed_bench_agent.yaml \
    --config benchmarks/speed-bench/configs/local_vllm_metrics.yaml \
    +policy_model._inherit_from=Qwen3-30B-A3B-Instruct-2507-ngram-specdec

# Collecting rollouts (5-example smoke test)
gym eval run --no-serve \
    --agent speed_bench_simple_agent \
    --input resources_servers/speed_bench/data/example.jsonl \
    --output results/speed_bench_rollouts.jsonl \
    --num-repeats 1
```

If you'd rather use the lighter-weight `vllm_model` config (which expects
an externally-launched vLLM endpoint), make sure to pass
`--speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'`
and `--per-request-spec-decode-metrics summary`
when starting the upstream `vllm serve` process.

## Tests

```bash
gym env test --resources-server speed_bench
```

The unit tests cover response aggregation, varying draft lengths, and the retained Prometheus parser, running-delta math, and metrics-URL resolver. They do not need a live vLLM server.
