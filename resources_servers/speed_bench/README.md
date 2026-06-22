# speed_bench

Speculative-decoding throughput resources server. Reads vLLM's
Prometheus `/metrics` counters before and after generation to compute
**acceptance length (AL)** and **acceptance rate (AR)** across the
benchmark window. Ported from
[`nemo_skills/inference/eval/specdec.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/eval/specdec.py).

## What it measures

This server does **not** check answer correctness. Each task's `verify()`
records:

- `num_generated_tokens` — total tokens emitted by the model for this task
  (sum across multi-turn replies)
- `gen_seconds` — wall-clock seconds since the benchmark window started
- `acceptance_length`, `acceptance_rate`, `num_drafts`, `draft_tokens`,
  `accepted_tokens`, `per_position_acceptance_rates` — the *running*
  spec-decode aggregate at the moment this task's `verify()` ran (delta
  between the first-task `/metrics` scrape and now)

`compute_metrics()` then takes the running aggregate from the task with
the largest accumulated `draft_tokens` (the latest-completing one) as the
headline `spec_acceptance_length` / `spec_acceptance_rate`.

For multi-rollout runs (e.g. `+num_repeats=N` on the rollout CLI),
`compute_metrics()` also emits Skills-equivalent variance estimators for
each spec score key (`acceptance_length`, `acceptance_rate`,
`num_drafts`, `draft_tokens`, `accepted_tokens`):

- `spec_<key>_avg` — mean across all (task, rollout) pairs
- `spec_<key>_std_dev_across_runs` — std-dev of per-run averages
  (run *i* = take rollout *i* from each task and average)
- `spec_<key>_std_err_across_runs` — `std_dev / sqrt(num_repeats)`
- `spec_<key>_avg_sample_std_dev` — mean of per-task std-devs across
  that task's `num_repeats` rollouts

These mirror Skills' `BaseMetrics._add_std_metrics`. Single-seed runs
(default for parity with Skills' default speed-bench eval) skip this
block — Skills also doesn't emit them at `max_k=1`.

## Verification

The server's `verify()` always returns `reward = 0.0`. Spec-decode metrics
live in the per-row response fields and the aggregate `compute_metrics()`
output. The model server must be vLLM with speculative decoding enabled
(`--speculative-config '{"method": "ngram", "num_speculative_tokens": ...}'`
or an Eagle/MTP method). Without spec-decode, the server's `/metrics`
endpoint omits `vllm:spec_decode_*` lines and every row carries
`spec_decode_unavailable: true`.

SGLang is supported via Prometheus delta on
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

- `vllm_base_url` (default `${policy_base_url}`) — model server's OpenAI
  base URL. The `/v1` suffix is stripped when deriving `<base>/metrics`.
- `vllm_metrics_url` (optional) — explicit override; takes precedence over
  `vllm_base_url`.
- `server_type_for_metrics` — `vllm` (default) or `sglang` (stub).
- `snapshot_at_init` — if true, take the "before" snapshot at server init
  time. Default false (lazy on first `verify()` call), which more
  precisely brackets the benchmark window when the server warmed up
  earlier.

## Starting the model server with spec-decode enabled

speed-bench requires the model server to expose
`vllm:spec_decode_*` (or `sglang:spec_accept_*`) Prometheus counters. By
default vLLM does not enable speculative decoding — you must pass a
`--speculative-config` flag (or set `vllm_serve_kwargs.speculative_config`
on a `local_vllm_model` config). Without this the resources server
records `spec_decode_unavailable: true` on every row and
`spec_acceptance_length` / `spec_acceptance_rate` come back null.

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
config_paths="responses_api_models/local_vllm_model/configs/Qwen/Qwen3-30B-A3B-Instruct-2507-ngram-specdec.yaml,\
resources_servers/speed_bench/configs/speed_bench.yaml,\
responses_api_agents/speed_bench_agent/configs/speed_bench_agent.yaml"
ng_run "+config_paths=[$config_paths]" \
    +policy_model=Qwen3-30B-A3B-Instruct-2507-ngram-specdec

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=speed_bench_simple_agent \
    +input_jsonl_fpath=resources_servers/speed_bench/data/example.jsonl \
    +output_jsonl_fpath=results/speed_bench_rollouts.jsonl \
    +num_repeats=1
```

If you'd rather use the lighter-weight `vllm_model` config (which expects
an externally-launched vLLM endpoint), make sure to pass
`--speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'`
when starting the upstream `vllm serve` process.

## Tests

```bash
ng_test +entrypoint=resources_servers/speed_bench
```

The unit tests cover the Prometheus parser, the running-delta math, the
metrics-URL resolver, and `compute_metrics` aggregation. They do not need
a live vLLM server.
