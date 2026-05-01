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

## Verification

The server's `verify()` always returns `reward = 0.0`. Spec-decode metrics
live in the per-row response fields and the aggregate `compute_metrics()`
output. The model server must be vLLM with speculative decoding enabled
(`--speculative-config '{"method": "ngram", "num_speculative_tokens": ...}'`
or an Eagle/MTP method). Without spec-decode, the server's `/metrics`
endpoint omits `vllm:spec_decode_*` lines and every row carries
`spec_decode_unavailable: true`.

SGLang is currently a stub — set `server_type_for_metrics: sglang` to
exercise the code path, but it raises `NotImplementedError`.

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

## Example usage

```bash
# Running servers (vLLM-backed; needs --speculative-config to produce real metrics)
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/speed_bench/configs/speed_bench.yaml,\
responses_api_agents/speed_bench_agent/configs/speed_bench_agent.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=speed_bench_simple_agent \
    +input_jsonl_fpath=resources_servers/speed_bench/data/example.jsonl \
    +output_jsonl_fpath=results/speed_bench_rollouts.jsonl \
    +num_repeats=1
```

## Tests

```bash
ng_test +entrypoint=resources_servers/speed_bench
```

The unit tests cover the Prometheus parser, the running-delta math, the
metrics-URL resolver, and `compute_metrics` aggregation. They do not need
a live vLLM server.
