# SPEED-Bench

SPEED-Bench measures speculative-decoding (SD) throughput — primarily
**acceptance length (AL)** and **acceptance rate (AR)** — over a curated
mixture of multi-turn prompts drawn from 14 external benchmarks (BAMBOO,
HLE, LiveCodeBench, MMLU-Pro, MT-Bench-101, OPUS-100, …). Source:
[nvidia/SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench).

This benchmark is governed by the
[NVIDIA Evaluation Dataset License Agreement](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/main/License.pdf).

## Configs

Skills' upstream prepare supports six configs:

- `qualitative` — single fixed mixture of qualitative prompts.
- `throughput_{1k,2k,8k,16k,32k}` — token-budgeted mixtures used to
  measure SD throughput at varying prompt lengths.

This Gym port defaults to preparing **`qualitative` and `throughput_2k`**
to keep iteration cheap; `prepare.py` accepts `--config all` to prepare
the full set.

## Multi-turn shape

Each row's `responses_create_params.input` is a list of
`{"role": "user", "content": "<turn>"}` messages with **no interspersed
assistant messages**. The `speed_bench_agent` replays these one turn at
a time at rollout time, mirroring Skills'
`SpecdecGenerationTask.process_single_datapoint`.

## Verification

Verification is server-side: the `speed_bench` resources server scrapes
the model server's `/metrics` endpoint before and after the benchmark
window and reports `spec_acceptance_length` /
`spec_acceptance_rate`. There is no notion of answer correctness;
`verify()` always returns `reward = 0.0`.

vLLM must be launched with speculative decoding enabled. Two common
configurations:

- **ngram (model-agnostic, no draft model)**:
  `--speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'`
- **Eagle3 / MTP** (when the target model has a paired draft):
  `--speculative-config '{"method": "eagle3", "num_speculative_tokens": 3, "model": "<draft model id>"}'`

## Example usage

```bash
# Prepare benchmark data (downloads the upstream HF dataset
# nvidia/SPEED-Bench plus the 14 source datasets it interpolates from).
# Run on a host that has internet access — see prepare.py for details.
ng_prepare_benchmark "+config_paths=[benchmarks/speed-bench/config_qualitative.yaml]"

# Running servers — uses the local_vllm_model demo config that bakes
# ngram speculative decoding into vllm_serve_kwargs.speculative_config.
# To use a different target model, swap this for any local_vllm_model
# config that includes a `speculative_config:` block.
config_paths="responses_api_models/local_vllm_model/configs/Qwen/Qwen3-30B-A3B-Instruct-2507-ngram-specdec.yaml,\
benchmarks/speed-bench/config_qualitative.yaml"
ng_run "+config_paths=[$config_paths]" \
    +policy_model=Qwen3-30B-A3B-Instruct-2507-ngram-specdec

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=speed_bench_qualitative_simple_agent \
    +input_jsonl_fpath=benchmarks/speed-bench/data/speed_bench_qualitative_benchmark.jsonl \
    +output_jsonl_fpath=results/speed_bench_qualitative_rollouts.jsonl \
    +num_repeats=1
```

If you're using the lighter-weight `vllm_model` config (external vLLM
endpoint), make sure to launch `vllm serve` with
`--speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'`
— without it, every rollout records `spec_decode_unavailable: true`.

## Deferred items

- **`throughput_{1k,8k,16k,32k}`**: `prepare.py` knows how to prepare
  these but they're not in the default config; pass
  `--config throughput_8k` etc. to prepare them explicitly.
- **Per-position acceptance rates**: recorded per-row but not surfaced in
  `get_key_metrics()` — they're only useful for deeper SD analysis.
- **SGLang per-request metrics file**: the Prometheus-delta SGLang path is
  fully ported (set `server_type_for_metrics: sglang` in the resources
  server config). Skills' per-request metrics-file fallback
  (`--export-metrics-to-file`) gives finer per-task attribution; not
  ported.
