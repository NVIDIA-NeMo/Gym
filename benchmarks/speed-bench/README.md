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
ng_prepare_benchmark "+config_paths=[benchmarks/speed-bench/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/speed-bench/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=speed_bench_qualitative_simple_agent \
    +input_jsonl_fpath=benchmarks/speed-bench/data/speed_bench_qualitative_benchmark.jsonl \
    +output_jsonl_fpath=results/speed_bench_qualitative_rollouts.jsonl \
    +num_repeats=1
```

## Stubbed pieces

- **SGLang**: the resources server's SGLang scrape path is a
  `NotImplementedError` stub. Skills supports SGLang via Prometheus delta
  and a per-request metrics file; both are deferred to a follow-up.
- **`throughput_{1k,8k,16k,32k}`**: `prepare.py` knows how to prepare
  these but they're not in the default config; pass
  `--config throughput_8k` etc. to prepare them explicitly.
- **Per-position acceptance rates**: recorded per-row but not surfaced in
  `get_key_metrics()` — they're only useful for deeper SD analysis.
