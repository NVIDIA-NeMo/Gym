# SupChain-Bench (SC-bench) Resource Server

NeMo Gym integration for [SupChain-Bench](https://github.com/Damon-GSY/SC-bench), a supply chain order management tool-use benchmark (ACL 2026 Findings).

## Description

Models answer natural-language questions about trade, fulfillment, and warehouse orders by calling eight simulated supply chain APIs. Rewards use the official SC-bench entity-level evaluator: tool traces are reconstructed into structured order state and compared to ground truth in `verifier_metadata`.

## Data


| Dataset                            | Source                                                           |
| ---------------------------------- | ---------------------------------------------------------------- |
| Benchmark (100 tool-use questions) | `benchmarks/sc_bench/prepare.py` |
| Example (5 rows)                   | `data/example.jsonl` (committed)                                 |
| Example rollouts (5 rows)          | `data/example_rollouts.jsonl` (committed)                        |
| Train / validation (80/20 split)   | `data/train.jsonl`, `data/validation.jsonl`                      |
| CSV tables                         | `data/csv/` (copied by prepare)                                  |


### Prepare data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/sc_bench/config.yaml]"
```

Validate example data:

```bash
ng_prepare_data "+config_paths=[resources_servers/sc_bench/configs/sc_bench.yaml]" \
  +output_dirpath=/tmp/sc_bench_prepare +mode=example_validation
```

## Run benchmark

### Open-source models (default — `vllm_model` + HUD chat-completions)

Set `policy_model_name` in `env.yaml` (e.g. `kimi-k2.5` for Kimi K2.5 instruct), then **restart `ng_run`** — servers do not pick up env changes while running.

```bash
cd Gym && source .venv/bin/activate

# Instruct models (config.yaml disables reasoning parser)
ng_run "+config_paths=[benchmarks/sc_bench/config.yaml]"

# Thinking models
ng_run "+config_paths=[benchmarks/sc_bench/config_thinking.yaml]"

# Smoke test
ng_collect_rollouts +agent_name=sc_bench_benchmark_simple_agent \
  +input_jsonl_fpath=resources_servers/sc_bench/data/example.jsonl \
  +output_jsonl_fpath=results/sc_bench_example_rollouts.jsonl \
  +num_repeats=1 \
  +num_samples_in_parallel=1 \
  "+responses_create_params={max_output_tokens: 8192, temperature: 0.7}"

# Full benchmark — name outputs by model slug under results/<model>/
MODEL=kimi-k2.5
ng_collect_rollouts +agent_name=sc_bench_benchmark_simple_agent \
  +input_jsonl_fpath=benchmarks/sc_bench/data/sc_bench_benchmark.jsonl \
  +output_jsonl_fpath=results/${MODEL}/sc_bench_${MODEL}_benchmark.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=1 \
  "+responses_create_params={max_output_tokens: 8192, temperature: 0.7}"

ng_reward_profile \
  ++materialized_inputs_jsonl_fpath=results/${MODEL}/sc_bench_${MODEL}_benchmark_materialized_inputs.jsonl \
  ++rollouts_jsonl_fpath=results/${MODEL}/sc_bench_${MODEL}_benchmark.jsonl

python scripts/print_aggregate_results.py \
  +jsonl_fpath=results/${MODEL}/sc_bench_${MODEL}_benchmark_reward_profiling.jsonl
```

**Open-source setup:** `vllm_model` → HUD `/v1/chat/completions`; `max_output_tokens: 8192`; `temperature: 0.7`; `num_samples_in_parallel: 1`. Do **not** pass `openai_model.yaml` on the CLI when using this config (duplicate `policy_model` error).

### Closed-source models (`openai_model` + HUD `/v1/responses`)

Use `benchmarks/sc_bench/config_openai.yaml` instead of `config.yaml`. Omit `temperature` for gpt-5 / gpt-5-mini on HUD.

```bash
ng_run "+config_paths=[benchmarks/sc_bench/config_openai.yaml]"
# policy_model_name: gpt-5-mini or gpt-5 in env.yaml
```

### Closed-source baselines (HUD)

**Setup (shared):** `openai_model` → HUD `/v1/responses`; 100 tasks × 5 repeats; `max_output_tokens: 8192`; `num_samples_in_parallel: 1`; **omit `temperature`** for gpt-5 / gpt-5-mini on HUD. Restart `ng_run` after changing `policy_model_name` in `env.yaml`.


| Metric                                 | gpt-5-mini         | gpt-5              |
| -------------------------------------- | ------------------ | ------------------ |
| **pass@1** (mean/reward)               | **0.726**          | **0.726**          |
| **pass@5** (≥1 perfect rollout / task) | **0.820** (82/100) | **0.820** (82/100) |
| mean/line_match_rate                   | 0.726              | 0.726              |
| mean/input_tokens                      | 25,380             | 30,584             |
| mean/output_tokens                     | 3,538              | 3,429              |
| mean/total_tokens                      | 28,918             | 34,012             |
| std/reward (per rollout)               | 0.446              | 0.446              |
| mean within-task std (5 repeats)       | 0.096              | 0.103              |
| median within-task std                 | 0.000              | 0.000              |
| Tasks: 5/5 pass                        | 62                 | 61                 |
| Tasks: mixed                           | 20                 | 21                 |
| Tasks: 5/5 fail                        | 18                 | 18                 |
| Rollouts completed                     | 500/500            | 500/500            |


### Open-source instruct baselines (HUD)

**Setup:** `config.yaml`; `vllm_model` → HUD `/v1/chat/completions`; 100 tasks × 5 repeats; `max_output_tokens: 8192`; `temperature: 0.7`; `num_samples_in_parallel: 1`. Restart `ng_run` after changing `policy_model_name` in `env.yaml`.


| Metric                                 | kimi-k2.5 (Kimi K2.5 instruct) |
| -------------------------------------- | ------------------------------ |
| **pass@1** (mean/reward)               | **0.392**                      |
| **pass@5** (≥1 perfect rollout / task) | **0.510** (51/100)             |
| mean/line_match_rate                   | 0.392                          |
| mean/input_tokens                      | 3,540                          |
| mean/output_tokens                     | 1,120                          |
| mean/total_tokens                      | 4,660                          |
| std/reward (per rollout)               | 0.489                          |
| mean within-task std (5 repeats)       | 0.109                          |
| median within-task std                 | 0.000                          |
| Tasks: 5/5 pass                        | 26                             |
| Tasks: mixed                           | 25                             |
| Tasks: 5/5 fail                        | 49                             |
| Rollouts completed                     | 500/500                        |


### Open-source thinking baselines (HUD / OpenRouter)

Models with **Provider = OpenRouter** on [hud.ai/models](https://hud.ai/models) are still called through the HUD gateway — no separate OpenRouter API key. Use the HUD **Model ID** slug in `policy_model_name` (e.g. `qwen3.6-plus`, not `qwen/qwen3.6-plus`).

**Setup:** `config_thinking.yaml` (reasoning parser on); `policy_model_name: qwen3.6-plus` in `env.yaml`; same rollout params as instruct; **always** `+num_samples_in_parallel=1`.

```bash
# env.yaml: policy_model_name: qwen3.6-plus
ng_run "+config_paths=[benchmarks/sc_bench/config_thinking.yaml]"

MODEL=qwen3.6-plus
ng_collect_rollouts +agent_name=sc_bench_benchmark_simple_agent \
  +input_jsonl_fpath=benchmarks/sc_bench/data/sc_bench_benchmark.jsonl \
  +output_jsonl_fpath=results/${MODEL}/sc_bench_${MODEL}_benchmark.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=1 \
  "+responses_create_params={max_output_tokens: 8192, temperature: 0.7}"
```


| Metric                                 | qwen3.6-plus (Qwen3.6 Plus, OpenRouter) |
| -------------------------------------- | --------------------------------------- |
| **pass@1** (mean/reward)               | **0.376**                               |
| **pass@5** (≥1 perfect rollout / task) | **0.510** (51/100)                      |
| mean/line_match_rate                   | 0.376                                   |
| mean/input_tokens                      | 4,796                                   |
| mean/output_tokens                     | 2,011                                   |
| mean/total_tokens                      | 6,807                                   |
| std/reward (per rollout)               | 0.485                                   |
| mean within-task std (5 repeats)       | 0.111                                   |
| median within-task std                 | 0.000                                   |
| Tasks: 5/5 pass                        | 27                                      |
| Tasks: mixed                           | 24                                      |
| Tasks: 5/5 fail                        | 49                                      |
| Rollouts completed                     | 500/500                                 |


### Comparison to the SC-bench paper

Published tool-calling accuracies are in [arXiv:2602.07342](https://arxiv.org/pdf/2602.07342) Table 1 (Information Retrieval Accuracy). Our benchmark JSONL includes the upstream **SOP system prompt** (`prompt_tool_sessions_with_sop.yaml`) in each rollout's `responses_create_params.input`, so the closest paper column is **w/ SOP**:


| Model            | Paper (w/ SOP) | NeMo Gym pass@1            | NeMo Gym pass@5 |
| ---------------- | -------------- | -------------------------- | --------------- |
| GPT-5-mini       | 86.7%          | **72.6%**                  | **82.0%**       |
| GPT-5            | 84.7%          | **72.6%**                  | **82.0%**       |
| Kimi-K2-Instruct | 10.2%          | **39.2%** (`kimi-k2.5`)    | **51.0%**       |
| Qwen3-Max        | 28.6%          | **37.6%** (`qwen3.6-plus`) | **51.0%**       |


We do **not** use SupChain-ReAct (5-trajectory voting).

## JSONL schema

Each row uses `verifier_metadata` for ground truth:

```json
{
  "id": "sc_bench_T1001_0",
  "responses_create_params": {
    "input": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "tools": [{"type": "function", "name": "query_buyer_and_related", "...": "..."}],
    "parallel_tool_calls": false
  },
  "verifier_metadata": {
    "trade_order_id": "T1001",
    "gt_lines": [{"trade_order_id": "T1001", "fulfillment_id": "FO2001", "...": "..."}],
    "expected_result": {"trade_order_id": "T1001", "buyer_id": {}, "fulfillments": []}
  }
}
```

## Licensing


| Component                    | License    |
| ---------------------------- | ---------- |
| SC-bench code/data (adapted) | Apache 2.0 |
| NeMo Gym integration         | Apache 2.0 |


Citation:

```bibtex
@misc{guan2026supchainbenchbenchmarkinglargelanguage,
  title={SupChain-Bench: Benchmarking Large Language Models for Real-World Supply Chain Management},
  author={Shengyue Guan and Yihao Liu and Lang Cao},
  year={2026},
  eprint={2602.07342},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2602.07342},
}
```

