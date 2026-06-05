# SupChain-Bench (SC-bench)

---

## Summary

- Integrates [SupChain-Bench (SC-bench)](https://github.com/Damon-GSY/SC-bench) into NeMo Gym as a native resources server + benchmark config.
- Models answer natural-language questions about trade, fulfillment, and warehouse orders via **8 simulated supply-chain tool APIs**.
- Verification uses the official SC-bench entity-level evaluator: tool traces are reconstructed into structured order state and compared against `verifier_metadata` ground truth (binary 0/1 reward).
- Uses `simple_agent` for multi-step tool calling; benchmark eval set is **100 tasks** with an **80/20 train/validation split** for RL workflows.

## What was added


| Area                       | Path                                                                 |
| -------------------------- | -------------------------------------------------------------------- |
| Resources server           | `resources_servers/sc_bench/`                                        |
| Benchmark config + prepare | `benchmarks/sc_bench/`                                               |
| Agent                      | `sc_bench_benchmark_simple_agent` (inherits `sc_bench_simple_agent`) |
| Example data (committed)   | `resources_servers/sc_bench/data/example.jsonl` (5 rows)             |
| README table entry         | `README.md`                                                          |


**Key files:**

- `app.py` — 8 tool endpoints + async `verify()`
- `supchain_tools.py` — CSV-backed tool implementations
- `evaluation.py`, `compare.py`, `verify_utils.py`, `get_results.py` — SC-bench scoring (adapted from upstream)
- `benchmarks/sc_bench/prepare.py` — clones/copies upstream data, delegates JSONL conversion to `SC-bench/scripts/convert_to_nemo_gym.py`

**Data conversion** lives in the upstream SC-bench repo (`SC-bench/scripts/convert_to_nemo_gym.py`), not in NeMo Gym, per contribution guidelines.

## Verification logic

1. Agent calls supply-chain tools exposed by the resources server.
2. `verify()` reconstructs the model's tool trace into structured order state.
3. Output is compared to `verifier_metadata.expected_result` / `gt_lines` using the SC-bench entity-level evaluator.
4. Returns `reward` ∈ {0.0, 1.0}. Empty or unparseable model output → 0.0 (no crash).

## Prompts / tasks

- **Source:** [Damon-GSY/SC-bench](https://github.com/Damon-GSY/SC-bench) (ACL 2026 Findings, [arXiv:2602.07342](https://arxiv.org/abs/2602.07342))
- **Domain:** `agent` — multi-step tool use over supply chain order management
- **Benchmark:** 100 tool-use questions over trade / fulfillment / warehouse orders
- **Train / validation:** 80/20 split (gitignored locally; GitLab registry — see open items)

## Licensing


| Component                    | License    |
| ---------------------------- | ---------- |
| SC-bench code/data (adapted) | Apache 2.0 |
| NeMo Gym integration         | Apache 2.0 |


## Baseline results

### Closed-source baselines (HUD) — complete

**Setup:** HUD gateway, `openai_model` → `/v1/responses`; 100 tasks × 5 repeats; `max_output_tokens: 8192`; `num_samples_in_parallel: 1`; omit `temperature` for gpt-5 / gpt-5-mini (unsupported on HUD responses API). Set `policy_model_name` in `env.yaml` and restart `ng_run` before each model run. Artifacts under `results/<model>/sc_bench_<model>_benchmark*.jsonl`.


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


### Open-source baselines (HUD) — complete

**Instruct setup:** `config.yaml`; `vllm_model` → HUD `/v1/chat/completions`; 100 tasks × 5 repeats; `max_output_tokens: 8192`; `temperature: 0.7`; `num_samples_in_parallel: 1`.

| Metric | kimi-k2.5 (Kimi K2.5 instruct, OpenRouter) |
| --- | --- |
| **pass@1** (mean/reward) | **0.392** |
| **pass@5** (≥1 perfect rollout / task) | **0.510** (51/100) |
| mean/line_match_rate | 0.392 |
| mean/input_tokens | 3,540 |
| mean/output_tokens | 1,120 |
| mean/total_tokens | 4,660 |
| std/reward (per rollout) | 0.489 |
| mean within-task std (5 repeats) | 0.109 |
| Tasks: 5/5 pass / mixed / fail | 26 / 25 / 49 |
| Rollouts completed | 500/500 |

Artifacts: `results/kimi-k2.5/sc_bench_kimi-k2.5_benchmark*.jsonl`

**Thinking setup:** `config_thinking.yaml`; `policy_model_name: qwen3.6-plus` (Provider=OpenRouter on HUD); same rollout params.

| Metric | qwen3.6-plus (Qwen3.6 Plus thinking, OpenRouter) |
| --- | --- |
| **pass@1** (mean/reward) | **0.376** |
| **pass@5** (≥1 perfect rollout / task) | **0.510** (51/100) |
| mean/line_match_rate | 0.376 |
| mean/input_tokens | 4,796 |
| mean/output_tokens | 2,011 |
| mean/total_tokens | 6,807 |
| std/reward (per rollout) | 0.485 |
| mean within-task std (5 repeats) | 0.111 |
| Tasks: 5/5 pass / mixed / fail | 27 / 24 / 49 |
| Rollouts completed | 500/500 |

Artifacts: `results/qwen3.6-plus/sc_bench_qwen3.6-plus_benchmark*.jsonl`

### Comparison to published SC-bench results

Source: [SupChain-Bench paper](https://arxiv.org/pdf/2602.07342) (Table 1), **tool-calling** split. The paper reports **Information Retrieval Accuracy** (%): entity-level match between the model's tool trace and oracle ground truth (same scoring family as our binary `reward` / `line_match_rate`). Paper conditions:

- **w/ SOP** — human-authored Standard Operating Procedure in the system prompt (our setup; see below)
- **w/o SOP** — SOP-free prompts (paper ablation only)
- **SupChain-ReAct** (Table 4) — 5 parallel ReAct trajectories + majority vote (not used in our NeMo Gym baselines)

Our benchmark JSONL embeds the upstream SOP system prompt from `SC-bench/config/prompt_tool_sessions_with_sop.yaml` in each row's `responses_create_params.input` (deterministic tool-orchestration workflow: query → status → cancel/error branches → warehouse enrichment).

| Model | Paper (w/ SOP) | Paper (w/o SOP) | NeMo Gym pass@1 | NeMo Gym pass@5 | Δ pass@1 vs paper w/ SOP |
| --- | --- | --- | --- | --- | --- |
| GPT-5-mini | 86.73% | 46.93% | **72.6%** | **82.0%** | −14.1 pp |
| GPT-5 | 84.69% | 35.71% | **72.6%** | **82.0%** | −12.1 pp |
| Kimi K2 instruct† | 10.20% | 12.24% | **39.2%** (`kimi-k2.5`) | **51.0%** | +29.0 pp |
| Qwen thinking‡ | 28.57% (`Qwen3-Max`) | 7.14% | **37.6%** (`qwen3.6-plus`) | **51.0%** | +9.0 pp |

† Paper evaluates **Kimi-K2-Instruct**; we use HUD slug `kimi-k2.5` (OpenRouter-routed Kimi K2.5 instruct).

‡ Paper evaluates **Qwen3-Max**; we use **Qwen3.6 Plus** (`qwen3.6-plus`) as the open-source thinking baseline — a newer model not in the paper tables.

**Qualitative read:**

- NeMo Gym **pass@1** for GPT-5 family is **below** paper **w/ SOP** (72.6% vs ~85–87%), but **pass@5** (82%) is close to paper SOP-guided closed-source numbers.
- OSS **pass@1** and **pass@5** exceed paper **w/ SOP** for the listed Kimi/Qwen models, but those are different model generations and inference routes (HUD OpenRouter vs paper APIs).
- We do **not** use **SupChain-ReAct** (Table 4 multi-trajectory voting); paper peaks there (e.g. Claude-4-Sonnet 75.5%, Gemini-2.5-Pro 72.4%) are a separate harness.

**Known differences (why numbers may not match exactly):**

1. **Harness** — Same SOP text as the paper's w/ SOP condition, but NeMo Gym `simple_agent` (no SupChain-ReAct voting).
2. **Metric sampling** — Paper Table 1 is single-shot accuracy; we report **pass@1** (mean over 500 rollouts = 100 tasks × 5 repeats) and **pass@5** (≥1 perfect rollout per task).
3. **Inference route** — Our baselines run via **HUD** (`inference.hud.ai`); the paper does not specify the same gateway. Model snapshots and routing may differ.
4. **API surface** — Closed-source: HUD `/v1/responses` (`openai_model`); OSS: HUD `/v1/chat/completions` (`vllm_model` + reasoning parser for `qwen3.6-plus`).
5. **Evaluator port** — We adapted upstream `evaluation.py` / `get_results.py` into the resources server; logic is intended to match entity-level IR accuracy but runs inside Gym's async `verify()` path.

Overall, the integration reproduces the benchmark task, SOP-guided prompts, and scoring semantics; closed-source GPT pass@5 is in range of paper w/ SOP results.

## Test plan

### Automated

- `ng_test +entrypoint=resources_servers/sc_bench` — 45 tests, ≥95% coverage
- `pytest tests/unit_tests/ -x`
- `pre-commit run --files benchmarks/sc_bench/**/* resources_servers/sc_bench/**/`*

### Data validation

```bash
ng_prepare_data "+config_paths=[resources_servers/sc_bench/configs/sc_bench.yaml]" \
  +output_dirpath=/tmp/sc_bench_prepare +mode=example_validation

ng_prepare_benchmark "+config_paths=[benchmarks/sc_bench/config.yaml]"
```

### End-to-end reproduction

```bash
cd Gym && source .venv/bin/activate

# Start servers
ng_run "+config_paths=[benchmarks/sc_bench/config.yaml]"

# Smoke test (5 example tasks)
ng_collect_rollouts +agent_name=sc_bench_benchmark_simple_agent \
  +input_jsonl_fpath=resources_servers/sc_bench/data/example.jsonl \
  +output_jsonl_fpath=results/sc_bench_example_rollouts.jsonl \
  +num_repeats=1

# Full benchmark — set policy_model_name in env.yaml, restart ng_run, then:
MODEL=gpt-5-mini  # or gpt-5
ng_collect_rollouts +agent_name=sc_bench_benchmark_simple_agent \
  +input_jsonl_fpath=benchmarks/sc_bench/data/sc_bench_benchmark.jsonl \
  +output_jsonl_fpath=results/${MODEL}/sc_bench_${MODEL}_benchmark.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=1 \
  "+responses_create_params={max_output_tokens: 8192}"

ng_reward_profile \
  ++materialized_inputs_jsonl_fpath=results/${MODEL}/sc_bench_${MODEL}_benchmark_materialized_inputs.jsonl \
  ++rollouts_jsonl_fpath=results/${MODEL}/sc_bench_${MODEL}_benchmark.jsonl

python scripts/print_aggregate_results.py \
  +jsonl_fpath=results/${MODEL}/sc_bench_${MODEL}_benchmark_reward_profiling.jsonl
```

### Open-source baseline reproduction (when ready)

Swap `openai_model.yaml` → `vllm_model.yaml` in `benchmarks/sc_bench/config.yaml`, restart `ng_run`, then:

```bash
ng_collect_rollouts +agent_name=sc_bench_benchmark_simple_agent \
  +input_jsonl_fpath=benchmarks/sc_bench/data/sc_bench_benchmark.jsonl \
  +output_jsonl_fpath=results/sc_bench_<model-slug>_benchmark.jsonl \
  +num_repeats=5 \
  +num_samples_in_parallel=1 \
  "+responses_create_params={max_output_tokens: 8192, temperature: 0.7}"
```

## JSONL schema (per row)

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

## Citation

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

