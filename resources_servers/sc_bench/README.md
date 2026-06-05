# SupChain-Bench (SC-bench) Resource Server

NeMo Gym integration for [SupChain-Bench](https://github.com/Damon-GSY/SC-bench), a supply chain order management tool-use benchmark (ACL 2026 Findings).

## Description

Models answer natural-language questions about trade, fulfillment, and warehouse orders by calling eight simulated supply chain APIs. Rewards use the official SC-bench entity-level evaluator: tool traces are reconstructed into structured order state and compared to ground truth in `verifier_metadata`.

## Data

| Dataset | Source |
| --- | --- |
| Conversion script | `SC-bench/scripts/convert_to_nemo_gym.py` (upstream source repo) |
| Benchmark (100 tool-use questions) | `benchmarks/sc_bench/prepare.py` |
| Example (5 rows) | `data/example.jsonl` (committed) |
| Train / validation (80/20 split) | `data/train.jsonl`, `data/validation.jsonl` (gitignored; upload to GitLab) |
| CSV tables | `data/csv/` (copied by prepare) |

### Prepare data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/sc_bench/config.yaml]"
```

Validate example data:

```bash
ng_prepare_data "+config_paths=[resources_servers/sc_bench/configs/sc_bench.yaml]" \
  +output_dirpath=/tmp/sc_bench_prepare +mode=example_validation
```

### Upload train/validation to GitLab (before PR merge)

```bash
ng_upload_dataset_to_gitlab \
  +dataset_name=sc_bench \
  +version=0.0.1 \
  +input_jsonl_fpath=resources_servers/sc_bench/data/train.jsonl

ng_upload_dataset_to_gitlab \
  +dataset_name=sc_bench \
  +version=0.0.1 \
  +input_jsonl_fpath=resources_servers/sc_bench/data/validation.jsonl
```

## Run benchmark

```bash
# Start servers (includes openai_model / policy_model; set policy_* in env.yaml)
ng_run "+config_paths=[benchmarks/sc_bench/config.yaml]"

# Smoke test on example data first
ng_collect_rollouts +agent_name=sc_bench_benchmark_simple_agent \
  +input_jsonl_fpath=resources_servers/sc_bench/data/example.jsonl \
  +output_jsonl_fpath=results/sc_bench_example_rollouts.jsonl \
  +num_repeats=1

# Full benchmark evaluation
ng_collect_rollouts +agent_name=sc_bench_benchmark_simple_agent \
  +input_jsonl_fpath=benchmarks/sc_bench/data/sc_bench_benchmark.jsonl \
  +output_jsonl_fpath=results/sc_bench_rollouts.jsonl \
  +num_repeats=1
```

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

| Component | License |
| --- | --- |
| SC-bench code/data (adapted) | Apache 2.0 |
| NeMo Gym integration | Apache 2.0 |

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
