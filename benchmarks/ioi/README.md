# IOI (International Olympiad in Informatics)

Gym benchmark for IOI'24, evaluated via the `resources_servers/ioi` server
(itself a thin subclass of `competitive_coding_challenges`).

This benchmark contributes:

- `prepare.py` — downloads `open-r1/ioi` + `open-r1/ioi-test-cases` from
  HuggingFace and emits CCC-shaped artifacts (one row per (problem, subtask)
  + a JSONL metadata file wrapped in CCC's competition shape, keyed by
  lowercase ioi_id so a single `problem_id` string serves both metadata
  lookup and IOI's `graders/{problem_id}.cpp` filename convention).

- `config.yaml` — inherits the `ioi_resources_server` + `ioi_simple_agent`
  configs from `resources_servers/ioi/configs/ioi.yaml`, overrides
  `test_file` / `shared_dir` to point at the benchmark's own data dir, and
  wires the `benchmark`-type dataset with its `prepare_script`.

## Metrics

Inherited from `resources_servers/ioi`:

- `ioi_total_score` — sum across problems of max per-subtask score pooled
  across rollouts. On the 0-600 IOI'24 scale.
- `per_problem_subtask_scores` — per-problem breakdown, each with
  `total.{score,max_score}` plus per-subtask `{score, max_score}`.
- Plus the standard pass@k/accuracy stats from
  `compute_pass_majority_metrics`.

## Running

```bash
ng_prepare_data +config_paths=[benchmarks/ioi/config.yaml] \
  +output_dirpath=benchmarks/ioi/data \
  +mode=benchmark_preparation

ng_run +config_paths=[benchmarks/ioi/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]

ng_collect_rollouts +agent_name=ioi_benchmark_simple_agent \
  +input_jsonl_fpath=benchmarks/ioi/data/ioi24_benchmark.jsonl \
  +output_jsonl_fpath=results/ioi_rollouts.jsonl \
  +num_repeats=50 +num_repeats_add_seed=true \
  "+responses_create_params={max_output_tokens: 131072, temperature: 1.0, top_p: 0.95}"
```
