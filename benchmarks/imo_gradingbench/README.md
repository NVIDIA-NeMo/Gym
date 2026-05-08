# IMO GradingBench

[IMO-GradingBench](https://github.com/google-deepmind/superhuman/blob/main/imobench/gradingbench.csv)
framed as a **four-class proof-grading task**: the policy model reads a
mathematical problem plus a candidate proof and must emit one of
`incorrect | partial | almost | correct` as the **last word** of its
response. Verification is fully deterministic — no external LLM judge —
so the same regex extracts the grade from the model's output that
NeMo Skills' `GradingBenchMetrics._extract_grade` uses.

The gold grade word lives in each row's `expected_answer`, copied
verbatim from the upstream `Reward` column.

## Prerequisites

The prepare script downloads `gradingbench.csv` from a pinned commit
of `google-deepmind/superhuman` (no HuggingFace login required). For
reasoning models, **serve with `--reasoning-parser`** so
`<think>…</think>` CoT is split out at the server layer and the
last-word extractor only sees the model's committed grade:

```bash
vllm serve <model> --reasoning-parser deepseek_r1
```

Skipping the parser for a reasoning model collapses accuracy because
the "last word" of the response is the last word of the reasoning
trace, not the final grade. See the [`imo_gradingbench`
server](../../resources_servers/imo_gradingbench/README.md) for
details.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/imo_gradingbench/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/imo_gradingbench/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (4 rollouts per task)
ng_collect_rollouts \
    +agent_name=imo_gradingbench_imo_gradingbench_simple_agent \
    +input_jsonl_fpath=benchmarks/imo_gradingbench/data/imo_gradingbench_benchmark.jsonl \
    +output_jsonl_fpath=results/imo_gradingbench_rollouts.jsonl \
    +prompt_config=benchmarks/imo_gradingbench/prompts/default.yaml \
    +num_repeats=4
```

## Metrics

Reported via the `imo_gradingbench` server's `compute_metrics()`:

* `pass@1[avg-of-k]/{exact_accuracy,binarized_accuracy,no_answer}` —
  primary parity targets
* `pass@k/{exact_accuracy,binarized_accuracy}` — best-of-K
* `majority@k/{exact_accuracy,binarized_accuracy}` — majority vote
* `pass@1[avg-of-k]/mae` and `pass@k/mae` — Skills-parity MAE over the
  ordinal `GRADE_TO_SCORE = {correct:7, almost:6, partial:1, incorrect:0}`
  mapping
* `mae` / `mae_count` — Skills-parity all-rollouts pooled MAE

## Upstream source

* CSV: [google-deepmind/superhuman/imobench/gradingbench.csv](https://github.com/google-deepmind/superhuman/blob/c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8/imobench/gradingbench.csv)
* Paper: [IMO-Bench](https://arxiv.org/abs/2511.01846), Appendix 7.3
* Skills counterpart: `nemo_skills/dataset/imo-gradingbench`
