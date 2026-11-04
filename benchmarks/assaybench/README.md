# AssayBench

[AssayBench](https://arxiv.org/abs/2605.10876): given a plain-text description of a CRISPR screen, rank the
100 genes most likely to be hits. Scored with the paper's adjusted nDCG@100 by the authors' own metric
package. The paper's three cohorts share this one benchmark; pick the cohort at prepare time:

| `+prepare_script_args.split=` | Cohort | Rows |
|---|---|---|
| `test` (default) | temporal test split, the paper's Table 2 | 334 |
| `validation` | temporal validation split | 218 |
| `LaTest` | screens published after Sept 2025, the paper's memorization probe | 19 |

Everything about the task, the prompt (the reference harness's DSPy chain-of-thought framing), the metrics,
the published numbers, the sampling settings per model, and what does and does not line up with the paper
is in [`resources_servers/assaybench/README.md`](../../resources_servers/assaybench/README.md).

## Usage

```bash
# Fetch the pinned Hugging Face revision and write data/assaybench_benchmark.jsonl (test split, 56 MB)
gym eval prepare --benchmark assaybench
# ...or another cohort (overwrites the same file):
gym eval prepare --benchmark assaybench +prepare_script_args.split=validation

# Start servers
gym env start --benchmark assaybench --model-type vllm_model

# Collect rollouts: 5 runs per screen, the open-weight protocol. Sampling settings are per model;
# these are GPT-OSS-120B's (see the server README for the others).
gym eval run --no-serve --benchmark assaybench --model-type vllm_model \
    --temperature 1.0 --max-output-tokens 16000 \
    --output results/assaybench-test.jsonl
```

Read `pass@1[avg-of-5]/adjusted_ndcg@100` from the aggregate metrics and divide by 100 to compare with the
paper (GPT-OSS-120B: 0.1211 on the test split).
