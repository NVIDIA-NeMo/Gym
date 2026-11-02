# SWE Atlas Q&A resources server

Serves SWE Atlas Codebase Q&A for `harness_agent`. See the [benchmark README](../../benchmarks/swe_atlas_qa/README.md)
to prepare data and run evaluations.

`/seed_session` creates the task sandbox from the task's pinned image with `/app` as its working directory, and
`harness_agent` attaches to it. `/verify` collects `/logs/agent/answer.txt`, closes the agent sandbox, and runs the
upstream evaluator in a separate sandbox. Rubrics, reference answers, and judge credentials never enter the agent
sandbox, and agent network access is limited to the policy-model host.

## Scoring

`reward` is the upstream strict pass. `agg_score` is the fraction of scored rubrics passed. As upstream, unscored
rubrics are excluded from strict pass. They are reported in `num_unscored` and `all_rubrics_scored`. Missing answers
score zero. Infrastructure and verifier failures set `mask_sample=true` and `evaluation_completed=false`.

## Tests

```bash
pytest resources_servers/swe_atlas_qa/tests
gym env test +entrypoint=resources_servers/swe_atlas_qa +should_validate_data=true
```

## Attribution

The dataset and evaluator are Scale AI's SWE Atlas, licensed under Apache-2.0. Cite *SWE Atlas: Benchmarking Coding
Agents Beyond Issue Resolution* ([paper](https://arxiv.org/abs/2605.08366)). Task images and harness runtimes keep
their own licenses.
