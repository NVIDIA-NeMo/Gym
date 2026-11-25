# SWE Atlas Q&A resources server

This resources server retains data preparation and the upstream rubric evaluator for SWE Atlas Codebase Q&A.
Run `python resources_servers/swe_atlas_qa/prepare.py` from the repository root to materialize the evaluation
rows from the pinned upstream source. The [benchmark README](../../benchmarks/swe_atlas_qa/README.md) records
historical results from the retired run presets.

No shipped agent can run these tasks yet. `/seed_session` returns the task sandbox as an untyped
`sandbox_descriptor`, which only the retired `harness_agent` read; session agents paired through
`single_agent_turn` expect the sandbox through `sandbox_access` and refuse the extra seed field. A run needs this
server to lend its sandbox first.

## Configuration

`configs/swe_atlas_qa.yaml` reads these values. Set them in `env.yaml` or on the command line:

| Key | Value |
|---|---|
| `judge_base_url`, `judge_model_name` | Rubric judge, reachable from the evaluator sandbox. Required. |
| `judge_api_key` | Judge key, default `unused`. |
| `sandbox` | Sandbox provider config, for example `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`. |

Task sandboxes may reach only the model host, so the Gym model server must listen on an address they can reach
(`+use_absolute_ip=true`); a seed with a loopback model host fails. When they cannot reach this machine, set
`swe_atlas_qa_resources_server.resources_servers.swe_atlas_qa.sandbox_model_base_url` to a relay URL.

`/seed_session` creates the task sandbox from the task's pinned image with `/app` as its working directory.
`/verify` collects `/logs/agent/answer.txt`, closes the agent sandbox, and runs the
upstream evaluator in a separate sandbox. Rubrics, reference answers, and judge credentials never enter the agent
sandbox, and agent network access is limited to the policy-model host.

## Scoring

`reward` is the upstream strict pass. `agg_score` is the fraction of scored rubrics passed. As upstream, unscored
rubrics are excluded from strict pass. They are reported in `num_unscored` and `all_rubrics_scored`. Missing answers
score zero. Infrastructure and verifier failures set `mask_sample=true` and `evaluation_completed=false`, with
`failure_kind` `session_lost` when the task session is missing, `judge_failed` when the evaluator scores no rubric
because every judge attempt failed, and `verifier_error` otherwise.

## Tests

```bash
pytest resources_servers/swe_atlas_qa/tests
gym env test +entrypoint=resources_servers/swe_atlas_qa +should_validate_data=true
```

## Attribution

The dataset and evaluator are Scale AI's SWE Atlas, licensed under Apache-2.0. Cite *SWE Atlas: Benchmarking Coding
Agents Beyond Issue Resolution* ([paper](https://arxiv.org/abs/2605.08366)). Task images and harness runtimes keep
their own licenses.
