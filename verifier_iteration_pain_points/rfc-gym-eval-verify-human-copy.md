epic: recompute rewards from existing rollouts — `gym eval verify` (#987)

Author: Martyna Patelka
GitHub: https://github.com/NVIDIA-NeMo/Gym/issues/987
Linear: https://linear.app/nvidia/issue/FEP-1195/can-you-recompute-reward-using-existing-rollout-and-a-new-verifier

# Background

## Pain points

### 1. Rewards are frozen at collection time; any verifier change re-pays all policy inference

Rollout collection fuses two very differently priced halves into a single `/run` request: the
expensive half (policy inference + the multi-step agent loop) and the cheap half (the verifier
scoring the finished trajectory). The reward is written into `rollouts.jsonl` at collection time.
If you then change a verifier hyperparameter — a grading mode, a threshold, a judge prompt — the
only supported way to get new rewards is to re-run `gym eval run`, which re-pays all of the policy
inference. 

For many stateless environments the data needed to re-verify is already on disk (`rollouts.jsonl` + `*_materialized_inputs.jsonl`),
and every resources server exposes a plain HTTP `/verify` endpoint — but there is no command that replays
stored rows through it.

### 2. Re-running judge failures means re-running inference

Re-running a full eval just to retry the judge wastes the expensive inference pass, which is already complete.
With limited judge quota (e.g. rate-limited endpoints), re-running everything burns the budget on samples that already scored fine.

Related issue: https://github.com/NVIDIA-NeMo/Gym/issues/1606


## Architecture


### Artifacts on disk after a run

What `gym eval run` writes:

| File | Content |
|---|---|
| `rollouts_materialized_inputs.jsonl` | final `/run` request rows; row identifier `(_ng_task_index, _ng_rollout_index)`; name of the agent server used for each row |
| `rollouts.jsonl` | one verify response per rollout; row identifier `(_ng_task_index, _ng_rollout_index)`; name of the agent server used for each row; the frozen rewards **and** the expensive `response` |
| `rollouts_failures.jsonl` | one row per failed attempt (`_ng_failure_class`), today fed only by `stirrup_agent` |
| `rollouts_aggregate_metrics.json` (unless `disable_aggregation` is set) | per-agent metrics |


Together, `rollouts_materialized_inputs.jsonl` and `rollouts.jsonl` contain everything a stateless verifier needs and can be joined losslessly on
`(_ng_task_index, _ng_rollout_index)`.


### The stirrup precedent: one team already built this in NeMo Gym

`stirrup_agent` is the `responses_api_agent` for GDPVal-style environments: the model spends a long, expensive
multi-step session producing deliverable *files* (documents, spreadsheets), which the `gdpval`
resources server then judges. The stirrup team solved it for
themselves with three config flags (`execute_only`, `judge_only`, `rerun_incomplete`,
[stirrup_agent/app.py:821-858](../responses_api_agents/stirrup_agent/app.py#L821)) that split one
environment's run into two phases:

- **Expensive phase.** With `persist_deliverables_dir` set, every rollout saves its deliverables; `execute_only: true`
  additionally skips judging entirely — generate now, score later.
- **Cheap phase.** A later run with `judge_only: true` skips the agent entirely and scores the cached deliverables from `deliverables_dir`.

#### Comment
That is a working judge-only resume and verifier-iteration loop — for exactly one agent/environment
pair. Its limits are what make it a precedent rather than a solution: the replay contract is
stirrup's private deliverables tree (you must have set `persist_deliverables_dir` and other flags correctly *before* the original run — nothing recovers a run that didn't), not the rollout artifacts every Gym run already
writes. The proposed solution below is the same two-phase idea, generalized:
keyed on `rollouts.jsonl` + `*_materialized_inputs.jsonl` (available for any past run), no agent
or model server in the loop, one implementation for every stateless environment.

# Proposed solution

## New command: `gym eval reverify`

Replay stored rollouts through resources-server `/verify` and write a fresh rollouts file plus
recomputed aggregate metrics (unless disabled).

### Iterate on a verifier hyperparameter (pain point 1) 
One command, one terminal. First way of running it (A):

```bash
gym eval reverify \
  --config resources_servers/mcqa/configs/mcqa.yaml \
  "++mcqa.resources_servers.mcqa.grading_mode=lenient_boxed" \
  --inputs results/rollouts_materialized_inputs.jsonl \
  --rollouts results/rollouts.jsonl \
  --output results/rollouts_lenient.jsonl
```
This starts **only** the mcqa resources server with the overridden config.
For convenience, the same config files previously used for `gym env start` will work too. Second way (B):
```bash
gym eval reverify  \
  --config benchmarks/birdbench/config.yaml \
  --inputs birdbench_results/birdbench_rollouts_materialized_inputs.jsonl \
  --rollouts birdbench_results/birdbench_rollouts.jsonl \
  --output birdbench_results/birdbench_rollouts_t120.jsonl \
  ++birdbench_bird_sql_resources_server.resources_servers.bird_sql.sql_execution_timeout_s=120
```

It won't be possible to use the `--no-serve` flag.

### Judge failures (pain point 2)
After fixing the judge, re-verify only the failed rows:

```bash
gym eval reverify \
  --config resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml \
  --judge-failed-only \
  --inputs results/rollouts_materialized_inputs.jsonl \
  --rollouts results/rollouts.jsonl \
  --output results/rollouts_recovered.jsonl
```

Successful rows are copied; rows with the judge-failure flag (to be added in https://github.com/NVIDIA-NeMo/Gym/issues/1606) are re-verified.

## When reverify is not possible
For some resources servers this simple solution is not possible. For example,
`aviary` calculates the reward as an aggregation over all steps, so reverify can't simply call `verify` with a request built from rollouts and materialized inputs.

We need a way to mark which servers support reverify in the simple form added in this epic. For now an enum is enough: `reverify_mode` with two values, defaulting to `UNSUPPORTED`; servers where the simple form is correct flip it to `STATELESS`.

A new class attribute, `reverify_mode`, plus a lightweight endpoint returning its value, can be used to
decide whether reverify is possible:
```python
class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    config: BaseResourcesServerConfig
    reverify_mode: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED # or only STATELESS for now

    ....

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass
```
## Force

A server whose `reverify_mode` is `UNSUPPORTED` can't guarantee correct re-verification. Sometimes we want to run
it anyway — for example when approximate rewards are good enough for an exploratory run. The `--force` flag overrides the guard:
```bash
gym eval reverify --force ...
```
With `--force`, a warning is printed before the replay starts
and repeated in the closing summary, and all output filenames are prefixed with `unsafe_`
(e.g. `unsafe_rollouts_lenient.jsonl`, `unsafe_rollouts_lenient_aggregate_metrics.json`) so the
artifacts stay clearly marked long after the console output is gone.

### How to set it correctly for all servers?
There are 95 resources servers that need this value set, so the plan is: default everything to `UNSUPPORTED`, generate an AI-proposed list of `STATELESS` candidates, and then use human feedback from #swdl-nemo-gym-core-devs to sign off on the value for each server.

## Points which need agreement

1. **Per-agent routing.**
A rollouts file may mix rows from several agents. During rollout collection the appropriate agent is picked by `server_name=row["agent_ref"]["name"]` and each agent instance then sends requests to the single resources server named in its own config block. The simplified reverify config case (A) does not contain agent blocks, so it is not possible to distinguish between different resources servers. When multiple values of `["agent_ref"]["name"]` are present in the data, this config format is insufficient and should raise an error. Config case (B) preserves the agent config blocks, and each block names its resources server (`resources_server.name`), so rows can be routed directly: `row["agent_ref"]["name"]` → agent block → its resources server ref — no naming conventions needed. I'd like confirmation that this plan is correct, and agreement on whether multi-agent rollouts files are in scope for this issue.

2. **Per-row calls.** POST `/verify` directly; `/seed_session` is **not** called. Resources servers that need it for scoring are mostly not `STATELESS` anyway.


# Out of scope

1. **Fully self-contained agents** that don't use resources servers at all and own everything, including the verifier (e.g. SWE), won't have re-verification enabled.

2. **Preventing inconsistent scores with `--judge-failed-only`**. When running:
    ```
    gym eval reverify \
      --config ... \
      --judge-failed-only \
      ...
    ```
if the judge configuration is changed, only the re-verified rows reflect it, and the file might end up with inconsistent scores. This could be addressed by always dumping the config with the results and validating it when running reverify, but I would put it out of scope for this issue.


