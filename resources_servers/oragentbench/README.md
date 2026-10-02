# ORAgentBench benchmark environment

[Code and tasks](https://github.com/ORAgentBench/ORAgentBench) (code MIT; documentation and
data CC BY 4.0 per the upstream README), [paper](https://arxiv.org/abs/2606.19787).

End-to-end operations research: an agent reads an operational brief and multi-file data under
`/app`, writes a mathematical model, implements a PySCIPOpt solver, runs it, and writes a decision
artefact under `/app/submissions`. Upstream's own hidden validator then checks submission schema,
hard constraints and objective quality against a verified reference solution inside the same
container. 107 tasks: 32 easy / 41 medium / 34 hard. Eight tasks are Harbor multi-step tasks
(3 or 4 replanning steps with event notices between them).

Pinned revision (no upstream tag or release exists; this is a pin we chose):

| Artifact | Pin |
| --- | --- |
| `ORAgentBench/ORAgentBench` (tasks, validators, reference solutions, base Dockerfile) | `c9eb952435a4352f33daa2a35efe0f8c76d31b28` |

## Scope

| What | Value |
| --- | --- |
| Tasks | 107, one Gym row each; `difficulty` from upstream `difficulty.json` |
| Agent budget | 2700 s per single-step task; per-step `steps.agent.timeout_sec` for multi-step tasks (2700 / 1200 / 1200 / 1200), from `task.toml` |
| Verifier budget | 360 s (single-step) / 300 s (per step), from `task.toml` |
| Container | 4 CPUs, 8192 MiB, no network, per `task.toml` and the paper's "no internet access" |
| Reward | 1.0 iff `feasibility > 0` and `quality / 2 > 0.4` (the paper's pass predicate), else 0.0 |

The reward is a literal port of `scripts/summarize_results.py::is_pass` at the pinned commit.
Upstream's `reward.json` carries `quality` on [0, 2], scored by `tests/test.sh` in one of two
regimes: when the reference comes with a best bound whose distance from the reference objective
exceeds the MIP gap, 1.0 = the reference objective and 2.0 = reaches the best bound
(`quality_status: scored_with_best_bound`); when the reference is proven optimal within the MIP
gap or no bound is reported, matching the reference scores 2.0 and shortfalls are scaled by the
MIP gap (`bound_gap_below_mip_gap`, `scored_with_mip_gap_fallback`; the committed example
rollouts are in this regime). The paper's `q` on [0, 1] is `quality / 2` and is returned as
`quality`. For multi-step tasks feasibility is the conjunction over steps and quality the mean
over steps with a missing step counting 0 (`summarize_trial`). `upstream_scalar_reward` is the
`(F + q) / 3` value upstream writes to `reward.txt`; it is returned for reference and is not the
reward.

Every verify response carries a `status`. A zero reward that is not a judgement on the policy
also carries a `failure_reason` and `harness_failure: 1.0`:

| Status | Meaning | Harness fault |
| --- | --- | :---: |
| `scored` | Validator ran and wrote `reward.json` for every step | no |
| `missing_solution` | Validator found no solution file (upstream `quality_status`) | no |
| `verifier_timeout` | `tests/test.sh` exceeded the task's verifier budget | no |
| `verifier_output_missing` | `test.sh` returned but wrote no readable `reward.json` | no |
| `step_aborted` | A step fell below its `min_reward` gate; later steps count 0 | no |
| `step_incomplete` | The agent loop ended before every step was verified | no |
| `bad_task_folder` | `task.toml`, `instruction.md` or `tests/` missing or unreadable | yes |
| `sandbox_failed` | Task container could not start (model-free modes only) | yes |
| `step_setup_failed` | A step's `workdir/setup.sh` exited non-zero before the agent ran | yes |
| `tests_upload_failed` | The step's `tests/` could not be copied into the container | yes |
| `verifier_exec_failed` | The container runtime could not run `tests/test.sh` (`docker exec` runtime error, or the server raised while running it); no validator verdict exists | yes |
| `no_session` | `verify()` called without a seeded session | yes |

Verifier timeouts and missing verifier output are charged to the policy because an agent's
artefact or container changes can cause both. Harness-fault rows are not masked: they stay in
the `pass_rate/<band>` denominator at reward 0 (as the Surveyor how-to and `terminal_bench_2_1`
do) and are distinguishable through `mean/harness_failure` and the row's `failure_kind`
(`provider_unavailable`, `verifier_error`, `session_lost`, or `oragentbench:<status>`), so a
run with infrastructure incidents reads as a lower bound rather than a clean score. This is a
deliberate departure from the `mask_sample` guidance in `nemo_gym/base_resources_server.py`,
which names infrastructure failures (a lost session, an unavailable judge, an OOM-killed
container) as the cases to mask: the published pass rates are over all 107 tasks, so every task
stays in the denominator and a harness incident costs the run its point rather than shrinking
the sample; the incident is surfaced through `mean/harness_failure` and `failure_kind` instead.
A consumer who wants harness rows excluded can filter on `harness_failure == 1.0` or on
`failure_kind`.

Headline selection: `mean/reward`, `mean/feasibility`, `mean/quality`, `mean/harness_failure`,
token counts, and the per-stratum `pass_rate/<band>`, `feasibility_rate/<band>`,
`mean_quality/<band>`, `count/<band>` are promoted to `key_metrics`; published baselines differ
by more than 2x across strata, so the pooled figure alone describes no published quantity.

## Agent path and multi-step tasks

The agent is Gym's sandboxed Terminus 2 loop
(`responses_api_agents/terminus_2_sandboxed_agent`), run once per Harbor step by
`responses_api_agents/terminus_2_multi_step_sandboxed_agent`. The server owns the task files and
the container: `/seed_session` starts the container and returns the step list, `/prepare_step`
uploads a step's `workdir` and runs its `setup.sh` (Harbor's per-step preparation),
`/verify_step` runs that step's validator in the same container (the step verifiers write the
carried-forward state the next step reads), and `/verify` aggregates. Single-step tasks are the
one-step case; the stock `terminus_2_sandboxed_agent` also works for them.

Harbor 0.22.0 gates a float `min_reward` on the `reward` key of `reward.json`, which upstream's
`reward.json` does not have (it has `feasibility` and `quality`), so a literal Harbor run aborts
every multi-step task after its first gated step. This server gates on the `(F + q) / 3` scalar
upstream writes to `reward.txt`, which makes `min_reward = 0.3` mean "continue only when the
step is feasible", the evident intent.

Upstream's `task.toml` declares `[environment] skills_dir = "/skills"`; each task image carries
four base skills there and Terminus 2 lists them in the initial prompt (`skills_dir: /skills`),
so the setting is zero-shot with a skill library, as upstream's harnesses had.

## Departures from upstream

- **Scaffold.** Upstream's published rows come from the Codex harness (GPT models) and the
  Claude Code harness (others). Results from this environment measure a different agent
  system and are a comparison, not a reproduction.
- **Image.** `image/Dockerfile` is upstream's `docker/base/Dockerfile` verbatim plus one layer
  installing `tmux` and `procps`, which Terminus 2 needs and the no-network container cannot
  fetch. Upstream pins no package versions (`FROM python:3.11-slim` and unversioned `pip install`),
  so the solver stack is whatever `pip` resolves on build day, and a different SCIP build can
  move `quality`. The build that produced the recorded validation results (reference sweep,
  negative controls, example rollouts) resolved the versions below (author-reported, resolved
  2026-09-22); `prepare_oragentbench.py --build-images` prints the same probe for the image it
  just built so a rebuild can be compared against this table.

  | Component | Resolved version |
  | --- | --- |
  | Base image | `python:3.11-slim` (Debian 13) |
  | PySCIPOpt | 6.2.1 |
  | SCIP | 10.0 |
  | numpy | 2.4.6 |
  | pandas | 3.0.6 |
  | scipy | 1.17.1 |
  | networkx | 3.6.1 |
  | pyomo | 6.10.1 |
- **Network.** Containers run with `--network none`. `task.toml` says `allow_internet = true`
  but the instruction and the paper both forbid internet use.
- **min_reward gating** as described above.

## Model-free validation

Set `validation_mode` and post every row to `/verify` with
`scripts/validate_model_free.py`; no model is involved.

| Mode | What runs in place of the agent | Expected |
| --- | --- | --- |
| `reference` | upstream `solution/solve.sh` per step | reward 1.0 on every task |
| `no_action` | nothing | reward 0.0, `missing_solution` |
| `wrong_file` | reference solve, then every new artefact renamed `*.wrong` | reward 0.0, `missing_solution` |
| `hung_process` | a background `sleep`, then an exec that times out | reward 0.0; container still removed |

Two knobs exist for the gold sweep only and never touch what an agent sees.
`REFERENCE_SOLUTION_PATCHES` in `app.py` fixes upstream's `sterile_processing_robust_schedule`
reference solver, which at the pin references two undefined names and crashes before writing
a solution. `reference_solve_time_limit_s` overrides the `ORCLAW_SOLVE_TIME_LIMIT_SECONDS`
(300) that `solve.sh` passes to the reference solver; several hard-task references run to that
cap and the objective they reach depends on the host, so a longer limit separates "the
validator is wrong" from "this host does not reach `reference_metrics.json` in 300 s".

Results of these sweeps over all 107 tasks are recorded in the Surveyor implementation note,
not here.

## Setup

```bash
cd resources_servers/oragentbench
uv venv --python 3.13
uv pip install -r requirements.txt
cd ../..
source resources_servers/oragentbench/.venv/bin/activate
```

**Every command below runs from the repository root with that environment active** and needs a
local Docker daemon. Preparation clones upstream at the pin into
`resources_servers/oragentbench/data/ORAgentBench/` (ignored by git) and builds the base image
plus one image per task.

## Quickstart

```bash
python resources_servers/oragentbench/scripts/prepare_oragentbench.py --build-images

# 1. start servers (leave running); env.yaml supplies policy_base_url / policy_api_key /
#    policy_model_name, and ORAGENTBENCH_POLICY_MODEL overrides the model id.
gym env start --config resources_servers/oragentbench/configs/oragentbench.yaml

# 2. collect rollouts against them
gym eval run --no-serve \
    --agent oragentbench_agent \
    --input resources_servers/oragentbench/data/benchmark.jsonl \
    --output results/oragentbench.jsonl --concurrency 6
```

Regenerate the committed example artifacts. `example.jsonl` is five synthetic rows written for
this repository against the fixture task under `tests/fixtures/toy_assignment`; upstream's task
packets are third-party content and are not redistributed here. Two stages: the script writes
preparer-schema rows (`agent_ref`), and `gym dataset collate` rewrites the file in place into the
collated schema (`task_source`) and writes `example_metrics.json`. The tracked `example.jsonl`
carries the collated schema.

```bash
docker build -t oragentbench-fixture:toy_assignment resources_servers/oragentbench/tests/fixtures/toy_assignment/environment
python resources_servers/oragentbench/scripts/make_example_data.py
gym dataset collate \
    "+config_paths=[resources_servers/oragentbench/configs/oragentbench.yaml]" \
    +output_dirpath=resources_servers/oragentbench/data \
    +mode=example_validation
```

## Tests

```bash
gym env test +entrypoint=resources_servers/oragentbench +should_validate_data=true
```

## Licensing

Code: Apache 2.0.

ORAgentBench: code MIT (root `LICENSE`, (c) 2026 ORAgentBench); the upstream README states that
benchmark documentation and data are CC BY 4.0. Task content derives from OR papers, IndustryOR
seeds, public datasets and authored scenarios whose provenance upstream asserts but does not
itemise, and a licence declaration is not a rights determination, so the benchmark dataset is
declared `license: TBD` in the config until a rights decision is recorded. Tasks, validators and
reference solutions are fetched at preparation time and not committed. `tests/fixtures/toy_assignment/tests/test.sh` is a verbatim copy of upstream's shared
scoring script (MIT) with a provenance header; the fixture task around it is synthetic.
