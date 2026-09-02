# EnterpriseOps-Gym Benchmark

Oracle-mode public split of [ServiceNow EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
(Apache 2.0): stateful enterprise tool-use tasks across 8 domains (Calendar, CSM, Drive, Email,
HR, ITSM, Teams, Hybrid), graded by SQL verifiers on final database state. See
`resources_servers/enterpriseops_gym/README.md` for architecture and parity notes.

## Prerequisites

1. Apptainer on the evaluation host. Gym owns the seven domain services and materializes
   the pinned EnterpriseOps-Gym checkout and `gym_dbs.zip` itself; no external Docker
   containers or `seed_sql_root` setting are needed. On ARM64, build the seven native SIFs
   first as described in the resources-server README and set `ENTERPRISEOPS_NATIVE_SIF_DIR`.
2. Hub egress at prepare time (`huggingface.co`) for two datasets: the task split
   (`ServiceNow-AI/EnterpriseOps-Gym`) and the tool schemas
   (`nvidia/NeMo-Gym-EnterpriseOps-Assets`, fetched by the resources server's
   `prepare.py`). On machines without egress, set `NEMO_GYM_EOG_LOCAL_TASKS=<EOG
   checkout>/data/revised` for the tasks and `NEMO_GYM_EOG_TOOLS_DIR` for the schemas
   (see the resources server README for the `hf download` one-liner).

## Usage

### Tool-set splits

Four upstream tool-set modes are registered. `oracle` gives each task exactly the tools it
needs; the `plus_N_tools` splits add that many distractors, which is the difficulty ladder
the benchmark is designed around.

| `--benchmark` | tasks | mean tools/task |
|---|---:|---:|
| `enterpriseops` | 649 | 8.6 |
| `enterpriseops/plus_5_tools` | 637 | 12.6 |
| `enterpriseops/plus_10_tools` | 637 | ~17.6 |
| `enterpriseops/plus_15_tools` | 637 | ~22.4 |

Only the tool set differs: verifiers, scoring, and the resources server are identical across
modes. Note the `plus_N` splits ship **fewer tasks than oracle upstream** (637 vs 649 — `teams`
61→52, `calendar` 61→59, `csm` 103→102), so raw row counts are not comparable across modes.
Per-domain and macro success rates are, since those are rates.

```bash
gym eval prepare --benchmark enterpriseops

gym env start --benchmark enterpriseops --model-type openai_model \
    --model-url http://127.0.0.1:8000/v1 --model-api-key EMPTY \
    --model <served-model-name>

# Swap in a distractor split with, e.g.:
#   gym eval prepare --benchmark enterpriseops/plus_10_tools
#   gym eval run --no-serve --agent enterpriseops_plus_10_tools_simple_agent \
#     --input benchmarks/enterpriseops/data/enterpriseops_plus_10_tools_benchmark.jsonl ...
gym eval run --no-serve \
    --agent enterpriseops_benchmark_simple_agent \
    --input benchmarks/enterpriseops/data/enterpriseops_oracle_benchmark.jsonl \
    --output results/enterpriseops_oracle.jsonl
```

Scoring: `reward` = EOG leaderboard parity (all name-collapsed verifiers pass). Aggregate
metrics include per-domain `{domain}/success_rate`, `{domain}/verifier_pass_rate`, and
`macro_success_rate` (mean over domains, the leaderboard "Avg").
