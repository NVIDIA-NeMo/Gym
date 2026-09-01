# EnterpriseOps-Gym Benchmark

Oracle-mode public split of [ServiceNow EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
(Apache 2.0): stateful enterprise tool-use tasks across 8 domains (Calendar, CSM, Drive, Email,
HR, ITSM, Teams, Hybrid), graded by SQL verifiers on final database state. See
`resources_servers/enterpriseops_gym/README.md` for architecture and parity notes.

## Prerequisites

1. The upstream MCP gym Docker containers running (7 domain servers; see the resources
   server README for ports) and the EOG checkout with `gym_dbs.zip` unzipped
   (`seed_sql_root` config).
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
    "++enterpriseops_benchmark_resources_server.resources_servers.enterpriseops_gym.seed_sql_root=<abs path to EOG checkout>"

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
