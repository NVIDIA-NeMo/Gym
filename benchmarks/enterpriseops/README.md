# EnterpriseOps-Gym Benchmark

Oracle-mode public split of [ServiceNow EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
(Apache 2.0): stateful enterprise tool-use tasks across 8 domains (Calendar, CSM, Drive, Email,
HR, ITSM, Teams, Hybrid), graded by SQL verifiers on final database state. See
`resources_servers/enterpriseops_gym/README.md` for architecture and parity notes.

## Prerequisites

1. The upstream MCP gym Docker containers running (7 domain servers; see the resources
   server README for ports) and the EOG checkout with `gym_dbs.zip` unzipped
   (`seed_sql_root` config).
2. Hub egress at prepare time (`huggingface.co`). On machines without egress, set
   `NEMO_GYM_EOG_LOCAL_TASKS=<EOG checkout>/data/revised` to convert the locally
   committed task subset instead.

## Usage

```bash
gym eval prepare --benchmark enterpriseops

gym env start --benchmark enterpriseops --model-type openai_model \
    "++enterpriseops_benchmark_resources_server.resources_servers.enterpriseops_gym.seed_sql_root=<abs path to EOG checkout>"

gym eval run --no-serve \
    --agent enterpriseops_benchmark_simple_agent \
    --input benchmarks/enterpriseops/data/enterpriseops_oracle_benchmark.jsonl \
    --output results/enterpriseops_oracle.jsonl
```

Scoring: `reward` = EOG leaderboard parity (all name-collapsed verifiers pass). Aggregate
metrics include per-domain `{domain}/success_rate`, `{domain}/verifier_pass_rate`, and
`macro_success_rate` (mean over domains, the leaderboard "Avg").
