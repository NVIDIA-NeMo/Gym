# EnterpriseOps-Gym Benchmark

Pinned oracle-mode public split of [ServiceNow EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
(Apache 2.0): 649 stateful enterprise tool-use tasks across 8 domains (Calendar, CSM, Drive,
Email, HR, ITSM, Teams, Hybrid), graded by SQL verifiers on final database state. See
`resources_servers/enterpriseops_gym/README.md` for architecture and parity notes.

## Prerequisites

1. The seven pinned MCP gym containers running and the pinned EOG checkout with
   `gym_dbs.zip` unzipped (`seed_sql_root` config). Exact pins and commands are in
   `resources_servers/enterpriseops_gym/RUNTIME.md`.
2. Hub egress at prepare time (`huggingface.co`). Preparation pins dataset revision
   `c8e538eae8a6205294f0a86675fefdc1fac408f6`, validates exact per-domain counts, and
   atomically writes the combined file. `NEMO_GYM_EOG_LOCAL_TASKS` is accepted only when
   it points to a complete eight-domain fallback, so a partial canonical file cannot be
   produced silently.

## Usage

```bash
gym eval prepare --benchmark enterpriseops

gym env start --benchmark enterpriseops --model-type openai_model \
    --model-url <OpenAI-compatible base URL> --model-api-key <API key> --model <model> \
    "++enterpriseops_benchmark_resources_server.resources_servers.enterpriseops_gym.seed_sql_root=<abs path to EOG checkout>"

gym eval run --no-serve --benchmark enterpriseops --model-type openai_model \
    --model-url <OpenAI-compatible base URL> --model-api-key <API key> --model <model> \
    --agent enterpriseops_benchmark_simple_agent \
    --input benchmarks/enterpriseops/data/enterpriseops_oracle_benchmark.jsonl \
    --output results/enterpriseops_oracle.jsonl
```

Scoring: `reward` = EOG leaderboard parity (all name-collapsed verifiers pass). Aggregate
metrics include per-domain `{domain}/success_rate`, `{domain}/verifier_pass_rate`, and
`macro_success_rate` (mean over domains, the leaderboard "Avg").
