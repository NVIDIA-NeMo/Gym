# EnterpriseOps-Gym Resources Server

Adapts the [ServiceNow EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
benchmark (Apache 2.0) to NeMo Gym: 1,150 stateful enterprise tool-use tasks across 8 domains
(Calendar, CSM, Drive, Email, HR, ITSM, Teams, Hybrid), executed against the upstream
Dockerized MCP gym servers and graded by SQL verifiers on final database state.

## Architecture

- The upstream MCP gym servers stay external (Docker containers, one per domain). This server
  is a thin adapter: per-rollout `/seed_session` seeds a fresh database from the task's SQL
  snapshot and pins `{gym -> database_id}` to the session cookie; a catch-all `POST /{tool_name}`
  proxies tool calls with the session's `x-database-id` and task context headers; `/verify`
  runs the task's verifiers (ported verbatim from EOG) and deletes the databases.
- `mcp_client.py` — pooled-aiohttp port of EOG's MCP/JSON-RPC client (one MCP session per gym
  server; per-call database ids; in-memory seed SQL cache).
- `verifier_engine.py` — line-for-line port of EOG's verifier engine (`database_state`,
  `response_check` LLM judge, `tool_execution`), including its **name-collapse scoring quirk**:
  duplicate-named verifiers overwrite each other and only the last one per name is scored.
  The headline `reward` preserves that for leaderboard parity; strict every-verifier metrics
  (`strict_success`, `strict_pass_rate`) are emitted alongside for RL reward shaping
  (`strict_verifiers: true` switches the reward to strict).
- `convert_tasks.py` / `snapshot_tools.py` — convert EOG task JSONs (local or the
  `ServiceNow-AI/EnterpriseOps-Gym` HF dataset) into NeMo Gym JSONL rows, baking in tool
  schemas from per-domain `tools/list` snapshots.

## Prerequisites

1. The EnterpriseOps-Gym checkout with `gym_dbs.zip` unzipped (`seed_sql_root` config points at it).
2. The upstream MCP gym Docker containers running (default ports: csm 8001, teams 8002,
   calendar 8003, email 8004, itsm 8006, hr 8008, drive 8009).

## Usage

```bash
# Snapshot tool schemas from a running gym server (one-time, per domain)
python resources_servers/enterpriseops_gym/snapshot_tools.py \
    --gym-url http://localhost:8001 --gym-name sn-csm-server \
    --output resources_servers/enterpriseops_gym/data/tools/csm.json

# Convert EOG task JSONs to a NeMo Gym dataset
python resources_servers/enterpriseops_gym/convert_tasks.py \
    --tasks-dir ../enterpriseops-gym/data/revised/csm \
    --tools-snapshot resources_servers/enterpriseops_gym/data/tools/csm.json \
    --domain csm --mode oracle \
    --output resources_servers/enterpriseops_gym/data/csm_revised.jsonl

# Run servers + collect rollouts
ng_run "+config_paths=[resources_servers/enterpriseops_gym/configs/enterpriseops_gym.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
ng_collect_rollouts +agent_name=enterpriseops_gym_simple_agent \
    +input_jsonl_fpath=resources_servers/enterpriseops_gym/data/csm_revised.jsonl \
    +output_jsonl_fpath=results/enterpriseops_csm.jsonl
```

## Parity notes

Ported bug-for-bug from EOG (do not "fix" here; see `verifier_engine.py` docstring):
SQL result extraction/comparison semantics, the verifier name-collapse, skipping verifiers
with unknown `gym_name`, judge prompts, and the model-observation encoding of tool results.
EOG judges `response_check` with the policy model itself; `judge_model_server` defaults to
`policy_model` to match.
