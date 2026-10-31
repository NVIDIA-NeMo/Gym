# EnterpriseOps-Gym benchmark

[EnterpriseOps-Gym](https://arxiv.org/abs/2603.13594) (ServiceNow) evaluates agents on stateful, multi-step planning
and tool use in enterprise workflows: Customer Service Management (CSM), HR, IT Service Management (ITSM), Email,
Calendar, Teams and Drive, plus Hybrid tasks that span two of them. The agent operates live systems through MCP tools
and is graded on the final state of their databases.

This benchmark runs the public oracle-mode tasks the way Artificial Analysis runs
[EnterpriseOps-Gym-AA](https://artificialanalysis.ai/methodology/intelligence-benchmarking#enterprise-ops-gym-aa): the
[Stirrup](https://github.com/ArtificialAnalysis/Stirrup) agent harness in a reason-and-act loop with a 100-turn cap,
only the tools each task needs, 3 repeats, and strict pass@1 (a task passes only when every verifier passes).

## How it works

```
┌──────────────────────┐  1. seed session        ┌─────────────────────────────┐
│  Environment server  │  4. /verify             │  enterpriseops_gym server   │
│                      │ ──────────────────────► │ task tools, SQL verifiers   │
└──────────┬───────────┘                         └──────┬───────────────▲──────┘
           │ 2. seed agent session                      │ starts, seeds │ task tools
           │    (sandbox + tool access)                 ▼               │
┌──────────────────────┐  3. installs Stirrup,   ┌──────────────────────┴──────┐
│    Stirrup agent     │     runs the episode    │  Gym server sandbox(es)     │
│                      │ ──────────────────────► │  MCP server + task database │
└──────────────────────┘                         └──────────────┬──────────────┘
                                                                │ model calls
                                                                ▼
                                                 ┌─────────────────────────────┐
                                                 │        Model server         │
                                                 └─────────────────────────────┘
```

1. The resources server starts one sandbox per gym server of the task from the upstream images and seeds a database
   for the task in each (see `resources_servers/enterpriseops_gym/README.md`).
2. The environment server lends the task's first sandbox to the Stirrup agent and grants it the resources server's
   tool routes.
3. Stirrup runs in that sandbox. Each tool call goes to the resources server, which forwards it to the gym server that
   owns the tool. The episode ends when the model calls Stirrup's `finish` tool, or after 100 turns.
4. `/verify` runs the task's SQL verifiers against the final database state and stops the sandboxes.

## Measurement contract

**Provenance**

- Tasks: [`ServiceNow-AI/EnterpriseOps-Gym`](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) at
  revision `c8e538e`, config `oracle`, all eight domain splits: 649 tasks (Calendar 61, CSM 103, Drive 64, Email 67,
  HR 102, Hybrid 88, ITSM 103, Teams 61), Apache 2.0. This is the public 60% of the benchmark; Artificial Analysis
  evaluates 1,117 tasks, so its numbers are not directly comparable.
- Tool schemas: the gym servers' `tools/list`, snapshotted in
  [`nvidia/NeMo-Gym-EnterpriseOps-Assets`](https://huggingface.co/datasets/nvidia/NeMo-Gym-EnterpriseOps-Assets) at
  revision `8918dc6`; identical to what the pinned images serve.
- Task databases: `gym_dbs.zip` from [ServiceNow/EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
  at `271f2c3`, pinned by sha256.
- Gym servers: the upstream images `shivakrishnareddyma225/enterpriseops-gym-mcp-<domain>` (amd64), pinned by digest.

**Task contract**

- Model input: the task's domain policy (`system_prompt`) as the system message and its `user_prompt` as the user
  message. Stirrup puts its own instructions first: complete the task with the tools in 100 steps, call a finish tool
  as the last step, and do not interact with the user.
- Tools: the task's oracle tools (`selected_tools` minus `restricted_tools`) with the gym servers' schemas, and
  Stirrup's `finish(reason, paths)`. No shell. Two tasks select a tool no image serves (`list_team_apps`,
  `remove_channel_member`) and run without it, as upstream.
- Each gym call carries the task database's ID and the task's context headers, mapped from the task's `context` as the
  upstream harness maps them.

**Scoring and run conditions**

- Reward is 1 when every verifier of the task passes. Aggregate metrics report `pass@1[avg-of-3]/success` (the
  headline), `pass@1[avg-of-3]/verifier_pass_rate` (the share of verifier checks passed), and both per domain.
- Verifier semantics follow upstream, with two deliberate differences (every verifier counts; numbers equal their
  numeric strings); see `resources_servers/enterpriseops_gym/README.md`.
- Harness: Stirrup in Gym's `stirrup_agent`, 100 turns, thinking on, `top_p` 0.95. The temperature defaults to 0.6,
  Artificial Analysis's setting for reasoning models; use the model lab's recommendation where there is one (see
  [Run](#run)).

## Prepare data

```bash
gym eval prepare --benchmark enterpriseops_gym
```

## Run

The gym servers run on OpenSandbox. Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`; put the scheme in the domain
(`https://...`) for a cell that only serves HTTPS. The agent runs inside a sandbox and calls the model and resources
servers from there, so the benchmark config sets `use_absolute_ip: true`, and the Gym host must be reachable from the
sandboxes.

```bash
gym eval run \
    --benchmark enterpriseops_gym \
    --model-type openai_model \
    --model nvidia/zai-org/glm-5.3-flash \
    --model-url https://inference-api.nvidia.com/v1 \
    --model-api-key <key> \
    --split benchmark \
    --concurrency 64 \
    --output results/enterpriseops_gym.jsonl \
    ++enterpriseops_gym_stirrup_agent.responses_api_agents.stirrup_agent.temperature=1.0
```

Tasks run as environment-server episodes (`taskset: enterpriseops_gym`), so sampling is set on the agent rather than
with `--temperature`, which overrides request fields of flat rows only. Add `++observability_enabled=true
++model_call_capture_dir=<absolute dir>` to record the model calls that rollout health checks and the harness evidence
gate read.

Each rollout uses one sandbox (two for Hybrid tasks) with 2 CPUs and 4 GiB of memory for the length of the episode.

## Results

GLM-5.3-Flash (`nvidia/zai-org/glm-5.3-flash`), temperature 1.0, 649 tasks x 3 repeats on OpenSandbox; 1,947 of
1,947 rollouts completed with no infrastructure failure in 44 minutes at concurrency 64.

| Domain | Success (this benchmark) | Success (EnterpriseOps-Gym-AA) |
|---|---:|---:|
| CSM | 30.7 | 47.3 |
| HR | 31.4 | 34.8 |
| ITSM | 37.2 | 38.1 |
| Email | 60.7 | 37.5 |
| Calendar | 36.1 | 5.8 |
| Teams | 51.4 | 7.0 |
| Drive | 54.2 | 58.1 |
| Hybrid | 36.4 | 24.2 |
| **Overall** | **40.5** | **33.2** |

- Verifier pass rate 74.2%; success per repeat 40.4, 39.8, 41.3. Two earlier full runs scored 41.0 and 41.0.
- A rollout averages 7.6 model calls, 12.2 tool calls and 2.3k output tokens (1.1k reasoning). Artificial Analysis
  reports 11.4 turns and 13.8k output tokens per task for this model.
- On the six domains other than Calendar and Teams, weighted by the per-domain task counts that Artificial
  Analysis's scores imply, the success rate is 39.2% here and 39.2% at Artificial Analysis. On Calendar and Teams,
  Artificial Analysis reports identical scores (5.8% and 7.0%) for GLM-5.3 and GLM-5.3-Flash, while its other models
  solve 36-52% there; that floor does not appear here.
- The run recorded model calls: all 1,947 rollouts are healthy and pass the `gym-p0/v1` evidence gate.

## Validation

The public task set has no reference solutions, so the integration is checked against the upstream harness's own
code instead:

- **Verifier goldens.** `resources_servers/enterpriseops_gym/tests/verifier_golden.json` records what upstream's
  `VerifierEngine` extracts from 12 kinds of `/api/sql-runner` answer and decides for 660 combinations of actual
  value, expected value and comparison. The server's verifier matches all of them except the 21 comparisons of a
  number with a numeric string.
- **Replay.** The first repeat of every task in the results run was replayed with upstream's `MCPClient`,
  `create_database_from_file` and `VerifierEngine` against fresh gym servers. 647 of 649 tasks replay (upstream
  cannot send the two whitespace headers). After normalizing generated IDs and timestamps, 99.8% of the tool outputs
  are identical; the others differ only in generated values (random IDs and the order they sort in, PDF bytes).
  Upstream's verifiers reach the same outcome on every task once numbers equal their numeric strings.
- **Do-nothing control.** Scored on the freshly seeded databases, 1 of 647 tasks passes, under both harnesses.

## Known issues in the public task set

| Issue | Tasks | Here |
|---|---:|---|
| A verifier expects a numeric string (`"1"`) from an integer column | 20 | Passable: numbers equal their numeric strings |
| A context header key has surrounding whitespace, which upstream cannot send | 2 | Passable: the whitespace is stripped |
| A verifier query fails (unknown column, or the table is on the task's other gym) | 3 | Unpassable; `evaluation_completed: false` |
| A selected tool is not served by the images | 2 | The task runs without it |
| The email image's `create_send_as_alias` fails with an internal server error for some inputs | 5 | As upstream |
