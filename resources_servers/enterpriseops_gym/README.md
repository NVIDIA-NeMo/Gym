# EnterpriseOps-Gym resources server

Runs the tasks of ServiceNow's [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym) against live
enterprise systems in sandboxes. A task names one or two *gym servers* (Calendar, CSM, Drive, Email, HR, ITSM, Teams):
each is an MCP server over a SQLite database seeded for the task. The agent works through the task's tools, and the
task is scored on the final database state with the task's SQL verifiers. `benchmarks/enterpriseops_gym/` pairs this
server with the Stirrup agent; see its README for the benchmark and its results.

## How it works

1. **`/seed_session`** starts one sandbox per gym server of the task, from the image `gym_images` pins for it, with the
   gym's HTTP server as the sandbox entrypoint and its port declared. Once `/health` answers through
   `sandbox.endpoint(port)`, it seeds a fresh database from the task's SQL dump (`/api/seed-database`). The dumps come
   from the upstream `gym_dbs.zip`, downloaded once and checked against `seed_archive_sha256`. The first sandbox is
   lent to the agent (`sandbox_access`), which runs its harness there.
2. **`POST /{tool_name}`** serves the task's tools (the oracle tools of the row's `tool_servers`, nothing else). Each
   call goes to the gym that owns the tool as an MCP `tools/call`, with the task database's `x-database-id` and the
   task's context headers (the acting user's token). The gym's text answer is returned. A tool error is a 400 whose
   detail is the gym's message, which the model sees; an unreachable gym is a 502, which fails the episode for a retry.
   A call that reached the gym is never resent, since tools change the database.
3. **`/verify`** runs every verifier query against its gym's database (`/api/sql-runner`) and compares the answer with
   the expected value. The reward is 1 only when every verifier passes; `verifier_pass_rate` is the share that pass and
   `verifier_results` holds each one's expected and actual value. A query the database rejects fails its verifier, as
   upstream scores it, and the response sets `evaluation_completed: false` and `failure_kind: verifier_error`, so the
   row is not mistaken for a task the model failed. A gym that cannot be reached fails the request, so the episode is
   retried rather than scored. Verification stops the sandboxes, and a repeated `/verify` returns the same results.
4. **`/close_session`** stops any sandbox still running. Every sandbox also has a TTL (`sandbox_config.ttl_s`).

A seed or close that conflicts with the session's episode, or a seed after its session closed, is a final 409; a task
the server cannot run is a final 422. For a session the server no longer holds, a tool call is a 404
(`session_lost`) and `/verify` returns a masked sample, since no database is left to score.

Aggregate metrics report `success` and `verifier_pass_rate` overall and per domain (`<domain>/pass@1[avg-of-k]/...`).

## Task fields

Rows are written by `benchmarks/enterpriseops_gym/prepare.py`; `task_data.py` is their schema. Besides
`responses_create_params` (the task's system and user prompts and its tool schemas), the server reads:

| Field | Meaning |
|---|---|
| `task_id`, `domain` | Upstream task ID and domain (`hybrid` tasks use two gym servers). |
| `gym_servers` | `[{name, seed_database_file, headers}]`: each gym server's dump in the seed archive and the lowercase `x-` context headers of its calls, in the task's order. |
| `tool_servers` | `{tool: gym server}` for every tool the agent may call. |
| `verifiers` | Upstream `database_state` verifiers: `{name, gym_name, validation_config: {query, expected_value, comparison_type}}`. |

## Verifier semantics

Extraction and comparison follow upstream `benchmark/verifier.py`, checked against goldens produced by running it
(`tests/verifier_golden.json`, regenerated with `tests/generate_verifier_golden.py`): a one-row, one-column answer is
compared as its value, `equals` is Python equality, `greater_than`/`less_than` are Python ordering (values that cannot
be ordered fail), and `contains` is a substring test. The queries run one at a time, as upstream runs them. The
verifier fixture (`VERIFIER_FIXTURE`, `tests/verifier_cases.jsonl`) scores the same way against final database states
built in SQLite, so it needs no gym servers. Two differences from the upstream harness, both deliberate:

- **Every verifier counts.** Upstream keys results by verifier name, so the last of several same-named verifiers
  overwrites the others (34 public tasks have same-named verifiers with different checks). Here a task passes only when
  all of its verifiers pass, as Artificial Analysis grades EnterpriseOps-Gym-AA.
- **Numbers equal their numeric strings** in `equals`, `greater_than` and `less_than`. 20 public oracle tasks expect
  `"1"` from a `COUNT(*)`, which upstream's strict equality never matches; the same tasks in the dataset's
  distractor-tool configs (`plus_5_tools`, `plus_10_tools`, `plus_15_tools`) expect `1`.

## Configuration

| Key | Meaning |
|---|---|
| `sandbox_provider` | The top-level sandbox provider block, e.g. `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`. |
| `sandbox_config` | `ttl_s`, `resources`, `env`, `metadata` and `provider_options` of every gym sandbox. |
| `gym_images` | Per gym server name: `image`, `command` (the entrypoint that starts its HTTP server) and `port`. The defaults pin the upstream images by digest. |
| `agent_workdir` | Working directory of the sandbox lent to the agent. |
| `seed_archive_url`, `seed_archive_sha256` | The upstream task database dumps; cached under `seed_archive_dir`, by default `<cache_dir>/enterpriseops_gym`. |
| `gym_ready_timeout_s`, `gym_request_timeout_s` | Bound on a gym becoming healthy, and on each request to it. |
| `verify_endpoint_tls` | Verify the TLS certificate of sandbox endpoints; off by default like the OpenSandbox provider's `tls_verify`. |

The upstream gym images are published only on Docker Hub, for amd64, by a personal account
(`shivakrishnareddyma225/enterpriseops-gym-mcp-<domain>`). A deployment that mirrors them keeps the digests and maps
the names with the provider's image rewrites (`images.rewrites` in the OpenSandbox config).

The server keeps sessions in its process, so it requires `num_workers: 1`.

## Licensing

- Code: Apache 2.0.
- EnterpriseOps-Gym (task data, gym server images, database dumps): Apache 2.0, by ServiceNow.
- Dependencies: `nemo_gym` (Apache 2.0).
